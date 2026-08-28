"""
Uplift model builders + scoring (S-, T-, X-learners).

Kept in src/ so notebooks and the Streamlit app share the same logic.
Base learner is a gradient-boosted tree by default; pass any sklearn-style
classifier/regressor (e.g. RandomForest, XGBoost) to swap.

Requires: scikit-learn (+ scikit-uplift for Qini metrics). No compiler needed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# Module-level factories (picklable — lambdas are not, which breaks model saving).
def default_clf():
    return GradientBoostingClassifier(random_state=0)

def default_reg():
    return GradientBoostingRegressor(random_state=0)


# ----------------------------------------------------------------------
# S-learner: one model, treatment as a feature, predict twice & difference
# ----------------------------------------------------------------------
class SLearner:
    def __init__(self, base=None):
        self.model = base if base is not None else GradientBoostingClassifier(random_state=0)

    def fit(self, X, treatment, y):
        Xt = X.copy()
        Xt["treatment"] = np.asarray(treatment)
        self.model.fit(Xt, y)
        return self

    def predict_uplift(self, X):
        X1 = X.copy(); X1["treatment"] = 1
        X0 = X.copy(); X0["treatment"] = 0
        p1 = self.model.predict_proba(X1)[:, 1]
        p0 = self.model.predict_proba(X0)[:, 1]
        return p1 - p0


# ----------------------------------------------------------------------
# T-learner: separate treated & control models, difference predictions
# ----------------------------------------------------------------------
class TLearner:
    def __init__(self, base_factory=None):
        self._mk = base_factory if base_factory is not None else default_clf
        self.m_t = self._mk()
        self.m_c = self._mk()

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        self.m_t.fit(X[treatment == 1], y[treatment == 1])
        self.m_c.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X):
        return self.m_t.predict_proba(X)[:, 1] - self.m_c.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------
# X-learner: hand-rolled from the meta-learner equations (pure sklearn,
# no compiler / no causalml). Chosen for the 2:1 treatment:control
# imbalance — the propensity-weighted blend leans on the better-estimated
# side, which is exactly where the X-learner beats the T-learner.
#
# Algorithm:
#   Stage 1  outcome models:   mu_t on treated, mu_c on control
#   Stage 2  impute effects:   D_t = Y - mu_c(X)   (treated rows)
#                              D_c = mu_t(X) - Y   (control rows)
#   Stage 3  effect models:    tau_t on D_t,  tau_c on D_c   (regressors)
#   Stage 4  blend:            tau = g*tau_c + (1-g)*tau_t   (g = propensity)
# ----------------------------------------------------------------------
class XLearner:
    def __init__(self, outcome_factory=None, effect_factory=None, propensity=None):
        self._mk_out = outcome_factory if outcome_factory is not None else default_clf
        self._mk_eff = effect_factory if effect_factory is not None else default_reg
        self.mu_t = self._mk_out()
        self.mu_c = self._mk_out()
        self.tau_t = self._mk_eff()
        self.tau_c = self._mk_eff()
        # propensity g(x) = P(treated). Constant by default (randomized experiment),
        # but accepts a fitted classifier for the general case.
        self.propensity = propensity
        self._g_const = None

    def fit(self, X, treatment, y):
        X = np.asarray(X); treatment = np.asarray(treatment); y = np.asarray(y)
        t_mask = treatment == 1
        c_mask = treatment == 0

        # Stage 1: outcome models
        self.mu_t.fit(X[t_mask], y[t_mask])
        self.mu_c.fit(X[c_mask], y[c_mask])

        # Stage 2: imputed treatment effects
        D_t = y[t_mask] - self.mu_c.predict_proba(X[t_mask])[:, 1]   # treated rows
        D_c = self.mu_t.predict_proba(X[c_mask])[:, 1] - y[c_mask]   # control rows

        # Stage 3: effect models (regressors on the imputed effects)
        self.tau_t.fit(X[t_mask], D_t)
        self.tau_c.fit(X[c_mask], D_c)

        # Stage 4 prep: propensity g(x)
        if self.propensity is None:
            self._g_const = float(treatment.mean())   # randomized -> constant
        else:
            self.propensity.fit(X, treatment)
        return self

    def predict_uplift(self, X):
        X = np.asarray(X)
        tau_t = self.tau_t.predict(X)
        tau_c = self.tau_c.predict(X)
        if self._g_const is not None:
            g = self._g_const
        else:
            g = self.propensity.predict_proba(X)[:, 1]
        # blend: weight each side by the OTHER group's share (lean on better-estimated side)
        return g * tau_c + (1.0 - g) * tau_t


def qini_auc(y_true, uplift, treatment):
    """scikit-uplift Qini AUC score (no compiler dependency)."""
    from sklift.metrics import qini_auc_score
    return qini_auc_score(np.asarray(y_true), np.asarray(uplift), np.asarray(treatment))
