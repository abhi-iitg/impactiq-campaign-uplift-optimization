"""
ImpactIQ: Campaign Intelligence & Uplift Optimization Engine.

An interactive decision-support application for:
- campaign impact analysis
- customer uplift scoring
- profit-optimized targeting

Run: streamlit run app/streamlit_app.py
Requires (once): python src/train_model.py


Run:  streamlit run app/streamlit_app.py
Requires (once):  python src/train_model.py   (trains + saves the X-learner)
"""
from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
sys.path.append(str(ROOT / "src"))

st.set_page_config(page_title="ImpactIQ | Campaign Intelligence",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------- styling -----------------------------
st.markdown("""
<style>
  :root { --ink:#10151c; --slate:#1d2731; --mist:#5b6b7a; --line:#dde3e9;
          --accent:#0f7b6c; --accent-soft:#e3f1ee; --warn:#b4451f; }
  .stApp { background:#f7f9fb; }
  h1,h2,h3 { color:var(--ink); font-family:"Georgia",serif; letter-spacing:-.01em; }
  .eyebrow { color:var(--accent); font-weight:700; text-transform:uppercase;
             letter-spacing:.14em; font-size:.72rem; }
  .metric-num { font-family:"SFMono-Regular",ui-monospace,Menlo,monospace;
                font-size:2.0rem; color:var(--ink); font-weight:600; }
  .metric-lab { color:var(--mist); font-size:.82rem; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px;
          padding:1.1rem 1.3rem; }
  .rec-yes { background:var(--accent-soft); border-left:4px solid var(--accent);
             padding:.9rem 1.1rem; border-radius:6px; color:var(--ink); }
  .rec-no  { background:#f6ece7; border-left:4px solid var(--warn);
             padding:.9rem 1.1rem; border-radius:6px; color:var(--ink); }
  .stTabs [data-baseweb="tab"] { font-family:Georgia,serif; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- data/model loaders -----------------------------
@st.cache_data
def load_scores():
    return pd.read_csv(PROC / "uplift_scores_test.csv")

@st.cache_resource
def load_model():
    with open(MODELS / "xlearner.pkl", "rb") as f:
        model = pickle.load(f)
    feats = json.load(open(MODELS / "feature_cols.json"))
    cats = json.load(open(MODELS / "categories.json"))
    return model, feats, cats


# ----------------------------- header -----------------------------
st.markdown('<div class="eyebrow">ImpactIQ · Causal inference · Uplift modeling</div>',
            unsafe_allow_html=True)
st.title("ImpactIQ: Campaign Intelligence & Uplift Optimization Engine")
st.caption("Deciding *who* to target with a marketing email to maximize incremental "
           "revenue — not just predicting who buys. Demonstration on the public, "
           "randomized Hillstrom dataset.")

tab1, tab2, tab3 = st.tabs(["Campaign Optimizer", "Customer Uplift Scoring", "ImpactIQ Methodology"])


# ============================ TAB 1: POLICY ============================
with tab1:
    try:
        scores = load_scores()
    except Exception:
        st.error("Run the notebooks first to create data/processed/uplift_scores_test.csv "
                 "(Sprint 4 saves it).")
        st.stop()

    st.markdown("#### Campaign economics and optimal targeting policy")
    c1, c2 = st.columns(2)
    with c1:
        cost = st.slider("Cost per email ($)", 0.01, 2.00, 0.10, 0.01,
                         help="Cheap channels (email) justify targeting almost "
                              "everyone; expensive channels (direct mail, calls) "
                              "make tight targeting essential.")
    with c2:
        value = st.number_input("Value per conversion ($)", 1.0, 1000.0, 116.0, 1.0,
                                help="Default ≈ Hillstrom's real average spend among "
                                     "spenders.")

    # ---- compute the profit curve (same logic as Sprint 6) ----
    d = scores[["uplift_X", "treatment", "conversion"]].copy()
    d = d.sort_values("uplift_X", ascending=False).reset_index(drop=True)
    d["is_t"] = (d.treatment == 1).astype(int)
    d["is_c"] = (d.treatment == 0).astype(int)
    d["cum_t"] = d.is_t.cumsum().clip(lower=1)
    d["cum_c"] = d.is_c.cumsum().clip(lower=1)
    d["cum_tconv"] = (d.conversion * d.is_t).cumsum()
    d["cum_cconv"] = (d.conversion * d.is_c).cumsum()
    d["inc_conv"] = d.cum_tconv - d.cum_cconv * (d.cum_t / d.cum_c)
    d["n"] = np.arange(1, len(d) + 1)
    d["profit"] = d.inc_conv * value - d.n * cost

    best = int(d.profit.idxmax())
    best_n, best_pct = d.n[best], 100 * d.n[best] / len(d)
    targeted_profit = float(d.profit[best])
    blanket_profit = float(d.profit.iloc[-1])
    overall_inc = d.inc_conv.iloc[-1] / len(d)
    random_profit = overall_inc * best_n * value - best_n * cost

    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="card"><div class="metric-lab">Optimal cutoff</div>'
                f'<div class="metric-num">{best_pct:.0f}%</div>'
                f'<div class="metric-lab">target top {best_n:,} customers</div></div>',
                unsafe_allow_html=True)
    m2.markdown(f'<div class="card"><div class="metric-lab">Profit at cutoff</div>'
                f'<div class="metric-num">${targeted_profit:,.0f}</div>'
                f'<div class="metric-lab">vs blanket ${blanket_profit:,.0f}</div></div>',
                unsafe_allow_html=True)
    m3.markdown(f'<div class="card"><div class="metric-lab">Beats random by</div>'
                f'<div class="metric-num">${targeted_profit-random_profit:,.0f}</div>'
                f'<div class="metric-lab">same-size random targeting</div></div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d.n, d.profit, color="#0f7b6c", lw=1.6, label="Uplift-targeted profit")
    ax.axvline(best_n, color="#b4451f", ls="--", lw=1.2,
               label=f"optimal cutoff (top {best_pct:.0f}%)")
    ax.axhline(0, color="#5b6b7a", lw=0.6)
    ax.set_xlabel("Customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative profit ($)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    st.pyplot(fig)
    st.caption("Drag the cost slider: as cost rises, the optimal cutoff shrinks — "
               "the value of targeting scales with per-contact cost.")


# ============================ TAB 2: SCORE ============================
with tab2:
    try:
        model, feats, cats = load_model()
    except Exception:
        st.error("Run `python src/train_model.py` first to train and save the model.")
        st.stop()

    st.markdown("#### Estimate an individual customer’s campaign uplift")
    cc1, cc2, cc3 = st.columns(3)
    recency = cc1.slider("Months since last purchase", 1, 12, 4)
    history = cc2.number_input("Past-year spend ($)", 0.0, 3000.0, 240.0, 10.0)
    newbie  = cc3.selectbox("New customer (last 12 mo)?", ["No", "Yes"]) == "Yes"
    cc4, cc5, cc6 = st.columns(3)
    mens   = cc4.selectbox("Bought men's merch?", ["No", "Yes"]) == "Yes"
    womens = cc5.selectbox("Bought women's merch?", ["No", "Yes"]) == "Yes"
    zip_code = cc6.selectbox("Zip type", cats["zip_code"])
    channel  = st.selectbox("Channel", cats["channel"])

    # build one-row feature frame matching training columns
    row = {"recency": recency, "history": history,
           "mens": int(mens), "womens": int(womens), "newbie": int(newbie),
           "zip_code": zip_code, "channel": channel}
    Xrow = pd.get_dummies(pd.DataFrame([row]))
    Xrow = Xrow.reindex(columns=feats, fill_value=0)

    uplift = float(model.predict_uplift(Xrow.values)[0])

    # tier thresholds from the test-set distribution
    scores = load_scores()
    q33, q66 = scores["uplift_X"].quantile([0.33, 0.66])
    tier = "High" if uplift >= q66 else ("Medium" if uplift >= q33 else "Low / negative")

    st.markdown(f'<div class="metric-lab">Predicted uplift (visit probability)</div>'
                f'<div class="metric-num">{uplift:+.4f}</div>', unsafe_allow_html=True)

    if tier == "High":
        st.markdown(f'<div class="rec-yes"><b>Recommendation: Target.</b> This customer '
                    f'is in the <b>High</b> uplift tier — the email is estimated to move '
                    f'them meaningfully. A persuadable worth the spend.</div>',
                    unsafe_allow_html=True)
    elif tier == "Medium":
        st.markdown(f'<div class="rec-yes"><b>Recommendation: Target if cheap.</b> '
                    f'<b>Medium</b> uplift — worth emailing on a low-cost channel, '
                    f'reconsider on an expensive one.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="rec-no"><b>Recommendation: Skip first.</b> '
                    f'<b>Low / negative</b> uplift — the email is estimated to move this '
                    f'customer little (or they may convert anyway / be a sleeping dog). '
                    f'Spend the budget on higher-uplift customers.</div>',
                    unsafe_allow_html=True)
    st.caption("Uplift is an *estimate* of the unobservable individual treatment effect, "
               "based on customers with similar features — not a guaranteed outcome.")


# ============================ TAB 3: STORY ============================
with tab3:
    st.markdown("#### How ImpactIQ turns campaign data into decisions")
    st.markdown("""
**The question.** A marketing email lifts conversions *on average* — but who should we
actually send it to? The average can hide customers the email doesn't move, or even
*annoys*. Targeting the right people is where the money is.

**The method.**
1. **Designed & validated the experiment** — power analysis, then balance checks
   confirming randomization held (treatment and control are statistical twins).
2. **Measured the average effect** — email lifts conversion 0.57% → 1.07%
   (~87% relative, p ≈ 4e-10). Real, but *not constant* across customers (~3× spread
   by recency) — so an average is not a targeting rule.
3. **Modeled the individual effect** — S-, T-, and a hand-rolled **X-learner**.
4. **An honest pivot.** On the rare *conversion* outcome, no model beat random
   (Qini ≈ 0) — the signal was too sparse. I diagnosed it as outcome density (not
   algorithm), and switched to the denser *visit* outcome, where models beat random and
   the **X-learner won** (consistent with its robustness to the 2:1 class imbalance).
5. **Turned scores into a decision** — ranked customers by uplift, found the
   profit-maximizing cutoff, and showed targeting beats same-size random sending by 2×.

**The judgment.** The most important result is #4: knowing when *not* to trust a model
is part of the analysis. Reported honestly, a model that doesn't beat random is a
finding — and it led to the principled fix.
""")
    st.caption(
    "ImpactIQ pipeline: experiment validation → campaign impact analysis → "
    "uplift modeling → Qini evaluation → profit-optimized targeting."
)
