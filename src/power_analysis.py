"""
Sprint 1 — Power analysis utilities for the uplift / A-B experiment.

Binary-outcome power analysis for a two-group (treatment vs control) experiment.
Works in both directions:
  - given n per group -> minimum detectable effect (MDE)
  - given a target lift -> required n per group

Usage:
    python power_analysis.py
or import the functions into a notebook.
"""

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.optimize import brentq


def required_n(baseline, target, alpha=0.05, power=0.80):
    """Required sample size PER GROUP to detect baseline -> target conversion."""
    h = proportion_effectsize(target, baseline)
    return NormalIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )


def min_detectable_rate(baseline, n_per_group, alpha=0.05, power=0.80):
    """Smallest treatment conversion rate detectable given n per group."""
    effect = NormalIndPower().solve_power(
        nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    # invert Cohen's h back to a treatment rate above baseline
    return brentq(
        lambda p2: abs(proportion_effectsize(p2, baseline)) - effect,
        baseline + 1e-6,
        0.5,
    )


if __name__ == "__main__":
    # Replace baseline with the REAL control conversion rate in Sprint 2.
    baseline = 0.010
    n_per_group = 64000 / 3

    mde_rate = min_detectable_rate(baseline, n_per_group)
    print(f"n per group: {n_per_group:,.0f}")
    print(
        f"MDE: {baseline*100:.2f}% -> {mde_rate*100:.3f}% "
        f"({(mde_rate-baseline)*100:.3f}pp, {(mde_rate/baseline-1)*100:.1f}% relative)"
    )

    n = required_n(baseline, 0.013)
    print(f"\nTo detect 1.0% -> 1.3%: {n:,.0f} per group required "
          f"({'enough' if n_per_group >= n else 'not enough'})")
