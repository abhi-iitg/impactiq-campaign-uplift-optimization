# ImpactIQ: Campaign Intelligence & Uplift Optimization Engine

> **Turning campaign data into causal insights—identifying high-impact customers and optimizing targeting for incremental revenue.**

ImpactIQ is an end-to-end **campaign intelligence and uplift optimization engine** built with causal machine learning. Instead of simply predicting which customers are most likely to convert, it estimates **who is most likely to change their behavior because of a marketing intervention**.

The project analyzes treatment effects, identifies high-uplift customer segments, compares uplift modeling strategies, and converts model outputs into a **profit-aware targeting policy**.

> **Demonstration project** built on the public, randomized Hillstrom MineThatData email dataset. The project focuses on independently implementing and demonstrating experimentation, causal inference, uplift modeling, evaluation, and campaign decision optimization.

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Scope: Experiment Design and Analysis](#scope-experiment-design-and-analysis)
5. [Key Findings](#key-findings)
6. [Visual Results](#visual-results)
  - [Model Comparison: Qini Curves](#1-model-comparison-qini-curves)
  - [Incremental Lift by Customer Decile](#2-incremental-lift-by-customer-decile)
  - [Profit-Optimized Targeting Policy](#3-profit-optimized-targeting-policy)
  - [Campaign Lift by Customer Recency](#4-campaign-lift-varies-by-customer-recency)
  - [Estimated Uplift Segmentation](#5-estimated-uplift-segmentation)
  - [Actionable Targeting Tiers](#6-actionable-targeting-tiers)
7. [ImpactIQ App](#impactiq-app)
8. [Roadmap](#roadmap)
9. [References and Data](#references-and-data)
10. [Tech Stack](#tech-stack)
11. [Setup](#setup)
12. [Run the App](#run-the-app)
13. [Conclusion](#conclusion)
14. [Connect](#connect)
15. [License](#license)

## The Core Idea

A campaign's customers can be viewed through four potential treatment-response segments:

| | Buys if treated | Doesn't buy if treated |
|---|---|---|
| **Buys if not treated** | Sure Thing (wasted spend) | **Sleeping Dog** (treatment *hurts*) |
| **Doesn't buy if not treated** | **Persuadable** (target these) | Lost Cause |

Uplift modeling helps identify **Persuadables** and avoid **Sleeping Dogs**—a distinction ordinary conversion models cannot make because they rank customers by predicted outcomes rather than the **incremental effect of treatment**.

## Dataset

Hillstrom MineThatData email challenge: 64,000 customers randomly assigned to a
men's-merchandise email, a women's-merchandise email, or no email; outcomes
(visit, conversion, spend) tracked over the following two weeks. I combine the two
email arms into a single "treated" group vs. the no-email control.

Get the data:
```bash
python src/download_data.py        # via scikit-uplift
# or download Hillstrom.csv from Kaggle into data/raw/
```

## Project structure

```
impactiq-campaign-uplift-optimization/
├── data/
│   ├── raw/            # downloaded data (gitignored)
│   └── processed/      # cleaned data (gitignored)
├── notebooks/          # one per sprint
│   ├── 02_eda_balance_checks.ipynb
│   ├── 03_average_treatment_effect.ipynb
│   ├── 04_uplift_models.ipynb
│   ├── 05_uplift_evaluation.ipynb
│   └── 06_targeting_policy.ipynb
├── src/                # reusable modules (imported by notebooks + app)
│   ├── download_data.py
│   ├── power_analysis.py
│   └── uplift_models.py    # S-, T-, and hand-rolled X-learner
├── reports/
│   ├── sprint1_experiment_design.md   # Sprint 1: design + power analysis
│   └── figures/
├── app/                # Streamlit app (Sprint 7)
├── requirements.txt
└── README.md
```

## Scope: Experiment Design and Analysis

In production, the data scientist owns experiment **design** (eligible population,
stable-hash randomization, power-based sample sizing, guardrails) *and* the
**analysis**. The Hillstrom data is pre-randomized, so this repo analyzes an existing
experiment — but `reports/sprint1_experiment_design.md` documents my full from-scratch
design approach, so the project demonstrates both halves of the skill.

## Key Findings

**1. The email works — on average.** Conversion lifts from 0.57% to 1.07% (a 0.50pp
absolute, ~87% relative lift; 95% CI [0.355, 0.636]pp, p ≈ 4e-10). Statistically solid.

**2. But the average hides who responds.** Broken down by recency, the lift ranges
~0.25–0.73pp — a ~3× spread. A single average is a verdict on the campaign, not a
targeting rule. This motivates modeling the *individual* treatment effect.

**3. An honest modeling result.** Modeling uplift on the rare **conversion** outcome
(~1%), no meta-learner reliably beat random targeting (Qini AUC ≈ 0). I diagnosed the
bottleneck as **outcome density**, not algorithm strength — a rare outcome leaves the
per-customer effect buried in noise, and a stronger learner only overfits that noise.
So I switched the modeling target to the denser **visit** outcome, where the signal is
detectable. Conversion stays the business goal; **visit is the modeling proxy** the
data can support.

> Reporting the conversion result honestly — a model that does *not* beat random — and
> diagnosing *why*, is part of the analysis. Knowing when not to trust a model is the
> point of evaluating it.

**4. On visit, the models beat random — and the X-learner wins.** Qini AUC: X = 0.038,
S = 0.035, T = 0.034. The X-learner's edge is consistent with its propensity-weighted
blend handling the 2:1 treatment:control imbalance, as I expected.

**5. Targeting the right customers pays off.** Valuing a conversion at Hillstrom's real
average spend (~$116) and assuming $0.10/email, the profit-maximizing cutoff targets
the top ~57% of customers. That beats blanket sending, and beats random targeting of the
*same size* by more than 2× — the value is in emailing the *right* people, not fewer
people. The optimal cutoff tightens as per-contact cost rises, so uplift targeting
matters most for expensive channels (direct mail, outbound calls).

## Visual Results

### 1. Model Comparison: Qini Curves

![Qini curves comparing S-, T-, and X-learners](reports/figures/qini_curves.png)

The Qini curve evaluates whether the model ranks customers by expected
incremental response better than random targeting. All three uplift models
outperform the random baseline on the denser visit outcome, with the
X-learner achieving the highest Qini AUC.

---

### 2. Incremental Lift by Customer Decile

![Uplift by decile](reports/figures/uplift_by_decile.png)

Customers are ranked from highest to lowest predicted uplift and grouped into
deciles. The top-ranked groups show the strongest incremental response,
supporting a prioritization strategy rather than sending the campaign to
every eligible customer.

---

### 3. Profit-Optimized Targeting Policy

![Profit curve with optimal targeting cutoff](reports/figures/profit_curve.png)

The profit curve shows the cumulative value of targeting customers in predicted
uplift order. Under the assumed campaign economics, profit peaks when targeting
approximately the top 57% of customers; contacting additional lower-uplift
customers reduces expected value.

---

### 4. Campaign Lift Varies by Customer Recency

![Lift by recency](reports/figures/lift_by_recency.png)

Average campaign impact differs meaningfully across recency groups. This
variation shows why one overall treatment-effect estimate is not sufficient
for customer-level targeting decisions.

---

### 5. Estimated Uplift Segmentation

![Estimated uplift quadrant segmentation](reports/figures/quadrant_segmentation.png)

The model categorizes customers into estimated treatment-response segments:
persuadables, sure things, lost causes, and sleeping dogs. The recommended
targeting policy focuses on persuadable customers, where campaign contact is
most likely to create incremental value.

---

### 6. Actionable Targeting Tiers

![Uplift targeting tiers](reports/figures/uplift_tiers.png)

Customers can be translated into practical targeting tiers based on predicted
uplift, enabling marketing teams to prioritize high-value audiences and avoid
spending on low- or negative-uplift segments.

## ImpactIQ App

The ImpactIQ Streamlit app turns causal analysis into an interactive campaign decision tool. *(Run `streamlit run app/streamlit_app.py` after `python src/train_model.py`.)*

![Campaign Optimizer with cost slider](assets/app_targeting_policy.pdf)
*Targeting-policy view. Drag the cost-per-email slider and the optimal cutoff, profit, and profit curve update live. At $0.10/email and $116/conversion, targeting the top ~57% of customers beats blanket sending — and beats random targeting of the same size by more than 2×. The value is in emailing the **right** people, not fewer people.*

![Cost sensitivity](assets/app_cost_sensitivity.pdf)
*Raising the per-contact cost tightens the optimal cutoff — the policy responds to channel economics. Uplift targeting is marginal for cheap email but decisive for expensive channels (direct mail, outbound calls), where contacting a non-responder wastes real budget.*

![Per-customer scoring](assets/app_customer_scoring.pdf)
*Per-customer scoring. Enter a customer's features and the app returns a predicted uplift and a plain-language target / target-if-cheap / skip recommendation — the model's decision for one individual, framed for a non-technical user.*

![Method story](assets/app_story.pdf)
*The "Story" tab. A plain-language walkthrough of the method — experiment validation, the average effect, the three uplift models, the honest conversion→visit pivot, and the targeting policy — so a non-technical stakeholder understands what the tool does and why to trust it.*

## Roadmap

- [x] **Sprint 1** — Experiment design & power analysis
- [x] **Sprint 2** — Load, EDA & randomization balance checks
- [x] **Sprint 3** — Average treatment effect (did it work overall?)
- [x] **Sprint 4** — Uplift models (S-, T-, hand-rolled X-learner)
- [x] **Sprint 5** — Uplift evaluation (Qini curve, uplift-by-decile)
- [x] **Sprint 6** — Campaign Optimizer, cost sensitivity & incremental revenue
- [x] **Sprint 7** — Streamlit app & polish

## References and Data

**Dataset**
- Kevin Hillstrom, *The MineThatData E-Mail Analytics And Data Mining Challenge* (2008)
  — the public, randomized email dataset used here.
  or also loadable via [`sklift.datasets.fetch_hillstrom`](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html).

**Methodology**
- Künzel, Sekhon, Bickel & Yu (2019), *Metalearners for estimating heterogeneous
  treatment effects using machine learning*, PNAS 116(10):4156–4165 — the S-, T-, and
  X-learner framework.
  [PNAS / DOI](https://doi.org/10.1073/pnas.1804597116) · [free arXiv PDF](https://arxiv.org/abs/1706.03461)
- Gutierrez & Gérardy (2017), *Causal Inference and Uplift Modelling: A Review of the
  Literature*, PMLR 67:1–13 — uplift evaluation, Qini/uplift curves, and why standard
  classification metrics don't apply.
  [PMLR](https://proceedings.mlr.press/v67/gutierrez17a.html) · [PDF](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)

  **Tutorials & learning resources**
- Juan Camilo Orduz, *Introduction to Uplift Modeling* — a hands-on walkthrough of
  meta-learners and Qini evaluation that informed this project's approach.
  [Write-up](https://juanitorduz.github.io/uplift/) · [PyData talk](https://www.youtube.com/watch?v=VWjsi-5yc3w)

**Key libraries**
- [`scikit-uplift`](https://www.uplift-modeling.com/) — Qini curve / Qini AUC and
  uplift-by-percentile metrics.
- [`statsmodels`](https://www.statsmodels.org/) — power analysis and the two-proportion
  / proportion-effect-size tools.
- [`scikit-learn`](https://scikit-learn.org/) — base learners for the meta-learners
  (the X-learner is implemented from scratch in `src/uplift_models.py`).

## Tech Stack

Python · pandas · statsmodels · scikit-uplift · scikit-learn · Streamlit

*(The X-learner is implemented from scratch in `src/uplift_models.py` — pure
scikit-learn, no compiler/causalml dependency, so the repo clones and runs anywhere.)*

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
```

Then run the notebooks in order (`02` -> `06`). Sprint 1 is the design document in
`reports/`.


## Run the App

```bash
pip install -r requirements.txt
python src/download_data.py        # if not done yet
# run notebooks 02-06 once to create data/processed/*.csv
python src/train_model.py          # trains & saves the X-learner
streamlit run app/streamlit_app.py
```

The app has three core experiences: a live **campaign optimizer** with a cost slider (watch the
optimal cutoff move), a per-customer **uplift scoring** tool with a target/skip recommendation,
and a plain-language **story** of the method and the honest conversion→visit pivot.

## Conclusion

### Beyond Prediction: Turning Causal Insight into Business Decisions

This project demonstrates how data science can move beyond prediction to:
- Support decision-making
- Optimize resource allocation
- Drive measurable business impact

## Connect

If you found this project interesting and have feedback, feel free to star and fork the repository, and follow for more such insightful projects!

My Portfolio & Profiles: 
- **Email : mr.abhishekaaa@gmail.com**
- **[Portfolio]()**
- **[LinkedIn](https://www.linkedin.com/in/abhishekkumargond/)**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.