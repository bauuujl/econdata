"""Do 401(k) plans increase household savings? OLS vs Double Machine Learning.

Replicates the 401(k) example in Chernozhukov et al. (2018), "Double/debiased
machine learning for treatment and structural parameters", The Econometrics
Journal 21(1), using 1991 SIPP data on 9,915 US households.

Outcome  Y = net financial assets (US$)
Treatment D = eligible for a 401(k) plan through an employer (1/0)
Controls X = age, income, family size, education, married, two earners,
             defined-benefit pension, IRA participation, home owner

Run: python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

Y, D = "net_tfa", "e401"
CONTROLS = ["age", "inc", "fsize", "educ", "marr", "twoearn", "db", "pira", "hown"]
PAPER_ESTIMATE = 9247  # Chernozhukov et al. (2018), Table 2, partially linear, forest, 5 folds

# Chart style: validated categorical palette (slot 1) and recessive ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    "text.parse_math": False,  # so "$" in labels prints as a dollar sign
})
DOLLARS = plt.FuncFormatter(lambda v, _: f"${v:,.0f}")


# --------------------------------------------------------------------------
# 1. Traditional econometrics: OLS
# --------------------------------------------------------------------------
def ols_estimates(df):
    naive = smf.ols(f"{Y} ~ {D}", df).fit(cov_type="HC1")
    controls = smf.ols(f"{Y} ~ {D} + " + " + ".join(CONTROLS), df).fit(cov_type="HC1")
    (RESULTS / "ols_with_controls_summary.txt").write_text(str(controls.summary()))
    return [
        {"method": "OLS, no controls", "estimate": naive.params[D], "std_error": naive.bse[D]},
        {"method": "OLS, linear controls", "estimate": controls.params[D],
         "std_error": controls.bse[D]},
    ]


# --------------------------------------------------------------------------
# 2. Double Machine Learning (partially linear model)
# --------------------------------------------------------------------------
def make_learner(name):
    if name == "Lasso":
        # Lasso on all squares and interactions of the controls.
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LassoCV(cv=5, max_iter=20000))
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=4,
                                     random_state=0, n_jobs=-1)
    if name == "Gradient boosting":
        return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                         random_state=0)
    raise ValueError(name)


def dml_plr(y, d, X, learner, n_folds=5, seed=0):
    """Partially linear model  y = theta*d + g(X) + e  estimated by DML.

    Step 1: with ML, predict y from X and d from X, using cross-fitting
            (each household's prediction comes from a model that never saw it).
    Step 2: keep the residuals (the part of y and d that X can't explain).
    Step 3: regress y-residual on d-residual. The slope is theta.
    """
    y_res, d_res = np.zeros(len(y)), np.zeros(len(y))
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        y_res[test] = y[test] - learner().fit(X[train], y[train]).predict(X[test])
        d_res[test] = d[test] - learner().fit(X[train], d[train]).predict(X[test])

    theta = np.sum(d_res * y_res) / np.sum(d_res ** 2)
    psi = (y_res - theta * d_res) * d_res  # score function
    se = np.sqrt(np.mean(psi ** 2)) / np.mean(d_res ** 2) / np.sqrt(len(y))
    return theta, se


def dml_estimates(df, n_repeats=5):
    y, d, X = df[Y].to_numpy(), df[D].to_numpy(float), df[CONTROLS].to_numpy(float)
    rows = []
    for name in ["Lasso", "Random forest", "Gradient boosting"]:
        # Repeat with different random fold splits and take the median, as in the paper.
        runs = [dml_plr(y, d, X, lambda: make_learner(name), seed=s) for s in range(n_repeats)]
        thetas, ses = np.array(runs).T
        theta = np.median(thetas)
        se = np.sqrt(np.median(ses ** 2 + (thetas - theta) ** 2))  # paper's split-adjusted SE
        rows.append({"method": f"DML: {name}", "estimate": theta, "std_error": se})
        print(f"  done: DML {name}")
    return rows


def check_with_doubleml_package(df):
    """Cross-check the hand-written DML against the official DoubleML package."""
    import doubleml as dml

    np.random.seed(0)  # DoubleML draws its fold split with numpy's global random state
    data = dml.DoubleMLData(df, y_col=Y, d_cols=D, x_cols=CONTROLS)
    model = dml.DoubleMLPLR(data, make_learner("Random forest"), make_learner("Random forest"),
                            n_folds=5)
    model.fit()
    return {"method": "DoubleML package: Random forest",
            "estimate": model.coef[0], "std_error": model.se[0]}


# --------------------------------------------------------------------------
# 3. Heterogeneity: does the effect differ by income?
# --------------------------------------------------------------------------
def effect_by_income(df):
    df = df.assign(income_quintile=pd.qcut(df["inc"], 5, labels=[1, 2, 3, 4, 5]))
    rows = []
    for q, sub in df.groupby("income_quintile", observed=True):
        theta, se = dml_plr(sub[Y].to_numpy(), sub[D].to_numpy(float),
                            sub[CONTROLS].to_numpy(float), lambda: make_learner("Random forest"))
        rows.append({"income_quintile": int(q),
                     "income_range": f"${sub['inc'].min():,.0f}-${sub['inc'].max():,.0f}",
                     "estimate": theta, "std_error": se, "households": len(sub)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_estimates(table):
    t = table[~table["method"].str.startswith("DoubleML")].iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [ORANGE if m.startswith("OLS") else BLUE for m in t["method"]]
    ax.errorbar(t["estimate"], y, xerr=1.96 * t["std_error"], fmt="none", ecolor=MUTED,
                elinewidth=1.5, capsize=3)
    ax.scatter(t["estimate"], y, s=60, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    for yi, v, se in zip(y, t["estimate"], t["std_error"]):
        ax.text(v + 1.96 * se + 400, yi, f"${v:,.0f}", va="center", color=INK)
    ax.axvline(PAPER_ESTIMATE, color=INK, lw=1, ls="--")
    ax.text(PAPER_ESTIMATE, len(t) - 0.4, " Paper (forest DML): $9,247", color=INK, fontsize=9)
    ax.set_yticks(y, t["method"])
    ax.set_ylim(-0.6, len(t) - 0.2)
    ax.set_xlim(0, 26000)
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect of 401(k) eligibility on net financial assets (US$, 95% CI)")
    ax.set_title("Controlling for confounders halves the estimated effect", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "estimates_comparison.png")
    plt.close(fig)


def plot_by_income(het):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = het["income_quintile"]
    ax.bar(x, het["estimate"], color=BLUE, width=0.6)
    ax.errorbar(x, het["estimate"], yerr=1.96 * het["std_error"], fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=3)
    ax.set_xticks(x, [f"Q{q}\n{r}" for q, r in zip(x, het["income_range"])], fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Effect on net financial assets (US$)")
    ax.set_title("The effect is largest for higher earners (DML, random forest)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "effect_by_income.png")
    plt.close(fig)


def main():
    df = pd.read_csv(HERE / "data" / "sipp1991.csv")

    summary = df.groupby(D)[[Y] + CONTROLS].mean().T.round(2)
    summary.columns = ["Not eligible", "Eligible"]
    summary.to_csv(RESULTS / "summary_by_eligibility.csv")

    print("Running models...")
    table = pd.DataFrame(ols_estimates(df) + dml_estimates(df) + [check_with_doubleml_package(df)])
    table["ci_low"] = table["estimate"] - 1.96 * table["std_error"]
    table["ci_high"] = table["estimate"] + 1.96 * table["std_error"]
    table = table.round(0)
    table.to_csv(RESULTS / "estimates.csv", index=False)

    het = effect_by_income(df).round(0)
    het.to_csv(RESULTS / "effect_by_income.csv", index=False)

    plot_estimates(table)
    plot_by_income(het)

    print("\n=== Average households, by 401(k) eligibility ===")
    print(summary.to_string())
    print("\n=== Effect of 401(k) eligibility on net financial assets ===")
    print(table.to_string(index=False))
    print(f"(Paper's estimate, DML random forest: ${PAPER_ESTIMATE:,})")
    print("\n=== Effect by income quintile (DML, random forest) ===")
    print(het.to_string(index=False))


if __name__ == "__main__":
    main()
