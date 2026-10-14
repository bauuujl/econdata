"""Okun's Law across countries: econometrics + machine learning.

Question: when real GDP grows faster, how much does unemployment fall?

Part 1 (econometrics): estimate Okun's law
    change in unemployment = a + b * real GDP growth
  with pooled OLS, country fixed effects, by income group, and for the US.
Part 2 (machine learning): predict the change in unemployment with a linear
  regression and a random forest, tested on years the models never saw.

Run fetch_data.py once first, then: python okun_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

HERE = Path(__file__).parent
REPO = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Chart colours (validated categorical palette, slots 1-3) and recessive ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
})

# Indicator ids used in mastersheet2.csv (see Dataset/indicators.csv).
INDICATORS = {
    2: "female_lf_share",   # Labor force, female (% of total labor force)
    3: "unemp_female",      # Unemployment, female (%)
    4: "unemp_male",        # Unemployment, male (%)
    5: "population",        # Population, total
    7: "inflation",         # Inflation, consumer prices (annual %)
}


# --------------------------------------------------------------------------
# 1. Build the country-year panel
# --------------------------------------------------------------------------
def build_panel():
    countries = pd.read_csv(REPO / "Dataset" / "countries.csv", encoding="latin-1")
    countries.columns = ["country", "country_code", "region", "income_group"]
    countries["country_id"] = np.arange(1, len(countries) + 1)  # SQL AUTO_INCREMENT order

    long = pd.read_csv(REPO / "mastersheet2.csv")
    long = long[long["indicator_id"].isin(INDICATORS)]
    wide = (long.pivot_table(index=["country_id", "Year"], columns="indicator_id",
                             values="Value")
                .rename(columns=INDICATORS).reset_index().rename(columns={"Year": "year"}))

    # Total unemployment = labour-force-weighted average of the female and male rates.
    f = wide["female_lf_share"] / 100
    wide["unemployment"] = f * wide["unemp_female"] + (1 - f) * wide["unemp_male"]

    growth = pd.read_csv(HERE / "data" / "real_gdp_growth.csv")
    df = (wide.merge(countries, on="country_id")
              .merge(growth, on=["country_code", "year"], how="left")
              .sort_values(["country", "year"]))

    g = df.groupby("country")
    df["d_unemp"] = g["unemployment"].diff()                 # change in unemployment (pp)
    df["unemp_lag"] = g["unemployment"].shift(1)
    df["growth_lag"] = g["real_gdp_growth"].shift(1)
    df["inflation_lag"] = g["inflation"].shift(1)
    df["log_pop"] = np.log(df["population"])
    return df


# --------------------------------------------------------------------------
# 2. Econometrics: Okun's law regressions
# --------------------------------------------------------------------------
def okun_regressions(df):
    data = df.dropna(subset=["d_unemp", "real_gdp_growth"])
    # Drop extreme outliers (wars, collapses) so a few points don't drive the line.
    data = data[data["real_gdp_growth"].between(-15, 20) & data["d_unemp"].abs().lt(10)]
    cluster = {"groups": data["country"].astype("category").cat.codes}

    models = {
        "Pooled OLS": smf.ols("d_unemp ~ real_gdp_growth", data).fit(
            cov_type="cluster", cov_kwds=cluster),
        "Country fixed effects": smf.ols("d_unemp ~ real_gdp_growth + C(country)", data).fit(
            cov_type="cluster", cov_kwds=cluster),
        "Country + year fixed effects": smf.ols(
            "d_unemp ~ real_gdp_growth + C(country) + C(year)", data).fit(
            cov_type="cluster", cov_kwds=cluster),
    }
    us = data[data["country_code"] == "USA"]
    models["United States only"] = smf.ols("d_unemp ~ real_gdp_growth", us).fit(cov_type="HC1")

    rows = []
    for name, m in models.items():
        rows.append({"model": name, "okun_coefficient": m.params["real_gdp_growth"],
                     "std_error": m.bse["real_gdp_growth"],
                     "p_value": m.pvalues["real_gdp_growth"],
                     "r_squared": m.rsquared, "observations": int(m.nobs)})

    # Same fixed-effects model separately for each income group.
    order = ["High income", "Upper middle income", "Lower middle income", "Low income"]
    for grp in order:
        sub = data[data["income_group"] == grp]
        m = smf.ols("d_unemp ~ real_gdp_growth + C(country)", sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["country"].astype("category").cat.codes})
        rows.append({"model": f"FE: {grp}", "okun_coefficient": m.params["real_gdp_growth"],
                     "std_error": m.bse["real_gdp_growth"],
                     "p_value": m.pvalues["real_gdp_growth"],
                     "r_squared": m.rsquared, "observations": int(m.nobs)})

    table = pd.DataFrame(rows).round(4)
    table.to_csv(RESULTS / "okun_regressions.csv", index=False)
    (RESULTS / "us_okun_summary.txt").write_text(str(models["United States only"].summary()))
    return data, models, table


def plot_okun(data, models, table):
    # Scatter: all countries vs US, with fitted lines.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["real_gdp_growth"], data["d_unemp"], s=8, color=BLUE, alpha=0.15,
               linewidths=0, label="All countries (country-years)")
    us = data[data["country_code"] == "USA"]
    ax.scatter(us["real_gdp_growth"], us["d_unemp"], s=40, color=ORANGE,
               edgecolor="white", linewidth=1.5, label="United States", zorder=3)
    # Draw each fitted line only over the range of its own data.
    for name, color, sample in [("Pooled OLS", BLUE, data), ("United States only", ORANGE, us)]:
        p = models[name].params
        x = np.linspace(sample["real_gdp_growth"].min(), sample["real_gdp_growth"].max(), 50)
        ax.plot(x, p["Intercept"] + p["real_gdp_growth"] * x, color=color, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Real GDP growth (%)")
    ax.set_ylabel("Change in unemployment rate (pp)")
    ax.set_title("Okun's law: faster growth, falling unemployment", loc="left", color=INK)
    leg = ax.legend(frameon=False, loc="upper right")
    leg.legend_handles[0].set_alpha(1)
    fig.tight_layout()
    fig.savefig(RESULTS / "okun_scatter.png")
    plt.close(fig)

    # Bar chart: Okun coefficient by income group with 95% CIs.
    grp = table[table["model"].str.startswith("FE: ")].copy()
    grp["label"] = grp["model"].str.replace("FE: ", "")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(grp))[::-1]
    ax.barh(y, grp["okun_coefficient"], color=BLUE, height=0.55)
    ax.errorbar(grp["okun_coefficient"], y, xerr=1.96 * grp["std_error"], fmt="none",
                ecolor=INK, elinewidth=1.2, capsize=3)
    for yi, v, se in zip(y, grp["okun_coefficient"], grp["std_error"]):
        ax.text(v - 1.96 * se - 0.004, yi, f"{v:.2f}", va="center", ha="right", color=INK)
    ax.set_yticks(y, grp["label"])
    ax.set_xlim(-0.21, 0.005)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Okun coefficient (pp change in unemployment per 1% growth)")
    ax.set_title("Unemployment responds most to growth in rich countries", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "okun_by_income_group.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Machine learning: predict the change in unemployment
# --------------------------------------------------------------------------
# Everything a model could know in year t except this year's unemployment itself.
FEATURES = ["real_gdp_growth", "growth_lag", "inflation", "inflation_lag",
            "unemp_lag", "female_lf_share", "log_pop", "high_income"]


def machine_learning(df):
    data = df.copy()
    data["high_income"] = (data["income_group"] == "High income").astype(int)
    data = data.dropna(subset=FEATURES + ["d_unemp"])
    data = data[data["inflation"].between(-10, 100) & data["d_unemp"].abs().lt(10)]

    # Time-based split: learn from the past, test on the future (no peeking).
    train = data[data["year"] <= 2012]
    test = data[data["year"] > 2012]
    X_train, y_train = train[FEATURES], train["d_unemp"]
    X_test, y_test = test[FEATURES], test["d_unemp"]

    models = {
        "Naive (predict 0 change)": None,
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=10,
                                               random_state=42, n_jobs=-1),
    }
    rows, preds = [], {}
    for name, model in models.items():
        if model is None:
            pred = np.zeros(len(y_test))
        else:
            pred = model.fit(X_train, y_train).predict(X_test)
        preds[name] = pred
        rows.append({"model": name,
                     "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                     "r_squared": r2_score(y_test, pred)})
    scores = pd.DataFrame(rows).round(4)
    scores["train_rows"], scores["test_rows"] = len(train), len(test)
    scores.to_csv(RESULTS / "ml_scores.csv", index=False)

    rf = models["Random forest"]
    imp = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                    .sort_values("importance", ascending=False).round(4))
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)

    # Chart: permutation importance.
    fig, ax = plt.subplots(figsize=(7, 4))
    imp_sorted = importance.iloc[::-1]
    ax.barh(imp_sorted["feature"], imp_sorted["importance"], color=AQUA, height=0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title("What the random forest relies on", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_importance.png")
    plt.close(fig)

    # Chart: US actual vs predicted over the test years.
    us = test[test["country_code"] == "USA"]
    mask = test["country_code"] == "USA"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(us["year"], us["d_unemp"], color=INK, lw=2, marker="o", ms=5,
            label="Actual")
    ax.plot(us["year"], preds["Linear regression"][mask.values], color=BLUE, lw=2,
            marker="o", ms=5, label="Linear regression")
    ax.plot(us["year"], preds["Random forest"][mask.values], color=ORANGE, lw=2,
            marker="o", ms=5, label="Random forest")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Change in unemployment (pp)")
    ax.set_title("US unemployment change: out-of-sample predictions", loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "us_forecast.png")
    plt.close(fig)
    return scores, importance


def main():
    df = build_panel()
    data, models, table = okun_regressions(df)
    plot_okun(data, models, table)
    scores, importance = machine_learning(df)

    print("\n=== Okun's law regressions ===")
    print(table.to_string(index=False))
    print("\n=== Machine learning (test years 2013-2024) ===")
    print(scores.to_string(index=False))
    print("\n=== Random forest feature importance ===")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
