# Code Guide: what each piece of code does

The project has two Python scripts:

| File | Purpose |
|---|---|
| `fetch_data.py` | Downloads real GDP growth from the World Bank API (run once). |
| `okun_analysis.py` | Builds the dataset, runs the econometrics and ML, and saves tables and charts. |

---

## Libraries used

| Library | What it's for |
|---|---|
| `pandas` | Loading CSVs and reshaping or merging tables (like Excel/SQL in Python). |
| `numpy` | Math helpers: `np.log`, `np.sqrt`, `np.linspace`. |
| `statsmodels` | Econometrics: OLS regressions with robust and clustered standard errors. |
| `scikit-learn` | Machine learning: linear regression, random forest, error metrics. |
| `matplotlib` | Charts. |
| `urllib` / `json` | Built into Python, used to call the World Bank API. |

---

## `fetch_data.py`

```python
URL = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD.ZG?format=json&date=1990:2024&per_page=20000"
```
- `NY.GDP.MKTP.KD.ZG` is the World Bank code for **real GDP growth (annual %)**.
  We need *real* growth because the GDP in `mastersheet2.csv` is nominal US$, which
  also moves with inflation and exchange rates.

```python
with urllib.request.urlopen(URL, timeout=60) as resp:
    meta, rows = json.load(resp)
```
- Downloads the data. The API returns two parts: page info (`meta`) and the data (`rows`).

```python
df = pd.DataFrame({...} for r in rows if r["value"] is not None and r["countryiso3code"])
```
- Keeps the country code, year and value, and skips empty values.

```python
df.sort_values(["country_code", "year"]).to_csv(OUT, index=False)
```
- Saves `data/real_gdp_growth.csv`.

---

## `okun_analysis.py`

### Setup
```python
plt.rcParams.update({...})
```
- Sets a clean chart style once: no top or right borders and a light grid.

```python
INDICATORS = {2: "female_lf_share", 3: "unemp_female", 4: "unemp_male", 5: "population", 7: "inflation"}
```
- Maps the `indicator_id` numbers in `mastersheet2.csv` to readable names.
  They come from `Dataset/indicators.csv`, the same IDs the SQL project uses.

### Step 1: `build_panel()` builds the country-year dataset

```python
countries = pd.read_csv(REPO / "Dataset" / "countries.csv", encoding="latin-1")
countries["country_id"] = np.arange(1, len(countries) + 1)
```
- Loads the country list. `encoding="latin-1"` handles accented names such as Côte d'Ivoire.
- `country_id` copies the SQL `AUTO_INCREMENT` order (Afghanistan = 1 … United States = 203).

```python
wide = long.pivot_table(index=["country_id", "Year"], columns="indicator_id", values="Value")
```
- **Pivot:** turns the long table (one row per country-year-indicator) into a wide
  one (one row per country-year, one column per indicator). In SQL this would take
  several self-joins.

```python
wide["unemployment"] = f * wide["unemp_female"] + (1 - f) * wide["unemp_male"]
```
- Rebuilds **total unemployment** as the female rate times the female share of
  the labour force, plus the male rate times the male share.

```python
df = wide.merge(countries, on="country_id").merge(growth, on=["country_code", "year"], how="left")
```
- **Merge = SQL JOIN.** Adds country names and income groups, then real GDP growth.

```python
g = df.groupby("country")
df["d_unemp"]    = g["unemployment"].diff()     # this year − last year
df["unemp_lag"]  = g["unemployment"].shift(1)   # last year's value
df["growth_lag"] = g["real_gdp_growth"].shift(1)
```
- `groupby("country")` makes sure we never subtract one country's value from another's.
- `.diff()` gives the **change in unemployment**, the left-hand side of Okun's law.
- `.shift(1)` creates **lags** (last year's values), used as ML features.

```python
df["log_pop"] = np.log(df["population"])
```
- Takes the log of population so giants like China and India don't dominate.

### Step 2: `okun_regressions()` runs the econometrics

```python
data = data[data["real_gdp_growth"].between(-15, 20) & data["d_unemp"].abs().lt(10)]
```
- Drops extreme outliers (wars, collapses) so a few points don't drive the results.

```python
smf.ols("d_unemp ~ real_gdp_growth", data).fit(cov_type="cluster", cov_kwds=cluster)
```
- **OLS regression** written as a formula: "explain `d_unemp` using `real_gdp_growth`".
- `cov_type="cluster"` gives **clustered standard errors by country**. Years within
  one country are related to each other, and ordinary standard errors would make us
  overconfident.

```python
"d_unemp ~ real_gdp_growth + C(country)"
```
- `C(country)` adds a dummy for every country. These are **country fixed effects**:
  each country is compared only with itself, which removes fixed differences.

```python
"d_unemp ~ real_gdp_growth + C(country) + C(year)"
```
- Adds **year fixed effects**, which remove shocks that hit everyone at once (2009, 2020).

```python
models["United States only"] = smf.ols(..., us).fit(cov_type="HC1")
```
- A US-only regression. `HC1` gives **heteroskedasticity-robust** standard errors.

```python
for grp in order: ...
```
- Re-runs the fixed-effects model separately for each income group to see whether
  the Okun coefficient differs, following Ball, Leigh & Loungani (2017).

```python
m.params["real_gdp_growth"], m.bse[...], m.pvalues[...], m.rsquared, m.nobs
```
- Pulls out the coefficient, standard error, p-value, R² and sample size, then saves
  them to `results/okun_regressions.csv`. The full US regression table goes to
  `results/us_okun_summary.txt`.

### `plot_okun()` draws the econometrics charts
- `ax.scatter(...)` plots every country-year as a faint blue dot and the US as orange dots.
- `ax.plot(x, a + b*x)` draws each fitted regression line, only over the range of its own data.
- `ax.barh(...)` plus `ax.errorbar(... xerr=1.96*std_error)` draw the coefficient
  for each income group with its **95% confidence interval**. A CI that doesn't
  cross zero means the coefficient is statistically significant.

### Step 3: `machine_learning()` runs the ML part

```python
FEATURES = ["real_gdp_growth", "growth_lag", "inflation", "inflation_lag",
            "unemp_lag", "female_lf_share", "log_pop", "high_income"]
```
- The inputs (X) the models use to predict `d_unemp` (y). This year's unemployment
  level is left out because it would give away the answer (**data leakage**).

```python
train = data[data["year"] <= 2012]
test  = data[data["year"] > 2012]
```
- A **time-based train/test split**: learn from the past, score on the future.
  A random split would let the model "see" neighbouring years and look better than it is.

```python
LinearRegression()
RandomForestRegressor(n_estimators=400, min_samples_leaf=10, random_state=42)
```
- **Linear regression:** a straight-line model, like OLS.
- **Random forest:** averages 400 decision trees, so it can capture curves and
  interactions. `min_samples_leaf=10` stops trees from memorising noise, and
  `random_state=42` makes results repeatable.
- **Naive baseline:** always predicts 0 ("no change"). Any useful model must beat it.

```python
model.fit(X_train, y_train).predict(X_test)
mean_squared_error(...), r2_score(...)
```
- `fit` trains the model and `predict` makes predictions on the unseen test years.
- **RMSE** is the typical prediction error in percentage points (lower is better).
- **R²** is the share of variation explained (higher is better). Below 0 means
  the model is worse than predicting the average.

```python
permutation_importance(rf, X_test, y_test, n_repeats=20)
```
- **Permutation importance:** shuffle one feature and measure how much the test R²
  drops. A big drop means the model relies on that feature. This is more
  trustworthy than the forest's built-in importance.

### Charts saved to `results/`
| File | Shows |
|---|---|
| `okun_scatter.png` | All countries vs US, with fitted Okun lines |
| `okun_by_income_group.png` | Okun coefficient by income group with 95% CIs |
| `feature_importance.png` | What the random forest relies on |
| `us_forecast.png` | US actual vs predicted unemployment change, 2013–2023 |

### `main()`
Runs the three steps in order and prints the tables to the screen. The same output
is saved in `results/console_output.txt`.
