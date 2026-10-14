# Code Guide: what each piece of code does

Everything is in one script, `analysis.py`. Run it with `python analysis.py`. It takes about
5 minutes, writes tables and charts to `results/`, and prints the results to the screen.

---

## Libraries used

| Library | What it's for |
|---|---|
| `pandas` | Loading the CSV and working with tables (like Excel or SQL in Python). |
| `numpy` | Math on arrays: sums, square roots, medians. |
| `statsmodels` | Econometrics: OLS regressions with robust standard errors. |
| `scikit-learn` | Machine learning: Lasso, random forest, gradient boosting, and K-fold splitting. |
| `doubleml` | The official Double Machine Learning package, used only to double-check my own code. |
| `matplotlib` | Charts. |

---

## Setup

```python
Y, D = "net_tfa", "e401"
CONTROLS = ["age", "inc", "fsize", "educ", "marr", "twoearn", "db", "pira", "hown"]
```
- **Y (outcome):** `net_tfa`, the household's net financial assets in dollars.
- **D (treatment):** `e401` = 1 if the household's employer offers a 401(k) plan.
- **X (controls):** age, income, family size, years of education, married, two earners,
  has a defined-benefit pension, has an IRA, owns a home. These are the same controls
  the paper uses.

```python
PAPER_ESTIMATE = 9247
```
- The published estimate (Chernozhukov et al. 2018, Table 2), used to check my results.

```python
plt.rcParams.update({... "text.parse_math": False})
DOLLARS = plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
```
- Sets the chart style once. `parse_math: False` makes `$` print as a dollar sign
  instead of starting a math formula. `DOLLARS` formats axis numbers as `$5,000`.

---

## Part 1: Traditional econometrics, `ols_estimates()`

```python
naive = smf.ols("net_tfa ~ e401", df).fit(cov_type="HC1")
```
- **OLS with no controls:** compares the average savings of eligible and non-eligible
  households. This is just the difference in means.
- `cov_type="HC1"` gives **robust standard errors**, which stay valid when the spread of
  wealth differs across households (it does, a lot).

```python
controls = smf.ols("net_tfa ~ e401 + age + inc + ...", df).fit(cov_type="HC1")
```
- **OLS with controls:** compares households that have the same age, income, education
  and so on. The catch is that OLS assumes each control affects savings in a
  **straight line**. In reality, savings rise much faster at high incomes, for example.
- The full regression table is saved to `results/ols_with_controls_summary.txt`.

---

## Part 2: Machine learning + econometrics, Double Machine Learning

### `make_learner(name)` builds the three ML models

| Model | Code | Idea |
|---|---|---|
| Lasso | `PolynomialFeatures(2)` → `StandardScaler()` → `LassoCV(cv=5)` | Creates every square and interaction of the controls (e.g. income², age × education), then lets Lasso keep only the useful ones. `LassoCV` picks the penalty by cross-validation. |
| Random forest | `RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=4)` | Averages 300 decision trees. `min_samples_leaf=5` stops trees from memorising single households. |
| Gradient boosting | `GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)` | Builds small trees one after another, each fixing the previous ones' mistakes. |

`random_state=0` makes the results repeatable.

### `dml_plr()` is the core method

The **partially linear model** is:

```
savings = θ × 401(k)-eligible + g(age, income, ...) + error
```

- `θ` (theta) is the effect we want.
- `g(...)` is "everything else". The ML models learn it, so it can have any shape.

The code works in three steps.

```python
for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
    y_res[test] = y[test] - learner().fit(X[train], y[train]).predict(X[test])
    d_res[test] = d[test] - learner().fit(X[train], d[train]).predict(X[test])
```
1. **Predict savings from X, and predict eligibility from X, with ML.**
2. **Keep the residuals** (actual minus predicted). `y_res` is the part of savings that
   the controls can't explain, and `d_res` is the same for eligibility.
- **Cross-fitting (`KFold`):** the data is split into 5 parts. Each part is predicted by a
  model trained on the other 4, so no household is predicted by a model that saw it.
  This stops overfitting from biasing θ, one of the paper's two key ideas.

```python
theta = np.sum(d_res * y_res) / np.sum(d_res ** 2)
```
3. **Regress the savings residual on the eligibility residual.** The slope is θ. This
   "partialling out" is the Frisch–Waugh–Lovell theorem from econometrics, with ML
   doing the partialling. Using residuals of *both* Y and D is the paper's other key
   idea. It is what makes the estimate "debiased".

```python
psi = (y_res - theta * d_res) * d_res
se = np.sqrt(np.mean(psi ** 2)) / np.mean(d_res ** 2) / np.sqrt(len(y))
```
- The **standard error**, computed from the "score function" `psi`. It is the
  robust standard error of that final regression.

### `dml_estimates()` repeats DML and combines the runs

```python
runs = [dml_plr(..., seed=s) for s in range(n_repeats)]
theta = np.median(thetas)
se = np.sqrt(np.median(ses ** 2 + (thetas - theta) ** 2))
```
- The answer depends a little on how the 5 folds are drawn, so DML is run 5 times
  with different random splits (the paper uses 100).
- It reports the **median** estimate, and a standard error that also counts how much the
  estimate moved between splits. This is the paper's formula.

### `check_with_doubleml_package()` double-checks my code

```python
data  = dml.DoubleMLData(df, y_col=Y, d_cols=D, x_cols=CONTROLS)
model = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=5)
model.fit()
```
- Runs the same model with the official `doubleml` package. If my hand-written
  version gives about the same answer, the implementation is right.
- `np.random.seed(0)` comes first because the package picks its folds with numpy's
  random generator.

---

## Part 3: Heterogeneity, `effect_by_income()`

```python
df.assign(income_quintile=pd.qcut(df["inc"], 5, labels=[1, 2, 3, 4, 5]))
```
- `pd.qcut` splits households into 5 equal-sized income groups (quintiles).
- DML with a random forest is then run separately in each group, to see who benefits
  most from 401(k) eligibility.

---

## Charts: `plot_estimates()` and `plot_by_income()`
- `ax.errorbar(..., xerr=1.96 * std_error)` draws the **95% confidence interval**
  (estimate ± 1.96 standard errors).
- `ax.axvline(PAPER_ESTIMATE, ls="--")` draws the published estimate as a dashed line,
  so you can see the replication match.
- Orange dots are OLS and blue dots are DML.

## `main()`
1. Loads `data/sipp1991.csv`.
2. Saves average characteristics of eligible vs non-eligible households. This shows they
   differ (eligible households earn ~$15k more), which is why controls matter.
3. Runs Parts 1–3, saves CSVs and charts, and prints everything.

## Output files in `results/`
| File | Contents |
|---|---|
| `estimates.csv` | All estimates with standard errors and 95% CIs |
| `effect_by_income.csv` | DML effect for each income quintile |
| `summary_by_eligibility.csv` | Average household characteristics, eligible vs not |
| `ols_with_controls_summary.txt` | Full statsmodels OLS table |
| `estimates_comparison.png` | Chart: OLS vs DML vs the paper |
| `effect_by_income.png` | Chart: effect by income quintile |
| `console_output.txt` | Everything the script printed |
