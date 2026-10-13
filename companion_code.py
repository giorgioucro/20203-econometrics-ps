"""Python companion for Assignment 2 of Econometrics 20203.

The assignment models Italian quarterly year-on-year real GDP growth and compares
static regressions, dynamic ARDL specifications, an AR(2), and a three-variable
VAR(2).  This script reconstructs the EViews workflow recorded in the submitted
workfile and reproduces the numerical tables and figures in
``assignment2/PS2_Results_group5.pdf`` from ``assignment2/data_final.xlsx``.

Run from any working directory with::

    python companion_code.py

All paths are resolved relative to this file.  Deterministic CSV, text, and PNG
outputs are written to ``assignment2/output``.  The validation routine compares
every identifiable reported number with the Python result and explicitly records
the slide-level transcription and last-digit display discrepancies found during
reconstruction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Keep Matplotlib's font cache portable and writable in restricted environments.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    recursive_olsresiduals,
)
from statsmodels.stats.stattools import jarque_bera


# -----------------------------------------------------------------------------
# 1. Paths, samples, and reference conventions
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "assignment2" / "data_final.xlsx"
OUTPUT_DIR = ROOT / "assignment2" / "output"

FULL_DESCRIPTIVE_SAMPLE = ("1997Q1", "2026Q1")
ESTIMATION_SAMPLE = ("1999Q1", "2009Q4")
FORECAST_SAMPLE = ("2010Q1", "2019Q4")
FULL_STABILITY_SAMPLE = ("1999Q1", "2019Q4")
SHORT_SAMPLE = ("1999Q1", "2006Q4")

GROWTH_VARIABLES = [
    "gdp_yoy",
    "hicp_yoy",
    "ip_yoy",
    "esi_yoy",
    "ftsemib_yoy",
]

DISPLAY_NAMES = {
    "gdp_yoy": "GDP",
    "hicp_yoy": "HICP",
    "ip_yoy": "IP",
    "esi_yoy": "ESI",
    "ftsemib_yoy": "FTSE MIB",
}

MODEL_SPECS = {
    "M1": ["ip_yoy", "esi_yoy", "ftsemib_yoy", "hicp_yoy"],
    "M2": ["ip_yoy", "esi_yoy", "ftsemib_yoy"],
    "M3": ["ip_yoy", "esi_yoy"],
    "D1": [
        "gdp_yoy_l1",
        "ip_yoy",
        "ip_yoy_l1",
        "esi_yoy",
        "esi_yoy_l1",
        "ftsemib_yoy",
        "ftsemib_yoy_l1",
        "hicp_yoy",
        "hicp_yoy_l1",
    ],
    "D2": [
        "gdp_yoy_l1",
        "ip_yoy",
        "esi_yoy_l1",
        "ftsemib_yoy_l1",
        "hicp_yoy",
    ],
    "D3": ["gdp_yoy_l1", "ip_yoy", "esi_yoy_l1"],
    "D4": ["gdp_yoy_l1", "ip_yoy", "ip_yoy_l1"],
}


def ensure_inputs() -> None:
    """Fail early with a useful message when the repository is incomplete."""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Required input not found: {DATA_FILE}\n"
            "Keep data_final.xlsx in the assignment2 directory."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Read the consolidated quarterly database and construct EViews-style lags."""

    data = pd.read_excel(DATA_FILE, sheet_name="db", engine="openpyxl")
    expected = {"date", *GROWTH_VARIABLES, "euribor"}
    missing_columns = expected.difference(data.columns)
    if missing_columns:
        raise ValueError(f"data_final.xlsx is missing columns: {sorted(missing_columns)}")

    labels = data["date"].astype(str).str.replace("-", "", regex=False)
    data["date"] = pd.PeriodIndex(labels, freq="Q")
    data = data.set_index("date").sort_index()

    if not data.index.is_unique:
        raise ValueError("The quarterly date index contains duplicates.")
    if data.index.min() != pd.Period("1997Q1", freq="Q"):
        raise ValueError(f"Unexpected first quarter: {data.index.min()}")
    if data.index.max() != pd.Period("2026Q1", freq="Q"):
        raise ValueError(f"Unexpected last quarter: {data.index.max()}")

    for variable in GROWTH_VARIABLES:
        data[f"{variable}_l1"] = data[variable].shift(1)
    data["gdp_yoy_l2"] = data["gdp_yoy"].shift(2)
    data["hicp_yoy_l2"] = data["hicp_yoy"].shift(2)
    data["euribor_l1"] = data["euribor"].shift(1)
    data["euribor_l2"] = data["euribor"].shift(2)
    return data


def reference_descriptive_data(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the descriptive sample used in slides 5-8.

    The submitted workbook's ``db_`` sheet multiplies cells on ``db`` by 100.
    Excel therefore cached blank formula references as zeros.  The EViews history
    then reset pre-1999 FTSE MIB zeros to missing, but retained the cached 2026Q1
    zeros for GDP, IP, and FTSE MIB.  The report's counts, means, and correlations
    prove that this convention was used.  It is reproduced here only for the
    descriptive results; regressions use the unmodified raw ``db`` observations.
    """

    descriptive = data.loc[
        FULL_DESCRIPTIVE_SAMPLE[0] : FULL_DESCRIPTIVE_SAMPLE[1], GROWTH_VARIABLES
    ].copy()
    descriptive *= 100.0

    final_period = pd.Period("2026Q1", freq="Q")
    for variable in ("gdp_yoy", "ip_yoy", "ftsemib_yoy"):
        if pd.isna(descriptive.loc[final_period, variable]):
            descriptive.loc[final_period, variable] = 0.0
    descriptive.loc[:"1998Q4", "ftsemib_yoy"] = np.nan
    return descriptive


# -----------------------------------------------------------------------------
# 2. Descriptive statistics and figures (slides 5-8)
# -----------------------------------------------------------------------------


def descriptive_statistics(descriptive: pd.DataFrame) -> pd.DataFrame:
    """Use EViews' population-moment definitions for skewness and kurtosis."""

    rows: dict[str, list[float]] = {}
    rows["Mean"] = [descriptive[c].mean() for c in GROWTH_VARIABLES]
    rows["Std. Dev."] = [descriptive[c].std(ddof=1) for c in GROWTH_VARIABLES]
    rows["Min"] = [descriptive[c].min() for c in GROWTH_VARIABLES]
    rows["Max"] = [descriptive[c].max() for c in GROWTH_VARIABLES]
    rows["Skewness"] = [
        stats.skew(descriptive[c].dropna(), bias=True) for c in GROWTH_VARIABLES
    ]
    rows["Kurtosis"] = [
        stats.kurtosis(descriptive[c].dropna(), fisher=False, bias=True)
        for c in GROWTH_VARIABLES
    ]
    rows["Obs."] = [int(descriptive[c].count()) for c in GROWTH_VARIABLES]
    table = pd.DataFrame(rows, index=GROWTH_VARIABLES).T
    table.columns = [DISPLAY_NAMES[c] for c in GROWTH_VARIABLES]
    return table


def plot_descriptive_series(descriptive: pd.DataFrame) -> None:
    """Recreate the two time-series figure groups in the report."""

    raw_plot = descriptive / 100.0
    x = raw_plot.index.to_timestamp(how="end")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        ("gdp_yoy", "Real GDP growth"),
        ("hicp_yoy", "HICP inflation"),
        ("ip_yoy", "Industrial production"),
    ]
    for axis, (variable, title) in zip(axes.flat, panels, strict=False):
        axis.plot(x, raw_plot[variable], color="#2F61D5", linewidth=1.1)
        axis.axhline(0, color="0.65", linewidth=0.7)
        axis.set_title(title)
        axis.grid(alpha=0.18)
    axes[1, 1].axis("off")
    fig.suptitle("Italy: quarterly year-on-year growth rates, 1997Q1-2026Q1")
    fig.savefig(OUTPUT_DIR / "time_series_hard_indicators.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    panels = [("esi_yoy", "ESI Italy"), ("ftsemib_yoy", "FTSE MIB")]
    for axis, (variable, title) in zip(axes, panels, strict=True):
        axis.plot(x, raw_plot[variable], color="#2F61D5", linewidth=1.1)
        axis.axhline(0, color="0.65", linewidth=0.7)
        axis.set_title(title)
        axis.grid(alpha=0.18)
    fig.suptitle("Italy: survey and financial indicators")
    fig.savefig(OUTPUT_DIR / "time_series_soft_financial.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Static and dynamic model specifications (slides 9-13)
# -----------------------------------------------------------------------------


def fit_ols(data: pd.DataFrame, regressors: list[str], sample: tuple[str, str]):
    """Fit OLS after applying the EViews date sample and listwise deletion."""

    frame = data.loc[sample[0] : sample[1], ["gdp_yoy", *regressors]].dropna()
    exog = sm.add_constant(frame[regressors], has_constant="add")
    return sm.OLS(frame["gdp_yoy"], exog).fit()


def information_criteria_per_observation(model) -> tuple[float, float, float]:
    """Return the EViews AIC, SIC, and HQ values reported per observation."""

    nobs = float(model.nobs)
    parameters = len(model.params)
    aic = model.aic / nobs
    sic = model.bic / nobs
    hq = (-2.0 * model.llf + 2.0 * parameters * np.log(np.log(nobs))) / nobs
    return float(aic), float(sic), float(hq)


def fit_assignment_models(data: pd.DataFrame):
    """Estimate M1-M3 and D1-D4 on the model-selection sample."""

    return {
        name: fit_ols(data, regressors, ESTIMATION_SAMPLE)
        for name, regressors in MODEL_SPECS.items()
    }


def model_results_table(models: dict[str, object]) -> pd.DataFrame:
    """Create a tidy coefficient and fit-statistics table for CSV output."""

    records: list[dict[str, object]] = []
    for name, model in models.items():
        aic, sic, hq = information_criteria_per_observation(model)
        for term in model.params.index:
            records.append(
                {
                    "model": name,
                    "term": term,
                    "coefficient": model.params[term],
                    "standard_error": model.bse[term],
                    "t_statistic": model.tvalues[term],
                    "p_value": model.pvalues[term],
                    "observations": int(model.nobs),
                    "adjusted_r_squared": model.rsquared_adj,
                    "aic_per_observation": aic,
                    "sic_per_observation": sic,
                    "hq_per_observation": hq,
                }
            )
    return pd.DataFrame.from_records(records)


def short_sample_results(data: pd.DataFrame) -> pd.DataFrame:
    """Check the report's qualitative robustness claim for 1999Q1-2006Q4."""

    rows = []
    for name in ("M1", "M2", "M3"):
        model = fit_ols(data, MODEL_SPECS[name], SHORT_SAMPLE)
        aic, sic, hq = information_criteria_per_observation(model)
        rows.append(
            {
                "model": f"{name}_short",
                "observations": int(model.nobs),
                "adjusted_r_squared": model.rsquared_adj,
                "aic_per_observation": aic,
                "sic_per_observation": sic,
                "hq_per_observation": hq,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. D4 diagnostics and parameter stability (slides 14-16)
# -----------------------------------------------------------------------------


def chow_breakpoint_test(
    data: pd.DataFrame, breakpoint: str
) -> tuple[float, float, int, int]:
    """EViews-style Chow breakpoint test with the breakpoint in subsample two."""

    regressors = MODEL_SPECS["D4"]
    frame = data.loc[
        FULL_STABILITY_SAMPLE[0] : FULL_STABILITY_SAMPLE[1],
        ["gdp_yoy", *regressors],
    ].dropna()
    x_full = sm.add_constant(frame[regressors], has_constant="add")
    pooled = sm.OLS(frame["gdp_yoy"], x_full).fit()

    breakpoint_period = pd.Period(breakpoint, freq="Q")
    before = frame.loc[: breakpoint_period - 1]
    after = frame.loc[breakpoint_period:]
    model_before = sm.OLS(
        before["gdp_yoy"], sm.add_constant(before[regressors], has_constant="add")
    ).fit()
    model_after = sm.OLS(
        after["gdp_yoy"], sm.add_constant(after[regressors], has_constant="add")
    ).fit()

    k = len(pooled.params)
    n = len(frame)
    unrestricted_ssr = model_before.ssr + model_after.ssr
    f_statistic = ((pooled.ssr - unrestricted_ssr) / k) / (
        unrestricted_ssr / (n - 2 * k)
    )
    p_value = stats.f.sf(f_statistic, k, n - 2 * k)
    return float(f_statistic), float(p_value), k, n - 2 * k


def diagnostic_results(data: pd.DataFrame, d4_model) -> tuple[pd.DataFrame, dict]:
    """Reproduce EViews' reported diagnostic-test p-values.

    The slide table reports the F-test probabilities, not the LM probabilities,
    for BPG, White, and Breusch-Godfrey.  ``robust=False`` selects the original
    Breusch-Pagan-Godfrey LM statistic; its F form is identical to EViews here.
    """

    bpg_lm, bpg_lm_p, bpg_f, bpg_f_p = het_breuschpagan(
        d4_model.resid, d4_model.model.exog, robust=False
    )
    white_lm, white_lm_p, white_f, white_f_p = het_white(
        d4_model.resid, d4_model.model.exog
    )
    bg_lm, bg_lm_p, bg_f, bg_f_p = acorr_breusch_godfrey(
        d4_model, nlags=4, result_object=False
    )
    jb_stat, jb_p, resid_skew, resid_kurtosis = jarque_bera(d4_model.resid)
    chow_2008 = chow_breakpoint_test(data, "2008Q4")
    chow_2009 = chow_breakpoint_test(data, "2009Q1")

    rows = [
        {
            "test": "Breusch-Pagan-Godfrey homoskedasticity",
            "sample": "1999Q1-2009Q4",
            "statistic": bpg_f,
            "p_value": bpg_f_p,
            "reported_statistic_type": "F",
        },
        {
            "test": "White homoskedasticity",
            "sample": "1999Q1-2009Q4",
            "statistic": white_f,
            "p_value": white_f_p,
            "reported_statistic_type": "F",
        },
        {
            "test": "Breusch-Godfrey serial correlation LM(4)",
            "sample": "1999Q1-2009Q4",
            "statistic": bg_f,
            "p_value": bg_f_p,
            "reported_statistic_type": "F",
        },
        {
            "test": "Jarque-Bera residual normality",
            "sample": "1999Q1-2009Q4",
            "statistic": jb_stat,
            "p_value": jb_p,
            "reported_statistic_type": "chi-square",
        },
        {
            "test": "Chow breakpoint at 2008Q4",
            "sample": "1999Q1-2019Q4",
            "statistic": chow_2008[0],
            "p_value": chow_2008[1],
            "reported_statistic_type": f"F({chow_2008[2]},{chow_2008[3]})",
        },
        {
            "test": "Chow breakpoint at 2009Q1",
            "sample": "1999Q1-2019Q4",
            "statistic": chow_2009[0],
            "p_value": chow_2009[1],
            "reported_statistic_type": f"F({chow_2009[2]},{chow_2009[3]})",
        },
    ]
    details = {
        "bpg_lm": bpg_lm,
        "bpg_lm_p": bpg_lm_p,
        "white_lm": white_lm,
        "white_lm_p": white_lm_p,
        "bg_lm": bg_lm,
        "bg_lm_p": bg_lm_p,
        "residual_skewness": resid_skew,
        "residual_kurtosis": resid_kurtosis,
    }
    return pd.DataFrame(rows), details


def residual_statistics(d4_model) -> dict[str, float]:
    """Statistics shown beside the D4 residual histogram on slide 15."""

    residuals = pd.Series(d4_model.resid)
    jb_stat, jb_p, skewness, kurtosis = jarque_bera(residuals)
    return {
        "Mean": float(residuals.mean()),
        "Median": float(residuals.median()),
        "Maximum": float(residuals.max()),
        "Minimum": float(residuals.min()),
        "Std. Dev.": float(residuals.std(ddof=1)),
        "Skewness": float(skewness),
        "Kurtosis": float(kurtosis),
        "Jarque-Bera": float(jb_stat),
        "Probability": float(jb_p),
    }


def plot_residual_histogram(d4_model, statistics: dict[str, float]) -> None:
    """Recreate the residual histogram and its summary panel."""

    residuals = np.asarray(d4_model.resid)
    fig, (axis, text_axis) = plt.subplots(
        1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": [3, 1.45]}
    )
    axis.hist(residuals, bins=10, color="#8BB6E9", edgecolor="#4677A8")
    axis.set_title("D4 residuals, 1999Q1-2009Q4")
    axis.set_xlabel("Residual")
    axis.set_ylabel("Frequency")
    axis.grid(axis="y", alpha=0.18)

    text_axis.axis("off")
    lines = [f"Observations  {int(d4_model.nobs)}", ""]
    for label, value in statistics.items():
        if label == "Mean":
            lines.append(f"{label:<13} {value:.2e}")
        else:
            lines.append(f"{label:<13} {value:.6f}")
    text_axis.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d4_residual_histogram.png", dpi=180)
    plt.close(fig)


def expanding_recursions(d4_model) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute recursive coefficients, one-step residuals, and forecast bands."""

    endog = np.asarray(d4_model.model.endog)
    exog = np.asarray(d4_model.model.exog)
    index = d4_model.model.data.row_labels
    k = exog.shape[1]
    records = []
    recursive_residuals = []
    recursive_se = []

    for current in range(k, len(endog)):
        prior_y = endog[:current]
        prior_x = exog[:current]
        prior_model = sm.OLS(prior_y, prior_x).fit()
        x_current = exog[current]
        prediction = float(x_current @ prior_model.params)
        innovation = float(endog[current] - prediction)
        leverage = float(x_current @ np.linalg.pinv(prior_x.T @ prior_x) @ x_current)
        forecast_se = float(np.sqrt(prior_model.mse_resid * (1.0 + leverage)))
        recursive_residuals.append(innovation)
        recursive_se.append(forecast_se)

        current_model = sm.OLS(endog[: current + 1], exog[: current + 1]).fit()
        row = {"date": index[current]}
        for position, term in enumerate(d4_model.params.index):
            row[f"{term}_coefficient"] = current_model.params[position]
            row[f"{term}_standard_error"] = current_model.bse[position]
        records.append(row)

    result = pd.DataFrame(records).set_index("date")
    return result, np.asarray(recursive_residuals), np.asarray(recursive_se)


def plot_recursive_diagnostics(d4_model) -> None:
    """Recreate CUSUM, CUSUMSQ, recursive residual, and slope paths."""

    rec_table, rec_residuals, rec_se = expanding_recursions(d4_model)
    rec_table.to_csv(OUTPUT_DIR / "recursive_coefficients.csv", index_label="date")

    output = recursive_olsresiduals(d4_model, skip=len(d4_model.params), alpha=0.95)
    # statsmodels prepends the initializing zero to CUSUM but returns confidence
    # bounds only for the subsequent n-k recursive observations.
    rcusum = np.asarray(output[5])[1:]
    rcusum_bounds = np.asarray(output[6])
    dates = rec_table.index.to_timestamp(how="end")

    # The recursive residual sequence contains n-k observations.  CUSUM has one
    # fewer element in statsmodels because the first term initializes the sum.
    cusum_dates = dates[-len(rcusum) :]
    if rcusum_bounds.shape[0] == 2:
        lower, upper = rcusum_bounds
    else:
        lower, upper = rcusum_bounds[:, 0], rcusum_bounds[:, 1]
    if len(lower) != len(rcusum):
        lower = np.interp(
            np.linspace(0, 1, len(rcusum)), np.linspace(0, 1, len(lower)), lower
        )
        upper = np.interp(
            np.linspace(0, 1, len(rcusum)), np.linspace(0, 1, len(upper)), upper
        )

    standardized = rec_residuals / rec_se
    squares = standardized**2
    cusum_sq = np.cumsum(squares) / np.sum(squares)
    positions = np.arange(1, len(cusum_sq) + 1) / len(cusum_sq)
    boundary_offset = 0.948 / np.sqrt(len(cusum_sq))
    cusum_sq_lower = positions - boundary_offset
    cusum_sq_upper = positions + boundary_offset

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)

    axes[0, 0].plot(cusum_dates, rcusum, color="#2F61D5", label="CUSUM")
    axes[0, 0].plot(cusum_dates, lower, "--", color="#D9534F", label="5% boundary")
    axes[0, 0].plot(cusum_dates, upper, "--", color="#D9534F")
    axes[0, 0].axhline(0, color="0.55", linewidth=0.7)
    axes[0, 0].set_title("CUSUM")
    axes[0, 0].legend(fontsize=8)

    axes[1, 0].plot(dates, cusum_sq, color="#2F61D5", label="CUSUM of squares")
    axes[1, 0].plot(dates, cusum_sq_lower, "--", color="#D9534F", label="5% boundary")
    axes[1, 0].plot(dates, cusum_sq_upper, "--", color="#D9534F")
    axes[1, 0].set_title("CUSUM of squares")
    axes[1, 0].legend(fontsize=8)

    axes[2, 0].plot(dates, rec_residuals, color="#2F61D5", label="Recursive residual")
    axes[2, 0].plot(dates, 2 * rec_se, "--", color="#D9534F", label="+/- 2 S.E.")
    axes[2, 0].plot(dates, -2 * rec_se, "--", color="#D9534F")
    axes[2, 0].axhline(0, color="0.55", linewidth=0.7)
    axes[2, 0].set_title("Recursive residuals")
    axes[2, 0].legend(fontsize=8)

    for axis, term in zip(
        axes[:, 1], ["gdp_yoy_l1", "ip_yoy", "ip_yoy_l1"], strict=True
    ):
        coefficient = rec_table[f"{term}_coefficient"]
        standard_error = rec_table[f"{term}_standard_error"]
        axis.plot(dates, coefficient, color="#2F61D5", label=f"Recursive {term}")
        axis.plot(
            dates,
            coefficient + 2 * standard_error,
            "--",
            color="#D9534F",
            label="+/- 2 S.E.",
        )
        axis.plot(dates, coefficient - 2 * standard_error, "--", color="#D9534F")
        axis.axhline(0, color="0.55", linewidth=0.7)
        axis.set_title(f"Recursive coefficient: {term}")
        axis.legend(fontsize=8)

    fig.suptitle("D4 recursive stability diagnostics")
    fig.savefig(OUTPUT_DIR / "d4_recursive_diagnostics.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. One-step-ahead forecast comparison (slides 17-18)
# -----------------------------------------------------------------------------


def fit_ar2(data: pd.DataFrame):
    """Estimate the AR(2) benchmark, using presample lags as EViews does."""

    return fit_ols(data, ["gdp_yoy_l1", "gdp_yoy_l2"], ESTIMATION_SAMPLE)


def fit_var2_equations(data: pd.DataFrame) -> dict[str, object]:
    """Estimate the VAR(2) equation-by-equation with EViews sample semantics.

    EViews estimates dependent observations from 1999Q1 through 2009Q4 while
    taking the two lags from 1998Q3-1998Q4, hence it reports 44 observations.
    ``statsmodels.VAR`` would instead discard the first two observations after
    subsetting and report 42, so the equivalent OLS system is implemented here.
    """

    endogenous = ["gdp_yoy", "hicp_yoy", "euribor"]
    regressors = [
        "gdp_yoy_l1",
        "hicp_yoy_l1",
        "euribor_l1",
        "gdp_yoy_l2",
        "hicp_yoy_l2",
        "euribor_l2",
    ]
    sample = data.loc[ESTIMATION_SAMPLE[0] : ESTIMATION_SAMPLE[1]]
    exog = sm.add_constant(sample[regressors], has_constant="add")
    return {variable: sm.OLS(sample[variable], exog).fit() for variable in endogenous}


def static_prediction(model, exog: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return EViews-style static predictions and prediction intervals."""

    x = sm.add_constant(exog, has_constant="add")
    prediction = model.get_prediction(x)
    frame = prediction.summary_frame(alpha=0.05)
    return pd.Series(frame["mean"], index=exog.index), frame.set_axis(exog.index)


def forecast_statistics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Compute the EViews forecast-evaluation measures shown on slide 17.

    Inputs are percentage points.  EViews' Theil U2 calculation loses the first
    evaluation quarter because its relative-error formula uses a within-sample lag.
    """

    actual, forecast = actual.align(forecast, join="inner")
    error = forecast - actual
    mse = float(np.mean(error**2))
    actual_sd = float(np.sqrt(np.mean((actual - actual.mean()) ** 2)))
    forecast_sd = float(np.sqrt(np.mean((forecast - forecast.mean()) ** 2)))
    correlation = float(np.corrcoef(actual, forecast)[0, 1])

    previous_actual = actual.shift(1)
    relative_forecast_error = ((forecast - actual) / previous_actual).iloc[1:]
    relative_naive_error = ((previous_actual - actual) / previous_actual).iloc[1:]

    return {
        "RMSFE": float(np.sqrt(mse)),
        "MAFE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100.0),
        "Theil inequality coefficient": float(
            np.sqrt(mse)
            / (np.sqrt(np.mean(actual**2)) + np.sqrt(np.mean(forecast**2)))
        ),
        "Bias proportion": float((forecast.mean() - actual.mean()) ** 2 / mse),
        "Variance proportion": float((forecast_sd - actual_sd) ** 2 / mse),
        "Covariance proportion": float(
            2.0 * (1.0 - correlation) * forecast_sd * actual_sd / mse
        ),
        "Theil U2 coefficient": float(
            np.sqrt(
                np.sum(relative_forecast_error**2)
                / np.sum(relative_naive_error**2)
            )
        ),
        "Symmetric MAPE": float(
            np.mean(2.0 * np.abs(error) / (np.abs(actual) + np.abs(forecast)))
            * 100.0
        ),
    }


def forecast_comparison(data: pd.DataFrame, models: dict[str, object]):
    """Generate static D4, AR(2), and VAR(2) forecasts and accuracy measures."""

    evaluation = data.loc[FORECAST_SAMPLE[0] : FORECAST_SAMPLE[1]]
    actual = evaluation["gdp_yoy"] * 100.0

    d4_exog = evaluation[MODEL_SPECS["D4"]]
    d4_forecast_raw, d4_interval_raw = static_prediction(models["D4"], d4_exog)

    ar2_model = fit_ar2(data)
    ar2_exog = evaluation[["gdp_yoy_l1", "gdp_yoy_l2"]]
    ar2_forecast_raw, ar2_interval_raw = static_prediction(ar2_model, ar2_exog)

    var2_models = fit_var2_equations(data)
    var2_regressors = [
        "gdp_yoy_l1",
        "hicp_yoy_l1",
        "euribor_l1",
        "gdp_yoy_l2",
        "hicp_yoy_l2",
        "euribor_l2",
    ]
    var2_x = sm.add_constant(evaluation[var2_regressors], has_constant="add")
    var2_forecast_raw = var2_models["gdp_yoy"].predict(var2_x)

    forecasts = pd.DataFrame(
        {
            "actual_gdp_yoy": actual,
            "D4": d4_forecast_raw * 100.0,
            "AR2": ar2_forecast_raw * 100.0,
            "VAR2": var2_forecast_raw * 100.0,
        },
        index=evaluation.index,
    )
    intervals = {
        "D4": d4_interval_raw * 100.0,
        "AR2": ar2_interval_raw * 100.0,
    }
    accuracy = {
        model_name: forecast_statistics(actual, forecasts[model_name])
        for model_name in ("D4", "AR2", "VAR2")
    }
    return forecasts, accuracy, intervals, ar2_model, var2_models


def plot_forecasts(
    forecasts: pd.DataFrame, intervals: dict[str, pd.DataFrame]
) -> None:
    """Recreate the D4/AR(2) and VAR(2) forecast figures."""

    x = forecasts.index.to_timestamp(how="end")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, model_name in zip(axes, ("D4", "AR2"), strict=True):
        frame = intervals[model_name]
        axis.plot(x, forecasts["actual_gdp_yoy"], color="#4C9A57", label="Actual")
        axis.plot(x, forecasts[model_name], color="#2F61D5", label=model_name)
        axis.plot(x, frame["obs_ci_lower"], color="#E07A73", linewidth=0.8, label="95% PI")
        axis.plot(x, frame["obs_ci_upper"], color="#E07A73", linewidth=0.8)
        axis.axhline(0, color="0.6", linewidth=0.7)
        axis.set_ylabel("Percentage points")
        axis.set_title(f"One-step-ahead GDP growth forecast: {model_name}")
        axis.legend(ncol=3, fontsize=8)
        axis.grid(alpha=0.18)
    fig.savefig(OUTPUT_DIR / "forecast_d4_ar2.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.plot(x, forecasts["actual_gdp_yoy"], color="#2F61D5", label="Actual GDP growth")
    axis.plot(x, forecasts["VAR2"], color="#E05252", label="VAR(2) forecast")
    axis.axhline(0, color="0.6", linewidth=0.7)
    axis.set_ylabel("Percentage points")
    axis.set_title("One-step-ahead GDP growth forecast: VAR(2)")
    axis.legend()
    axis.grid(alpha=0.18)
    fig.savefig(OUTPUT_DIR / "forecast_var2.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Numerical validation against PS2_Results_group5.pdf
# -----------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    section: str
    result: str
    reference_value: float | str
    python_value: float | str
    reported_decimals: int | str
    status: str
    note: str = ""


def rounded_match(actual: float, reference: float, decimals: int) -> bool:
    """Compare at the precision printed in the report."""

    tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
    return bool(np.isclose(actual, reference, atol=tolerance, rtol=0.0))


def add_numeric_check(
    checks: list[ValidationCheck],
    section: str,
    result: str,
    reference: float,
    actual: float,
    decimals: int,
    known_discrepancy: str = "",
    absolute_tolerance: float | None = None,
) -> None:
    if absolute_tolerance is not None:
        matches = bool(np.isclose(actual, reference, atol=absolute_tolerance, rtol=0.0))
    else:
        matches = rounded_match(actual, reference, decimals)
    if matches:
        status = "exact to reported precision"
        note = ""
    elif known_discrepancy:
        status = "documented report discrepancy"
        note = known_discrepancy
    else:
        status = "unexpected mismatch"
        note = "Requires investigation."
    checks.append(
        ValidationCheck(section, result, reference, actual, decimals, status, note)
    )


def build_validation(
    descriptive_stats: pd.DataFrame,
    correlations: pd.DataFrame,
    models: dict[str, object],
    diagnostics: pd.DataFrame,
    residual_stats: dict[str, float],
    forecast_accuracy: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compare every identifiable reported number with the Python result."""

    checks: list[ValidationCheck] = []

    reported_descriptive = {
        "Mean": [0.710178, 2.087664, -0.374246, 1.281098, 3.389088],
        "Std. Dev.": [3.507082, 1.937929, 6.384778, 14.53332, 21.91750],
        "Min": [-18.13402, -0.399202, -25.94433, -38.54712, -50.64535],
        "Max": [19.31833, 12.43243, 33.28859, 79.55272, 49.35048],
        "Skewness": [-0.222550, 2.543296, -0.208938, 1.514986, -0.292952],
        "Kurtosis": [15.54186, 12.01236, 12.65136, 9.636316, 2.414696],
        "Obs.": [117, 117, 117, 117, 109],
    }
    row_decimals = {
        "Mean": 6,
        "Std. Dev.": 5,
        "Min": 5,
        "Max": 5,
        "Skewness": 6,
        "Kurtosis": 5,
        "Obs.": 0,
    }
    for row, references in reported_descriptive.items():
        for column, reference in zip(descriptive_stats.columns, references, strict=True):
            add_numeric_check(
                checks,
                "Slide 7: descriptive statistics",
                f"{row} - {column}",
                float(reference),
                float(descriptive_stats.loc[row, column]),
                row_decimals[row],
            )

    reported_correlations = np.array(
        [
            [1.0, 0.245244, 0.855675, 0.682679, 0.330636],
            [0.245244, 1.0, 0.033101, -0.175248, -0.132161],
            [0.855675, 0.033101, 1.0, 0.710253, 0.395200],
            [0.682679, -0.175248, 0.710253, 1.0, 0.558875],
            [0.330636, -0.132161, 0.395200, 0.558875, 1.0],
        ]
    )
    for row_index, row_name in enumerate(correlations.index):
        for column_index, column_name in enumerate(correlations.columns):
            add_numeric_check(
                checks,
                "Slide 8: correlation matrix",
                f"{row_name} with {column_name}",
                float(reported_correlations[row_index, column_index]),
                float(correlations.iloc[row_index, column_index]),
                6,
            )

    reported_models = {
        "M1": {
            "coef": [0.0187, 0.3698, 0.0123, -0.0011, -0.2692],
            "se": [0.0065, 0.0357, 0.0153, 0.0096, 0.2680],
            "fit": [0.8821, -6.5661, -6.3633, -6.4908, 44],
        },
        "M2": {
            "coef": [0.0123, 0.3448, 0.0126, 0.0046],
            "se": [0.0013, 0.0256, 0.0153, 0.0078],
            "fit": [0.8820, -6.5860, -6.4237, -6.5258, 44],
        },
        "M3": {
            "coef": [0.0123, 0.3492, 0.0171],
            "se": [0.0013, 0.0242, 0.0131],
            "fit": [0.8839, -6.6229, -6.5012, -6.5777, 44],
        },
        "D1": {
            "coef": [
                0.0054,
                0.7722,
                0.2396,
                -0.1919,
                0.0007,
                0.0095,
                0.0025,
                0.0066,
                0.1428,
                -0.2730,
            ],
            "se": [
                0.0057,
                0.1047,
                0.0444,
                0.0598,
                0.0161,
                0.0152,
                0.0084,
                0.0093,
                0.2426,
                0.2326,
            ],
            "fit": [0.9548, -7.4073, -6.9977, -7.2563, 43],
        },
        "D2": {
            "coef": [0.0172, 0.4414, 0.2144, 0.0091, 0.0063, -0.4450],
            "se": [0.0048, 0.0863, 0.0401, 0.0128, 0.0080, 0.2026],
            "fit": [0.9327, -7.0813, -6.8355, -6.9907, 43],
        },
        "D3": {
            "coef": [0.0080, 0.3565, 0.2239, 0.0198],
            "se": [0.0016, 0.0894, 0.0362, 0.0133],
            "fit": [0.9175, -6.9440, -6.7818, -6.8838, 44],
        },
        "D4": {
            "coef": [0.0024, 0.7979, 0.3120, -0.2410],
            "se": [0.0015, 0.1003, 0.0286, 0.0422],
            "fit": [0.9521, -7.4865, -7.3243, -7.4264, 44],
        },
    }
    information_criterion_display_differences = {
        ("M1", "HQ"),
        ("M2", "SIC"),
        ("M3", "SIC"),
        ("M3", "HQ"),
    }
    for model_name, reference in reported_models.items():
        model = models[model_name]
        for term, reference_value, actual in zip(
            model.params.index, reference["coef"], model.params, strict=True
        ):
            discrepancy = ""
            if model_name == "D4" and term == "ip_yoy":
                discrepancy = (
                    "The slide prints 0.3120, while the fitted model is 0.311696 "
                    "(0.3117 at four decimals). The latter is confirmed by the exact "
                    "reported D4 forecast statistics, so 0.3120 is a transcription error."
                )
            add_numeric_check(
                checks,
                f"Slides 10-11: {model_name}",
                f"coefficient {term}",
                reference_value,
                float(actual),
                4,
                discrepancy,
            )
        for term, reference_value, actual in zip(
            model.params.index, reference["se"], model.bse, strict=True
        ):
            add_numeric_check(
                checks,
                f"Slides 10-11: {model_name}",
                f"standard error {term}",
                reference_value,
                float(actual),
                4,
            )
        aic, sic, hq = information_criteria_per_observation(model)
        fit_actual = [model.rsquared_adj, aic, sic, hq, model.nobs]
        fit_names = ["Adjusted R2", "AIC", "SIC", "HQ", "Observations"]
        for name, reference_value, actual in zip(
            fit_names, reference["fit"], fit_actual, strict=True
        ):
            discrepancy = ""
            if (model_name, name) in information_criterion_display_differences:
                discrepancy = (
                    "The slide differs from the full-precision EViews-equivalent "
                    "calculation by less than 0.0001, consistent with manual "
                    "truncation or transcription in the presentation table."
                )
            add_numeric_check(
                checks,
                f"Slides 10-11: {model_name}",
                name,
                reference_value,
                float(actual),
                0 if name == "Observations" else 4,
                discrepancy,
            )

    reported_diagnostics = {
        "Breusch-Pagan-Godfrey homoskedasticity": 0.7872,
        "White homoskedasticity": 0.9663,
        "Breusch-Godfrey serial correlation LM(4)": 0.3367,
        "Chow breakpoint at 2008Q4": 0.3755,
        "Chow breakpoint at 2009Q1": 0.1997,
    }
    for _, row in diagnostics.iterrows():
        if row["test"] not in reported_diagnostics:
            continue
        discrepancy = ""
        if row["test"] == "White homoskedasticity":
            discrepancy = (
                "The slide prints 0.9663. The EViews-style White auxiliary regression "
                "with squares and cross-products gives F p=0.956338 (0.9563), indicating "
                "a slide transcription error; the inference is unchanged."
            )
        add_numeric_check(
            checks,
            "Slide 15: diagnostic tests",
            f"p-value {row['test']}",
            reported_diagnostics[row["test"]],
            float(row["p_value"]),
            4,
            discrepancy,
        )

    reported_residuals = {
        "Mean": -7.27e-19,
        "Median": 7.27e-05,
        "Maximum": 0.011328,
        "Minimum": -0.012354,
        "Std. Dev.": 0.005292,
        "Skewness": -0.148390,
        "Kurtosis": 2.630458,
        "Jarque-Bera": 0.411838,
        "Probability": 0.813899,
    }
    for label, reference in reported_residuals.items():
        add_numeric_check(
            checks,
            "Slide 15: residual histogram",
            label,
            reference,
            residual_stats[label],
            6,
            absolute_tolerance=1e-15 if label == "Mean" else None,
        )

    reported_forecasts = {
        "D4": {
            "RMSFE": 0.643893,
            "MAFE": 0.509110,
            "MAPE": 61.87589,
            "Theil inequality coefficient": 0.202658,
            "Bias proportion": 0.128715,
            "Variance proportion": 0.016799,
            "Covariance proportion": 0.854486,
            "Theil U2 coefficient": 0.662093,
            "Symmetric MAPE": 64.19367,
        },
        "AR2": {
            "RMSFE": 0.511406,
            "MAFE": 0.415279,
            "MAPE": 66.25868,
            "Theil inequality coefficient": 0.164105,
            "Bias proportion": 0.074128,
            "Variance proportion": 0.055957,
            "Covariance proportion": 0.869915,
            "Theil U2 coefficient": 0.785583,
            "Symmetric MAPE": 55.66256,
        },
        "VAR2": {"RMSFE": 2.2400, "MAFE": 1.9962},
    }
    for model_name, references in reported_forecasts.items():
        for metric, reference in references.items():
            decimals = 4 if model_name == "VAR2" else (5 if metric in {"MAPE", "Symmetric MAPE"} else 6)
            add_numeric_check(
                checks,
                f"Slides 17-18: {model_name} forecasts",
                metric,
                reference,
                forecast_accuracy[model_name][metric],
                decimals,
            )

    checks.extend(
        [
            ValidationCheck(
                "Slides 5-6: time series",
                "GDP, HICP, IP, ESI, and FTSE MIB plots",
                "five quarterly series",
                "time_series_hard_indicators.png and time_series_soft_financial.png",
                "visual",
                "recreated",
                "Same variables, sample, raw-rate scale, and reference zero handling.",
            ),
            ValidationCheck(
                "Slide 16: recursive diagnostics",
                "CUSUM, CUSUMSQ, recursive residuals, and C(2)-C(4) paths",
                "six panels",
                "d4_recursive_diagnostics.png",
                "visual",
                "recreated",
                "Econometrically equivalent recursive OLS calculations; drawing style and confidence-bound interpolation may differ from EViews.",
            ),
            ValidationCheck(
                "Slides 17-18: forecast plots",
                "D4, AR(2), and VAR(2) forecast series",
                "three forecast panels",
                "forecast_d4_ar2.png and forecast_var2.png",
                "visual",
                "recreated",
                "Forecast series generate the exact reported accuracy statistics.",
            ),
        ]
    )

    validation = pd.DataFrame(asdict(check) for check in checks)
    unexpected = validation[validation["status"] == "unexpected mismatch"]
    if not unexpected.empty:
        labels = ", ".join(unexpected["result"].astype(str))
        raise AssertionError(f"Unexpected validation mismatches: {labels}")
    return validation


def write_validation_report(validation: pd.DataFrame) -> None:
    """Save a concise human-readable validation summary."""

    counts = validation["status"].value_counts()
    lines = [
        "Assignment 2 Python replication - validation summary",
        "=" * 58,
        "",
        f"Exact to reported precision: {counts.get('exact to reported precision', 0)}",
        f"Recreated figures: {counts.get('recreated', 0)}",
        f"Documented report discrepancies: {counts.get('documented report discrepancy', 0)}",
        f"Unexpected mismatches: {counts.get('unexpected mismatch', 0)}",
        "",
        "Documented discrepancies",
        "------------------------",
    ]
    discrepancies = validation[
        validation["status"] == "documented report discrepancy"
    ]
    for _, row in discrepancies.iterrows():
        lines.append(
            f"- {row['section']} / {row['result']}: report={row['reference_value']}, "
            f"Python={row['python_value']}. {row['note']}"
        )
    lines.extend(
        [
            "",
            "Figure scope",
            "------------",
            "- Slides 5-6: five time-series plots recreated.",
            "- Slide 15: residual histogram and all printed residual statistics recreated.",
            "- Slide 16: six recursive-diagnostic panels recreated.",
            "- Slides 17-18: D4, AR(2), and VAR(2) forecast plots recreated.",
            "",
            "See validation_summary.csv for every numerical comparison.",
        ]
    )
    (OUTPUT_DIR / "validation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# 7. Orchestration and console output
# -----------------------------------------------------------------------------


def print_model_summary(models: dict[str, object]) -> None:
    for name, model in models.items():
        aic, sic, hq = information_criteria_per_observation(model)
        table = pd.DataFrame(
            {
                "coefficient": model.params,
                "standard_error": model.bse,
                "p_value": model.pvalues,
            }
        )
        print(f"\n{name} ({int(model.nobs)} observations)")
        print(table.to_string(float_format=lambda value: f"{value:.6f}"))
        print(
            f"Adjusted R2={model.rsquared_adj:.6f}  "
            f"AIC={aic:.6f}  SIC={sic:.6f}  HQ={hq:.6f}"
        )


def main() -> None:
    ensure_inputs()
    data = load_data()

    descriptive = reference_descriptive_data(data)
    stats_table = descriptive_statistics(descriptive)
    correlation_table = descriptive.corr()
    correlation_table.index = [DISPLAY_NAMES[c] for c in correlation_table.index]
    correlation_table.columns = [DISPLAY_NAMES[c] for c in correlation_table.columns]

    models = fit_assignment_models(data)
    model_table = model_results_table(models)
    short_table = short_sample_results(data)
    diagnostics, _ = diagnostic_results(data, models["D4"])
    residual_stats = residual_statistics(models["D4"])
    forecasts, forecast_accuracy, intervals, ar2_model, var2_models = forecast_comparison(
        data, models
    )

    # Deterministic numerical outputs.
    stats_table.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index_label="statistic")
    correlation_table.to_csv(OUTPUT_DIR / "correlation_matrix.csv", index_label="variable")
    model_table.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    short_table.to_csv(OUTPUT_DIR / "short_sample_robustness.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "diagnostic_tests.csv", index=False)
    pd.Series(residual_stats, name="value").to_csv(
        OUTPUT_DIR / "residual_statistics.csv", index_label="statistic"
    )
    forecasts.to_csv(OUTPUT_DIR / "forecast_series.csv", index_label="date")
    pd.DataFrame(forecast_accuracy).T.to_csv(
        OUTPUT_DIR / "forecast_accuracy.csv", index_label="model"
    )

    ar2_parameters = pd.DataFrame(
        {"coefficient": ar2_model.params, "standard_error": ar2_model.bse}
    )
    ar2_parameters.to_csv(OUTPUT_DIR / "ar2_parameters.csv", index_label="term")
    var2_rows = []
    for equation, model in var2_models.items():
        for term in model.params.index:
            var2_rows.append(
                {
                    "equation": equation,
                    "term": term,
                    "coefficient": model.params[term],
                    "standard_error": model.bse[term],
                    "t_statistic": model.tvalues[term],
                    "p_value": model.pvalues[term],
                    "observations": int(model.nobs),
                }
            )
    pd.DataFrame(var2_rows).to_csv(OUTPUT_DIR / "var2_parameters.csv", index=False)

    # Figures.
    plot_descriptive_series(descriptive)
    plot_residual_histogram(models["D4"], residual_stats)
    plot_recursive_diagnostics(models["D4"])
    plot_forecasts(forecasts, intervals)

    # Validation is deliberately last so all evidence remains available if a new
    # unexpected mismatch appears after a future dependency or data change.
    validation = build_validation(
        stats_table,
        correlation_table,
        models,
        diagnostics,
        residual_stats,
        forecast_accuracy,
    )
    validation.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    write_validation_report(validation)

    print("Assignment 2 Python companion")
    print("================================")
    print(f"Input:  {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("\nDescriptive statistics (percentage points)")
    print(stats_table.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nCorrelation matrix")
    print(correlation_table.to_string(float_format=lambda value: f"{value:.6f}"))
    print_model_summary(models)
    print("\nDiagnostic tests")
    print(
        diagnostics[["test", "statistic", "p_value"]].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print("\nForecast accuracy")
    print(
        pd.DataFrame(forecast_accuracy).T.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )
    print("\nValidation status")
    print(validation["status"].value_counts().to_string())
    print("\nCompleted successfully. See assignment2/output/validation_report.txt.")


if __name__ == "__main__":
    main()
