"""
CSV dashboard for quick inspection, sorting, filtering, and plotting.

Run with:
    shiny run --reload app.py
"""

from __future__ import annotations

import ast
import base64
import os
import io
import shutil
import uuid
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
from typing import Optional

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo


sns.set_theme(style="whitegrid")
INDEX_COL = "__index__"
ALL_COLUMNS_DUPLICATES = "__all_columns__"
DEMO_SOURCE_CHOICE = "__demo_data__"
DEMO_DATA_FILE = Path(__file__).with_name("df_monthly.csv")
PLOTLY_FIG_HEIGHT = 600
UPLOAD_CACHE_DIR = Path("/private/tmp") / "csv_dashboard_uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def add_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the current index exposed as a column."""
    out = df.copy()
    out[INDEX_COL] = out.index
    return out


def materialize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select columns, allowing the synthetic index column."""
    out = df.copy()
    if INDEX_COL in cols:
        out = add_index_column(out)
    valid_cols = [col for col in cols if col in out.columns]
    if valid_cols:
        out = out.loc[:, valid_cols]
    return out


def resolve_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Map the synthetic index column back to the dataframe index."""
    if col == INDEX_COL:
        return pd.Series(df.index, index=df.index, name="index")
    return df[col]


def matplotlib_to_html(fig: plt.Figure) -> ui.Tag:
    """Convert a Matplotlib figure into embeddable HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return ui.HTML(
        f'<img src="data:image/png;base64,{encoded}" style="max-width:100%; height:auto;" />'
    )


def plotly_to_html(fig) -> ui.Tag:
    """Convert a Plotly figure into embeddable HTML."""
    return ui.HTML(
        fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width="100%",
            default_height=f"{PLOTLY_FIG_HEIGHT}px",
            config={"responsive": True},
        )
    )


def make_demo_data(n: int = 400) -> pd.DataFrame:
    """Fallback dataset so the app is useful before any CSV is uploaded."""
    if DEMO_DATA_FILE.exists():
        return pd.read_csv(DEMO_DATA_FILE, parse_dates=["date"])

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "category": rng.choice(["A", "B", "C", "D"], size=n),
            "region": rng.choice(["North", "South", "East", "West"], size=n),
            "value": np.round(rng.normal(100, 20, n), 2),
            "cost": np.round(rng.gamma(2.2, 25, n), 2),
            "score": np.round(rng.uniform(0, 100, n), 2),
        }
    )


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and try to coerce obvious numeric/date columns."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]

    def _looks_date_like(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(10)
        if sample.empty:
            return False

        date_like = sample.str.contains(
            r"[-/:]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
            case=False,
            regex=True,
        )
        return bool(date_like.mean() >= 0.5)

    for col in out.columns:
        series = out[col]

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.notna().sum()
            if non_null == 0:
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() / non_null >= 0.9:
                out[col] = numeric
                continue

            if _looks_date_like(series):
                parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
                if parsed_dates.notna().sum() / non_null >= 0.8:
                    out[col] = parsed_dates

    return out


def evaluate_formula_expression(formula: str, variables: dict[str, pd.Series | float | int]) -> pd.Series | float:
    """Safely evaluate a small arithmetic expression over named Series."""
    tree = ast.parse(str(formula), mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) or node.value is None:
                return node.value
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression")

    return _eval(tree)


def coerce_column_dtype(df: pd.DataFrame, column: str, dtype_target: str, datetime_format: str = "") -> pd.DataFrame:
    """Return a copy of df with one column coerced to a target dtype."""
    out = df.copy()
    if column not in out.columns:
        return out

    if dtype_target == "numeric":
        out[column] = pd.to_numeric(out[column], errors="coerce")
    elif dtype_target == "datetime":
        fmt = datetime_format.strip()
        if fmt:
            out[column] = pd.to_datetime(out[column], errors="coerce", format=fmt)
        else:
            out[column] = pd.to_datetime(out[column], errors="coerce", format="mixed")
    elif dtype_target == "string":
        out[column] = out[column].astype("string")

    return out


def series_to_stat_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values for statistics, supporting datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        valid = series.notna()
        if valid.any():
            dt = pd.to_datetime(series.loc[valid], errors="coerce").dropna()
            if not dt.empty:
                numeric = dt.astype("int64") / 86_400_000_000_000.0
                numeric = numeric - float(np.nanmin(numeric))
                out.loc[numeric.index] = numeric.astype(float)
        return out
    return pd.to_numeric(series, errors="coerce")


def stat_selected_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Select a dataframe column, supporting the synthetic index."""
    if column == INDEX_COL:
        return resolve_column(df, INDEX_COL)
    if column in df.columns:
        return df[column]
    return pd.Series(dtype="float64")


def stat_selected_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Select and coerce a dataframe column to numeric for statistics."""
    return series_to_stat_numeric(stat_selected_series(df, column))


def stat_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build a numeric dataframe from selected columns."""
    if not columns:
        return pd.DataFrame(index=df.index)
    frame = {col: stat_selected_numeric(df, col) for col in columns}
    return pd.DataFrame(frame)


def stat_logistic_predictor_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build a complete-case design matrix for logistic regression predictors."""
    if not columns:
        return pd.DataFrame(index=df.index)

    raw = pd.DataFrame(index=df.index)
    categorical_cols: list[str] = []

    for col in columns:
        series = stat_selected_series(df, col)
        if pd.api.types.is_datetime64_any_dtype(series):
            raw[col] = series_to_stat_numeric(series)
        elif pd.api.types.is_numeric_dtype(series):
            raw[col] = pd.to_numeric(series, errors="coerce")
        else:
            raw[col] = series.astype("string")
            categorical_cols.append(col)

    raw = raw.dropna()
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    parts: list[pd.DataFrame] = []
    numeric_cols = [col for col in columns if col not in categorical_cols]
    if numeric_cols:
        parts.append(raw[numeric_cols].astype(float))
    if categorical_cols:
        encoded = pd.get_dummies(
            raw[categorical_cols].astype("string"),
            prefix=categorical_cols,
            prefix_sep="=",
            drop_first=True,
            dummy_na=False,
        )
        if not encoded.empty:
            parts.append(encoded.astype(float))

    if not parts:
        return pd.DataFrame(index=raw.index)
    return pd.concat(parts, axis=1)


def exact_find_replace(series: pd.Series, find_text: str, replace_text: str) -> pd.Series:
    """Replace exact matches in a series while respecting numeric columns."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_like = pd.api.types.is_numeric_dtype(series) or (
        series.notna().any() and numeric_series.notna().sum() == series.notna().sum()
    )
    if numeric_like:
        find_value = pd.to_numeric(pd.Series([find_text]), errors="coerce").iloc[0]
        if pd.isna(find_value):
            return series.astype("string").where(series.astype("string") != str(find_text), replace_text)

        replacement_value = pd.to_numeric(pd.Series([replace_text]), errors="coerce").iloc[0]
        if pd.isna(replacement_value):
            return series.astype("string").where(numeric_series != find_value, replace_text)
        return numeric_series.where(numeric_series != find_value, replacement_value)

    if pd.api.types.is_datetime64_any_dtype(series):
        find_value = pd.to_datetime(pd.Series([find_text]), errors="coerce", format="mixed").iloc[0]
        if pd.isna(find_value):
            return series.astype("string").where(series.astype("string") != str(find_text), replace_text)

        replacement_value = pd.to_datetime(pd.Series([replace_text]), errors="coerce", format="mixed").iloc[0]
        if pd.isna(replacement_value):
            return series.astype("string").where(series != find_value, replace_text)
        return series.where(series != find_value, replacement_value)

    return series.astype("string").where(series.astype("string") != str(find_text), replace_text)


def _binary_auc(x: pd.Series, y: pd.Series) -> float:
    """Compute a rank-based AUC for a binary target."""
    x = pd.Series(x)
    y = pd.Series(y)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return float("nan")

    x_valid = x.loc[valid]
    y_valid = y.loc[valid].astype(float)
    if x_valid.nunique(dropna=True) < 2 or y_valid.nunique(dropna=True) < 2:
        return float("nan")

    n_pos = float(y_valid.sum())
    n_neg = float(len(y_valid) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    ranks = x_valid.rank(method="average")
    u_stat = float(ranks[y_valid >= 0.5].sum() - n_pos * (n_pos + 1.0) / 2.0)
    return u_stat / (n_pos * n_neg)


def logistic_separation_diagnostics(
    predictors: pd.DataFrame,
    y: pd.Series,
    *,
    fit: Optional[dict[str, object]] = None,
    max_signal_rows: int = 6,
) -> tuple[pd.DataFrame, str]:
    """Build a diagnostic report for separation and leakage signals."""
    y = pd.Series(y).astype(float).reset_index(drop=True)
    predictors = predictors.reset_index(drop=True)
    if predictors.empty or y.empty:
        return pd.DataFrame(columns=["Metric", "Value", "Details"]), ""

    diagnostics: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    complete_hits: list[str] = []
    quasi_hits: list[str] = []

    for col in predictors.columns:
        series = predictors[col]
        series_str = series.astype("string")
        numeric_candidate = pd.to_numeric(series, errors="coerce")
        numeric_like = pd.api.types.is_numeric_dtype(series) or (
            series.notna().sum() > 0 and numeric_candidate.notna().sum() == series.notna().sum()
        )

        if numeric_like:
            numeric_series = numeric_candidate
            valid = numeric_series.notna() & y.notna()
            if int(valid.sum()) < 2:
                continue
            x = numeric_series.loc[valid].astype(float)
            yy = y.loc[valid].astype(float)
            if x.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
                continue

            auc = _binary_auc(x, yy)
            if np.isnan(auc):
                continue
            strength = max(auc, 1.0 - auc)
            direction = "positive" if auc >= 0.5 else "inverse"
            class0 = x.loc[yy < 0.5]
            class1 = x.loc[yy >= 0.5]
            range_separated = bool(
                len(class0) > 0
                and len(class1) > 0
                and (
                    float(np.nanmax(class0)) < float(np.nanmin(class1))
                    or float(np.nanmax(class1)) < float(np.nanmin(class0))
                )
            )

            if range_separated:
                complete_hits.append(col)
                signal_rows.append(
                    {
                        "Metric": "Complete separation signal",
                        "Value": col,
                        "Details": f"numeric/datetime range does not overlap; AUC={auc:.4f}",
                    }
                )
            elif strength >= 0.95:
                if strength >= 0.995:
                    quasi_hits.append(col)
                signal_rows.append(
                    {
                        "Metric": "Near-separation signal",
                        "Value": col,
                        "Details": f"AUC={auc:.4f} ({direction})",
                    }
                )
            continue

        ctab = pd.crosstab(series_str, y, dropna=False)
        if ctab.empty or ctab.shape[1] < 2:
            continue

        counts = ctab.to_numpy(dtype=float)
        row_totals = counts.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_purity = np.nanmax(counts / row_totals[:, None], axis=1)
        best_idx = int(np.nanargmax(row_purity)) if np.isfinite(row_purity).any() else -1
        best_level = str(ctab.index[best_idx]) if best_idx >= 0 else str(col)
        best_purity = float(row_purity[best_idx]) if best_idx >= 0 else float("nan")

        pure_levels: list[str] = []
        for level, row in ctab.iterrows():
            row_values = row.to_numpy(dtype=float)
            if float(row_values.sum()) <= 0:
                continue
            if np.any(row_values == 0.0):
                pure_levels.append(f"{level} ({', '.join(f'{int(v)}' for v in row_values)})")

        if pure_levels:
            complete_hits.append(col)
            signal_rows.append(
                {
                    "Metric": "Complete separation signal",
                    "Value": col,
                    "Details": f"pure level(s): {', '.join(pure_levels[:2])}",
                }
            )
        elif best_purity >= 0.95:
            quasi_hits.append(col)
            signal_rows.append(
                {
                    "Metric": "Near-separation signal",
                    "Value": col,
                    "Details": f"best level {best_level} purity={best_purity:.4f}",
                }
            )

    complete = bool(complete_hits)
    quasi = bool(quasi_hits)

    diagnostics.append(
        {
            "Metric": "Complete separation",
            "Value": "Yes" if complete else "No",
            "Details": ", ".join(dict.fromkeys(complete_hits)) if complete_hits else "No single predictor is perfectly pure.",
        }
    )
    diagnostics.append(
        {
            "Metric": "Quasi-separation",
            "Value": "Yes" if quasi else "No",
            "Details": ", ".join(dict.fromkeys(quasi_hits)) if quasi_hits else "No strong single-predictor signal detected.",
        }
    )

    if fit is not None:
        probs = np.asarray(fit.get("probabilities", []), dtype=float)
        beta = np.asarray(fit.get("fitted", []), dtype=float)
        extreme_rate = float(np.mean((probs < 0.01) | (probs > 0.99))) if probs.size else float("nan")
        max_abs_coef = float(np.nanmax(np.abs(beta))) if beta.size else float("nan")
        diagnostics.append(
            {
                "Metric": "Model stability",
                "Value": "watch" if (np.isfinite(extreme_rate) and extreme_rate >= 0.95) else "ok",
                "Details": (
                    f"extreme probability rate={extreme_rate:.3f}, max |coef|={max_abs_coef:.3f}"
                    if np.isfinite(extreme_rate)
                    else "No fitted probabilities available."
                ),
            }
        )
        if np.isfinite(extreme_rate) and extreme_rate >= 0.95 and not complete and not quasi:
            diagnostics.append(
                {
                    "Metric": "Combination / leakage signal",
                    "Value": "Suspected",
                    "Details": "Model probabilities are extremely concentrated even though no single predictor is close to pure.",
                }
            )

    diagnostics.extend(signal_rows[:max_signal_rows])
    report = pd.DataFrame(diagnostics, columns=["Metric", "Value", "Details"])
    warning_parts = []
    if complete:
        warning_parts.append(f"complete separation: {', '.join(dict.fromkeys(complete_hits))}")
    if quasi:
        warning_parts.append(f"quasi-separation: {', '.join(dict.fromkeys(quasi_hits))}")
    warning = "; ".join(warning_parts)
    return report, warning


def prepare_logistic_analysis(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    positive_class: str,
    *,
    l2_penalty: float = 1.0,
) -> dict[str, object]:
    """Prepare the encoded design matrix, fit, and diagnostics for logistic regression."""
    predictors = stat_logistic_predictor_frame(df, predictor_cols)
    target_raw = stat_selected_series(df, target_col).astype("string")
    logit_df = predictors.copy()
    logit_df["_target_raw"] = target_raw
    logit_df = logit_df.dropna()
    if logit_df.empty:
        return {"predictors": pd.DataFrame(), "logit_df": pd.DataFrame(), "fit": None, "diagnostics": pd.DataFrame(), "warning": "", "positive_class": positive_class}

    label_order = list(dict.fromkeys(logit_df["_target_raw"].astype(str).tolist()))
    if positive_class not in label_order and label_order:
        positive_class = label_order[1] if len(label_order) > 1 else label_order[0]

    logit_df["_y"] = (logit_df["_target_raw"].astype(str) == positive_class).astype(float)
    if logit_df["_y"].nunique(dropna=True) < 2:
        return {"predictors": predictors, "logit_df": logit_df, "fit": None, "diagnostics": pd.DataFrame(), "warning": "", "positive_class": positive_class}

    fit = fit_logistic_regression(
        logit_df[predictors.columns].to_numpy(dtype=float),
        logit_df["_y"].to_numpy(dtype=float),
        list(predictors.columns),
        l2_penalty=l2_penalty,
    )
    raw_predictors = df.loc[logit_df.index, predictor_cols].copy()
    diagnostics, warning = logistic_separation_diagnostics(
        raw_predictors,
        logit_df["_y"],
        fit=fit,
    )
    return {
        "predictors": predictors,
        "logit_df": logit_df,
        "fit": fit,
        "diagnostics": diagnostics,
        "warning": warning,
        "positive_class": positive_class,
    }


def fit_multiple_linear_regression(X: np.ndarray, y: np.ndarray, predictor_names: list[str]) -> dict[str, object]:
    """Ordinary least squares multiple regression with summary stats."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < p + 2:
        raise ValueError("Need more rows than predictors for multiple regression.")
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    resid = y - y_hat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 and not np.isnan(r2) else float("nan")
    df_resid = n - p - 1
    sigma2 = rss / df_resid if df_resid > 0 else float("nan")
    xtx_inv = np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, np.inf))
    t_stats = beta / se
    mse = rss / df_resid if df_resid > 0 else float("nan")
    f_stat = ((tss - rss) / p) / mse if p > 0 and df_resid > 0 and mse > 0 else float("nan")
    coef_table = pd.DataFrame(
        {
            "Term": ["Intercept"] + predictor_names,
            "Coefficient": beta,
            "StdErr": se,
            "t stat": t_stats,
        }
    )
    return {
        "n": n,
        "p": p,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "resid_sd": float(np.std(resid, ddof=p + 1)) if df_resid > 0 else float(np.std(resid, ddof=1)),
        "f_stat": float(f_stat),
        "coef_table": coef_table,
        "predicted": y_hat,
        "residuals": resid,
    }


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    predictor_names: list[str],
    l2_penalty: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, object]:
    """Binary logistic regression via L2-regularized iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1, dtype=float)
    penalty = max(float(l2_penalty), 0.0)
    penalty_vec = np.concatenate([[0.0], np.full(p, penalty, dtype=float)])
    hess = np.eye(p + 1, dtype=float)

    for iteration in range(1, max_iter + 1):
        eta = np.clip(X1 @ beta, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = prob * (1.0 - prob)
        grad = X1.T @ (y - prob) - penalty_vec * beta
        hess = X1.T @ (X1 * w[:, None]) + np.diag(penalty_vec + 1e-8)
        step = np.linalg.pinv(hess) @ grad
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        iteration = max_iter

    eta = np.clip(X1 @ beta, -500, 500)
    prob = 1.0 / (1.0 + np.exp(-eta))
    pred = (prob >= 0.5).astype(float)
    eps = 1e-12
    loglik = float(np.sum(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps)))
    p0 = float(np.clip(np.mean(y), eps, 1.0 - eps))
    null_prob = np.full_like(y, p0, dtype=float)
    null_loglik = float(np.sum(y * np.log(null_prob + eps) + (1.0 - y) * np.log(1.0 - null_prob + eps)))
    mcfadden_r2 = 1.0 - loglik / null_loglik if null_loglik < 0 else float("nan")
    accuracy = float(np.mean(pred == y))
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z_stats = beta / se
    coef_table = pd.DataFrame(
        {
            "Term": ["Intercept"] + predictor_names,
            "Coefficient": beta,
            "StdErr": se,
            "z stat": z_stats,
            "Odds ratio": np.exp(beta),
        }
    )
    confusion = pd.DataFrame(
        {
            "Pred 0": [int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 1) & (pred == 0)))],
            "Pred 1": [int(np.sum((y == 0) & (pred == 1))), int(np.sum((y == 1) & (pred == 1)))],
        },
        index=["Actual 0", "Actual 1"],
    )
    return {
        "n": n,
        "p": p,
        "iterations": iteration,
        "loglik": loglik,
        "null_loglik": null_loglik,
        "mcfadden_r2": float(mcfadden_r2),
        "accuracy": accuracy,
        "coef_table": coef_table,
        "confusion": confusion,
        "probabilities": prob,
        "predicted": pred,
        "fitted": beta,
        "penalty": penalty,
        "condition_number": float(np.linalg.cond(hess)),
    }


def fit_pca(X: np.ndarray, column_names: list[str], n_components: int = 2, standardize: bool = True) -> dict[str, object]:
    """Principal components via covariance eigen-decomposition."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("PCA needs at least one numeric column and two rows.")
    n_components = max(1, min(int(n_components), p))
    means = np.mean(X, axis=0)
    centered = X - means
    scales = np.std(centered, axis=0, ddof=1)
    scales = np.where(np.isclose(scales, 0), 1.0, scales)
    X_use = centered / scales if standardize else centered
    cov = np.cov(X_use, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    elif np.ndim(cov) == 1:
        cov = np.diag(np.asarray(cov, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.sum(eigvals))
    explained = eigvals / total if total > 0 else np.full_like(eigvals, np.nan, dtype=float)
    scores = X_use @ eigvecs[:, :n_components]
    loadings = eigvecs[:, :n_components] * np.sqrt(np.clip(eigvals[:n_components], 0, np.inf))
    explained_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(len(eigvals))],
            "Eigenvalue": eigvals,
            "Explained variance": explained,
            "Cumulative": np.cumsum(explained),
        }
    )
    loading_df = pd.DataFrame(
        loadings,
        index=column_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "Feature"})
    score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    return {
        "n": n,
        "p": p,
        "n_components": n_components,
        "explained_df": explained_df,
        "loading_df": loading_df,
        "scores": score_df,
        "means": means,
        "scales": scales,
    }


def fit_kmeans(X: np.ndarray, n_clusters: int = 3, max_iter: int = 100, seed: int = 42) -> dict[str, object]:
    """Simple k-means clustering with standardized features."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("K-means needs at least two rows and one feature.")

    n_clusters = max(1, min(int(n_clusters), n))
    means = np.mean(X, axis=0)
    scales = np.std(X, axis=0, ddof=0)
    scales = np.where(np.isclose(scales, 0), 1.0, scales)
    X_use = (X - means) / scales

    if n_clusters == 1:
        center = np.mean(X_use, axis=0, keepdims=True)
        labels = np.zeros(n, dtype=int)
        inertia = float(np.sum((X_use - center[0]) ** 2))
        return {
            "labels": labels,
            "centers": center,
            "inertia": inertia,
            "iterations": 0,
            "means": means,
            "scales": scales,
        }

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = X_use[init_idx].copy()
    labels = np.full(n, -1, dtype=int)

    for iteration in range(1, max_iter + 1):
        distances = np.sum((X_use[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            members = X_use[new_labels == k]
            if len(members) == 0:
                new_centers[k] = X_use[rng.integers(0, n)]
            else:
                new_centers[k] = np.mean(members, axis=0)
        if np.array_equal(new_labels, labels) or np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    else:
        iteration = max_iter

    inertia = float(np.sum((X_use - centers[labels]) ** 2))
    return {
        "labels": labels,
        "centers": centers,
        "inertia": inertia,
        "iterations": iteration,
        "means": means,
        "scales": scales,
    }


def suggest_k_by_elbow(X: np.ndarray, max_k: int = 8, seed: int = 42) -> dict[str, object]:
    """Estimate a good k by finding the elbow in the k-means inertia curve."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < 2 or p < 1:
        return {
            "suggested_k": 1,
            "curve": pd.DataFrame(columns=["k", "inertia"]),
        }

    max_k = max(1, min(int(max_k), n))
    ks = list(range(1, max_k + 1))
    inertias: list[float] = []
    for k in ks:
        try:
            fit = fit_kmeans(X, n_clusters=k, seed=seed)
            inertias.append(float(fit["inertia"]))
        except ValueError:
            inertias.append(float("nan"))

    curve = pd.DataFrame({"k": ks, "inertia": inertias}).dropna()
    if len(curve) == 0:
        return {
            "suggested_k": 1,
            "curve": curve,
        }
    if len(curve) == 1:
        return {
            "suggested_k": int(curve.iloc[0]["k"]),
            "curve": curve,
        }

    points = curve[["k", "inertia"]].to_numpy(dtype=float)
    start = points[0]
    end = points[-1]
    line = end - start
    denom = float(np.linalg.norm(line))
    if denom <= 0:
        suggested_k = int(curve.iloc[0]["k"])
    else:
        offsets = points - start
        distances = np.abs(np.cross(line, offsets)) / denom
        suggested_k = int(curve.iloc[int(np.argmax(distances))]["k"])

    return {
        "suggested_k": max(1, suggested_k),
        "curve": curve,
    }


def suggest_pca_components_by_elbow(X: np.ndarray, standardize: bool = True) -> dict[str, object]:
    """Estimate a good PCA component count from the cumulative explained variance elbow."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < 2 or p < 1:
        return {
            "suggested_components": 1,
            "curve": pd.DataFrame(columns=["component", "cumulative"]),
            "basis": "Not enough data to estimate an elbow.",
        }

    fit = fit_pca(X, [f"X{i+1}" for i in range(p)], n_components=p, standardize=standardize)
    curve = fit["explained_df"][["Component", "Cumulative"]].copy()
    curve["component"] = np.arange(1, len(curve) + 1)
    curve = curve.loc[:, ["component", "Cumulative"]].rename(columns={"Cumulative": "cumulative"})
    curve = curve.dropna()
    if len(curve) == 0:
        return {
            "suggested_components": 1,
            "curve": curve,
            "basis": "No valid PCA variance curve could be computed.",
        }
    if len(curve) == 1:
        return {
            "suggested_components": int(curve.iloc[0]["component"]),
            "curve": curve,
            "basis": "Only one usable principal component is available.",
        }

    points = curve[["component", "cumulative"]].to_numpy(dtype=float)
    start = points[0]
    end = points[-1]
    line = end - start
    denom = float(np.linalg.norm(line))
    if denom <= 0:
        suggested = int(curve.iloc[0]["component"])
    else:
        offsets = points - start
        distances = np.abs(np.cross(line, offsets)) / denom
        suggested = int(curve.iloc[int(np.argmax(distances))]["component"])

    return {
        "suggested_components": max(1, suggested),
        "curve": curve,
        "basis": (
            "the elbow point of the cumulative explained-variance curve, measured as "
            "the point with maximum perpendicular distance from the line joining the "
            "first and last curve values, derived from the eigenvalues of the selected columns"
        ),
    }


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least two observations")
    if np.allclose(x, x[0]):
        raise ValueError("X values must vary")

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {
        "n": float(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "resid_sd": float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float(np.std(residuals, ddof=1)),
    }


def welch_t_stat(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least two observations per group")
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    denom = np.sqrt(v1 / n1 + v2 / n2)
    if denom == 0:
        mean_diff = float(np.mean(a) - np.mean(b))
        return float("inf") if not np.isclose(mean_diff, 0) else 0.0
    return (float(np.mean(a)) - float(np.mean(b))) / denom


def permutation_p_value(a: np.ndarray, b: np.ndarray, n_perm: int = 2000, seed: int = 42) -> float:
    observed = abs(welch_t_stat(a, b))
    combined = np.concatenate([a, b])
    n1 = len(a)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        if abs(welch_t_stat(perm[:n1], perm[n1:])) >= observed:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def one_way_anova(groups: list[np.ndarray]) -> dict[str, float]:
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least two groups")
    if any(len(g) < 2 for g in groups):
        raise ValueError("Each ANOVA group needs at least two observations")

    all_vals = np.concatenate(groups)
    grand_mean = float(np.mean(all_vals))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups)
    ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else float("nan")
    f_stat = ms_between / ms_within if ms_within and ms_within > 0 else float("inf")
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else float("nan")
    return {
        "f": float(f_stat),
        "eta_sq": float(eta_sq),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
        "df_between": float(df_between),
        "df_within": float(df_within),
    }


def permutation_f_p_value(groups: list[np.ndarray], n_perm: int = 1000, seed: int = 42) -> float:
    observed = one_way_anova(groups)["f"]
    values = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(values)
        split = []
        start = 0
        for size in sizes:
            split.append(perm[start:start + size])
            start += size
        if one_way_anova(split)["f"] >= observed:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def permutation_p_value_anova(groups: list[np.ndarray], n_perm: int = 1000, seed: int = 42) -> float:
    """Compatibility wrapper for the ANOVA permutation p-value helper."""
    return permutation_f_p_value(groups, n_perm=n_perm, seed=seed)


def heteroscedasticity_diagnostics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    reg = fit_linear_regression(x, y)
    fitted = reg["slope"] * x + reg["intercept"]
    residuals = y - fitted
    abs_resid = np.abs(residuals)
    split = np.median(fitted)
    low = abs_resid[fitted <= split]
    high = abs_resid[fitted > split]
    var_low = float(np.var(low, ddof=1)) if len(low) > 1 else float("nan")
    var_high = float(np.var(high, ddof=1)) if len(high) > 1 else float("nan")
    var_ratio = float(var_high / var_low) if var_low and var_low > 0 else float("nan")
    p_perm = permutation_p_value(low, high, n_perm=1000) if len(low) > 1 and len(high) > 1 else float("nan")
    abs_corr = float(np.corrcoef(fitted, abs_resid)[0, 1]) if np.std(fitted) > 0 and np.std(abs_resid) > 0 else float("nan")
    return {"abs_resid_corr": abs_corr, "variance_ratio_high_low": var_ratio, "perm_p_abs_resid": p_perm}


def heteroscedasticity_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compatibility wrapper for heteroscedasticity diagnostics."""
    return heteroscedasticity_diagnostics(x, y)


def sanitize_filename_stem(name: str) -> str:
    """Return a filesystem-friendly filename stem."""
    stem = Path(str(name)).stem.strip() or "current_data"
    cleaned = []
    for char in stem:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "current_data"


def duplicate_subset_from_choice(choice: str, columns: list[str]) -> Optional[list[str]]:
    """Map a duplicate-scope choice to a pandas drop/duplicated subset."""
    if choice == ALL_COLUMNS_DUPLICATES or not choice:
        return None
    if choice in columns:
        return [choice]
    return None


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        :root {
            --bg-1: #f6f4ef;
            --bg-2: #edf3f7;
            --surface: rgba(255, 255, 255, 0.88);
            --border: rgba(15, 23, 42, 0.08);
            --text: #16202a;
            --muted: #667085;
            --accent: #0f766e;
            --accent-2: #e07a3f;
            --accent-3: #3757c6;
            --shadow: 0 18px 48px rgba(15, 23, 42, 0.10);
            --shadow-soft: 0 10px 28px rgba(15, 23, 42, 0.06);
            --radius: 22px;
        }
        body {
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(55, 87, 198, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(224, 122, 63, 0.10), transparent 26%),
                linear-gradient(180deg, var(--bg-1), var(--bg-2));
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }
        .container-fluid {
            padding-top: 1rem;
            padding-bottom: 1.25rem;
        }
        .card {
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--surface);
            box-shadow: var(--shadow-soft);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        .card-body {
            padding: 1rem 1.1rem;
        }
        .card-header {
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.10), rgba(55, 87, 198, 0.06));
            color: var(--text);
            font-weight: 700;
            letter-spacing: 0.01em;
            padding: 0.85rem 1rem;
        }
        .source-panel .card-header {
            background: linear-gradient(135deg, rgba(224, 122, 63, 0.18), rgba(224, 122, 63, 0.04));
        }
        .viz-controls-panel .card-header {
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.18), rgba(15, 118, 110, 0.05));
        }
        .plot-panel .card-header {
            background: linear-gradient(135deg, rgba(55, 87, 198, 0.16), rgba(55, 87, 198, 0.05));
        }
        .stats-panel .card-header {
            background: linear-gradient(135deg, rgba(224, 122, 63, 0.16), rgba(55, 87, 198, 0.05));
        }
        .plot-panel {
            min-height: 400px;
            position: sticky;
            top: 1rem;
            align-self: stretch;
            width: 100%;
            box-sizing: border-box;
            z-index: 3;
            max-height: calc(100vh - 2rem);
        }
        .plot-panel .card-body,
        .stats-panel .card-body {
            display: flex;
            flex-direction: column;
        }
        .plot-panel .card-body {
            overflow-x: hidden;
            overflow-y: visible;
            padding-bottom: 0.75rem;
        }
        .plot-stage {
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .plot-stage .html-fill-container,
        .plot-stage .html-fill-item {
            width: 100%;
            overflow: hidden;
        }
        .plot-stage .plotly,
        .plot-stage .plotly-graph-div {
            overflow: hidden;
        }
        .stats-panel {
            min-height: 420px;
        }
        .stat-box .card-body {
            padding: 0.85rem 1rem;
        }
        .stat-label {
            color: var(--muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            width: 100%;
        }
        .column-tools-save-row {
            display: flex;
            align-items: end;
            margin: 0.15rem 0 0.75rem;
            gap: 0.5rem;
        }
        .column-tools-download-row {
            margin: 0 0 0.75rem;
        }
        .top-layout {
            align-items: stretch;
        }
        .top-panel-stack {
            display: flex;
            flex-direction: column;
            min-height: 500px;
            height: 100%;
        }
        .source-panel {
            flex: 0 0 auto;
        }
        .viz-controls-panel {
            flex: 1 1 auto;
            min-height: 0;
        }
        .btn {
            border-radius: 999px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-3), #5d79d6);
            border-color: transparent;
        }
        .btn-secondary {
            background: linear-gradient(135deg, var(--accent), #19a08e);
            border-color: transparent;
        }
        .btn-outline-primary {
            color: var(--accent-3);
            border-color: rgba(55, 87, 198, 0.28);
            background: rgba(255, 255, 255, 0.72);
        }
        .save-preview-btn {
            font-size: 0.78rem;
            padding: 0.28rem 0.6rem;
            min-height: 2rem;
            line-height: 1.1;
        }
        .form-control,
        .form-select,
        .selectize-input,
        .selectize-control.single .selectize-input {
            border-radius: 14px;
            border-color: rgba(15, 23, 42, 0.12);
            box-shadow: none;
            background-color: rgba(255, 255, 255, 0.92);
        }
        .form-control:focus,
        .form-select:focus,
        .selectize-input.focus {
            border-color: rgba(55, 87, 198, 0.40);
            box-shadow: 0 0 0 0.2rem rgba(55, 87, 198, 0.10);
        }
        .html-fill-container {
            min-height: 0;
        }
        """
    ),
    ui.layout_columns(
        ui.div(
                ui.card(
                    ui.card_header("Data Source"),
                    ui.input_file("csv_file", "Upload CSV files", accept=[".csv"], multiple=True),
                    ui.help_text("Upload one or more CSV files, then choose which one is active."),
                    ui.input_select("active_csv_file", "Active CSV file", choices={}),
                    ui.output_ui("uploaded_files_list"),
                    ui.help_text("Use Join Tables in Column Tools to merge the active CSV with another upload."),
                    ui.help_text("If no CSV is uploaded, a demo dataset is loaded automatically."),
                    class_="source-panel",
                ),
            ui.card(
                ui.card_header("Visualisation Controls"),
                ui.input_select(
                    "viz_engine",
                    "Visualisation Library",
                    choices=["plotly", "seaborn", "matplotlib"],
                    selected="plotly",
                ),
                ui.input_select(
                    "plot_kind",
                    "Plot type",
                    choices=["scatter", "line", "bar", "stacked crosstab", "histogram", "pie", "box", "heatmap"],
                    selected="scatter",
                ),
                ui.panel_conditional(
                    "input.plot_kind == 'heatmap'",
                    ui.input_switch("heatmap_show_values", "Show correlation values", value=False),
                    ui.help_text("Displays the correlation coefficient inside each heatmap cell."),
                ),
                ui.layout_columns(
                    ui.input_select("x_col", "X axis", choices=[]),
                    ui.input_switch("x_log", "Log", value=False),
                    col_widths=[9, 3],
                ),
                ui.layout_columns(
                    ui.input_select("y_col", "Y axis", choices=[]),
                    ui.input_switch("y_log", "Log", value=False),
                    col_widths=[9, 3],
                ),
                ui.input_checkbox("use_twin_y_axis", "Use twin y axis", value=False),
                ui.panel_conditional(
                    "input.use_twin_y_axis",
                    ui.help_text("Twin y axis uses a second numeric column on the right-hand side."),
                    ui.help_text("This mode currently works for scatter and line charts. Group / color is ignored."),
                    ui.layout_columns(
                        ui.input_select("y2_col", "Secondary Y axis", choices=[]),
                        ui.input_switch("y2_log", "Log", value=False),
                        col_widths=[9, 3],
                    ),
                    ui.input_text("y2_axis_title", "Secondary axis title", placeholder="Leave blank for the default label"),
                ),
                ui.panel_conditional(
                    "input.use_twin_y_axis && input.plot_kind == 'line'",
                    ui.help_text("Set the primary and secondary line styles independently."),
                    ui.layout_columns(
                        ui.input_select(
                            "twin_y1_line_style",
                            "Primary line style",
                            choices={
                                "Solid": "solid",
                                "Dashed": "dash",
                                "Dotted": "dot",
                                "Dash-dot": "dashdot",
                            },
                            selected="solid",
                        ),
                        ui.input_select(
                            "twin_y1_line_color",
                            "Primary line color",
                            choices={
                                "Blue": "blue",
                                "Orange": "orange",
                                "Green": "green",
                                "Red": "red",
                                "Purple": "purple",
                                "Teal": "teal",
                                "Black": "black",
                            },
                            selected="blue",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.input_select(
                            "twin_y2_line_style",
                            "Secondary line style",
                            choices={
                                "Solid": "solid",
                                "Dashed": "dash",
                                "Dotted": "dot",
                                "Dash-dot": "dashdot",
                            },
                            selected="dash",
                        ),
                        ui.input_select(
                            "twin_y2_line_color",
                            "Secondary line color",
                            choices={
                                "Blue": "blue",
                                "Orange": "orange",
                                "Green": "green",
                                "Red": "red",
                                "Purple": "purple",
                                "Teal": "teal",
                                "Black": "black",
                            },
                            selected="orange",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.panel_conditional(
                    "input.plot_kind == 'line'",
                    ui.layout_columns(
                        ui.input_select(
                            "line_style",
                            "Line style",
                            choices={
                                "Solid": "solid",
                                "Dashed": "dash",
                                "Dotted": "dot",
                                "Dash-dot": "dashdot",
                            },
                            selected="solid",
                        ),
                        ui.input_numeric("line_width", "Width", value=2.5, min=0.5, max=10, step=0.5),
                        ui.input_select(
                            "line_marker",
                            "Marker",
                            choices={
                                "None": "none",
                                "Circle": "circle",
                                "Square": "square",
                                "Diamond": "diamond",
                                "Triangle": "triangle-up",
                                "X": "x",
                            },
                            selected="circle",
                        ),
                        ui.input_checkbox("line_show_markers", "Show markers", value=True),
                        col_widths=[3, 3, 3, 3],
                    ),
                ),
                ui.input_select("hue_col", "Group / color", choices=["None"], selected="None"),
                ui.output_ui("histogram_controls"),
                ui.output_ui("pie_controls"),
                ui.input_checkbox("show_plot_formatting", "Show plot formatting options", value=False),
                ui.panel_conditional(
                    "input.show_plot_formatting",
                    ui.card(
                        ui.card_header("Plot Formatting"),
                        ui.input_text("plot_title", "Plot title", placeholder="Leave blank for the default title"),
                        ui.input_text("x_axis_title", "X axis title", placeholder="Leave blank for the default label"),
                        ui.input_text("y_axis_title", "Y axis title", placeholder="Leave blank for the default label"),
                        ui.layout_columns(
                            ui.input_numeric("plot_title_font_size", "Title font size", value=18, min=8, max=40, step=1),
                            ui.input_numeric("axis_title_font_size", "Axis title font size", value=14, min=8, max=32, step=1),
                            col_widths=[6, 6],
                        ),
                        ui.layout_columns(
                            ui.input_numeric("x_tick_angle", "X tick rotation", value=0, min=-90, max=90, step=5),
                            ui.input_numeric("max_x_ticks", "Max x ticks", value=10, min=2, max=40, step=1),
                            col_widths=[6, 6],
                        ),
                        ui.layout_columns(
                            ui.input_select("grid_axis", "Grid axis", choices=["both", "x", "y", "none"], selected="both"),
                            ui.input_numeric("grid_alpha", "Grid opacity", value=0.35, min=0, max=1, step=0.05),
                            col_widths=[6, 6],
                        ),
                        ui.input_select("grid_style", "Grid style", choices=["solid", "dashed", "dotted"], selected="dashed"),
                    ),
                ),
                class_="viz-controls-panel",
            ),
            class_="top-panel-stack",
        ),
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("Summary and plot"),
                        ui.download_button("download_main_plot", "Save current plot"),
                        class_="panel-header",
                    )
                ),
                ui.output_text("summary"),
                ui.div(ui.output_ui("plot"), class_="plot-stage"),
                class_="plot-panel",
            ),
        col_widths=[3, 9],
        min_height="500px",
        class_="top-layout",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Data Preview"),
            ui.output_data_frame("preview"),
        ),
        ui.div(
                ui.card(
                    ui.card_header("Column Tools"),
                    ui.div(
                        ui.layout_columns(
                            ui.input_text(
                                "preview_dataset_name",
                                "New data set name",
                                placeholder="Leave blank to use the current table name",
                            ),
                            ui.div(
                                ui.input_action_button("save_preview_dataset", "Save new dataframe", class_="save-preview-btn"),
                                class_="column-tools-save-row",
                            ),
                            col_widths=[8, 4],
                        ),
                    ),
                    ui.div(
                        ui.download_button("download_preview_csv", "Download new dataframe"),
                        class_="column-tools-download-row",
                    ),
                ui.input_select(
                    "column_tool",
                    "Tool",
                    choices=[
                        "rename",
                        "convert_type",
                        "format_numeric",
                        "calculate_formula",
                        "find_replace",
                        "drop_rows",
                        "find_duplicates",
                        "drop_duplicates",
                        "z_score normalisation",
                        "pivot_wide",
                        "melt",
                        "join_tables",
                    ],
                    selected="rename",
                ),
                ui.panel_conditional(
                    "input.column_tool == 'rename'",
                    ui.input_select("rename_from", "Rename column", choices=[]),
                    ui.input_text("rename_to", "New column name", placeholder="e.g. release_date"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'convert_type'",
                    ui.input_select("dtype_col", "Change data type for column", choices=[]),
                    ui.input_select(
                        "dtype_target",
                        "Convert to",
                        choices=["numeric", "datetime", "string"],
                        selected="numeric",
                    ),
                    ui.input_text(
                        "datetime_format",
                        "Datetime format (optional)",
                        placeholder="%Y-%m-%d",
                    ),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'find_replace'",
                    ui.input_select("find_replace_col", "Column", choices=[]),
                    ui.input_text("find_text", "Find exact value", placeholder="value to match exactly"),
                    ui.input_text("replace_text", "Replace with", placeholder="replacement text"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'drop_rows'",
                    ui.input_select("drop_rows_col", "Column", choices=[]),
                    ui.input_select(
                        "drop_rows_mode",
                        "Drop rows where",
                        choices=["missing", "value"],
                        selected="missing",
                    ),
                    ui.input_text(
                        "drop_rows_value",
                        "Value to drop",
                        placeholder="e.g. 0 or missing text",
                    ),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'z_score normalisation'",
                    ui.help_text("Replaces the selected numeric column with z-scores: (x - mean) / standard deviation."),
                    ui.input_select("zscore_col", "Column", choices=[]),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'format_numeric'",
                    ui.help_text("Rounds the selected numeric column to a chosen number of decimal places."),
                    ui.input_select("format_numeric_col", "Column", choices=[]),
                    ui.input_numeric("format_numeric_decimals", "Decimal places", value=2, min=0, max=10, step=1),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'calculate_formula'",
                    ui.help_text("Create a new column by calculating from two numeric columns using A and B in the formula."),
                    ui.input_text("calculate_new_col", "New column name", placeholder="calculated"),
                    ui.layout_columns(
                        ui.input_select("calculate_col_a", "Column A", choices=[]),
                        ui.input_select("calculate_col_b", "Column B", choices=[]),
                        col_widths=[6, 6],
                    ),
                    ui.input_text("calculate_formula", "Formula", value="(A*B)/B", placeholder="(A*B)/B"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'pivot_wide'",
                    ui.help_text("Like an Excel PivotTable: rows become the index, unique values become columns."),
                    ui.input_selectize("pivot_index_cols", "Row fields", choices=[], multiple=True),
                    ui.input_select("pivot_column_col", "Column field", choices=[]),
                    ui.input_select("pivot_value_col", "Values field", choices=[]),
                    ui.input_select(
                        "pivot_aggfunc",
                        "Aggregation",
                        choices=["sum", "mean", "median", "min", "max", "count", "first", "last", "nunique"],
                        selected="sum",
                    ),
                    ui.output_ui("pivot_notice"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'melt'",
                    ui.help_text("Melt turns wide data into a long format."),
                    ui.input_selectize("melt_id_cols", "Identifier fields", choices=[], multiple=True),
                    ui.input_selectize("melt_value_cols", "Fields to unpivot", choices=[], multiple=True),
                    ui.input_text("melt_var_name", "Variable column name", placeholder="variable"),
                    ui.input_text("melt_value_name", "Value column name", placeholder="value"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'join_tables'",
                    ui.help_text("Join the current table to another uploaded CSV using a primary/foreign key mapping."),
                    ui.input_select("join_table", "Foreign table", choices=[]),
                    ui.input_select(
                        "join_how",
                        "Join type",
                        choices={
                            "Left join (keep current rows)": "left",
                            "Inner join (matches only)": "inner",
                            "Right join (keep foreign rows)": "right",
                            "Full outer join": "outer",
                        },
                        selected="left",
                    ),
                    ui.input_select("join_left_key", "Primary key column", choices=[]),
                    ui.input_select("join_right_key", "Foreign key column", choices=[]),
                    ui.output_ui("join_notice"),
                    ui.output_ui("join_download_ui"),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'find_duplicates'",
                    ui.input_select(
                        "find_duplicates_col",
                        "Duplicate scope",
                        choices={"All columns": ALL_COLUMNS_DUPLICATES},
                        selected=ALL_COLUMNS_DUPLICATES,
                    ),
                ),
                ui.panel_conditional(
                    "input.column_tool == 'drop_duplicates'",
                    ui.input_select(
                        "drop_duplicates_col",
                        "Duplicate scope",
                        choices={"All columns": ALL_COLUMNS_DUPLICATES},
                        selected=ALL_COLUMNS_DUPLICATES,
                    ),
                ),
                ui.input_action_button("undo_column", "Undo last change"),
                ui.input_action_button("apply_columns", "Apply column changes"),
                ui.help_text("Choose a tool, fill in the related fields, then click Apply."),
                ui.help_text("Use the save button to turn the current preview table into a new active data source."),
            ),
            ui.card(
                ui.card_header("Preview Options"),
                ui.input_checkbox("preview_all_cols", "Show all columns", value=False),
                ui.input_selectize(
                    "preview_cols",
                    "Preview columns",
                    choices=[],
                    multiple=True,
                ),
                ui.output_ui("preview_notice"),
            ),
        ),
        col_widths=[8, 4],
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Statistical Controls"),
            ui.input_select(
                "stat_mode",
                "Test",
                choices=[
                    "descriptive",
                    "correlation",
                    "regression",
                    "multiple regression",
                    "logistic regression",
                    "t-test",
                    "ANOVA",
                    "heteroscedasticity",
                    "PCA",
                ],
                selected="descriptive",
            ),
            ui.panel_conditional(
                "input.stat_mode == 'descriptive'",
                ui.input_select("stat_desc_col", "Column", choices=[]),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'correlation' || input.stat_mode == 'regression' || input.stat_mode == 'heteroscedasticity'",
                ui.input_select("stat_x_col", "X column", choices=[]),
                ui.input_select("stat_y_col", "Y column", choices=[]),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'multiple regression'",
                ui.input_select("stat_mr_target_col", "Target column", choices=[]),
                ui.input_selectize("stat_mr_predictor_cols", "Predictor columns", choices=[], multiple=True),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'logistic regression'",
                ui.input_select("stat_logit_target_col", "Target column", choices=[]),
                ui.input_select("stat_logit_positive_class", "Positive class", choices=[]),
                ui.input_selectize("stat_logit_predictor_cols", "Predictor columns", choices=[], multiple=True),
            ),
            ui.panel_conditional(
                "input.stat_mode == 't-test'",
                ui.input_select("stat_group_col", "Group column", choices=[]),
                ui.input_select("stat_value_col", "Value column", choices=[]),
                ui.input_select("stat_group_a", "Group A", choices=[]),
                ui.input_select("stat_group_b", "Group B", choices=[]),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'ANOVA'",
                ui.input_select("stat_anova_col", "ANOVA group column", choices=[]),
                ui.input_select("stat_anova_value_col", "Value column", choices=[]),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'PCA'",
                ui.input_selectize("stat_pca_cols", "Columns", choices=[], multiple=True),
                ui.input_numeric("stat_pca_components", "Components", value=2, min=1, max=10, step=1),
                ui.input_checkbox("stat_pca_standardize", "Standardize columns", value=True),
                ui.output_ui("stat_pca_hint"),
            ),
            ui.panel_conditional(
                "input.stat_mode != 'descriptive'",
                ui.input_checkbox("stat_cluster_enable", "Color points by K-means clusters", value=False),
                ui.input_numeric("stat_cluster_k", "Clusters (k)", value=3, min=1, max=20, step=1),
                ui.output_ui("stat_cluster_hint"),
            ),
            ui.panel_conditional(
                "input.stat_mode != 'descriptive'",
                ui.input_checkbox("stat_plot_x_log", "Log X axis", value=False),
                ui.input_checkbox("stat_plot_y_log", "Log Y axis", value=False),
                ui.help_text("Log axes only apply to numeric, positive values."),
            ),
            ui.panel_conditional(
                "input.stat_mode == 'descriptive'",
                ui.help_text("Descriptive statistics do not need column selectors."),
            ),
            ui.input_action_button("apply_stats", "Run statistics"),
            ui.help_text("Choose a test, pick the relevant columns, then click Run statistics."),
        ),
        ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("Statistics Output"),
                        ui.download_button("download_stats_plot", "Save current plot"),
                        ui.download_button("download_stats_results", "Save results to CSV"),
                        class_="panel-header",
                    )
                ),
            ui.output_ui("statistics_plot"),
            ui.output_ui("statistics_panel"),
            class_="stats-panel",
        ),
        col_widths=[4, 8],
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    uploaded_file_store = reactive.value([])

    @reactive.effect
    @reactive.event(input.csv_file, ignore_init=True)
    def _accumulate_uploaded_files() -> None:
        uploads = input.csv_file() or []
        if not uploads:
            if uploaded_file_store.get():
                uploaded_file_store.set([])
            return

        current = list(uploaded_file_store.get() or [])
        new_records: list[dict[str, object]] = []
        for index, file_info in enumerate(uploads, start=1):
            name = str(file_info["name"])
            source_path = Path(str(file_info["datapath"]))
            cached_path = UPLOAD_CACHE_DIR / f"{uuid.uuid4().hex}_{sanitize_filename_stem(name)}_{index}.csv"
            shutil.copy2(source_path, cached_path)
            new_records.append(
                {
                    "path": str(cached_path),
                    "name": name,
                    "df": normalize_dataframe(pd.read_csv(cached_path)),
                }
            )

        if new_records:
            uploaded_file_store.set(current + new_records)

    @reactive.effect
    @reactive.event(input.join_table, ignore_init=True)
    def _remember_join_table() -> None:
        selected = str(input.join_table() or "")
        if selected:
            join_table_state.set(selected)

    @reactive.effect
    @reactive.event(input.join_how, ignore_init=True)
    def _remember_join_how() -> None:
        selected = str(input.join_how() or "").strip()
        if selected in {"left", "inner", "right", "outer"}:
            join_how_state.set(selected)

    @reactive.effect
    @reactive.event(input.active_csv_file, ignore_init=True)
    def _remember_active_csv() -> None:
        selected = str(input.active_csv_file() or "")
        if selected:
            active_csv_state.set(selected)

    @reactive.calc
    def uploaded_file_records() -> list[dict[str, object]]:
        records = list(uploaded_file_store.get() or [])
        seen_stems: dict[str, int] = {}
        labeled_records: list[dict[str, object]] = []
        for index, record in enumerate(records):
            name = str(record["name"])
            stem = Path(name).stem.strip() or f"file_{index + 1}"
            seen_stems[stem] = seen_stems.get(stem, 0) + 1
            label = stem if seen_stems[stem] == 1 else f"{stem} ({seen_stems[stem]})"
            labeled_records.append({**record, "label": label})
        return labeled_records

    @reactive.calc
    def active_uploaded_record() -> Optional[dict[str, object]]:
        records = uploaded_file_records()
        if not records:
            return None

        active_label = str(active_csv_state.get() or input.active_csv_file() or "")
        for record in records:
            if str(record["label"]) == active_label:
                return record
        return records[0]

    @reactive.calc
    def base_data() -> pd.DataFrame:
        record = active_uploaded_record()
        if record is None:
            return normalize_dataframe(make_demo_data())
        return record["df"].copy()

    column_history = reactive.value([])
    column_history_source = reactive.value(None)
    main_plot_cache = reactive.value(None)
    stats_plot_cache = reactive.value(None)
    active_csv_state = reactive.value("")
    join_table_state = reactive.value("")
    join_how_state = reactive.value("left")
    join_result_cache = reactive.value(None)
    twin_y1_line_style_state = reactive.value("solid")
    twin_y1_line_color_state = reactive.value("blue")
    twin_y2_line_style_state = reactive.value("dash")
    twin_y2_line_color_state = reactive.value("orange")
    preview_cols_state = reactive.value(None)
    preview_all_cols_state = reactive.value(None)
    preview_state_source = reactive.value(None)

    twin_line_style_choices = {"solid", "dash", "dot", "dashdot"}
    twin_line_color_choices = {"blue", "orange", "green", "red", "purple", "teal", "black"}

    @reactive.effect
    @reactive.event(input.twin_y1_line_style, ignore_init=True)
    def _remember_twin_y1_line_style() -> None:
        selected = str(input.twin_y1_line_style() or "").strip().lower()
        if selected in twin_line_style_choices:
            twin_y1_line_style_state.set(selected)

    @reactive.effect
    @reactive.event(input.twin_y1_line_color, ignore_init=True)
    def _remember_twin_y1_line_color() -> None:
        selected = str(input.twin_y1_line_color() or "").strip().lower()
        if selected in twin_line_color_choices:
            twin_y1_line_color_state.set(selected)

    @reactive.effect
    @reactive.event(input.twin_y2_line_style, ignore_init=True)
    def _remember_twin_y2_line_style() -> None:
        selected = str(input.twin_y2_line_style() or "").strip().lower()
        if selected in twin_line_style_choices:
            twin_y2_line_style_state.set(selected)

    @reactive.effect
    @reactive.event(input.twin_y2_line_color, ignore_init=True)
    def _remember_twin_y2_line_color() -> None:
        selected = str(input.twin_y2_line_color() or "").strip().lower()
        if selected in twin_line_color_choices:
            twin_y2_line_color_state.set(selected)

    @reactive.effect
    @reactive.event(input.preview_cols, ignore_init=True)
    def _remember_preview_cols() -> None:
        selected = list(input.preview_cols() or [])
        preview_cols_state.set(selected)
        if bool(preview_all_cols_state.get()):
            current_columns = {INDEX_COL, *data().columns.tolist()}
            if set(selected) != current_columns:
                preview_all_cols_state.set(False)

    @reactive.effect
    @reactive.event(input.preview_all_cols, ignore_init=True)
    def _remember_preview_all_cols() -> None:
        checked = bool(input.preview_all_cols())
        preview_all_cols_state.set(checked)
        if checked:
            preview_cols_state.set([INDEX_COL, *data().columns.tolist()])

    @reactive.effect
    def _reset_preview_state_on_source_change() -> None:
        sig = source_signature()
        if preview_state_source.get() != sig:
            preview_state_source.set(sig)
            preview_cols_state.set(None)
            preview_all_cols_state.set(None)

    def append_dataframe_source(df: pd.DataFrame, base_name: str, activate: bool = True) -> dict[str, object]:
        stem = sanitize_filename_stem(base_name)
        saved_name = f"{stem}_{uuid.uuid4().hex[:8]}.csv"
        saved_path = UPLOAD_CACHE_DIR / saved_name
        saved_df = normalize_dataframe(df.copy())
        saved_df.to_csv(saved_path, index=False)
        record = {
            "path": str(saved_path),
            "name": saved_name,
            "df": saved_df,
        }
        records = list(uploaded_file_store.get() or [])
        uploaded_file_store.set(records + [record])
        if activate:
            active_csv_state.set(Path(saved_name).stem)
        return record

    @reactive.calc
    def preview_dataframe() -> pd.DataFrame:
        df = data()
        if input.column_tool() == "find_duplicates":
            choice = input.find_duplicates_col() or ALL_COLUMNS_DUPLICATES
            subset = duplicate_subset_from_choice(choice, df.columns.tolist())
            df = df.loc[df.duplicated(subset=subset, keep=False)].copy()
        stored_preview_cols = preview_cols_state.get()
        cols = stored_preview_cols if stored_preview_cols is not None else input.preview_cols()
        if cols:
            df = materialize_columns(df, list(cols))
        return df

    applied_stats_config = reactive.value(
        {
            "ready": False,
            "stat_mode": "descriptive",
            "stat_desc_col": "",
            "stat_x_col": "",
            "stat_y_col": "",
            "stat_mr_target_col": "",
            "stat_mr_predictor_cols": [],
            "stat_logit_target_col": "",
            "stat_logit_positive_class": "",
            "stat_logit_predictor_cols": [],
            "stat_pca_cols": [],
            "stat_pca_components": 2,
            "stat_pca_standardize": True,
            "stat_plot_x_log": False,
            "stat_plot_y_log": False,
            "stat_cluster_enable": False,
            "stat_cluster_k": 3,
            "stat_group_col": "",
            "stat_value_col": "",
            "stat_group_a": "",
            "stat_group_b": "",
            "stat_anova_col": "",
            "stat_anova_value_col": "",
        }
    )

    @reactive.calc
    def source_signature() -> tuple:
        record = active_uploaded_record()
        if record is None:
            return (DEMO_SOURCE_CHOICE,)
        return (str(record["label"]), str(record["name"]))

    @reactive.effect
    def _reset_column_history_on_source_change() -> None:
        sig = source_signature()
        if column_history_source.get() != sig:
            column_history_source.set(sig)
            column_history.set([])

    @reactive.effect
    @reactive.event(input.apply_columns, ignore_init=True)
    def _apply_column_tools() -> None:
        current_df = data()
        tool = input.column_tool()
        new_df = current_df.copy()
        uploaded_records = uploaded_file_records()
        stored_preview_cols = preview_cols_state.get()
        current_preview_cols = list(stored_preview_cols if stored_preview_cols is not None else (input.preview_cols() or []))

        if tool == "rename":
            rename_from = input.rename_from()
            rename_to = input.rename_to().strip()
            if rename_from in new_df.columns and rename_to and rename_to != rename_from:
                new_df = new_df.rename(columns={rename_from: rename_to})
        elif tool == "convert_type":
            dtype_col = input.dtype_col()
            dtype_target = input.dtype_target()
            datetime_format = input.datetime_format()
            if dtype_col in new_df.columns:
                new_df = coerce_column_dtype(new_df, dtype_col, dtype_target, datetime_format=datetime_format)
        elif tool == "find_replace":
            find_replace_col = input.find_replace_col()
            find_text = input.find_text()
            replace_text = input.replace_text()
            if find_replace_col in new_df.columns and find_text:
                new_df[find_replace_col] = exact_find_replace(new_df[find_replace_col], find_text, replace_text)
        elif tool == "drop_rows":
            drop_rows_col = input.drop_rows_col()
            drop_rows_mode = input.drop_rows_mode()
            drop_rows_value = input.drop_rows_value().strip()
            if drop_rows_col in new_df.columns:
                if drop_rows_mode == "missing":
                    new_df = new_df.loc[~new_df[drop_rows_col].isna()].copy()
                elif drop_rows_mode == "value":
                    series = new_df[drop_rows_col]
                    if pd.api.types.is_numeric_dtype(series):
                        target = pd.to_numeric(pd.Series([drop_rows_value]), errors="coerce").iloc[0]
                        if pd.notna(target):
                            numeric_series = pd.to_numeric(series, errors="coerce")
                            new_df = new_df.loc[~numeric_series.eq(target)].copy()
                    elif pd.api.types.is_datetime64_any_dtype(series):
                        target_dt = pd.to_datetime(drop_rows_value, errors="coerce", format="mixed")
                        if pd.notna(target_dt):
                            new_df = new_df.loc[~pd.to_datetime(series, errors="coerce").eq(target_dt)].copy()
                    else:
                        new_df = new_df.loc[series.astype("string") != drop_rows_value].copy()
        elif tool == "z_score normalisation":
            zscore_col = input.zscore_col()
            if zscore_col in new_df.columns:
                numeric_series = pd.to_numeric(new_df[zscore_col], errors="coerce")
                mean_value = numeric_series.mean()
                std_value = numeric_series.std(ddof=0)
                if pd.notna(std_value) and std_value != 0:
                    new_df[zscore_col] = (numeric_series - mean_value) / std_value
                else:
                    new_df[zscore_col] = numeric_series.where(numeric_series.isna(), 0.0)
        elif tool == "format_numeric":
            format_numeric_col = input.format_numeric_col()
            decimals_raw = input.format_numeric_decimals()
            try:
                decimals = int(decimals_raw)
            except Exception:
                decimals = 2
            if decimals < 0:
                decimals = 0
            if format_numeric_col in new_df.columns:
                numeric_series = pd.to_numeric(new_df[format_numeric_col], errors="coerce")
                new_df[format_numeric_col] = numeric_series.round(decimals)
        elif tool == "calculate_formula":
            calc_new_col = str(input.calculate_new_col() or "").strip() or "calculated"
            calc_col_a = input.calculate_col_a()
            calc_col_b = input.calculate_col_b()
            calc_formula = str(input.calculate_formula() or "").strip()
            if calc_col_a in new_df.columns and calc_col_b in new_df.columns and calc_formula:
                a_series = pd.to_numeric(new_df[calc_col_a], errors="coerce")
                b_series = pd.to_numeric(new_df[calc_col_b], errors="coerce")

                def unique_column_name(base: str, used: list[str]) -> str:
                    candidate = base
                    suffix = 2
                    while candidate in used:
                        candidate = f"{base}_{suffix}"
                        suffix += 1
                    return candidate

                try:
                    result = evaluate_formula_expression(calc_formula, {"A": a_series, "B": b_series})
                    if isinstance(result, pd.Series):
                        out_series = result.reindex(new_df.index)
                    else:
                        out_series = pd.Series(result, index=new_df.index)
                    target_col = unique_column_name(calc_new_col, list(new_df.columns))
                    new_df[target_col] = out_series
                except Exception:
                    pass
        elif tool == "pivot_wide":
            working_df = add_index_column(new_df)
            index_cols = [col for col in (input.pivot_index_cols() or []) if col in working_df.columns]
            pivot_column_col = input.pivot_column_col()
            pivot_value_col = input.pivot_value_col()
            pivot_aggfunc = str(input.pivot_aggfunc() or "sum").strip().lower()
            if index_cols and pivot_column_col in working_df.columns and pivot_value_col in working_df.columns:
                pivot_source = working_df.copy()
                if pivot_aggfunc in {"sum", "mean", "median", "min", "max"}:
                    pivot_source[pivot_value_col] = pd.to_numeric(pivot_source[pivot_value_col], errors="coerce")
                try:
                    pivoted = pd.pivot_table(
                        pivot_source,
                        index=index_cols,
                        columns=pivot_column_col,
                        values=pivot_value_col,
                        aggfunc=pivot_aggfunc,
                        dropna=False,
                    )
                    if isinstance(pivoted, pd.Series):
                        pivoted = pivoted.to_frame()
                    new_df = pivoted.reset_index().copy()
                    if isinstance(new_df.columns, pd.MultiIndex):
                        new_df.columns = [
                            "_".join(str(part) for part in col if str(part) != "").strip("_")
                            for col in new_df.columns.to_flat_index()
                        ]
                    if not new_df.empty:
                        new_df.columns = [str(col) if not isinstance(col, str) else col for col in new_df.columns]
                except Exception:
                    pass
        elif tool == "melt":
            working_df = add_index_column(new_df)
            id_vars = [col for col in (input.melt_id_cols() or []) if col in working_df.columns]
            value_vars = [col for col in (input.melt_value_cols() or []) if col in working_df.columns and col not in id_vars]
            if not value_vars:
                value_vars = [col for col in working_df.columns if col not in id_vars]
            var_name = str(input.melt_var_name() or "variable").strip() or "variable"
            value_name = str(input.melt_value_name() or "value").strip() or "value"

            def unique_column_name(base: str, used: list[str]) -> str:
                candidate = base
                suffix = 2
                while candidate in used:
                    candidate = f"{base}_{suffix}"
                    suffix += 1
                return candidate

            used_names = list(working_df.columns)
            var_name = unique_column_name(var_name, used_names)
            used_names.append(var_name)
            value_name = unique_column_name(value_name, used_names)
            if value_vars and (set(value_vars) - set(id_vars)):
                try:
                    new_df = pd.melt(
                        working_df,
                        id_vars=id_vars,
                        value_vars=value_vars,
                        var_name=var_name,
                        value_name=value_name,
                    )
                except Exception:
                    pass
        elif tool == "join_tables":
            join_table_path = str(join_table_state.get() or input.join_table() or "")
            join_left_key = input.join_left_key()
            join_right_key = input.join_right_key()
            join_how = str(join_how_state.get() or input.join_how() or "left").strip()
            if join_how not in {"left", "inner", "right", "outer"}:
                join_how = "left"
            secondary_df = next((record["df"] for record in uploaded_records if str(record["label"]) == join_table_path), None)
            if isinstance(secondary_df, pd.DataFrame) and join_left_key and join_right_key:
                left_df = add_index_column(new_df)
                right_df = add_index_column(secondary_df)
                if (join_left_key in left_df.columns or join_left_key == INDEX_COL) and (join_right_key in right_df.columns or join_right_key == INDEX_COL):
                    left_key_temp = "__join_left_key__"
                    right_key_temp = "__join_right_key__"

                    def join_series(frame: pd.DataFrame, key: str) -> pd.Series:
                        return resolve_column(frame, INDEX_COL) if key == INDEX_COL else frame[key]

                    left_merge = left_df.copy()
                    right_merge = right_df.copy()
                    left_merge[left_key_temp] = join_series(left_df, join_left_key)
                    right_merge[right_key_temp] = join_series(right_df, join_right_key)
                    left_merge = left_merge.drop(columns=[join_left_key], errors="ignore")
                    right_merge = right_merge.drop(columns=[join_right_key], errors="ignore")
                    joined = pd.merge(
                        left_merge,
                        right_merge,
                        left_on=left_key_temp,
                        right_on=right_key_temp,
                        how=join_how,
                        suffixes=("_left", "_right"),
                    )
                    if left_key_temp in joined.columns and right_key_temp in joined.columns:
                        joined["__join_key__"] = joined[left_key_temp].combine_first(joined[right_key_temp])
                        joined = joined.drop(columns=[left_key_temp, right_key_temp])
                        joined = joined.rename(columns={"__join_key__": join_left_key})
                    joined_df = normalize_dataframe(joined.copy())
                    current_source = sanitize_filename_stem(source_name())
                    foreign_source = sanitize_filename_stem(str(join_table_path))
                    joined_name = f"joined_{current_source}_to_{foreign_source}_{join_how}_{uuid.uuid4().hex[:6]}.csv"
                    join_result_cache.set({"df": joined_df, "name": joined_name})
                    new_df = joined_df.copy()
        elif tool == "drop_duplicates":
            drop_duplicates_choice = input.drop_duplicates_col()
            subset = duplicate_subset_from_choice(drop_duplicates_choice, list(new_df.columns))
            new_df = new_df.drop_duplicates(subset=subset, keep="first").copy()
        elif tool == "find_duplicates":
            pass

        if not new_df.equals(current_df):
            added_columns = [col for col in new_df.columns if col not in current_df.columns]
            if added_columns:
                updated_preview_cols = list(dict.fromkeys(current_preview_cols + added_columns))
                preview_cols_state.set(updated_preview_cols)
                preview_all_selected = set(updated_preview_cols) == set(new_df.columns)
                preview_all_cols_state.set(preview_all_selected)
                ui.update_selectize(
                    "preview_cols",
                    selected=updated_preview_cols,
                    session=session,
                )
                ui.update_checkbox(
                    "preview_all_cols",
                    value=preview_all_selected,
                    session=session,
                )
            history = list(column_history.get() or [])
            history.append(new_df)
            column_history.set(history)

    @reactive.effect
    @reactive.event(input.undo_column, ignore_init=True)
    def _undo_column_tool() -> None:
        history = list(column_history.get() or [])
        if history:
            history.pop()
            column_history.set(history)

    @reactive.effect
    @reactive.event(input.save_preview_dataset, ignore_init=True)
    def _save_preview_dataset() -> None:
        preview_df = preview_dataframe()
        preview_name = str(input.preview_dataset_name() or "").strip()
        base_name = preview_name or f"{sanitize_filename_stem(source_name())}_preview"
        append_dataframe_source(preview_df, base_name, activate=True)

    @render.ui
    def uploaded_files_list():
        records = uploaded_file_records()
        active_label = str(active_csv_state.get() or input.active_csv_file() or "")

        if not records:
            return ui.div(
                ui.div("No CSV files uploaded. Demo data is active.", class_="small-muted"),
            )

        items = []
        for record in records:
            is_active = str(record["label"]) == active_label
            items.append(
                ui.tags.li(
                    ui.tags.span(str(record["label"])),
                    ui.tags.span(f" - {record['name']}", class_="small-muted"),
                    ui.tags.span(" (active)", class_="small-muted") if is_active else "",
                )
            )

        return ui.div(
            ui.div(f"Uploaded files ({len(records)})", class_="small-muted"),
            ui.tags.ul(*items),
        )

    @reactive.effect
    @reactive.event(input.apply_stats, ignore_init=True)
    def _apply_stats() -> None:
        applied_stats_config.set(
            {
                "ready": True,
                "stat_mode": input.stat_mode(),
                "stat_desc_col": input.stat_desc_col(),
                "stat_x_col": input.stat_x_col(),
                "stat_y_col": input.stat_y_col(),
                "stat_mr_target_col": input.stat_mr_target_col(),
                "stat_mr_predictor_cols": list(input.stat_mr_predictor_cols() or []),
                "stat_logit_target_col": input.stat_logit_target_col(),
                "stat_logit_positive_class": input.stat_logit_positive_class(),
                "stat_logit_predictor_cols": list(input.stat_logit_predictor_cols() or []),
                "stat_pca_cols": list(input.stat_pca_cols() or []),
                "stat_pca_components": int(input.stat_pca_components() or 2),
                "stat_pca_standardize": bool(input.stat_pca_standardize()),
                "stat_plot_x_log": bool(input.stat_plot_x_log()),
                "stat_plot_y_log": bool(input.stat_plot_y_log()),
                "stat_cluster_enable": bool(input.stat_cluster_enable()),
                "stat_cluster_k": int(input.stat_cluster_k() or 3),
                "stat_group_col": input.stat_group_col(),
                "stat_value_col": input.stat_value_col(),
                "stat_group_a": input.stat_group_a(),
                "stat_group_b": input.stat_group_b(),
                "stat_anova_col": input.stat_anova_col(),
                "stat_anova_value_col": input.stat_anova_value_col(),
            }
        )

    @reactive.calc
    def data() -> pd.DataFrame:
        history = column_history.get() or []
        if history:
            return history[-1]
        return base_data()

    @reactive.calc
    def source_name() -> str:
        record = active_uploaded_record()
        if record is None:
            return "demo data"
        return str(record["name"])

    def store_plot_cache(cache: reactive.Value, kind: str, content: object, filename: str, media_type: str) -> None:
        cache.set(
            {
                "kind": kind,
                "content": content,
                "filename": filename,
                "media_type": media_type,
            }
        )

    def plotly_download_html(fig) -> str:
        return fig.to_html(full_html=True, include_plotlyjs="cdn")

    def matplotlib_download_bytes(fig: plt.Figure) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()

    def plot_download_filename(cache: Optional[dict], prefix: str) -> str:
        if cache and cache.get("media_type") == "text/html":
            return f"{prefix}.html"
        return f"{prefix}.png"

    def cache_plotly(cache: reactive.Value, fig, prefix: str) -> None:
        store_plot_cache(cache, "html", plotly_download_html(fig), f"{prefix}.html", "text/html")

    def cache_matplotlib(cache: reactive.Value, fig: plt.Figure, prefix: str) -> None:
        store_plot_cache(cache, "png", matplotlib_download_bytes(fig), f"{prefix}.png", "image/png")

    def cluster_feature_frame_from_selection(mode: str, selection: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
        if mode == "descriptive":
            cols = [str(selection.get("stat_desc_col", ""))]
        elif mode in {"correlation", "regression", "heteroscedasticity"}:
            cols = [str(selection.get("stat_x_col", "")), str(selection.get("stat_y_col", ""))]
        elif mode == "multiple regression":
            cols = [str(selection.get("stat_mr_target_col", ""))] + list(selection.get("stat_mr_predictor_cols") or [])
        elif mode == "logistic regression":
            cols = list(selection.get("stat_logit_predictor_cols") or [])
        elif mode == "PCA":
            cols = list(selection.get("stat_pca_cols") or [])
        elif mode == "t-test":
            cols = [str(selection.get("stat_value_col", ""))]
        elif mode == "ANOVA":
            cols = [str(selection.get("stat_anova_value_col", ""))]
        else:
            cols = []

        cols = [col for col in dict.fromkeys(cols) if col]
        if not cols:
            return pd.DataFrame(index=df.index)
        if mode == "logistic regression":
            return stat_logistic_predictor_frame(df, cols).dropna()
        return stat_numeric_frame(df, cols).dropna()

    @reactive.effect
    def _update_controls() -> None:
        df = data()
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        categorical_cols = [
            col for col in cols if col not in numeric_cols and col not in datetime_cols
        ]
        choices = {INDEX_COL: "Index", **{col: col for col in cols}}
        analysis_cols = list(dict.fromkeys([INDEX_COL] + numeric_cols + datetime_cols))
        analysis_choices = {col: ("Index" if col == INDEX_COL else col) for col in analysis_cols}
        pie_value_choices = {"Count": "Count", INDEX_COL: "Index", **{col: col for col in cols}}
        choice_keys = set(choices.keys())
        analysis_choice_keys = set(analysis_choices.keys())
        duplicate_choices = {ALL_COLUMNS_DUPLICATES: "All columns", **{col: col for col in cols}}
        duplicate_choice_keys = set(duplicate_choices.keys())

        current_x = input.x_col()
        current_y = input.y_col()
        current_hue = input.hue_col()
        current_use_twin_y_axis = bool(input.use_twin_y_axis())
        current_y2_col = input.y2_col()
        current_y2_axis_title = input.y2_axis_title()
        current_y2_log = bool(input.y2_log())
        current_twin_y1_line_style = str(input.twin_y1_line_style() or twin_y1_line_style_state.get() or "solid").strip().lower()
        current_twin_y1_line_color = str(input.twin_y1_line_color() or twin_y1_line_color_state.get() or "blue").strip().lower()
        current_twin_y2_line_style = str(input.twin_y2_line_style() or twin_y2_line_style_state.get() or "dash").strip().lower()
        current_twin_y2_line_color = str(input.twin_y2_line_color() or twin_y2_line_color_state.get() or "orange").strip().lower()
        current_column_tool = input.column_tool()
        current_rename_from = input.rename_from()
        current_dtype_col = input.dtype_col()
        current_find_replace_col = input.find_replace_col()
        current_drop_rows_col = input.drop_rows_col()
        current_zscore_col = input.zscore_col()
        current_format_numeric_col = input.format_numeric_col()
        current_format_numeric_decimals = input.format_numeric_decimals()
        current_calc_new_col = str(input.calculate_new_col() or "")
        current_calc_col_a = input.calculate_col_a()
        current_calc_col_b = input.calculate_col_b()
        current_calc_formula = input.calculate_formula()
        current_active_csv = str(active_csv_state.get() or input.active_csv_file() or "")
        current_join_table = str(join_table_state.get() or input.join_table() or "")
        current_join_left_key = input.join_left_key()
        current_join_right_key = input.join_right_key()
        current_join_how = str(join_how_state.get() or input.join_how() or "left").strip()
        current_pivot_index_cols = list(input.pivot_index_cols() or [])
        current_pivot_column_col = input.pivot_column_col()
        current_pivot_value_col = input.pivot_value_col()
        current_pivot_aggfunc = input.pivot_aggfunc()
        current_melt_id_cols = list(input.melt_id_cols() or [])
        current_melt_value_cols = list(input.melt_value_cols() or [])
        current_melt_var_name = input.melt_var_name()
        current_melt_value_name = input.melt_value_name()
        current_find_duplicates_col = input.find_duplicates_col()
        current_drop_duplicates_col = input.drop_duplicates_col()
        stored_preview_cols = preview_cols_state.get()
        if stored_preview_cols is None:
            current_preview_cols = list(input.preview_cols() or [])
        else:
            current_preview_cols = list(stored_preview_cols or [])
        stored_preview_all_cols = preview_all_cols_state.get()
        current_preview_all_cols = bool(stored_preview_all_cols if stored_preview_all_cols is not None else input.preview_all_cols())
        current_stat_desc_col = input.stat_desc_col()
        current_stat_x_col = input.stat_x_col()
        current_stat_y_col = input.stat_y_col()
        current_stat_mr_target_col = input.stat_mr_target_col()
        current_stat_mr_predictor_cols = list(input.stat_mr_predictor_cols() or [])
        current_stat_logit_target_col = input.stat_logit_target_col()
        current_stat_logit_positive_class = input.stat_logit_positive_class()
        current_stat_logit_predictor_cols = list(input.stat_logit_predictor_cols() or [])
        current_stat_pca_cols = list(input.stat_pca_cols() or [])
        current_stat_pca_components = input.stat_pca_components()
        current_stat_pca_standardize = bool(input.stat_pca_standardize())
        current_stat_plot_x_log = bool(input.stat_plot_x_log())
        current_stat_plot_y_log = bool(input.stat_plot_y_log())
        current_stat_cluster_enable = bool(input.stat_cluster_enable())
        current_stat_cluster_k = input.stat_cluster_k()
        current_stat_group_col = input.stat_group_col()
        current_stat_value_col = input.stat_value_col()
        current_stat_group_a = input.stat_group_a()
        current_stat_group_b = input.stat_group_b()
        current_stat_anova_col = input.stat_anova_col()
        current_stat_anova_value_col = input.stat_anova_value_col()
        uploaded_records = uploaded_file_records()
        uploaded_table_choices = [str(record["label"]) for record in uploaded_records]

        def pick_single(current: str, fallback: str) -> str:
            if current in choice_keys:
                return current
            return fallback

        def pick_multi(current_values: list[str], fallback_values: list[str]) -> list[str]:
            selected = [value for value in current_values if value in choice_keys]
            return selected or fallback_values

        def pick_duplicate(current: str, fallback: str) -> str:
            if current in duplicate_choice_keys:
                return current
            return fallback

        def pick_upload(current: str, fallback: str) -> str:
            if current in uploaded_table_choices:
                return current
            return fallback

        def pick_numeric(current: str, fallback: str) -> str:
            if current in numeric_cols:
                return current
            return fallback

        def pick_text(current: str, fallback: str) -> str:
            value = str(current).strip()
            return value if value else fallback

        def pick_choice(current: str, fallback: str, allowed: set[str]) -> str:
            value = str(current).strip()
            return value if value in allowed else fallback

        preview_defaults = [INDEX_COL] + cols[: min(7, len(cols))]
        x_default = pick_single(current_x, INDEX_COL if INDEX_COL in choices else (datetime_cols[0] if datetime_cols else cols[0]))
        y_default = pick_single(current_y, numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else cols[0]))
        y2_default = pick_numeric(current_y2_col, next((col for col in numeric_cols if col != y_default), (numeric_cols[0] if numeric_cols else "")))
        y2_axis_title_default = pick_text(current_y2_axis_title, y2_default if y2_default else "Secondary Y axis")
        y2_log_default = current_y2_log
        hue_choices = ["None", INDEX_COL] + categorical_cols
        hue_default = current_hue if current_hue in hue_choices else "None"
        column_tool_default = current_column_tool if current_column_tool in {"rename", "convert_type", "find_replace", "drop_rows", "z_score normalisation", "format_numeric", "calculate_formula", "pivot_wide", "melt", "join_tables", "find_duplicates", "drop_duplicates"} else "rename"
        pie_label_default = categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL)
        pie_value_default = "Count" if "Count" in pie_value_choices else (numeric_cols[0] if numeric_cols else INDEX_COL)
        rename_from_default = pick_single(current_rename_from, cols[0] if cols else INDEX_COL)
        dtype_col_default = pick_single(current_dtype_col, cols[0] if cols else INDEX_COL)
        find_replace_col_default = pick_single(current_find_replace_col, cols[0] if cols else INDEX_COL)
        drop_rows_col_default = pick_single(current_drop_rows_col, cols[0] if cols else INDEX_COL)
        zscore_col_default = pick_numeric(current_zscore_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        format_numeric_col_default = pick_numeric(current_format_numeric_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        calc_new_col_default = current_calc_new_col
        calc_col_a_default = pick_numeric(current_calc_col_a, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        calc_col_b_default = pick_numeric(current_calc_col_b, numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL)))
        calc_formula_default = current_calc_formula if current_calc_formula is not None else ""
        try:
            format_numeric_decimals_default = int(current_format_numeric_decimals)
        except Exception:
            format_numeric_decimals_default = 2
        if format_numeric_decimals_default < 0:
            format_numeric_decimals_default = 2
        active_csv_choices = ["Demo data (default)"] if not uploaded_records else uploaded_table_choices
        active_csv_default = current_active_csv if current_active_csv in active_csv_choices else next(iter(active_csv_choices), "Demo data (default)")
        join_table_default = pick_upload(current_join_table, next(iter(uploaded_table_choices), ""))
        pivot_index_fallback = [categorical_cols[0]] if categorical_cols else ([cols[0]] if cols else [INDEX_COL])
        if current_pivot_index_cols:
            pivot_index_default = [value for value in current_pivot_index_cols if value in choice_keys]
        else:
            pivot_index_default = []
        if not pivot_index_default:
            pivot_index_default = [value for value in pivot_index_fallback if value in choice_keys]
        pivot_column_default = pick_single(current_pivot_column_col, categorical_cols[1] if len(categorical_cols) > 1 else (cols[1] if len(cols) > 1 else (cols[0] if cols else INDEX_COL)))
        pivot_value_default = pick_single(current_pivot_value_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        pivot_aggfunc_default = current_pivot_aggfunc if str(current_pivot_aggfunc).strip() in {"sum", "mean", "median", "min", "max", "count", "first", "last", "nunique"} else ("sum" if numeric_cols else "count")
        melt_id_fallback = [categorical_cols[0]] if categorical_cols else ([cols[0]] if cols else [INDEX_COL])
        if current_melt_id_cols:
            melt_id_default = [value for value in current_melt_id_cols if value in choice_keys]
        else:
            melt_id_default = []
        if not melt_id_default:
            melt_id_default = [value for value in melt_id_fallback if value in choice_keys]
        melt_value_default = [value for value in current_melt_value_cols if value in choice_keys]
        if not melt_value_default:
            melt_value_default = [value for value in cols if value not in melt_id_default][: max(1, min(4, len(cols)))]
        melt_var_name_default = pick_text(current_melt_var_name, "variable")
        melt_value_name_default = pick_text(current_melt_value_name, "value")
        join_secondary_df = next((record["df"] for record in uploaded_records if str(record["label"]) == join_table_default), None)
        join_secondary_cols = list(join_secondary_df.columns) if isinstance(join_secondary_df, pd.DataFrame) else []
        join_left_default = pick_single(current_join_left_key, cols[0] if cols else INDEX_COL)
        join_right_default = join_left_default if join_left_default in join_secondary_cols or join_left_default == INDEX_COL else (join_secondary_cols[0] if join_secondary_cols else INDEX_COL)
        join_how_default = current_join_how if current_join_how in {"left", "inner", "right", "outer"} else "left"
        find_duplicates_col_default = pick_duplicate(current_find_duplicates_col, ALL_COLUMNS_DUPLICATES)
        drop_duplicates_col_default = pick_duplicate(current_drop_duplicates_col, ALL_COLUMNS_DUPLICATES)
        preview_default_selected = [value for value in current_preview_cols if value in choice_keys]
        if current_preview_all_cols:
            preview_default_selected = list(choices.keys())
        elif stored_preview_cols is None and not preview_default_selected:
            preview_default_selected = pick_multi(current_preview_cols, preview_defaults)
        preview_all_effective = current_preview_all_cols and set(preview_default_selected) == choice_keys
        stat_desc_col_default = pick_single(current_stat_desc_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        stat_x_default = pick_single(current_stat_x_col, datetime_cols[0] if datetime_cols else (numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL)))
        stat_y_default = pick_single(current_stat_y_col, numeric_cols[0] if numeric_cols else (cols[0] if len(cols) > 0 else INDEX_COL))
        stat_mr_target_default = pick_single(current_stat_mr_target_col, analysis_cols[0] if analysis_cols else INDEX_COL)
        stat_mr_predictor_choices = {k: v for k, v in analysis_choices.items() if k != stat_mr_target_default}
        stat_mr_predictor_default = [value for value in current_stat_mr_predictor_cols if value in stat_mr_predictor_choices]
        if not stat_mr_predictor_default:
            stat_mr_predictor_default = [value for value in analysis_cols if value != stat_mr_target_default][:2]
        stat_logit_target_default = pick_single(current_stat_logit_target_col, cols[0] if cols else INDEX_COL)
        stat_logit_positive_choices = []
        if stat_logit_target_default in choice_keys:
            stat_logit_positive_choices = list(dict.fromkeys(df[stat_logit_target_default].dropna().astype(str).tolist()))[:100]
        stat_logit_positive_default = (
            current_stat_logit_positive_class
            if current_stat_logit_positive_class in stat_logit_positive_choices
            else (stat_logit_positive_choices[1] if len(stat_logit_positive_choices) > 1 else (stat_logit_positive_choices[0] if stat_logit_positive_choices else ""))
        )
        stat_logit_predictor_choices = {k: v for k, v in choices.items() if k != stat_logit_target_default}
        stat_logit_predictor_default = [value for value in current_stat_logit_predictor_cols if value in stat_logit_predictor_choices]
        if not stat_logit_predictor_default:
            preferred_logit_cols = [value for value in analysis_cols if value != stat_logit_target_default]
            if not preferred_logit_cols:
                preferred_logit_cols = [value for value in cols if value != stat_logit_target_default]
            stat_logit_predictor_default = preferred_logit_cols[:2]
        stat_pca_default = [value for value in current_stat_pca_cols if value in analysis_choice_keys]
        if not stat_pca_default:
            stat_pca_default = analysis_cols[:3]
        try:
            stat_pca_components_default = int(current_stat_pca_components)
        except Exception:
            stat_pca_components_default = 2
        if stat_pca_components_default < 1:
            stat_pca_components_default = 2
        stat_pca_standardize_default = current_stat_pca_standardize
        stat_plot_x_log_default = current_stat_plot_x_log
        stat_plot_y_log_default = current_stat_plot_y_log
        stat_cluster_enable_default = current_stat_cluster_enable
        try:
            stat_cluster_k_default = int(current_stat_cluster_k)
        except Exception:
            stat_cluster_k_default = 3
        if stat_cluster_k_default < 1:
            stat_cluster_k_default = 3
        stat_group_col_default = pick_single(current_stat_group_col, categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL))
        stat_value_col_default = pick_single(current_stat_value_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        stat_anova_col_default = pick_single(current_stat_anova_col, categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL))
        stat_anova_value_col_default = pick_single(current_stat_anova_value_col, numeric_cols[0] if numeric_cols else (cols[0] if cols else INDEX_COL))
        stat_group_source = stat_group_col_default if stat_group_col_default in choice_keys else (categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL))
        stat_group_values = []
        if stat_group_source in df.columns:
            stat_group_values = list(dict.fromkeys(df[stat_group_source].dropna().astype(str).tolist()))
            stat_group_values = stat_group_values[:100]
        if len(stat_group_values) >= 2:
            stat_group_a_default = current_stat_group_a if current_stat_group_a in stat_group_values else stat_group_values[0]
            stat_group_b_default = current_stat_group_b if current_stat_group_b in stat_group_values and current_stat_group_b != stat_group_a_default else stat_group_values[1]
        elif len(stat_group_values) == 1:
            stat_group_a_default = stat_group_values[0]
            stat_group_b_default = stat_group_values[0]
        else:
            stat_group_a_default = ""
            stat_group_b_default = ""

        ui.update_selectize(
            "preview_cols",
            choices=choices,
            selected=list(choices.keys()) if preview_all_effective else preview_default_selected,
            session=session,
        )
        ui.update_checkbox("preview_all_cols", value=preview_all_effective, session=session)
        ui.update_select(
            "x_col",
            choices=choices,
            selected=x_default,
            session=session,
        )
        ui.update_select(
            "y_col",
            choices=choices,
            selected=y_default,
            session=session,
        )
        ui.update_checkbox(
            "use_twin_y_axis",
            value=current_use_twin_y_axis if numeric_cols else False,
            session=session,
        )
        ui.update_select(
            "y2_col",
            choices={col: col for col in numeric_cols} if numeric_cols else {"": "No numeric columns available"},
            selected=y2_default if y2_default in numeric_cols else (numeric_cols[0] if numeric_cols else ""),
            session=session,
        )
        ui.update_text(
            "y2_axis_title",
            value=y2_axis_title_default,
            session=session,
        )
        ui.update_checkbox(
            "y2_log",
            value=y2_log_default,
            session=session,
        )
        ui.update_select(
            "hue_col",
            choices=hue_choices,
            selected=hue_default,
            session=session,
        )
        ui.update_select(
            "rename_from",
            choices=choices,
            selected=rename_from_default,
            session=session,
        )
        ui.update_select(
            "column_tool",
            choices=["rename",
                        "convert_type",
                        "format_numeric",
                        "find_replace",
                        "drop_rows",
                        "find_duplicates",
                        "drop_duplicates",
                        "z_score normalisation",
                        "calculate_formula",
                        "pivot_wide",
                        "melt",
                        "join_tables"],
            selected=column_tool_default,
            session=session,
        )
        ui.update_select(
            "dtype_col",
            choices=choices,
            selected=dtype_col_default,
            session=session,
        )
        ui.update_select(
            "find_replace_col",
            choices=choices,
            selected=find_replace_col_default,
            session=session,
        )
        ui.update_select(
            "drop_rows_col",
            choices=choices,
            selected=drop_rows_col_default,
            session=session,
        )
        ui.update_select(
            "zscore_col",
            choices={col: col for col in numeric_cols} if numeric_cols else {"": "No numeric columns available"},
            selected=zscore_col_default if zscore_col_default in numeric_cols else (numeric_cols[0] if numeric_cols else ""),
            session=session,
        )
        ui.update_select(
            "format_numeric_col",
            choices={col: col for col in numeric_cols} if numeric_cols else {"": "No numeric columns available"},
            selected=format_numeric_col_default if format_numeric_col_default in numeric_cols else (numeric_cols[0] if numeric_cols else ""),
            session=session,
        )
        ui.update_numeric(
            "format_numeric_decimals",
            value=format_numeric_decimals_default,
            session=session,
        )
        ui.update_text(
            "calculate_new_col",
            value=calc_new_col_default,
            session=session,
        )
        ui.update_select(
            "calculate_col_a",
            choices={col: col for col in numeric_cols} if numeric_cols else {"": "No numeric columns available"},
            selected=calc_col_a_default if calc_col_a_default in numeric_cols else (numeric_cols[0] if numeric_cols else ""),
            session=session,
        )
        ui.update_select(
            "calculate_col_b",
            choices={col: col for col in numeric_cols} if numeric_cols else {"": "No numeric columns available"},
            selected=calc_col_b_default if calc_col_b_default in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else "")),
            session=session,
        )
        ui.update_text(
            "calculate_formula",
            value=calc_formula_default,
            session=session,
        )
        ui.update_select(
            "active_csv_file",
            choices=active_csv_choices,
            selected=active_csv_default,
            session=session,
        )
        ui.update_select(
            "join_table",
            choices=uploaded_table_choices,
            selected=join_table_default,
            session=session,
        )
        ui.update_select(
            "join_how",
            choices={
                "Left join (keep current rows)": "left",
                "Inner join (matches only)": "inner",
                "Right join (keep foreign rows)": "right",
                "Full outer join": "outer",
            },
            selected=join_how_default,
            session=session,
        )
        ui.update_select(
            "join_left_key",
            choices=choices,
            selected=join_left_default,
            session=session,
        )
        ui.update_select(
            "join_right_key",
            choices={INDEX_COL: "Index", **{col: col for col in join_secondary_cols}},
            selected=join_right_default if join_right_default in join_secondary_cols or join_right_default == INDEX_COL else (join_secondary_cols[0] if join_secondary_cols else INDEX_COL),
            session=session,
        )
        ui.update_selectize(
            "pivot_index_cols",
            choices=choices,
            selected=pivot_index_default,
            session=session,
        )
        ui.update_select(
            "pivot_column_col",
            choices=choices,
            selected=pivot_column_default,
            session=session,
        )
        ui.update_select(
            "pivot_value_col",
            choices=choices,
            selected=pivot_value_default,
            session=session,
        )
        ui.update_select(
            "pivot_aggfunc",
            choices=["sum", "mean", "median", "min", "max", "count", "first", "last", "nunique"],
            selected=pivot_aggfunc_default,
            session=session,
        )
        ui.update_selectize(
            "melt_id_cols",
            choices=choices,
            selected=melt_id_default,
            session=session,
        )
        ui.update_selectize(
            "melt_value_cols",
            choices=choices,
            selected=melt_value_default,
            session=session,
        )
        ui.update_text(
            "melt_var_name",
            value=melt_var_name_default,
            session=session,
        )
        ui.update_text(
            "melt_value_name",
            value=melt_value_name_default,
            session=session,
        )
        ui.update_select(
            "find_duplicates_col",
            choices=duplicate_choices,
            selected=find_duplicates_col_default,
            session=session,
        )
        ui.update_select(
            "drop_duplicates_col",
            choices=duplicate_choices,
            selected=drop_duplicates_col_default,
            session=session,
        )
        ui.update_select(
            "stat_desc_col",
            choices=choices,
            selected=stat_desc_col_default,
            session=session,
        )
        ui.update_select(
            "stat_x_col",
            choices=choices,
            selected=stat_x_default,
            session=session,
        )
        ui.update_select(
            "stat_y_col",
            choices=choices,
            selected=stat_y_default,
            session=session,
        )
        ui.update_select(
            "stat_mr_target_col",
            choices=analysis_choices,
            selected=stat_mr_target_default,
            session=session,
        )
        ui.update_selectize(
            "stat_mr_predictor_cols",
            choices=stat_mr_predictor_choices,
            selected=stat_mr_predictor_default,
            session=session,
        )
        ui.update_select(
            "stat_logit_target_col",
            choices=choices,
            selected=stat_logit_target_default,
            session=session,
        )
        ui.update_select(
            "stat_logit_positive_class",
            choices=stat_logit_positive_choices,
            selected=stat_logit_positive_default,
            session=session,
        )
        ui.update_selectize(
            "stat_logit_predictor_cols",
            choices=stat_logit_predictor_choices,
            selected=stat_logit_predictor_default,
            session=session,
        )
        ui.update_selectize(
            "stat_pca_cols",
            choices=analysis_choices,
            selected=stat_pca_default,
            session=session,
        )
        ui.update_numeric(
            "stat_pca_components",
            value=stat_pca_components_default,
            session=session,
        )
        ui.update_checkbox(
            "stat_pca_standardize",
            value=stat_pca_standardize_default,
            session=session,
        )
        ui.update_checkbox(
            "stat_plot_x_log",
            value=stat_plot_x_log_default,
            session=session,
        )
        ui.update_checkbox(
            "stat_plot_y_log",
            value=stat_plot_y_log_default,
            session=session,
        )
        ui.update_checkbox(
            "stat_cluster_enable",
            value=stat_cluster_enable_default,
            session=session,
        )
        ui.update_numeric(
            "stat_cluster_k",
            value=stat_cluster_k_default,
            session=session,
        )
        ui.update_select(
            "stat_group_col",
            choices=choices,
            selected=stat_group_col_default,
            session=session,
        )
        ui.update_select(
            "stat_value_col",
            choices=choices,
            selected=stat_value_col_default,
            session=session,
        )
        ui.update_select(
            "stat_anova_col",
            choices=choices,
            selected=stat_anova_col_default,
            session=session,
        )
        ui.update_select(
            "stat_anova_value_col",
            choices=choices,
            selected=stat_anova_value_col_default,
            session=session,
        )
        ui.update_select(
            "stat_group_a",
            choices=stat_group_values,
            selected=stat_group_a_default,
            session=session,
        )
        ui.update_select(
            "stat_group_b",
            choices=stat_group_values,
            selected=stat_group_b_default,
            session=session,
        )

    @render.text
    def summary() -> str:
        df = data()
        missing = int(df.isna().sum().sum())
        numeric_n = len(df.select_dtypes(include="number").columns)
        datetime_n = len(df.select_dtypes(include="datetime").columns)
        return (
            f"Source: {source_name()} | Rows: {df.shape[0]} | Columns: {df.shape[1]} | "
            f"Missing cells: {missing} | Numeric columns: {numeric_n} | "
            f"Datetime columns: {datetime_n}"
        )

    @render.ui
    def preview_notice():
        df = data()
        if input.column_tool() != "find_duplicates":
            return ui.div()

        choice = input.find_duplicates_col() or ALL_COLUMNS_DUPLICATES
        subset = duplicate_subset_from_choice(choice, df.columns.tolist())
        duplicate_mask = df.duplicated(subset=subset, keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        if duplicate_rows == 0:
            scope_label = "all columns" if choice == ALL_COLUMNS_DUPLICATES else choice
            return ui.div(ui.div(f"No duplicated rows found for {scope_label}.", class_="small-muted"))

        scope_label = "all columns" if choice == ALL_COLUMNS_DUPLICATES else choice
        return ui.div(
            ui.div(
                f"Showing {duplicate_rows} duplicated rows based on {scope_label}.",
                class_="small-muted",
            )
        )

    @render.ui
    def pivot_notice():
        if input.column_tool() != "pivot_wide":
            return ui.div()

        df = data()
        working_df = add_index_column(df)
        index_cols = [col for col in (input.pivot_index_cols() or []) if col in working_df.columns]
        pivot_column_col = input.pivot_column_col()
        pivot_value_col = input.pivot_value_col()
        pivot_aggfunc = str(input.pivot_aggfunc() or "sum").strip().lower()
        if not index_cols or pivot_column_col not in working_df.columns or pivot_value_col not in working_df.columns:
            return ui.div(ui.div("Choose row fields, a column field, and a values field.", class_="small-muted"))

        duplicate_rows = int(working_df.duplicated(subset=index_cols + [pivot_column_col], keep=False).sum())
        if duplicate_rows:
            return ui.div(
                ui.div(
                    f"{duplicate_rows} rows share the same row and column combination. "
                    f"Pivoting will aggregate them with {pivot_aggfunc}.",
                    class_="small-muted",
                )
            )
        return ui.div(
            ui.div(
                f"Row and column combinations are unique. Aggregation will still use {pivot_aggfunc} if needed.",
                class_="small-muted",
            )
        )

    @render.ui
    def join_notice():
        if input.column_tool() != "join_tables":
            return ui.div()

        records = uploaded_file_records()
        if not records:
            return ui.div(ui.div("Upload one or more CSV files to enable joins.", class_="small-muted"))

        join_table_path = str(join_table_state.get() or input.join_table() or "")
        secondary_record = next((record for record in records if str(record["label"]) == join_table_path), None)
        if secondary_record is None:
            return ui.div(ui.div("Choose a foreign table to join against.", class_="small-muted"))

        current_df = data()
        left_key = input.join_left_key()
        right_key = input.join_right_key()
        if not left_key or not right_key:
            return ui.div(ui.div("Choose both a primary key and a foreign key.", class_="small-muted"))

        left_df = add_index_column(current_df)
        right_df = add_index_column(secondary_record["df"]) if isinstance(secondary_record.get("df"), pd.DataFrame) else pd.DataFrame()
        if left_key not in left_df.columns and left_key != INDEX_COL:
            return ui.div(ui.div("Choose a valid primary key column.", class_="small-muted"))
        if right_key not in right_df.columns and right_key != INDEX_COL:
            return ui.div(ui.div("Choose a valid foreign key column.", class_="small-muted"))

        left_subset = [left_key] if left_key != INDEX_COL else [INDEX_COL]
        right_subset = [right_key] if right_key != INDEX_COL else [INDEX_COL]
        left_unique = not left_df.duplicated(subset=left_subset, keep=False).any()
        right_unique = not right_df.duplicated(subset=right_subset, keep=False).any()
        table_label = str(secondary_record["label"])
        status = "Keys look unique." if left_unique and right_unique else "One or both keys contain duplicates; the join will be one-to-many or many-to-many."
        return ui.div(
            ui.div(f"Joining current data to {table_label} with a {join_how_state.get() or 'left'} join.", class_="small-muted"),
            ui.div(status, class_="small-muted"),
        )

    @render.ui
    def join_download_ui():
        if input.column_tool() != "join_tables":
            return ui.div()

        cache = join_result_cache.get()
        if not cache or not isinstance(cache.get("df"), pd.DataFrame):
            return ui.div()

        joined_name = str(cache.get("name") or "joined_table.csv")
        return ui.div(
            ui.download_button("download_joined_csv", "Save joined CSV"),
            ui.help_text(f"Download the most recent joined table as {joined_name}."),
        )

    @render.data_frame
    def preview():
        return render.DataGrid(preview_dataframe(), filters=True, selection_mode="rows")

    @render.ui
    def histogram_controls():
        if input.plot_kind() != "histogram":
            return ui.div()
        return ui.card(
            ui.card_header("Histogram Controls"),
            ui.input_numeric("hist_bins", "Histogram bins", value=20, min=1, max=200, step=1),
        )

    @render.ui
    def pie_controls():
        if input.plot_kind() != "pie":
            return ui.div()
        df = data()
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        categorical_cols = [
            col for col in cols if col not in numeric_cols and col not in datetime_cols
        ]
        label_choices = {INDEX_COL: "Index", **{col: col for col in cols}}
        value_choices = {"Count": "Count", INDEX_COL: "Index", **{col: col for col in cols}}
        label_default = categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL)
        value_default = "Count" if "Count" in value_choices else (numeric_cols[0] if numeric_cols else INDEX_COL)
        return ui.card(
            ui.card_header("Pie Chart Controls"),
            ui.help_text("Pie charts use label and value columns only."),
            ui.input_select("pie_label_col", "Label column", choices=label_choices, selected=label_default),
            ui.input_select("pie_value_col", "Value column", choices=value_choices, selected=value_default),
            ui.input_checkbox("pie_donut", "Donut chart", value=False),
        )

    @render.ui
    def stat_cluster_hint():
        if not bool(input.stat_cluster_enable()):
            return ui.div()

        df = data()
        mode = input.stat_mode()
        selection = {
            "stat_desc_col": input.stat_desc_col(),
            "stat_x_col": input.stat_x_col(),
            "stat_y_col": input.stat_y_col(),
            "stat_mr_target_col": input.stat_mr_target_col(),
            "stat_mr_predictor_cols": list(input.stat_mr_predictor_cols() or []),
            "stat_logit_predictor_cols": list(input.stat_logit_predictor_cols() or []),
            "stat_pca_cols": list(input.stat_pca_cols() or []),
            "stat_value_col": input.stat_value_col(),
            "stat_anova_value_col": input.stat_anova_value_col(),
        }
        feature_df = cluster_feature_frame_from_selection(mode, selection, df)
        if feature_df.empty or len(feature_df) < 2:
            return ui.div(
                ui.p("Not enough numeric data to estimate a k suggestion for clustering."),
                class_="small-muted",
            )

        suggestion = suggest_k_by_elbow(feature_df.to_numpy(dtype=float), max_k=min(8, len(feature_df)))
        suggested_k = int(suggestion["suggested_k"])
        current_k = int(input.stat_cluster_k() or suggested_k)
        curve = suggestion["curve"]
        inertia_msg = ""
        if not curve.empty:
            best_row = curve.loc[curve["k"] == suggested_k]
            if not best_row.empty:
                inertia_msg = f" Elbow inertia: {round(float(best_row.iloc[0]['inertia']), 2)}."

        note = f"Suggested k - value by elbow plot: {suggested_k}."
        if current_k != suggested_k:
            note += f" Current value of k is {current_k}."
        note += inertia_msg

        return ui.div(
            ui.div(note, class_="small-muted"),
        )

    @render.ui
    def stat_pca_hint():
        if input.stat_mode() != "PCA":
            return ui.div()

        df = data()
        pca_cols = list(input.stat_pca_cols() or [])
        if not pca_cols:
            return ui.div(ui.div("Choose PCA columns to see a suggested number of components.", class_="small-muted"))

        feature_df = stat_numeric_frame(df, pca_cols).dropna()
        if feature_df.empty or len(feature_df) < 2:
            return ui.div(ui.div("Not enough numeric data to estimate PCA components.", class_="small-muted"))

        suggestion = suggest_pca_components_by_elbow(
            feature_df.to_numpy(dtype=float),
            standardize=bool(input.stat_pca_standardize()),
        )
        suggested_components = int(suggestion["suggested_components"])
        current_components = int(input.stat_pca_components() or suggested_components)
        curve = suggestion["curve"]
        basis = str(suggestion.get("basis", "Elbow point of the cumulative explained-variance curve."))
        cumulative_msg = ""
        if not curve.empty:
            best_row = curve.loc[curve["component"] == suggested_components]
            if not best_row.empty:
                cumulative_msg = f" Cumulative explained variance: {round(float(best_row.iloc[0]['cumulative']), 6)}."

        selected_cols = ", ".join(pca_cols[:6])
        if len(pca_cols) > 6:
            selected_cols += ", ..."

        standardize_note = "on" if bool(input.stat_pca_standardize()) else "off"
        note = f"Suggested PCA components: {suggested_components}. Basis: {basis}."
        note += f" Standardization: {standardize_note}."
        note += f" Selected columns: {selected_cols}."
        if current_components != suggested_components:
            note += f" Current components: {current_components}."
        note += cumulative_msg

        return ui.div(ui.div(note, class_="small-muted"))

    @render.ui
    def statistics_plot():
        df = data()
        config = applied_stats_config.get()
        stat_plot_x_log = bool(input.stat_plot_x_log())
        stat_plot_y_log = bool(input.stat_plot_y_log())

        if not config.get("ready"):
            return ui.div(
                ui.p("Choose a test, then click Run statistics to preview the data used by that test."),
            )

        stat_mode = config["stat_mode"]
        stat_desc_col = config["stat_desc_col"]
        stat_x_col = config["stat_x_col"]
        stat_y_col = config["stat_y_col"]
        stat_group_col = config["stat_group_col"]
        stat_value_col = config["stat_value_col"]
        stat_group_a = config["stat_group_a"]
        stat_group_b = config["stat_group_b"]
        stat_anova_col = config["stat_anova_col"]
        stat_anova_value_col = config["stat_anova_value_col"]

        def selected_series(column: str) -> pd.Series:
            if column == INDEX_COL:
                return resolve_column(df, INDEX_COL)
            if column in df.columns:
                return df[column]
            return pd.Series(dtype="float64")

        def selected_numeric(column: str) -> pd.Series:
            return series_to_stat_numeric(selected_series(column))

        def format_stats_fig(fig, title: str):
            fig.update_layout(
                template="plotly_white",
                title=title,
                height=480,
                margin=dict(l=55, r=25, t=70, b=90),
                autosize=True,
            )
            fig.update_xaxes(automargin=True)
            fig.update_yaxes(automargin=True)
            return fig

        def maybe_apply_log_axes(
            fig,
            *,
            x_values: Optional[pd.Series | np.ndarray] = None,
            y_values: Optional[pd.Series | np.ndarray] = None,
            x_allowed: bool = False,
            y_allowed: bool = False,
        ):
            def can_log(values: Optional[pd.Series | np.ndarray]) -> bool:
                if values is None:
                    return False
                try:
                    arr = np.asarray(values, dtype=float)
                except Exception:
                    return False
                arr = arr[np.isfinite(arr)]
                return bool(arr.size and float(np.min(arr)) > 0)

            if stat_plot_x_log and x_allowed and can_log(x_values):
                try:
                    fig.update_xaxes(type="log")
                except Exception:
                    pass
            if stat_plot_y_log and y_allowed and can_log(y_values):
                try:
                    fig.update_yaxes(type="log")
                except Exception:
                    pass
            return fig

        def clustered_labels(feature_df: pd.DataFrame) -> Optional[pd.Series]:
            if not bool(config.get("stat_cluster_enable")):
                return None
            if feature_df.empty:
                return None
            cleaned = feature_df.dropna()
            if cleaned.empty:
                return None
            try:
                fit = fit_kmeans(cleaned.to_numpy(dtype=float), n_clusters=int(config.get("stat_cluster_k", 3) or 3))
            except ValueError:
                return None
            return pd.Series([f"Cluster {int(label) + 1}" for label in fit["labels"]], index=cleaned.index, name="_cluster")

        def attach_cluster_labels(plot_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
            labels = clustered_labels(feature_frame)
            if labels is None:
                return plot_frame
            out = plot_frame.copy()
            out["_cluster"] = labels.reindex(out.index).fillna("No cluster").astype(str)
            return out

        if stat_mode == "descriptive":
            series = selected_series(stat_desc_col).dropna()
            if series.empty:
                return ui.div(ui.p("No data available for the selected descriptive column."))

            if pd.api.types.is_datetime64_any_dtype(series):
                plot_df = pd.DataFrame({"value": pd.to_datetime(series, errors="coerce")}).dropna()
                fig = px.histogram(plot_df, x="value", nbins=min(30, max(5, len(plot_df) // 5 or 5)), title=f"Distribution of {stat_desc_col}")
            else:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if numeric.empty:
                    return ui.div(ui.p("Choose a numeric or datetime column for the descriptive preview plot."))
                plot_df = pd.DataFrame({"value": numeric})
                if plot_df["value"].nunique(dropna=True) <= 20:
                    counts = plot_df["value"].value_counts().sort_index().reset_index()
                    counts.columns = ["value", "count"]
                    fig = px.bar(counts, x="value", y="count", title=f"Distribution of {stat_desc_col}")
                    maybe_apply_log_axes(fig, y_values=counts["count"], y_allowed=True)
                else:
                    fig = px.histogram(
                        plot_df,
                        x="value",
                        nbins=min(30, max(5, len(plot_df) // 5 or 5)),
                        title=f"Distribution of {stat_desc_col}",
                    )
                    maybe_apply_log_axes(fig, x_values=plot_df["value"], x_allowed=True)

            format_stats_fig(fig, f"Distribution of {stat_desc_col}")
            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        if stat_mode in {"correlation", "regression", "heteroscedasticity"}:
            x_raw = selected_series(stat_x_col)
            y_raw = selected_series(stat_y_col)
            x_num = selected_numeric(stat_x_col)
            y_num = selected_numeric(stat_y_col)
            plot_df = pd.DataFrame(
                {
                    "_x_raw": x_raw,
                    "_y_raw": y_raw,
                    "_x_num": x_num,
                    "_y_num": y_num,
                }
            ).dropna(subset=["_x_num", "_y_num"])

            if plot_df.empty:
                return ui.div(ui.p("Choose numeric or datetime X and Y columns with valid rows."))

            x_plot = plot_df["_x_raw"] if pd.api.types.is_datetime64_any_dtype(x_raw) else plot_df["_x_num"]
            y_plot = plot_df["_y_raw"] if pd.api.types.is_datetime64_any_dtype(y_raw) else plot_df["_y_num"]
            scatter_df = plot_df.assign(_x_plot=x_plot, _y_plot=y_plot)
            scatter_df = attach_cluster_labels(scatter_df, plot_df[["_x_num", "_y_num"]])
            scatter_kwargs = dict(
                data_frame=scatter_df,
                x="_x_plot",
                y="_y_plot",
                title="Statistics Plot",
                opacity=0.75,
            )
            if "_cluster" in scatter_df.columns:
                scatter_kwargs["color"] = "_cluster"
            fig = px.scatter(**scatter_kwargs)
            fig.update_traces(marker=dict(size=8))
            format_stats_fig(fig, "Statistics Plot")
            fig.update_xaxes(title_text=stat_x_col)
            fig.update_yaxes(title_text=stat_y_col)
            maybe_apply_log_axes(
                fig,
                x_values=plot_df["_x_num"] if not pd.api.types.is_datetime64_any_dtype(x_raw) else None,
                y_values=plot_df["_y_num"] if not pd.api.types.is_datetime64_any_dtype(y_raw) else None,
                x_allowed=not pd.api.types.is_datetime64_any_dtype(x_raw),
                y_allowed=not pd.api.types.is_datetime64_any_dtype(y_raw),
            )

            if stat_mode in {"regression", "heteroscedasticity"}:
                fit_x = plot_df["_x_num"].to_numpy(dtype=float)
                fit_y = plot_df["_y_num"].to_numpy(dtype=float)
                try:
                    stats = fit_linear_regression(fit_x, fit_y)
                    fitted = stats["slope"] * fit_x + stats["intercept"]
                    line_df = plot_df.copy().sort_values("_x_num")
                    line_df["_fitted"] = stats["slope"] * line_df["_x_num"] + stats["intercept"]
                    line_x = line_df["_x_raw"] if pd.api.types.is_datetime64_any_dtype(x_raw) else line_df["_x_num"]
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_df["_fitted"],
                            mode="lines",
                            name="Fit line",
                            line=dict(color="black", dash="dash"),
                        )
                    )
                    if stat_mode == "regression":
                        fig.add_annotation(
                            text=f"R² = {stats['r2']:.4f}",
                            xref="paper",
                            yref="paper",
                            x=0.01,
                            y=0.99,
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)",
                        )
                except ValueError:
                    pass

            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        if stat_mode == "multiple regression":
            predictor_cols = list(config["stat_mr_predictor_cols"] or [])
            target_col = config["stat_mr_target_col"]
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column."))
            if target_col in predictor_cols:
                predictor_cols = [col for col in predictor_cols if col != target_col]
            mr_frame = stat_numeric_frame(df, [target_col] + predictor_cols).dropna()
            if mr_frame.empty or len(mr_frame) < len(predictor_cols) + 2:
                return ui.div(ui.p("Choose numeric or datetime target and predictor columns with enough valid rows."))

            fit = fit_multiple_linear_regression(
                mr_frame[predictor_cols].to_numpy(dtype=float),
                mr_frame[target_col].to_numpy(dtype=float),
                predictor_cols,
            )
            actual = mr_frame[target_col].to_numpy(dtype=float)
            predicted = fit["predicted"]
            lims = [
                float(np.nanmin(np.concatenate([actual, predicted]))),
                float(np.nanmax(np.concatenate([actual, predicted]))),
            ]
            scatter_df = pd.DataFrame({"predicted": predicted, "actual": actual})
            scatter_df = attach_cluster_labels(scatter_df, mr_frame[[target_col] + predictor_cols])
            scatter_kwargs = dict(
                data_frame=scatter_df,
                x="predicted",
                y="actual",
                opacity=0.75,
                title="Multiple regression: actual vs predicted",
            )
            if "_cluster" in scatter_df.columns:
                scatter_kwargs["color"] = "_cluster"
            fig = px.scatter(**scatter_kwargs)
            fig.add_trace(
                go.Scatter(
                    x=lims,
                    y=lims,
                    mode="lines",
                    name="Perfect fit",
                    line=dict(color="black", dash="dash"),
                )
            )
            fig.update_layout(template="plotly_white")
            fig.update_xaxes(title_text="Predicted")
            fig.update_yaxes(title_text="Actual")
            format_stats_fig(fig, "Multiple regression preview")
            maybe_apply_log_axes(fig, x_values=predicted, y_values=actual, x_allowed=True, y_allowed=True)
            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        if stat_mode == "logistic regression":
            predictor_cols = list(config["stat_logit_predictor_cols"] or [])
            target_col = config["stat_logit_target_col"]
            positive_class = str(config["stat_logit_positive_class"] or "")
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column."))
            if target_col in predictor_cols:
                predictor_cols = [col for col in predictor_cols if col != target_col]
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column different from the target column."))

            analysis = prepare_logistic_analysis(df, target_col, predictor_cols, positive_class)
            predictors = analysis["predictors"]
            logit_df = analysis["logit_df"]
            fit = analysis["fit"]
            diagnostics = analysis["diagnostics"]
            warning = str(analysis["warning"] or "")
            positive_class = str(analysis["positive_class"] or positive_class)
            if fit is None or logit_df.empty or predictors.empty:
                return ui.div(ui.p("Choose a binary target with valid predictor rows."))

            plot_df = pd.DataFrame(
                {
                    "actual": np.where(logit_df["_y"].to_numpy(dtype=float) >= 0.5, positive_class, f"not {positive_class}"),
                    "probability": fit["probabilities"],
                }
            )
            plot_df = attach_cluster_labels(plot_df, logit_df[predictors.columns])
            box_kwargs = dict(
                data_frame=plot_df,
                x="actual",
                y="probability",
                points="all",
                title="Logistic regression: predicted probability by class",
            )
            if "_cluster" in plot_df.columns:
                box_kwargs["color"] = "_cluster"
            else:
                box_kwargs["color"] = "actual"
            fig = px.box(**box_kwargs)
            fig.add_hline(y=0.5, line_dash="dash", line_color="black")
            fig.update_layout(template="plotly_white")
            fig.update_yaxes(range=[0, 1], title_text=f"Probability of {positive_class}")
            if warning:
                fig.add_annotation(
                    text=warning,
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=1.12,
                    showarrow=False,
                    align="left",
                    font=dict(size=11, color="#6c757d"),
                )
            format_stats_fig(fig, "Logistic regression preview")
            maybe_apply_log_axes(fig, y_values=plot_df["probability"], y_allowed=True)
            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        if stat_mode == "PCA":
            pca_cols = list(config["stat_pca_cols"] or [])
            if not pca_cols:
                return ui.div(ui.p("Choose at least one numeric or datetime column for PCA."))
            pca_df = stat_numeric_frame(df, pca_cols).dropna()
            if pca_df.empty or len(pca_df) < 2:
                return ui.div(ui.p("Choose PCA columns with at least two valid rows."))

            suggestion = suggest_pca_components_by_elbow(
                pca_df.to_numpy(dtype=float),
                standardize=bool(config["stat_pca_standardize"]),
            )
            suggested_components = int(suggestion["suggested_components"])
            requested_components = int(config["stat_pca_components"] or suggested_components)
            requested_components = max(1, min(requested_components, pca_df.shape[1]))
            fit = fit_pca(
                pca_df.to_numpy(dtype=float),
                pca_cols,
                n_components=requested_components,
                standardize=bool(config["stat_pca_standardize"]),
            )
            scores = fit["scores"]
            if scores.shape[1] >= 2:
                scores = attach_cluster_labels(scores, pca_df[pca_cols])
                scatter_kwargs = dict(
                    data_frame=scores,
                    x="PC1",
                    y="PC2",
                    opacity=0.8,
                    title="PCA scores: PC1 vs PC2",
                )
                if "_cluster" in scores.columns:
                    scatter_kwargs["color"] = "_cluster"
                fig = px.scatter(**scatter_kwargs)
                fig.update_layout(template="plotly_white")
                fig.update_xaxes(title_text="PC1")
                fig.update_yaxes(title_text="PC2")
                fig.add_annotation(
                    text=f"Suggested components: {suggested_components}",
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=1.08,
                    showarrow=False,
                    align="right",
                    font=dict(size=11, color="#6c757d"),
                )
                maybe_apply_log_axes(fig, x_values=scores["PC1"], y_values=scores["PC2"], x_allowed=True, y_allowed=True)
            else:
                fig = px.bar(
                    fit["explained_df"].head(requested_components),
                    x="Component",
                    y="Explained variance",
                    title="PCA explained variance",
                )
                fig.update_layout(template="plotly_white")
                fig.update_yaxes(title_text="Explained variance")
                maybe_apply_log_axes(fig, y_values=fit["explained_df"].head(requested_components)["Explained variance"], y_allowed=True)
            format_stats_fig(fig, "PCA preview")
            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        if stat_mode in {"t-test", "ANOVA"}:
            if stat_mode == "t-test":
                group_series = selected_series(stat_group_col).astype(str)
                value_series = selected_numeric(stat_value_col)
                plot_df = pd.DataFrame({"group": group_series, "value": value_series}).dropna()
                plot_df = plot_df[plot_df["group"].isin([str(stat_group_a), str(stat_group_b)])]
                if plot_df.empty:
                    return ui.div(ui.p("Choose two populated groups and a numeric value column."))
                plot_df = attach_cluster_labels(plot_df, plot_df[["value"]])
                box_kwargs = dict(
                    data_frame=plot_df,
                    x="group",
                    y="value",
                    points="all",
                    title="T-test data preview",
                )
                if "_cluster" in plot_df.columns:
                    box_kwargs["color"] = "_cluster"
                else:
                    box_kwargs["color"] = "group"
                fig = px.box(**box_kwargs)
            else:
                group_series = selected_series(stat_anova_col).astype(str)
                value_series = selected_numeric(stat_anova_value_col)
                plot_df = pd.DataFrame({"group": group_series, "value": value_series}).dropna()
                if plot_df.empty:
                    return ui.div(ui.p("Choose a valid ANOVA group column and numeric value column."))
                plot_df = attach_cluster_labels(plot_df, plot_df[["value"]])
                box_kwargs = dict(
                    data_frame=plot_df,
                    x="group",
                    y="value",
                    points="all",
                    title="ANOVA data preview",
                )
                if "_cluster" in plot_df.columns:
                    box_kwargs["color"] = "_cluster"
                else:
                    box_kwargs["color"] = "group"
            fig = px.box(**box_kwargs)
            format_stats_fig(fig, fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Statistics Plot")
            maybe_apply_log_axes(fig, y_values=plot_df["value"], y_allowed=True)
            cache_plotly(stats_plot_cache, fig, "statistics_plot")
            return ui.card(
                ui.div("Statistics Plot", class_="stat-label"),
                plotly_to_html(fig),
                class_="stat-box",
            )

        return ui.div(ui.p("Select a statistical test to see the preview plot."))

    @render.download(filename=lambda: plot_download_filename(main_plot_cache.get(), "current_plot"))
    def download_main_plot():
        cache = main_plot_cache.get()
        if not cache:
            yield "No plot available."
            return
        content = cache.get("content")
        if cache.get("media_type") == "text/html":
            yield content if isinstance(content, str) else str(content)
        else:
            yield content if isinstance(content, (bytes, bytearray)) else bytes(content)

    @render.download(filename=lambda: plot_download_filename(stats_plot_cache.get(), "statistics_plot"))
    def download_stats_plot():
        cache = stats_plot_cache.get()
        if not cache:
            yield "No statistics plot available."
            return
        content = cache.get("content")
        if cache.get("media_type") == "text/html":
            yield content if isinstance(content, str) else str(content)
        else:
            yield content if isinstance(content, (bytes, bytearray)) else bytes(content)

    @render.download(
        filename=lambda: f"{sanitize_filename_stem(str(input.preview_dataset_name() or source_name()))}_preview.csv"
    )
    def download_preview_csv():
        yield preview_dataframe().to_csv(index=False)

    @render.download(filename=lambda: str((join_result_cache.get() or {}).get("name") or "joined_table.csv"))
    def download_joined_csv():
        cache = join_result_cache.get()
        if not cache:
            yield "No joined table available."
            return

        joined_df = cache.get("df")
        if not isinstance(joined_df, pd.DataFrame):
            yield "No joined table available."
            return
        yield joined_df.to_csv(index=False)

    @render.download(filename=lambda: f"{sanitize_filename_stem(source_name())}_statistics_results.csv")
    def download_stats_results():
        results = statistics_results_frame()
        if results.empty:
            yield "No statistics results available."
            return
        yield results.to_csv(index=False)

    @reactive.calc
    def statistics_results_frame() -> pd.DataFrame:
        df = data()
        config = applied_stats_config.get()
        if not config.get("ready"):
            return pd.DataFrame()

        stat_mode = config["stat_mode"]
        stat_desc_col = config["stat_desc_col"]
        stat_x_col = config["stat_x_col"]
        stat_y_col = config["stat_y_col"]
        stat_mr_target_col = config["stat_mr_target_col"]
        stat_mr_predictor_cols = list(config["stat_mr_predictor_cols"] or [])
        stat_logit_target_col = config["stat_logit_target_col"]
        stat_logit_positive_class = config["stat_logit_positive_class"]
        stat_logit_predictor_cols = list(config["stat_logit_predictor_cols"] or [])
        stat_pca_cols = list(config["stat_pca_cols"] or [])
        stat_pca_components = int(config["stat_pca_components"] or 2)
        stat_pca_standardize = bool(config["stat_pca_standardize"])
        stat_group_col = config["stat_group_col"]
        stat_value_col = config["stat_value_col"]
        stat_group_a = config["stat_group_a"]
        stat_group_b = config["stat_group_b"]
        stat_anova_col = config["stat_anova_col"]
        stat_anova_value_col = config["stat_anova_value_col"]

        def display_col(col: str) -> str:
            return "Index" if col == INDEX_COL else str(col)

        def selected_series(column: str) -> pd.Series:
            if column == INDEX_COL:
                return resolve_column(df, INDEX_COL)
            if column in df.columns:
                return df[column]
            return pd.Series(dtype="float64")

        def selected_numeric(column: str) -> pd.Series:
            return series_to_stat_numeric(selected_series(column))

        if stat_mode == "descriptive":
            desc_series = selected_series(stat_desc_col)
            desc_numeric = series_to_stat_numeric(desc_series).dropna()
            if desc_numeric.empty:
                return pd.DataFrame()
            q1 = float(np.quantile(desc_numeric, 0.25))
            q3 = float(np.quantile(desc_numeric, 0.75))
            return pd.DataFrame(
                [
                    ("Test", "Descriptive"),
                    ("Column", display_col(stat_desc_col)),
                    ("Count", len(desc_numeric)),
                    ("Missing", int(desc_series.isna().sum())),
                    ("Mean", float(np.mean(desc_numeric))),
                    ("Median", float(np.median(desc_numeric))),
                    ("Std dev", float(np.std(desc_numeric, ddof=1)) if len(desc_numeric) > 1 else float("nan")),
                    ("Min", float(np.min(desc_numeric))),
                    ("Q1", q1),
                    ("Q3", q3),
                    ("Max", float(np.max(desc_numeric))),
                    ("IQR", q3 - q1),
                    ("Variance", float(np.var(desc_numeric, ddof=1)) if len(desc_numeric) > 1 else float("nan")),
                    ("Skew", float(pd.Series(desc_numeric).skew()) if len(desc_numeric) > 2 else float("nan")),
                    ("Kurtosis", float(pd.Series(desc_numeric).kurt()) if len(desc_numeric) > 3 else float("nan")),
                ],
                columns=["Metric", "Value"],
            )

        x_series = selected_numeric(stat_x_col)
        y_series = selected_numeric(stat_y_col)

        if stat_mode == "correlation":
            corr_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if corr_df.empty or corr_df["_x"].nunique(dropna=True) < 2 or corr_df["_y"].nunique(dropna=True) < 2:
                return pd.DataFrame()
            x = corr_df["_x"].to_numpy(dtype=float)
            y = corr_df["_y"].to_numpy(dtype=float)
            r = float(np.corrcoef(x, y)[0, 1])
            covariance = float(np.cov(x, y, ddof=1)[0, 1]) if len(corr_df) > 1 else float("nan")
            return pd.DataFrame(
                [
                    ("Test", "Correlation"),
                    ("X column", display_col(stat_x_col)),
                    ("Y column", display_col(stat_y_col)),
                    ("n", len(corr_df)),
                    ("Pearson r", round(r, 6)),
                    ("R-squared", round(r**2, 6)),
                    ("Covariance", round(covariance, 6)),
                    ("Mean X", round(float(np.mean(x)), 6)),
                ],
                columns=["Metric", "Value"],
            )

        if stat_mode == "regression":
            reg_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if reg_df.empty:
                return pd.DataFrame()
            x = reg_df["_x"].to_numpy(dtype=float)
            y = reg_df["_y"].to_numpy(dtype=float)
            try:
                fit = fit_linear_regression(x, y)
            except ValueError:
                return pd.DataFrame()
            fitted = fit["slope"] * x + fit["intercept"]
            resid = y - fitted
            return pd.DataFrame(
                [
                    ("Test", "Regression"),
                    ("X column", display_col(stat_x_col)),
                    ("Y column", display_col(stat_y_col)),
                    ("n", round(fit["n"], 0)),
                    ("Slope", round(fit["slope"], 6)),
                    ("Intercept", round(fit["intercept"], 6)),
                    ("Pearson r", round(fit["r"], 6)),
                    ("R-squared", round(fit["r2"], 6)),
                    ("RMSE", round(fit["rmse"], 6)),
                    ("MAE", round(fit["mae"], 6)),
                    ("Residual SD", round(fit["resid_sd"], 6)),
                    ("Residual mean", round(float(np.mean(resid)), 6)),
                ],
                columns=["Metric", "Value"],
            )

        if stat_mode == "multiple regression":
            if not stat_mr_predictor_cols:
                return pd.DataFrame()
            if stat_mr_target_col in stat_mr_predictor_cols:
                stat_mr_predictor_cols = [col for col in stat_mr_predictor_cols if col != stat_mr_target_col]
            mr_frame = stat_numeric_frame(df, [stat_mr_target_col] + stat_mr_predictor_cols).dropna()
            if mr_frame.empty or len(mr_frame) < len(stat_mr_predictor_cols) + 2:
                return pd.DataFrame()
            try:
                fit = fit_multiple_linear_regression(
                    mr_frame[stat_mr_predictor_cols].to_numpy(dtype=float),
                    mr_frame[stat_mr_target_col].to_numpy(dtype=float),
                    stat_mr_predictor_cols,
                )
            except ValueError:
                return pd.DataFrame()
            coef_df = fit["coef_table"].copy()
            coef_df.insert(0, "Test", "Multiple regression")
            return coef_df

        if stat_mode == "logistic regression":
            if not stat_logit_predictor_cols:
                return pd.DataFrame()
            if stat_logit_target_col in stat_logit_predictor_cols:
                stat_logit_predictor_cols = [col for col in stat_logit_predictor_cols if col != stat_logit_target_col]
            analysis = prepare_logistic_analysis(df, stat_logit_target_col, stat_logit_predictor_cols, str(stat_logit_positive_class))
            predictors = analysis["predictors"]
            logit_df = analysis["logit_df"]
            fit = analysis["fit"]
            diagnostics = analysis["diagnostics"]
            if fit is None or predictors.empty or logit_df.empty:
                return pd.DataFrame()
            coef_df = fit["coef_table"].copy()
            diag_df = diagnostics.copy()
            diag_df.insert(0, "Test", "Logistic regression")
            coef_df.insert(0, "Test", "Logistic regression")
            conf_df = fit["confusion"].copy()
            conf_df.insert(0, "Test", "Logistic regression")
            return pd.concat([diag_df, coef_df, conf_df], ignore_index=True, sort=False)

        if stat_mode == "PCA":
            if not stat_pca_cols:
                return pd.DataFrame()
            pca_df = stat_numeric_frame(df, stat_pca_cols).dropna()
            if pca_df.empty or len(pca_df) < 2:
                return pd.DataFrame()
            try:
                fit = fit_pca(
                    pca_df.to_numpy(dtype=float),
                    stat_pca_cols,
                    n_components=stat_pca_components,
                    standardize=stat_pca_standardize,
                )
            except ValueError:
                return pd.DataFrame()
            explained_df = fit["explained_df"].copy()
            loading_df = fit["loading_df"].copy()
            scores_df = fit["scores"].head(10).copy()
            explained_df.insert(0, "Test", "PCA")
            loading_df.insert(0, "Test", "PCA")
            scores_df.insert(0, "Test", "PCA")
            return pd.concat([explained_df, loading_df, scores_df], ignore_index=True, sort=False)

        if stat_mode == "t-test":
            group_series = selected_series(stat_group_col).astype(str)
            value_series = selected_numeric(stat_value_col)
            test_df = pd.DataFrame({"_group": group_series, "_value": value_series}).dropna()
            test_df = test_df[test_df["_group"].isin([str(stat_group_a), str(stat_group_b)])]
            if test_df.empty or test_df["_group"].nunique() < 2:
                return pd.DataFrame()
            a = test_df.loc[test_df["_group"] == str(stat_group_a), "_value"].to_numpy(dtype=float)
            b = test_df.loc[test_df["_group"] == str(stat_group_b), "_value"].to_numpy(dtype=float)
            try:
                t_stat = welch_t_stat(a, b)
                p_value = permutation_p_value(a, b)
                d = cohens_d(a, b)
            except ValueError:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    ("Test", "T-test"),
                    ("Group column", display_col(stat_group_col)),
                    ("Value column", display_col(stat_value_col)),
                    ("Groups", f"{stat_group_a} / {stat_group_b}"),
                    ("Group A n", len(a)),
                    ("Group B n", len(b)),
                    ("Mean A", round(float(np.mean(a)), 6)),
                    ("Mean B", round(float(np.mean(b)), 6)),
                    ("Mean diff", round(float(np.mean(a) - np.mean(b)), 6)),
                    ("t statistic", round(float(t_stat), 6)),
                    ("Permutation p-value", round(float(p_value), 6)),
                    ("Cohen's d", round(float(d), 6)),
                ],
                columns=["Metric", "Value"],
            )

        if stat_mode == "ANOVA":
            group_series = selected_series(stat_anova_col).astype(str)
            value_series = selected_numeric(stat_anova_value_col)
            anova_df = pd.DataFrame({"_group": group_series, "_value": value_series}).dropna()
            groups = [g["_value"].to_numpy(dtype=float) for _, g in anova_df.groupby("_group") if len(g) > 0]
            if len(groups) < 2:
                return pd.DataFrame()
            try:
                result = one_way_anova(groups)
                p_perm = permutation_p_value_anova(groups)
            except ValueError:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    ("Test", "ANOVA"),
                    ("Group column", display_col(stat_anova_col)),
                    ("Value column", display_col(stat_anova_value_col)),
                    ("Groups", len(groups)),
                    ("Total observations", len(anova_df)),
                    ("F statistic", round(float(result["f"]), 6)),
                    ("Permutation p-value", round(float(p_perm), 6)),
                ],
                columns=["Metric", "Value"],
            )

        if stat_mode == "heteroscedasticity":
            het_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if het_df.empty:
                return pd.DataFrame()
            try:
                het = heteroscedasticity_diagnostics(het_df["_x"].to_numpy(dtype=float), het_df["_y"].to_numpy(dtype=float))
            except ValueError:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    ("Test", "Heteroscedasticity"),
                    ("X column", display_col(stat_x_col)),
                    ("Y column", display_col(stat_y_col)),
                    ("Abs resid corr", round(float(het["abs_resid_corr"]), 6)),
                    ("Variance ratio", round(float(het["variance_ratio_high_low"]), 6)),
                    ("Permutation p-value", round(float(het["perm_p_abs_resid"]), 6)),
                ],
                columns=["Metric", "Value"],
            )

        return pd.DataFrame()

    @render.ui
    def statistics_panel():
        df = data()
        config = applied_stats_config.get()

        if not config.get("ready"):
            return ui.div(
                ui.p("Choose a test and column settings, then click Run statistics."),
                ui.p("Results are shown here as text boxes only."),
            )

        stat_mode = config["stat_mode"]
        stat_desc_col = config["stat_desc_col"]
        stat_x_col = config["stat_x_col"]
        stat_y_col = config["stat_y_col"]
        stat_group_col = config["stat_group_col"]
        stat_value_col = config["stat_value_col"]
        stat_group_a = config["stat_group_a"]
        stat_group_b = config["stat_group_b"]
        stat_anova_col = config["stat_anova_col"]
        stat_anova_value_col = config["stat_anova_value_col"]

        def display_col(col: str) -> str:
            return "Index" if col == INDEX_COL else str(col)

        def stat_box(label: str, value: object) -> ui.Tag:
            return ui.card(
                ui.div(label, class_="stat-label"),
                ui.div("NA" if pd.isna(value) else str(value), class_="stat-value"),
                class_="stat-box",
            )

        def stat_row(items: list[tuple[str, object]]) -> ui.Tag:
            return ui.layout_columns(*[stat_box(label, value) for label, value in items])

        def table_block(title: str, frame: pd.DataFrame) -> ui.Tag:
            return ui.card(
                ui.div(title, class_="stat-label"),
                ui.HTML(frame.to_html(index=False, border=0, classes="table table-sm table-striped mb-0")),
                class_="stat-box",
            )

        def selected_series(column: str) -> pd.Series:
            if column == INDEX_COL:
                return resolve_column(df, INDEX_COL)
            if column in df.columns:
                return df[column]
            return pd.Series(dtype="float64")

        def selected_numeric(column: str) -> pd.Series:
            return series_to_stat_numeric(selected_series(column))

        if stat_mode == "descriptive":
            desc_series = selected_series(stat_desc_col)
            desc_numeric = series_to_stat_numeric(desc_series).dropna()
            if desc_numeric.empty:
                return ui.div(ui.p("Choose a numeric or datetime column with at least one valid value."))

            q1 = float(np.quantile(desc_numeric, 0.25))
            q3 = float(np.quantile(desc_numeric, 0.75))
            summary_df = pd.DataFrame(
                [
                    ("Count", len(desc_numeric)),
                    ("Missing", int(desc_series.isna().sum())),
                    ("Mean", float(np.mean(desc_numeric))),
                    ("Median", float(np.median(desc_numeric))),
                    ("Std dev", float(np.std(desc_numeric, ddof=1)) if len(desc_numeric) > 1 else float("nan")),
                    ("Min", float(np.min(desc_numeric))),
                    ("Q1", q1),
                    ("Q3", q3),
                    ("Max", float(np.max(desc_numeric))),
                    ("IQR", q3 - q1),
                    ("Variance", float(np.var(desc_numeric, ddof=1)) if len(desc_numeric) > 1 else float("nan")),
                    ("Skew", float(pd.Series(desc_numeric).skew()) if len(desc_numeric) > 2 else float("nan")),
                    ("Kurtosis", float(pd.Series(desc_numeric).kurt()) if len(desc_numeric) > 3 else float("nan")),
                ],
                columns=["Metric", "Value"],
            )
            return ui.div(
                stat_row(
                    [
                        ("Test", "Descriptive"),
                        ("Column", display_col(stat_desc_col)),
                        ("Valid values", len(desc_numeric)),
                        ("Missing", int(desc_series.isna().sum())),
                    ]
                ),
                ui.br(),
                table_block("Descriptive statistics", summary_df),
            )

        x_series = selected_numeric(stat_x_col)
        y_series = selected_numeric(stat_y_col)

        if stat_mode == "correlation":
            corr_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if corr_df.empty or corr_df["_x"].nunique(dropna=True) < 2 or corr_df["_y"].nunique(dropna=True) < 2:
                return ui.div(ui.p("Choose numeric or datetime X and Y columns with at least two valid rows."))

            x = corr_df["_x"].to_numpy(dtype=float)
            y = corr_df["_y"].to_numpy(dtype=float)
            r = float(np.corrcoef(x, y)[0, 1])
            covariance = float(np.cov(x, y, ddof=1)[0, 1]) if len(corr_df) > 1 else float("nan")
            return ui.div(
                stat_row(
                    [
                        ("Test", "Correlation"),
                        ("X column", display_col(stat_x_col)),
                        ("Y column", display_col(stat_y_col)),
                        ("n", len(corr_df)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Pearson r", round(r, 6)),
                        ("R-squared", round(r**2, 6)),
                        ("Covariance", round(covariance, 6)),
                        ("Mean X", round(float(np.mean(x)), 6)),
                    ]
                ),
            )

        if stat_mode == "regression":
            reg_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if reg_df.empty:
                return ui.div(ui.p("Select numeric or datetime X and Y columns with at least two valid rows."))

            x = reg_df["_x"].to_numpy(dtype=float)
            y = reg_df["_y"].to_numpy(dtype=float)
            try:
                stats = fit_linear_regression(x, y)
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            fitted = stats["slope"] * x + stats["intercept"]
            resid = y - fitted
            return ui.div(
                stat_row(
                    [
                        ("Test", "Regression"),
                        ("X column", display_col(stat_x_col)),
                        ("Y column", display_col(stat_y_col)),
                        ("n", round(stats["n"], 0)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Slope", round(stats["slope"], 6)),
                        ("Intercept", round(stats["intercept"], 6)),
                        ("Pearson r", round(stats["r"], 6)),
                        ("R-squared", round(stats["r2"], 6)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("RMSE", round(stats["rmse"], 6)),
                        ("MAE", round(stats["mae"], 6)),
                        ("Residual SD", round(stats["resid_sd"], 6)),
                        ("Residual mean", round(float(np.mean(resid)), 6)),
                    ]
                ),
            )

        if stat_mode == "multiple regression":
            predictor_cols = list(config["stat_mr_predictor_cols"] or [])
            target_col = config["stat_mr_target_col"]
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column."))
            if target_col in predictor_cols:
                predictor_cols = [col for col in predictor_cols if col != target_col]
            mr_frame = stat_numeric_frame(df, [target_col] + predictor_cols).dropna()
            if mr_frame.empty or len(mr_frame) < len(predictor_cols) + 2:
                return ui.div(ui.p("Choose numeric or datetime target and predictor columns with enough valid rows."))

            try:
                fit = fit_multiple_linear_regression(
                    mr_frame[predictor_cols].to_numpy(dtype=float),
                    mr_frame[target_col].to_numpy(dtype=float),
                    predictor_cols,
                )
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            coef_df = fit["coef_table"].copy()
            return ui.div(
                stat_row(
                    [
                        ("Test", "Multiple regression"),
                        ("Target column", display_col(target_col)),
                        ("Predictors", ", ".join(display_col(col) for col in predictor_cols)),
                        ("Rows used", fit["n"]),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("R-squared", round(float(fit["r2"]), 6)),
                        ("Adj. R-squared", round(float(fit["adj_r2"]), 6)),
                        ("F statistic", round(float(fit["f_stat"]), 6)),
                        ("RMSE", round(float(fit["rmse"]), 6)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Residual SD", round(float(fit["resid_sd"]), 6)),
                        ("Predictors", len(predictor_cols)),
                        ("Intercept", round(float(coef_df.loc[0, "Coefficient"]), 6)),
                        ("Residual mean", round(float(np.mean(fit["residuals"])), 6)),
                    ]
                ),
                ui.br(),
                table_block("Regression coefficients", coef_df.round(6)),
            )

        if stat_mode == "logistic regression":
            predictor_cols = list(config["stat_logit_predictor_cols"] or [])
            target_col = config["stat_logit_target_col"]
            positive_class = str(config["stat_logit_positive_class"] or "")
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column."))
            if target_col in predictor_cols:
                predictor_cols = [col for col in predictor_cols if col != target_col]
            if not predictor_cols:
                return ui.div(ui.p("Choose at least one predictor column different from the target column."))

            analysis = prepare_logistic_analysis(df, target_col, predictor_cols, positive_class)
            predictors = analysis["predictors"]
            logit_df = analysis["logit_df"]
            fit = analysis["fit"]
            diagnostics = analysis["diagnostics"]
            warning = str(analysis["warning"] or "")
            positive_class = str(analysis["positive_class"] or positive_class)
            if fit is None or predictors.empty or logit_df.empty:
                return ui.div(ui.p("Choose a binary target with valid predictor rows."))

            coef_df = fit["coef_table"].copy()
            conf_df = fit["confusion"].copy()
            diag_df = diagnostics.copy()
            diag_df.insert(0, "Test", "Logistic regression")
            return ui.div(
                stat_row(
                    [
                        ("Test", "Logistic regression"),
                        ("Target column", display_col(target_col)),
                        ("Positive class", positive_class),
                        ("Predictors", ", ".join(display_col(col) for col in predictor_cols)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Accuracy", round(float(fit["accuracy"]), 6)),
                        ("McFadden R²", round(float(fit["mcfadden_r2"]), 6)),
                        ("Log-likelihood", round(float(fit["loglik"]), 6)),
                        ("Null log-likelihood", round(float(fit["null_loglik"]), 6)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Iterations", fit["iterations"]),
                        ("Rows used", fit["n"]),
                        ("Predictors", fit["p"]),
                        ("Predicted positives", int(np.sum(fit["predicted"] >= 0.5))),
                    ]
                ),
                ui.br(),
                ui.div(warning or "No strong separation signal detected.", class_="small-muted"),
                ui.br(),
                table_block("Logistic diagnostics", diag_df),
                ui.br(),
                table_block("Logistic coefficients", coef_df.round(6)),
                ui.br(),
                table_block("Confusion matrix", conf_df.reset_index().rename(columns={"index": "Actual"})),
            )

        if stat_mode == "PCA":
            pca_cols = list(config["stat_pca_cols"] or [])
            if not pca_cols:
                return ui.div(ui.p("Choose at least one numeric or datetime column for PCA."))
            pca_df = stat_numeric_frame(df, pca_cols).dropna()
            if pca_df.empty or len(pca_df) < 2:
                return ui.div(ui.p("Choose PCA columns with at least two valid rows."))

            try:
                fit = fit_pca(
                    pca_df.to_numpy(dtype=float),
                    pca_cols,
                    n_components=int(config["stat_pca_components"] or 2),
                    standardize=bool(config["stat_pca_standardize"]),
                )
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            explained_df = fit["explained_df"].head(int(config["stat_pca_components"] or 2) if int(config["stat_pca_components"] or 2) > 0 else 2).copy()
            loading_df = fit["loading_df"].copy()
            scores_df = fit["scores"].copy()
            return ui.div(
                stat_row(
                    [
                        ("Test", "PCA"),
                        ("Columns", ", ".join(display_col(col) for col in pca_cols)),
                        ("Components", fit["n_components"]),
                        ("Rows used", fit["n"]),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Standardized", "Yes" if config["stat_pca_standardize"] else "No"),
                        ("Features", fit["p"]),
                        ("PC1 variance", round(float(explained_df.iloc[0]["Explained variance"]), 6) if not explained_df.empty else float("nan")),
                        ("Cumulative", round(float(explained_df.iloc[min(len(explained_df) - 1, fit["n_components"] - 1)]["Cumulative"]), 6) if not explained_df.empty else float("nan")),
                    ]
                ),
                ui.br(),
                table_block("Explained variance", explained_df.round(6)),
                ui.br(),
                table_block("Loadings", loading_df.round(6)),
                ui.br(),
                table_block("PCA scores preview", scores_df.head(10).round(6)),
            )

        if stat_mode == "t-test":
            group_series = selected_series(stat_group_col).astype(str)
            value_series = selected_numeric(stat_value_col)
            test_df = pd.DataFrame({"_group": group_series, "_value": value_series}).dropna()
            test_df = test_df[test_df["_group"].isin([str(stat_group_a), str(stat_group_b)])]
            if test_df.empty or test_df["_group"].nunique() < 2:
                return ui.div(ui.p("Choose a group column with at least two populated groups and a numeric value column."))

            a = test_df.loc[test_df["_group"] == str(stat_group_a), "_value"].to_numpy(dtype=float)
            b = test_df.loc[test_df["_group"] == str(stat_group_b), "_value"].to_numpy(dtype=float)
            try:
                t_stat = welch_t_stat(a, b)
                p_value = permutation_p_value(a, b)
                d = cohens_d(a, b)
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            return ui.div(
                stat_row(
                    [
                        ("Test", "T-test"),
                        ("Group column", display_col(stat_group_col)),
                        ("Value column", display_col(stat_value_col)),
                        ("Groups", f"{stat_group_a} / {stat_group_b}"),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Mean A", round(float(np.mean(a)), 6)),
                        ("Mean B", round(float(np.mean(b)), 6)),
                        ("Mean diff", round(float(np.mean(a) - np.mean(b)), 6)),
                        ("t stat", round(float(t_stat), 6)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("p value", round(float(p_value), 6)),
                        ("Cohen's d", round(float(d), 6)),
                        ("Group A n", len(a)),
                        ("Group B n", len(b)),
                    ]
                ),
            )

        if stat_mode == "ANOVA":
            group_series = selected_series(stat_anova_col).astype(str)
            value_series = selected_numeric(stat_anova_value_col)
            anova_df = pd.DataFrame({"_group": group_series, "_value": value_series}).dropna()
            groups = [g["_value"].to_numpy(dtype=float) for _, g in anova_df.groupby("_group") if len(g) > 0]
            if len(groups) < 2:
                return ui.div(ui.p("ANOVA needs at least two groups with numeric values."))
            try:
                result = one_way_anova(groups)
                p_value = permutation_f_p_value(groups)
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            return ui.div(
                stat_row(
                    [
                        ("Test", "ANOVA"),
                        ("Group column", display_col(stat_anova_col)),
                        ("Value column", display_col(stat_anova_value_col)),
                        ("Groups", len(groups)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("F statistic", round(float(result["f"]), 6)),
                        ("p - Value", round(float(p_value), 6)),
                        ("Eta squared", round(float(result["eta_sq"]), 6)),
                        ("df between", round(float(result["df_between"]), 0)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("df within", round(float(result["df_within"]), 0)),
                        ("SS between", round(float(result["ss_between"]), 6)),
                        ("SS within", round(float(result["ss_within"]), 6)),
                        ("Total observations", len(anova_df)),
                    ]
                ),
            )

        if stat_mode == "heteroscedasticity":
            het_df = pd.DataFrame({"_x": x_series, "_y": y_series}).dropna()
            if het_df.empty:
                return ui.div(ui.p("Select numeric or datetime X and Y columns with at least two valid rows."))

            x = het_df["_x"].to_numpy(dtype=float)
            y = het_df["_y"].to_numpy(dtype=float)
            try:
                diagnostics = heteroscedasticity_diagnostics(x, y)
            except ValueError as exc:
                return ui.div(ui.p(str(exc)))

            return ui.div(
                stat_row(
                    [
                        ("Test", "Heteroscedasticity"),
                        ("X column", display_col(stat_x_col)),
                        ("Y column", display_col(stat_y_col)),
                        ("n", len(het_df)),
                    ]
                ),
                ui.br(),
                stat_row(
                    [
                        ("Abs(resid) corr", round(float(diagnostics["abs_resid_corr"]), 6)),
                        ("Variance ratio", round(float(diagnostics["variance_ratio_high_low"]), 6)),
                        ("Permutation p", round(float(diagnostics["perm_p_abs_resid"]), 6)),
                        ("Residual SD", round(float(np.std(y - (np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1]), ddof=1)), 6)),
                    ]
                ),
            )

        return ui.div(ui.p("Select a statistical test to see results."))

    @render.ui
    def plot():
        df = data().copy()
        viz_engine = input.viz_engine()
        plot_kind = input.plot_kind()
        heatmap_show_values = bool(input.heatmap_show_values())
        x_col = input.x_col()
        y_col = input.y_col()
        x_log = bool(input.x_log())
        y_log = bool(input.y_log())
        hue_col = input.hue_col()
        hue_col = None if hue_col == "None" else hue_col
        twin_y_requested = bool(input.use_twin_y_axis())
        y2_col = input.y2_col()
        y2_log = bool(input.y2_log())
        y2_axis_title_override = input.y2_axis_title().strip()
        cols = df.columns.tolist()
        available_cols = cols + [INDEX_COL]

        if not cols:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            ax.text(0.5, 0.5, "No columns available", ha="center", va="center")
            ax.axis("off")
            return matplotlib_to_html(fig)

        if x_col not in available_cols:
            datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
            x_col = datetime_cols[0] if datetime_cols else cols[0]

        if y_col not in available_cols:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            fallback_y = next((c for c in numeric_cols if c != x_col), None)
            if fallback_y is None:
                fallback_y = next((c for c in cols if c != x_col), x_col)
            y_col = fallback_y

        if hue_col is not None and hue_col not in available_cols:
            hue_col = None

        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.axis("off")
            return matplotlib_to_html(fig)

        def mpl_message(message: str) -> ui.Tag:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
            ax.axis("off")
            return matplotlib_to_html(fig)

        def plotly_message(message: str) -> ui.Tag:
            fig = px.scatter(title=message)
            fig.update_layout(
                template="plotly_white",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
            return plotly_to_html(fig)

        if twin_y_requested:
            twin_y_allowed = plot_kind in {"scatter", "line"}
            twin_y_valid = y2_col in available_cols and y2_col != y_col
            if not twin_y_allowed:
                message = "Twin y axis is currently supported for scatter and line charts."
                return plotly_message(message) if viz_engine == "plotly" else mpl_message(message)
            if not twin_y_valid:
                message = "Choose a different secondary Y column to enable the twin y axis."
                return plotly_message(message) if viz_engine == "plotly" else mpl_message(message)

        def axis_payload(series: Optional[pd.Series], enabled: bool) -> tuple[Optional[pd.Series], str]:
            if not enabled or series is None:
                return series, "linear"
            if pd.api.types.is_datetime64_any_dtype(series):
                return series, "linear"
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()
            if valid.empty:
                return series, "linear"
            if (valid > 0).all():
                return numeric, "log"
            transformed = np.sign(numeric) * np.log10(np.abs(numeric) + 1.0)
            return transformed, "signed_log"

        def axis_title(base: str, mode: str) -> str:
            if mode == "log":
                return f"{base} (log)"
            if mode == "signed_log":
                return f"{base} (log-like)"
            return base

        def apply_plotly_axis_scale(fig, x_mode: str = "linear", y_mode: str = "linear") -> None:
            if x_mode == "log":
                fig.update_xaxes(type="log")
            if y_mode == "log":
                fig.update_yaxes(type="log")

        def apply_mpl_axis_scale(ax, x_mode: str = "linear", y_mode: str = "linear") -> None:
            if x_mode == "log":
                ax.set_xscale("log")
            if y_mode == "log":
                ax.set_yscale("log")

        def safe_int(value, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        def safe_float(value, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def display_col(col: str) -> str:
            return "Index" if col == INDEX_COL else str(col)

        plot_title_override = input.plot_title().strip()
        x_title_override = input.x_axis_title().strip()
        y_title_override = input.y_axis_title().strip()
        plot_title_font_size = max(8, safe_int(input.plot_title_font_size(), 18))
        axis_title_font_size = max(8, safe_int(input.axis_title_font_size(), 14))
        x_tick_angle = safe_int(input.x_tick_angle(), 0)
        max_x_ticks = max(2, safe_int(input.max_x_ticks(), 10))
        grid_axis = input.grid_axis()
        grid_alpha = min(1.0, max(0.0, safe_float(input.grid_alpha(), 0.35)))
        grid_linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}.get(input.grid_style(), "--")

        def resolve_text(override: str, fallback: str) -> str:
            return override.strip() if override and override.strip() else fallback

        def grid_enabled(axis: str) -> bool:
            return grid_axis == "both" or grid_axis == axis

        def apply_plotly_formatting(fig, default_title: str, x_title: Optional[str] = None, y_title: Optional[str] = None) -> None:
            fig.update_layout(
                template="plotly_white",
                title_text=resolve_text(plot_title_override, default_title),
                title_font=dict(size=plot_title_font_size),
                height=PLOTLY_FIG_HEIGHT,
                margin=dict(l=55, r=25, t=70, b=90),
                autosize=True,
            )
            if x_title is not None:
                fig.update_xaxes(
                    title_text=resolve_text(x_title_override, x_title),
                    tickangle=x_tick_angle,
                    nticks=max_x_ticks,
                    showgrid=grid_enabled("x"),
                    gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                    zeroline=False,
                    title_font=dict(size=axis_title_font_size),
                    automargin=True,
                )
            if y_title is not None:
                fig.update_yaxes(
                    title_text=resolve_text(y_title_override, y_title),
                    showgrid=grid_enabled("y"),
                    gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                    zeroline=False,
                    title_font=dict(size=axis_title_font_size),
                    automargin=True,
                )

        def apply_mpl_formatting(ax, default_title: str, x_title: Optional[str] = None, y_title: Optional[str] = None, x_locator_series: Optional[pd.Series] = None) -> None:
            ax.set_title(resolve_text(plot_title_override, default_title), fontsize=plot_title_font_size)
            if x_title is not None:
                ax.set_xlabel(resolve_text(x_title_override, x_title), fontsize=axis_title_font_size)
            if y_title is not None:
                ax.set_ylabel(resolve_text(y_title_override, y_title), fontsize=axis_title_font_size)
            ax.tick_params(axis="x", rotation=x_tick_angle)
            if grid_axis == "none":
                ax.grid(False)
            else:
                ax.grid(True, axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha)
            if x_locator_series is not None and pd.api.types.is_numeric_dtype(x_locator_series):
                try:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks))
                except Exception:
                    pass

        def normalize_choice(value: object) -> str:
            return str(value).strip().lower()

        def normalize_named_color(value: object, fallback: str) -> str:
            allowed = {"blue", "orange", "green", "red", "purple", "teal", "black"}
            color = str(value).strip().lower()
            return color if color in allowed else fallback

        line_style_choice = normalize_choice(input.line_style()) if plot_kind == "line" else "solid"
        line_width = max(0.5, safe_float(input.line_width(), 2.5)) if plot_kind == "line" else 2.5
        line_marker_choice = normalize_choice(input.line_marker()) if plot_kind == "line" else "circle"
        line_show_markers = bool(input.line_show_markers()) if plot_kind == "line" else True
        twin_primary_line_style_choice = normalize_choice(input.twin_y1_line_style()) if (plot_kind == "line" and twin_y_requested) else line_style_choice
        twin_primary_line_color_choice = normalize_named_color(input.twin_y1_line_color(), "blue") if (plot_kind == "line" and twin_y_requested) else "blue"
        twin_secondary_line_style_choice = normalize_choice(input.twin_y2_line_style()) if (plot_kind == "line" and twin_y_requested) else "dash"
        twin_secondary_line_color_choice = normalize_named_color(input.twin_y2_line_color(), "orange") if (plot_kind == "line" and twin_y_requested) else "orange"

        plotly_line_style_map = {
            "solid": "solid",
            "dash": "dash",
            "dot": "dot",
            "dashdot": "dashdot",
            "dashed": "dash",
            "dotted": "dot",
            "dash-dot": "dashdot",
        }
        mpl_line_style_map = {
            "solid": "-",
            "dash": "--",
            "dot": ":",
            "dashdot": "-.",
            "dashed": "--",
            "dotted": ":",
            "dash-dot": "-.",
        }
        plotly_marker_map = {
            "none": None,
            "circle": "circle",
            "square": "square",
            "diamond": "diamond",
            "triangle-up": "triangle-up",
            "x": "x",
            "triangle": "triangle-up",
        }
        mpl_marker_map = {
            "none": None,
            "circle": "o",
            "square": "s",
            "diamond": "D",
            "triangle-up": "^",
            "x": "x",
            "triangle": "^",
        }
        selected_plotly_line_style = plotly_line_style_map.get(line_style_choice, "solid")
        selected_mpl_line_style = mpl_line_style_map.get(line_style_choice, "-")
        selected_plotly_marker = plotly_marker_map.get(line_marker_choice, "circle")
        selected_mpl_marker = mpl_marker_map.get(line_marker_choice, "o")
        twin_primary_plotly_line_style = plotly_line_style_map.get(twin_primary_line_style_choice, selected_plotly_line_style)
        twin_primary_mpl_line_style = mpl_line_style_map.get(twin_primary_line_style_choice, selected_mpl_line_style)
        twin_secondary_plotly_line_style = plotly_line_style_map.get(twin_secondary_line_style_choice, "dash")
        twin_secondary_mpl_line_style = mpl_line_style_map.get(twin_secondary_line_style_choice, "--")

        plot_df = add_index_column(df)
        plot_df["_plot_x"], x_mode = axis_payload(plot_df[x_col], x_log)
        plot_df["_plot_y"], y_mode = axis_payload(plot_df[y_col], y_log)
        x_axis_label = axis_title(x_col, x_mode)
        y_axis_label = axis_title(y_col, y_mode)
        if twin_y_requested:
            plot_df["_plot_y2"], y2_mode = axis_payload(plot_df[y2_col], y2_log)
            y2_axis_label = resolve_text(y2_axis_title_override, axis_title(y2_col, y2_mode))
        else:
            y2_mode = "linear"
            y2_axis_label = ""

        if plot_kind == "scatter":
            if twin_y_requested:
                scatter_df = plot_df[["_plot_x", "_plot_y", "_plot_y2"]].dropna().copy()
                if viz_engine == "plotly":
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_df["_plot_x"],
                            y=scatter_df["_plot_y"],
                            mode="markers",
                            name=y_axis_label,
                            marker=dict(size=8, opacity=0.75),
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_df["_plot_x"],
                            y=scatter_df["_plot_y2"],
                            mode="markers",
                            name=y2_axis_label,
                            marker=dict(size=8, opacity=0.75, symbol="diamond"),
                        ),
                        secondary_y=True,
                    )
                    fig.update_layout(
                        template="plotly_white",
                        title_text=resolve_text(plot_title_override, "Scatter plot"),
                        title_font=dict(size=plot_title_font_size),
                        legend_title_text="",
                        height=PLOTLY_FIG_HEIGHT,
                        margin=dict(l=55, r=25, t=70, b=90),
                        autosize=True,
                    )
                    fig.update_xaxes(
                        title_text=resolve_text(x_title_override, x_axis_label),
                        tickangle=x_tick_angle,
                        nticks=max_x_ticks,
                        showgrid=grid_enabled("x"),
                        gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                        zeroline=False,
                        automargin=True,
                    )
                    fig.update_yaxes(
                        title_text=resolve_text(y_title_override, y_axis_label),
                        secondary_y=False,
                        showgrid=grid_enabled("y"),
                        gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                        zeroline=False,
                        title_font=dict(size=axis_title_font_size, color=twin_primary_line_color_choice),
                        type="log" if y_mode == "log" else None,
                    )
                    fig.update_yaxes(
                        title_text=y2_axis_label,
                        secondary_y=True,
                        showgrid=False,
                        zeroline=False,
                        title_font=dict(size=axis_title_font_size, color=twin_secondary_line_color_choice),
                        type="log" if y2_mode == "log" else None,
                    )
                    cache_plotly(main_plot_cache, fig, "current_plot")
                    return plotly_to_html(fig)
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                ax2 = ax.twinx()
                ax.scatter(scatter_df["_plot_x"], scatter_df["_plot_y"], alpha=0.75, s=35, label=y_axis_label, color="tab:blue")
                ax2.scatter(scatter_df["_plot_x"], scatter_df["_plot_y2"], alpha=0.75, s=35, label=y2_axis_label, color="tab:orange", marker="D")
                ax.set_title("Scatter plot")
                ax.set_xlabel(x_axis_label)
                ax.set_ylabel(y_axis_label, fontsize=axis_title_font_size, color=twin_primary_line_color_choice)
                ax2.set_ylabel(y2_axis_label, fontsize=axis_title_font_size, color=twin_secondary_line_color_choice)
                apply_mpl_axis_scale(ax, x_mode, y_mode)
                if y2_mode == "log":
                    ax2.set_yscale("log")
                ax.tick_params(axis="x", rotation=x_tick_angle)
                if grid_axis == "none":
                    ax.grid(False)
                else:
                    ax.grid(True, axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha)
                if pd.api.types.is_numeric_dtype(scatter_df["_plot_x"]):
                    try:
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks))
                    except Exception:
                        pass
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                if handles1 or handles2:
                    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
                apply_mpl_formatting(ax, "Scatter plot", x_axis_label, y_axis_label, scatter_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            if viz_engine == "plotly":
                fig = px.scatter(
                    plot_df,
                    x="_plot_x",
                    y="_plot_y",
                    color=hue_col,
                    opacity=0.75,
                    title="Scatter plot",
                )
                fig.update_layout(template="plotly_white", legend_title_text=hue_col or "")
                fig.update_xaxes(title_text=x_axis_label)
                fig.update_yaxes(title_text=y_axis_label)
                apply_plotly_axis_scale(fig, x_mode, y_mode)
                apply_plotly_formatting(fig, "Scatter plot", x_axis_label, y_axis_label)
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                sns.scatterplot(
                    data=plot_df,
                    x="_plot_x",
                    y="_plot_y",
                    hue=hue_col,
                    ax=ax,
                    alpha=0.75,
                    s=35,
                )
                ax.set_title("Scatter plot")
                ax.set_xlabel(x_axis_label)
                ax.set_ylabel(y_axis_label)
                apply_mpl_axis_scale(ax, x_mode, y_mode)
                apply_mpl_formatting(ax, "Scatter plot", x_axis_label, y_axis_label, plot_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            if hue_col:
                for group, group_df in plot_df.groupby(hue_col):
                    ax.scatter(group_df["_plot_x"], group_df["_plot_y"], alpha=0.75, s=35, label=str(group))
                ax.legend(title=hue_col)
            else:
                ax.scatter(plot_df["_plot_x"], plot_df["_plot_y"], alpha=0.75, s=35)
            ax.set_title("Scatter plot")
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            apply_mpl_axis_scale(ax, x_mode, y_mode)
            apply_mpl_formatting(ax, "Scatter plot", x_axis_label, y_axis_label, plot_df["_plot_x"])
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "line":
            line_df = plot_df[[x_col, y_col, y2_col] if twin_y_requested else [x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
            line_df["_plot_x"], line_x_mode = axis_payload(line_df[x_col], x_log)
            line_df["_plot_y"], line_y_mode = axis_payload(line_df[y_col], y_log)
            if twin_y_requested:
                line_df["_plot_y2"], line_y2_mode = axis_payload(line_df[y2_col], y2_log)
            else:
                line_y2_mode = "linear"
            if "_plot_x" in line_df.columns:
                line_df = line_df.sort_values("_plot_x")
            effective_line_markers = line_show_markers and line_marker_choice != "none"
            if twin_y_requested:
                twin_line_df = line_df[["_plot_x", "_plot_y", "_plot_y2"]].dropna().copy()
                if viz_engine == "plotly":
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(
                            x=twin_line_df["_plot_x"],
                            y=twin_line_df["_plot_y"],
                            mode="lines+markers" if effective_line_markers else "lines",
                            name=y_axis_label,
                            line=dict(color=twin_primary_line_color_choice, dash=twin_primary_plotly_line_style, width=line_width),
                            marker=dict(symbol=selected_plotly_marker, size=8, color=twin_primary_line_color_choice) if effective_line_markers else None,
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=twin_line_df["_plot_x"],
                            y=twin_line_df["_plot_y2"],
                            mode="lines+markers" if effective_line_markers else "lines",
                            name=y2_axis_label,
                            line=dict(color=twin_secondary_line_color_choice, dash=twin_secondary_plotly_line_style, width=line_width),
                            marker=dict(symbol="diamond", size=8, color=twin_secondary_line_color_choice) if effective_line_markers else None,
                        ),
                        secondary_y=True,
                    )
                    fig.update_layout(
                        template="plotly_white",
                        title_text=resolve_text(plot_title_override, "Line plot"),
                        title_font=dict(size=plot_title_font_size),
                        legend_title_text="",
                        height=PLOTLY_FIG_HEIGHT,
                        margin=dict(l=55, r=25, t=70, b=90),
                        autosize=True,
                    )
                    fig.update_xaxes(
                        title_text=resolve_text(x_title_override, axis_title(x_col, line_x_mode)),
                        tickangle=x_tick_angle,
                        nticks=max_x_ticks,
                        showgrid=grid_enabled("x"),
                        gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                        zeroline=False,
                        automargin=True,
                    )
                    fig.update_yaxes(
                        title_text=resolve_text(y_title_override, y_axis_label),
                        secondary_y=False,
                        showgrid=grid_enabled("y"),
                        gridcolor=f"rgba(15, 23, 42, {grid_alpha})",
                        zeroline=False,
                        title_font=dict(size=axis_title_font_size, color=twin_primary_line_color_choice),
                        type="log" if line_y_mode == "log" else None,
                    )
                    fig.update_yaxes(
                        title_text=y2_axis_label,
                        secondary_y=True,
                        showgrid=False,
                        zeroline=False,
                        title_font=dict(size=axis_title_font_size, color=twin_secondary_line_color_choice),
                        type="log" if line_y2_mode == "log" else None,
                    )
                    cache_plotly(main_plot_cache, fig, "current_plot")
                    return plotly_to_html(fig)
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                ax2 = ax.twinx()
                ax.plot(
                    twin_line_df["_plot_x"],
                    twin_line_df["_plot_y"],
                    linestyle=twin_primary_mpl_line_style,
                    linewidth=line_width,
                    marker=selected_mpl_marker if effective_line_markers else None,
                    color=twin_primary_line_color_choice,
                    label=y_axis_label,
                )
                ax2.plot(
                    twin_line_df["_plot_x"],
                    twin_line_df["_plot_y2"],
                    linestyle=twin_secondary_mpl_line_style,
                    linewidth=line_width,
                    marker="D" if effective_line_markers else None,
                    color=twin_secondary_line_color_choice,
                    label=y2_axis_label,
                )
                ax.set_title("Line plot")
                ax.set_xlabel(axis_title(x_col, line_x_mode))
                ax.set_ylabel(y_axis_label, fontsize=axis_title_font_size, color=twin_primary_line_color_choice)
                ax2.set_ylabel(y2_axis_label, fontsize=axis_title_font_size, color=twin_secondary_line_color_choice)
                apply_mpl_axis_scale(ax, line_x_mode, line_y_mode)
                if line_y2_mode == "log":
                    ax2.set_yscale("log")
                ax.tick_params(axis="x", rotation=x_tick_angle)
                if grid_axis == "none":
                    ax.grid(False)
                else:
                    ax.grid(True, axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha)
                if pd.api.types.is_numeric_dtype(twin_line_df["_plot_x"]):
                    try:
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks))
                    except Exception:
                        pass
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                if handles1 or handles2:
                    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
                apply_mpl_formatting(ax, "Line plot", axis_title(x_col, line_x_mode), y_axis_label, twin_line_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            if viz_engine == "plotly":
                fig = go.Figure()
                if hue_col:
                    grouped = line_df.groupby(hue_col, sort=False)
                    for group, group_df in grouped:
                        fig.add_trace(
                            go.Scatter(
                                x=group_df["_plot_x"],
                                y=group_df["_plot_y"],
                                mode="lines+markers" if effective_line_markers else "lines",
                                name=str(group),
                                line=dict(dash=selected_plotly_line_style, width=line_width),
                                marker=dict(symbol=selected_plotly_marker, size=8) if effective_line_markers else None,
                            )
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=line_df["_plot_x"],
                            y=line_df["_plot_y"],
                            mode="lines+markers" if effective_line_markers else "lines",
                            name="Series",
                            line=dict(dash=selected_plotly_line_style, width=line_width),
                            marker=dict(symbol=selected_plotly_marker, size=8) if effective_line_markers else None,
                        )
                    )
                fig.update_layout(template="plotly_white", legend_title_text=hue_col or "", title="Line plot")
                fig.update_xaxes(title_text=axis_title(x_col, line_x_mode))
                fig.update_yaxes(title_text=axis_title(y_col, line_y_mode))
                apply_plotly_axis_scale(fig, line_x_mode, line_y_mode)
                apply_plotly_formatting(fig, "Line plot", axis_title(x_col, line_x_mode), axis_title(y_col, line_y_mode))
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                if hue_col:
                    palette = sns.color_palette(n_colors=line_df[hue_col].nunique())
                    for color, (group, group_df) in zip(palette, line_df.groupby(hue_col, sort=False)):
                        ax.plot(
                            group_df["_plot_x"],
                            group_df["_plot_y"],
                            color=color,
                            linestyle=selected_mpl_line_style,
                            linewidth=line_width,
                            marker=selected_mpl_marker if effective_line_markers else None,
                            label=str(group),
                        )
                    ax.legend(title=hue_col)
                else:
                    ax.plot(
                        line_df["_plot_x"],
                        line_df["_plot_y"],
                        linestyle=selected_mpl_line_style,
                        linewidth=line_width,
                        marker=selected_mpl_marker if effective_line_markers else None,
                    )
                ax.set_title("Line plot")
                ax.set_xlabel(axis_title(x_col, line_x_mode))
                ax.set_ylabel(axis_title(y_col, line_y_mode))
                apply_mpl_axis_scale(ax, line_x_mode, line_y_mode)
                apply_mpl_formatting(ax, "Line plot", axis_title(x_col, line_x_mode), axis_title(y_col, line_y_mode), line_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            if hue_col:
                for group, group_df in line_df.groupby(hue_col):
                    ax.plot(
                        group_df["_plot_x"],
                        group_df["_plot_y"],
                        linestyle=selected_mpl_line_style,
                        linewidth=line_width,
                        marker=selected_mpl_marker if effective_line_markers else None,
                        label=str(group),
                    )
                ax.legend(title=hue_col)
            else:
                ax.plot(
                    line_df["_plot_x"],
                    line_df["_plot_y"],
                    linestyle=selected_mpl_line_style,
                    linewidth=line_width,
                    marker=selected_mpl_marker if effective_line_markers else None,
                )
            ax.set_title("Line plot")
            ax.set_xlabel(axis_title(x_col, line_x_mode))
            ax.set_ylabel(axis_title(y_col, line_y_mode))
            apply_mpl_axis_scale(ax, line_x_mode, line_y_mode)
            apply_mpl_formatting(ax, "Line plot", axis_title(x_col, line_x_mode), axis_title(y_col, line_y_mode), line_df["_plot_x"])
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "bar":
            bar_df = plot_df[[x_col] + ([y_col] if y_col else []) + ([hue_col] if hue_col else [])].dropna()
            if y_col and pd.api.types.is_numeric_dtype(bar_df[y_col]):
                if hue_col:
                    agg = bar_df.groupby([x_col, hue_col], as_index=False)[y_col].mean()
                else:
                    agg = bar_df.groupby(x_col, as_index=False)[y_col].mean()
            else:
                if hue_col:
                    agg = bar_df.groupby([x_col, hue_col]).size().reset_index(name="count")
                    y_plot_col = "count"
                else:
                    agg = bar_df[x_col].astype(str).value_counts().reset_index()
                    agg.columns = [x_col, "count"]
                    y_plot_col = "count"
            if y_col and pd.api.types.is_numeric_dtype(bar_df[y_col]):
                y_plot_col = y_col
            agg["_plot_x"], bar_x_mode = axis_payload(agg[x_col], x_log)
            agg["_plot_y"], bar_y_mode = axis_payload(agg[y_plot_col], y_log)
            if viz_engine == "plotly":
                fig = px.bar(
                    agg,
                    x="_plot_x",
                    y="_plot_y",
                    color=hue_col,
                    barmode="group" if hue_col else "relative",
                    title="Bar plot",
                )
                fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
                fig.update_xaxes(title_text=axis_title(x_col, bar_x_mode))
                fig.update_yaxes(title_text=axis_title(y_plot_col if y_col else "Count", bar_y_mode))
                apply_plotly_axis_scale(fig, bar_x_mode, bar_y_mode)
                apply_plotly_formatting(fig, "Bar plot", axis_title(x_col, bar_x_mode), axis_title(y_plot_col if y_col else "Count", bar_y_mode))
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                sns.barplot(data=agg, x="_plot_x", y="_plot_y", hue=hue_col, ax=ax, color="steelblue" if not hue_col else None)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title("Bar plot")
                ax.set_xlabel(axis_title(x_col, bar_x_mode))
                ax.set_ylabel(axis_title(y_plot_col if y_col else "Count", bar_y_mode))
                apply_mpl_axis_scale(ax, bar_x_mode, bar_y_mode)
                apply_mpl_formatting(ax, "Bar plot", axis_title(x_col, bar_x_mode), axis_title(y_plot_col if y_col else "Count", bar_y_mode), agg["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            if hue_col and y_plot_col != "count":
                pivot = agg.pivot(index="_plot_x", columns=hue_col, values="_plot_y").fillna(0)
                pivot.plot(kind="bar", ax=ax)
            elif hue_col and y_plot_col == "count":
                pivot = agg.pivot(index="_plot_x", columns=hue_col, values="_plot_y").fillna(0)
                pivot.plot(kind="bar", ax=ax)
            else:
                ax.bar(agg["_plot_x"].astype(str), agg["_plot_y"], color="steelblue")
            ax.tick_params(axis="x", rotation=45)
            ax.set_title("Bar plot")
            ax.set_xlabel(axis_title(x_col, bar_x_mode))
            ax.set_ylabel(axis_title(y_plot_col if y_col else "Count", bar_y_mode))
            apply_mpl_axis_scale(ax, bar_x_mode, bar_y_mode)
            apply_mpl_formatting(ax, "Bar plot", axis_title(x_col, bar_x_mode), axis_title(y_plot_col if y_col else "Count", bar_y_mode), agg["_plot_x"])
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "stacked crosstab":
            if not hue_col:
                return mpl_message("Choose a Group / color column for the stacked crosstab plot.")
            crosstab_df = plot_df[[x_col, hue_col]].dropna()
            if crosstab_df.empty:
                return mpl_message("No data available for the selected x and group columns.")

            crosstab = pd.crosstab(crosstab_df[x_col].astype(str), crosstab_df[hue_col].astype(str))
            if crosstab.empty:
                return mpl_message("No cross-tabulated counts available for the selected columns.")

            if viz_engine == "plotly":
                fig = go.Figure()
                group_totals = crosstab.sum(axis=1).reindex(crosstab.index).to_numpy(dtype=float)
                x_labels = crosstab.index.astype(str).tolist()
                for level in crosstab.columns:
                    counts = crosstab[level].to_numpy(dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        share = np.where(group_totals > 0, counts / group_totals, np.nan)
                    fig.add_trace(
                        go.Bar(
                            name=str(level),
                            x=x_labels,
                            y=counts,
                            customdata=np.column_stack([group_totals, share]),
                            hovertemplate=(
                                f"{display_col(x_col)}=%{{x}}<br>"
                                f"{display_col(hue_col)}=%{{fullData.name}}<br>"
                                "Count=%{y}<br>"
                                f"Total in {display_col(x_col)}=%{{customdata[0]:.0f}}<br>"
                                "Share of total=%{customdata[1]:.1%}<extra></extra>"
                            ),
                        )
                    )
                fig.update_layout(
                    barmode="stack",
                    template="plotly_white",
                    legend_title_text=display_col(hue_col),
                )
                fig.update_xaxes(title_text=display_col(x_col))
                fig.update_yaxes(title_text="Count")
                apply_plotly_formatting(fig, "Stacked crosstab count plot", display_col(x_col), "Count")
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)

            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_title("Stacked crosstab count plot")
            ax.set_xlabel(display_col(x_col))
            ax.set_ylabel("Count")
            ax.legend(title=display_col(hue_col), bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
            apply_mpl_formatting(ax, "Stacked crosstab count plot", display_col(x_col), "Count", None)
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "histogram":
            hist_bins = int(input.hist_bins() or 20)
            hist_df = plot_df[[x_col] + ([hue_col] if hue_col else [])].dropna()
            hist_df["_plot_x"], hist_x_mode = axis_payload(hist_df[x_col], x_log)
            is_numeric = pd.api.types.is_numeric_dtype(hist_df[x_col])
            is_datetime = pd.api.types.is_datetime64_any_dtype(hist_df[x_col])
            unique_n = hist_df[x_col].nunique(dropna=True)
            integer_like = False
            if is_numeric:
                numeric_values = pd.to_numeric(hist_df[x_col], errors="coerce").dropna()
                if not numeric_values.empty:
                    integer_like = bool(np.all(np.isclose(numeric_values, np.round(numeric_values))))

            if not is_numeric and not is_datetime:
                return mpl_message("Histogram requires a numeric or datetime column")

            if integer_like and unique_n <= 30 and not is_datetime:
                if viz_engine == "plotly":
                    fig = px.histogram(
                        hist_df,
                        x="_plot_x",
                        color=hue_col,
                        nbins=min(hist_bins, unique_n),
                        barmode="overlay" if hue_col else "group",
                        opacity=0.65,
                        title="Histogram plot",
                    )
                    fig.update_traces(marker_line_color="white", marker_line_width=1)
                    fig.update_layout(template="plotly_white")
                    fig.update_xaxes(title_text=axis_title(x_col, hist_x_mode))
                    apply_plotly_axis_scale(fig, hist_x_mode, "linear")
                    apply_plotly_formatting(fig, "Histogram plot", axis_title(x_col, hist_x_mode), "Count")
                    cache_plotly(main_plot_cache, fig, "current_plot")
                    return plotly_to_html(fig)
                if viz_engine == "seaborn":
                    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                    sns.histplot(
                        data=hist_df,
                        x="_plot_x",
                        hue=hue_col,
                        discrete=True,
                        bins=min(hist_bins, unique_n),
                        stat="count",
                        common_norm=False,
                        element="bars",
                        shrink=0.8,
                        alpha=0.65,
                        ax=ax,
                    )
                    ax.set_title("Histogram plot")
                    ax.set_xlabel(axis_title(x_col, hist_x_mode))
                    ax.set_ylabel("Count")
                    apply_mpl_axis_scale(ax, hist_x_mode, "linear")
                    apply_mpl_formatting(ax, "Histogram plot", axis_title(x_col, hist_x_mode), "Count", hist_df["_plot_x"])
                    fig.tight_layout()
                    cache_matplotlib(main_plot_cache, fig, "current_plot")
                    return matplotlib_to_html(fig)
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                if hue_col:
                    for group, group_df in hist_df.groupby(hue_col):
                        ax.hist(group_df["_plot_x"], bins=min(hist_bins, unique_n), alpha=0.5, label=str(group))
                    ax.legend(title=hue_col)
                else:
                    ax.hist(hist_df["_plot_x"], bins=min(hist_bins, unique_n), color="steelblue", edgecolor="white", alpha=0.8)
                ax.set_title("Histogram plot")
                ax.set_xlabel(axis_title(x_col, hist_x_mode))
                ax.set_ylabel("Count")
                apply_mpl_axis_scale(ax, hist_x_mode, "linear")
                apply_mpl_formatting(ax, "Histogram plot", axis_title(x_col, hist_x_mode), "Count", hist_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)

            if viz_engine == "plotly":
                fig = px.histogram(
                    hist_df,
                    x="_plot_x",
                    color=hue_col,
                    nbins=hist_bins,
                    opacity=0.65,
                    title="Histogram plot",
                )
                fig.update_layout(template="plotly_white")
                fig.update_xaxes(title_text=axis_title(x_col, hist_x_mode))
                apply_plotly_axis_scale(fig, hist_x_mode, "linear")
                apply_plotly_formatting(fig, "Histogram plot", axis_title(x_col, hist_x_mode), "Count")
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                sns.histplot(
                    data=hist_df,
                    x="_plot_x",
                    hue=hue_col,
                    bins=hist_bins,
                    kde=False,
                    element="step",
                    ax=ax,
                    alpha=0.6,
                )
                ax.set_title("Histogram plot")
                ax.set_xlabel(axis_title(x_col, hist_x_mode))
                ax.set_ylabel("Count")
                apply_mpl_axis_scale(ax, hist_x_mode, "linear")
                apply_mpl_formatting(ax, "Histogram plot", axis_title(x_col, hist_x_mode), "Count", hist_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            if hue_col:
                for group, group_df in hist_df.groupby(hue_col):
                    ax.hist(group_df["_plot_x"], bins=hist_bins, alpha=0.5, label=str(group))
                ax.legend(title=hue_col)
            else:
                ax.hist(hist_df["_plot_x"], bins=hist_bins, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title("Histogram plot")
            ax.set_xlabel(axis_title(x_col, hist_x_mode))
            ax.set_ylabel("Count")
            apply_mpl_axis_scale(ax, hist_x_mode, "linear")
            apply_mpl_formatting(ax, "Histogram plot", axis_title(x_col, hist_x_mode), "Count", hist_df["_plot_x"])
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "pie":
            pie_donut = bool(input.pie_donut())
            categorical_cols = [
                col for col in cols if col not in df.select_dtypes(include="number").columns and col not in df.select_dtypes(include="datetime").columns
            ]
            pie_label_col = input.pie_label_col() or (categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL))
            pie_value_col = input.pie_value_col() or "Count"
            if pie_label_col not in available_cols:
                pie_label_col = categorical_cols[0] if categorical_cols else (cols[0] if cols else INDEX_COL)
            if pie_value_col != "Count" and pie_value_col not in available_cols:
                pie_value_col = "Count"
            required_cols = [pie_label_col] + ([] if pie_value_col == "Count" else [pie_value_col])
            pie_df = plot_df[required_cols].dropna()
            if pie_df.empty:
                return mpl_message("No data available for pie chart")

            if pie_value_col == "Count":
                pie_data = (
                    pie_df[pie_label_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                pie_data.columns = ["label", "value"]
            else:
                if not pd.api.types.is_numeric_dtype(pie_df[pie_value_col]):
                    return mpl_message("Pie value column must be numeric or Count.")

                pie_data = (
                    pie_df.groupby(pie_label_col, as_index=False)[pie_value_col]
                    .sum()
                    .rename(columns={pie_label_col: "label", pie_value_col: "value"})
                )

            if viz_engine == "plotly":
                fig = px.pie(
                    pie_data,
                    names="label",
                    values="value",
                    hole=0.35 if pie_donut else 0,
                    title="Pie chart" if not pie_donut else "Donut chart",
                )
                fig.update_layout(template="plotly_white")
                apply_plotly_formatting(fig, "Donut chart" if pie_donut else "Pie chart")
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            ax.pie(
                pie_data["value"],
                labels=pie_data["label"].astype(str),
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width": 0.35} if pie_donut else None,
            )
            ax.set_title("Donut chart" if pie_donut else "Pie chart")
            ax.axis("equal")
            apply_mpl_formatting(ax, "Donut chart" if pie_donut else "Pie chart")
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "box":
            box_df = plot_df[[x_col, y_col]].copy()
            hue_alias = None
            if hue_col:
                hue_alias = hue_col
                if hue_col in {x_col, y_col}:
                    hue_alias = f"__box_hue_{hue_col}__"
                hue_series = resolve_column(df, hue_col) if hue_col != INDEX_COL else resolve_column(df, INDEX_COL)
                box_df[hue_alias] = hue_series.reindex(box_df.index)
                box_df = box_df.dropna(subset=[x_col, y_col, hue_alias])
            else:
                box_df = box_df.dropna(subset=[x_col, y_col])
            box_df["_plot_x"], box_x_mode = axis_payload(box_df[x_col], x_log)
            box_df["_plot_y"], box_y_mode = axis_payload(box_df[y_col], y_log)
            if viz_engine == "plotly":
                fig = px.box(
                    box_df,
                    x="_plot_x",
                    y="_plot_y",
                    color=hue_alias,
                    points="outliers",
                    title="Box plot",
                )
                fig.update_layout(template="plotly_white", legend_title_text=hue_col or "")
                fig.update_xaxes(title_text=axis_title(x_col, box_x_mode))
                fig.update_yaxes(title_text=axis_title(y_col, box_y_mode))
                apply_plotly_axis_scale(fig, box_x_mode, box_y_mode)
                apply_plotly_formatting(fig, "Box plot", axis_title(x_col, box_x_mode), axis_title(y_col, box_y_mode))
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                sns.boxplot(data=box_df, x="_plot_x", y="_plot_y", hue=hue_alias, ax=ax)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title("Box plot")
                ax.set_xlabel(axis_title(x_col, box_x_mode))
                ax.set_ylabel(axis_title(y_col, box_y_mode))
                apply_mpl_axis_scale(ax, box_x_mode, box_y_mode)
                apply_mpl_formatting(ax, "Box plot", axis_title(x_col, box_x_mode), axis_title(y_col, box_y_mode), box_df["_plot_x"])
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            if hue_alias:
                groups = []
                labels = []
                for group, group_df in box_df.groupby(hue_alias):
                    groups.append(group_df["_plot_y"].dropna().values)
                    labels.append(str(group))
                ax.boxplot(groups, labels=labels)
                ax.set_xlabel(hue_col)
            else:
                ax.boxplot([box_df["_plot_y"].dropna().values], labels=[y_col])
            ax.tick_params(axis="x", rotation=45)
            ax.set_title("Box plot")
            ax.set_ylabel(axis_title(y_col, box_y_mode))
            apply_mpl_axis_scale(ax, box_x_mode, box_y_mode)
            apply_mpl_formatting(ax, "Box plot", axis_title(x_col, box_x_mode), axis_title(y_col, box_y_mode), box_df["_plot_x"])
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)
        elif plot_kind == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return mpl_message("Need at least two numeric columns")

            corr = numeric_df.corr(numeric_only=True)
            if viz_engine == "plotly":
                fig = px.imshow(
                    corr,
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    title="Correlation heatmap",
                )
                apply_plotly_formatting(fig, "Correlation heatmap", "Variable", "Variable")
                if heatmap_show_values:
                    value_threshold = float((np.nanmin(corr.to_numpy()) + np.nanmax(corr.to_numpy())) / 2.0)
                    for row_idx, row_label in enumerate(corr.index):
                        for col_idx, col_label in enumerate(corr.columns):
                            value = float(corr.iat[row_idx, col_idx])
                            fig.add_annotation(
                                x=col_label,
                                y=row_label,
                                text=f"{value:.2f}",
                                showarrow=False,
                                font=dict(color="white" if value < value_threshold else "black", size=12),
                            )
                cache_plotly(main_plot_cache, fig, "current_plot")
                return plotly_to_html(fig)
            if viz_engine == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
                heatmap = sns.heatmap(corr, cmap="viridis", ax=ax, annot=heatmap_show_values, fmt=".2f")
                if heatmap_show_values:
                    flat_values = corr.to_numpy().ravel()
                    for text, value in zip(ax.texts, flat_values):
                        text.set_color("white" if heatmap.collections[0].norm(value) > 0.5 else "black")
                ax.set_title("Correlation heatmap")
                apply_mpl_formatting(ax, "Correlation heatmap", "Variable", "Variable")
                fig.tight_layout()
                cache_matplotlib(main_plot_cache, fig, "current_plot")
                return matplotlib_to_html(fig)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
            im = ax.imshow(corr.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            if heatmap_show_values:
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        value = float(corr.iat[i, j])
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if im.norm(value) > 0.5 else "black",
                            fontsize=8,
                        )
            fig.colorbar(im, ax=ax)
            ax.set_title("Correlation heatmap")
            apply_mpl_formatting(ax, "Correlation heatmap", "Variable", "Variable")
            fig.tight_layout()
            cache_matplotlib(main_plot_cache, fig, "current_plot")
            return matplotlib_to_html(fig)

        return mpl_message("Unsupported plot type")


app = App(app_ui, server)


if __name__ == "__main__":
    from shiny import run_app

    run_app(app)
