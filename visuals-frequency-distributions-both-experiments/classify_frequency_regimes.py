"""
Classify noun terms into statistically estimated frequency regimes.

Method
------
1. Clean each corpus.
2. Convert absolute counts to frequency per million noun tokens (FPM).
3. Rank terms from most to least frequent.
4. Fit continuous piecewise-linear regressions to:
       log10(FPM) ~ log10(rank)
5. Compare 1..MAX_SEGMENTS using BIC.
6. Select the model with the lowest BIC.
7. Estimate breakpoint stability by repeated subsampling.
8. Assign ordinal labels only when the selected model has four segments.
   If BIC selects another number, the script reports that four categories
   are not empirically supported. Set FORCE_FOUR_SEGMENTS=True only for an
   explicitly constrained four-class analysis.

Important limitation
--------------------
The input files contain aggregated term frequencies, not document-level data.
Therefore, the repeated subsampling intervals produced here are descriptive
stability intervals, not confidence intervals for corpus-sampling uncertainty.
A proper inferential bootstrap would require resampling documents and
recomputing the term frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PG19_PATH = Path("../term_frequencies-pg19/noun_frequencies_pg19.csv")
PUBMED_PATH = Path("../term_frequencies/noun_frequencies_pubmed.csv")

OUTPUT_DIR = Path("frequency_regime_results")

MAX_SEGMENTS = 6
MAX_FIT_POINTS = 6000
MIN_LOG_WIDTH = 0.15

STABILITY_REPETITIONS = 30
STABILITY_SAMPLE_FRACTION = 0.80

RANDOM_SEED = 42

# Keep False for a fully data-driven analysis.
# If True, the script will force four segments even when BIC prefers another
# number. The output will state that the four-class model was constrained.
FORCE_FOUR_SEGMENTS = False

CLASS_LABELS = ["Very high", "High", "Medium", "Low"]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
@dataclass
class PiecewiseFit:
    n_segments: int
    breakpoints_x: np.ndarray
    coefficients: np.ndarray
    rss: float
    bic: float
    predictions: np.ndarray


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def load_and_prepare(path: Path, corpus_name: str) -> pd.DataFrame:
    """Load, clean, normalize, sort and rank one frequency table."""
    df = pd.read_csv(path)

    if "frequency" not in df.columns:
        raise ValueError(
            f"{path} does not contain a 'frequency' column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df = df.dropna(subset=["frequency"])
    df = df[df["frequency"] > 0].copy()

    if df.empty:
        raise ValueError(f"No positive numeric frequencies were found in {path}.")

    total_noun_tokens = float(df["frequency"].sum())

    df["frequency_per_million"] = (
        df["frequency"] / total_noun_tokens
    ) * 1_000_000

    df = df.sort_values(
        "frequency_per_million",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["corpus"] = corpus_name

    return df


def logarithmic_rank_sample(
    df: pd.DataFrame,
    max_points: int,
) -> pd.DataFrame:
    """
    Select approximately log-spaced ranks for model fitting.

    This prevents the very large low-frequency tail from numerically
    overwhelming the fit while retaining observations across the full range.
    """
    n = len(df)

    if n <= max_points:
        return df.copy()

    sampled_ranks = np.unique(
        np.rint(
            np.geomspace(1, n, num=max_points)
        ).astype(np.int64)
    )

    return df.iloc[sampled_ranks - 1].copy()


# ---------------------------------------------------------------------
# Continuous segmented regression
# ---------------------------------------------------------------------
def hinge_design_matrix(
    x: np.ndarray,
    breakpoints: np.ndarray,
) -> np.ndarray:
    """
    Continuous piecewise-linear spline basis:

        y = beta_0 + beta_1*x
            + gamma_1*max(0, x-tau_1)
            + ...
    """
    columns = [
        np.ones_like(x),
        x,
    ]

    for breakpoint in breakpoints:
        columns.append(np.maximum(0.0, x - breakpoint))

    return np.column_stack(columns)


def valid_breakpoints(
    breakpoints: np.ndarray,
    x_min: float,
    x_max: float,
    min_width: float,
) -> bool:
    """Require ordered breakpoints and a minimum width for every segment."""
    if len(breakpoints) == 0:
        return True

    breakpoints = np.sort(breakpoints)
    edges = np.concatenate(([x_min], breakpoints, [x_max]))
    return bool(np.all(np.diff(edges) >= min_width))


def fit_given_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    breakpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the linear coefficients conditional on breakpoint locations."""
    design = hinge_design_matrix(x, breakpoints)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    residuals = y - predictions
    rss = float(np.sum(residuals**2))
    return coefficients, predictions, rss


def fit_piecewise_model(
    x: np.ndarray,
    y: np.ndarray,
    n_segments: int,
    min_log_width: float,
    seed: int,
) -> PiecewiseFit:
    """Fit a continuous piecewise-linear model with unknown breakpoints."""
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    n_breakpoints = n_segments - 1

    if n_breakpoints == 0:
        coefficients, predictions, rss = fit_given_breakpoints(
            x,
            y,
            np.array([], dtype=float),
        )
        # Intercept + slope
        k = 2
        bic = len(y) * np.log(max(rss / len(y), np.finfo(float).tiny))
        bic += k * np.log(len(y))

        return PiecewiseFit(
            n_segments=n_segments,
            breakpoints_x=np.array([], dtype=float),
            coefficients=coefficients,
            rss=rss,
            bic=float(bic),
            predictions=predictions,
        )

    available_width = x_max - x_min
    required_width = n_segments * min_log_width

    if available_width <= required_width:
        raise ValueError(
            f"The x-range is too narrow for {n_segments} segments with "
            f"MIN_LOG_WIDTH={min_log_width}."
        )

    lower = x_min + min_log_width
    upper = x_max - min_log_width
    bounds = [(lower, upper)] * n_breakpoints

    penalty_scale = max(float(np.var(y) * len(y)), 1.0) * 1e6

    def objective(candidate: np.ndarray) -> float:
        candidate = np.sort(candidate)

        if not valid_breakpoints(
            candidate,
            x_min=x_min,
            x_max=x_max,
            min_width=min_log_width,
        ):
            return penalty_scale

        _, _, rss = fit_given_breakpoints(x, y, candidate)
        return rss

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )

    if not result.success:
        warnings.warn(
            f"Optimization did not fully converge for {n_segments} segments: "
            f"{result.message}"
        )

    breakpoints = np.sort(result.x)
    coefficients, predictions, rss = fit_given_breakpoints(
        x,
        y,
        breakpoints,
    )

    # Parameters:
    # 2 linear coefficients + (m-1) slope changes + (m-1) breakpoints = 2m
    k = 2 * n_segments
    bic = len(y) * np.log(max(rss / len(y), np.finfo(float).tiny))
    bic += k * np.log(len(y))

    return PiecewiseFit(
        n_segments=n_segments,
        breakpoints_x=breakpoints,
        coefficients=coefficients,
        rss=rss,
        bic=float(bic),
        predictions=predictions,
    )


def select_number_of_segments(
    x: np.ndarray,
    y: np.ndarray,
    max_segments: int,
    min_log_width: float,
    seed: int,
) -> tuple[PiecewiseFit, pd.DataFrame, dict[int, PiecewiseFit]]:
    """Fit 1..max_segments and select the model with minimum BIC."""
    fitted_models: dict[int, PiecewiseFit] = {}
    rows: list[dict[str, float | int]] = []

    for n_segments in range(1, max_segments + 1):
        print(f"  Fitting {n_segments} segment(s)...")

        model = fit_piecewise_model(
            x=x,
            y=y,
            n_segments=n_segments,
            min_log_width=min_log_width,
            seed=seed + n_segments,
        )

        fitted_models[n_segments] = model

        rows.append(
            {
                "n_segments": n_segments,
                "rss": model.rss,
                "bic": model.bic,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("n_segments")
    best_segments = int(
        comparison.loc[comparison["bic"].idxmin(), "n_segments"]
    )

    return fitted_models[best_segments], comparison, fitted_models


# ---------------------------------------------------------------------
# Breakpoint conversion and classification
# ---------------------------------------------------------------------
def breakpoint_table(
    full_df: pd.DataFrame,
    fitted_model: PiecewiseFit,
) -> pd.DataFrame:
    """
    Convert log-rank breakpoints into observed rank and frequency thresholds.
    """
    rows = []

    for number, breakpoint_x in enumerate(
        fitted_model.breakpoints_x,
        start=1,
    ):
        estimated_rank = int(np.rint(10**breakpoint_x))
        estimated_rank = int(np.clip(estimated_rank, 1, len(full_df)))

        row = full_df.iloc[estimated_rank - 1]

        rows.append(
            {
                "breakpoint": number,
                "log10_rank": breakpoint_x,
                "rank": estimated_rank,
                "frequency": float(row["frequency"]),
                "frequency_per_million": float(
                    row["frequency_per_million"]
                ),
            }
        )

    return pd.DataFrame(rows)


def assign_frequency_classes(
    df: pd.DataFrame,
    breakpoint_ranks: list[int],
    labels: list[str],
) -> pd.DataFrame:
    """Assign one ordered class per term using estimated rank boundaries."""
    if len(breakpoint_ranks) != len(labels) - 1:
        raise ValueError(
            "The number of breakpoints must equal len(labels) - 1."
        )

    result = df.copy()

    bins = [0] + sorted(breakpoint_ranks) + [np.inf]

    result["frequency_class"] = pd.cut(
        result["rank"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )

    return result


# ---------------------------------------------------------------------
# Descriptive breakpoint stability
# ---------------------------------------------------------------------
def estimate_subsampling_stability(
    x: np.ndarray,
    y: np.ndarray,
    n_segments: int,
    repetitions: int,
    sample_fraction: float,
    min_log_width: float,
    seed: int,
) -> pd.DataFrame:
    """
    Refit the selected model on repeated random subsets.

    These are descriptive stability intervals, not formal confidence intervals
    for corpus-sampling uncertainty.
    """
    if n_segments == 1:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    sample_size = max(
        int(np.floor(len(x) * sample_fraction)),
        100,
    )

    estimates = []

    for repetition in range(repetitions):
        print(
            f"    Stability repetition {repetition + 1}/{repetitions}"
        )

        indices = np.sort(
            rng.choice(
                len(x),
                size=sample_size,
                replace=False,
            )
        )

        fitted = fit_piecewise_model(
            x=x[indices],
            y=y[indices],
            n_segments=n_segments,
            min_log_width=min_log_width,
            seed=seed + 10_000 + repetition,
        )

        estimates.append(fitted.breakpoints_x)

    estimates_array = np.vstack(estimates)

    rows = []

    for breakpoint_index in range(n_segments - 1):
        values = estimates_array[:, breakpoint_index]

        rows.append(
            {
                "breakpoint": breakpoint_index + 1,
                "median_log10_rank": float(np.median(values)),
                "lower_2_5_log10_rank": float(
                    np.quantile(values, 0.025)
                ),
                "upper_97_5_log10_rank": float(
                    np.quantile(values, 0.975)
                ),
                "median_rank": int(
                    np.rint(10 ** np.median(values))
                ),
                "lower_2_5_rank": int(
                    np.rint(10 ** np.quantile(values, 0.025))
                ),
                "upper_97_5_rank": int(
                    np.rint(10 ** np.quantile(values, 0.975))
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def predict_from_model(
    x: np.ndarray,
    model: PiecewiseFit,
) -> np.ndarray:
    design = hinge_design_matrix(x, model.breakpoints_x)
    return design @ model.coefficients


def plot_segmented_fit(
    fit_df: pd.DataFrame,
    model: PiecewiseFit,
    corpus_name: str,
    output_path: Path,
) -> None:
    """Plot observed log-rank/log-frequency data and fitted segments."""
    x = np.log10(fit_df["rank"].to_numpy(dtype=float))
    y = np.log10(
        fit_df["frequency_per_million"].to_numpy(dtype=float)
    )

    grid_x = np.linspace(x.min(), x.max(), 1500)
    grid_y = predict_from_model(grid_x, model)

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        10**x,
        10**y,
        s=8,
        alpha=0.35,
        label="Observed terms",
    )

    ax.plot(
        10**grid_x,
        10**grid_y,
        linewidth=2.5,
        label=f"Selected {model.n_segments}-segment model",
    )

    for breakpoint in model.breakpoints_x:
        ax.axvline(
            10**breakpoint,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noun rank, r (log scale)")
    ax.set_ylabel("Frequency per million noun tokens (log scale)")
    ax.set_title(
        f"{corpus_name}: segmented rank-frequency model"
    )
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bic(
    comparison: pd.DataFrame,
    corpus_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        comparison["n_segments"],
        comparison["bic"],
        marker="o",
    )

    best_row = comparison.loc[comparison["bic"].idxmin()]

    ax.scatter(
        [best_row["n_segments"]],
        [best_row["bic"]],
        s=90,
        zorder=3,
        label="Minimum BIC",
    )

    ax.set_xlabel("Number of segments")
    ax.set_ylabel("BIC")
    ax.set_title(f"{corpus_name}: model selection")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Corpus-level pipeline
# ---------------------------------------------------------------------
def analyse_corpus(
    path: Path,
    corpus_name: str,
    output_dir: Path,
) -> dict[str, object]:
    print(f"\nAnalysing {corpus_name}")

    full_df = load_and_prepare(path, corpus_name)
    fit_df = logarithmic_rank_sample(
        full_df,
        max_points=MAX_FIT_POINTS,
    )

    x = np.log10(fit_df["rank"].to_numpy(dtype=float))
    y = np.log10(
        fit_df["frequency_per_million"].to_numpy(dtype=float)
    )

    selected_model, comparison, fitted_models = (
        select_number_of_segments(
            x=x,
            y=y,
            max_segments=MAX_SEGMENTS,
            min_log_width=MIN_LOG_WIDTH,
            seed=RANDOM_SEED,
        )
    )

    empirically_selected_segments = selected_model.n_segments

    if FORCE_FOUR_SEGMENTS:
        final_model = fitted_models[4]
        model_status = (
            "Four segments were imposed by design; BIC selected "
            f"{empirically_selected_segments} segment(s)."
        )
    else:
        final_model = selected_model
        model_status = (
            f"BIC selected {empirically_selected_segments} segment(s)."
        )

    print(f"  {model_status}")

    breakpoints = breakpoint_table(full_df, final_model)

    stability = estimate_subsampling_stability(
        x=x,
        y=y,
        n_segments=final_model.n_segments,
        repetitions=STABILITY_REPETITIONS,
        sample_fraction=STABILITY_SAMPLE_FRACTION,
        min_log_width=MIN_LOG_WIDTH,
        seed=RANDOM_SEED,
    )

    safe_name = corpus_name.lower().replace(" ", "_")

    comparison.to_csv(
        output_dir / f"{safe_name}_bic_comparison.csv",
        index=False,
    )

    breakpoints.to_csv(
        output_dir / f"{safe_name}_breakpoints.csv",
        index=False,
    )

    stability.to_csv(
        output_dir / f"{safe_name}_breakpoint_stability.csv",
        index=False,
    )

    plot_segmented_fit(
        fit_df=fit_df,
        model=final_model,
        corpus_name=corpus_name,
        output_path=output_dir / f"{safe_name}_segmented_fit.png",
    )

    plot_bic(
        comparison=comparison,
        corpus_name=corpus_name,
        output_path=output_dir / f"{safe_name}_bic.png",
    )

    classification_supported = final_model.n_segments == 4

    if classification_supported:
        breakpoint_ranks = (
            breakpoints["rank"].astype(int).tolist()
        )

        classified_df = assign_frequency_classes(
            full_df,
            breakpoint_ranks=breakpoint_ranks,
            labels=CLASS_LABELS,
        )

        classified_df.to_csv(
            output_dir / f"{safe_name}_classified.csv",
            index=False,
        )

        class_summary = (
            classified_df.groupby(
                "frequency_class",
                observed=False,
            )
            .agg(
                n_terms=("rank", "size"),
                minimum_frequency=("frequency", "min"),
                maximum_frequency=("frequency", "max"),
                minimum_fpm=("frequency_per_million", "min"),
                maximum_fpm=("frequency_per_million", "max"),
            )
            .reset_index()
        )

        class_summary.to_csv(
            output_dir / f"{safe_name}_class_summary.csv",
            index=False,
        )

        print(
            f"  Four classes assigned and saved for {corpus_name}."
        )
    else:
        classified_df = full_df.copy()
        classified_df["frequency_class"] = pd.NA

        classified_df.to_csv(
            output_dir / f"{safe_name}_unclassified.csv",
            index=False,
        )

        print(
            "  Four classes were not assigned because the selected model "
            f"contains {final_model.n_segments} segment(s), not four."
        )

    return {
        "corpus": corpus_name,
        "n_terms": len(full_df),
        "selected_segments_bic": empirically_selected_segments,
        "final_segments_used": final_model.n_segments,
        "forced_four_segments": FORCE_FOUR_SEGMENTS,
        "classification_supported": classification_supported,
        "bic": final_model.bic,
        "model_status": model_status,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyses = [
        analyse_corpus(
            path=PG19_PATH,
            corpus_name="PG19",
            output_dir=OUTPUT_DIR,
        ),
        analyse_corpus(
            path=PUBMED_PATH,
            corpus_name="PubMed",
            output_dir=OUTPUT_DIR,
        ),
    ]

    summary = pd.DataFrame(analyses)
    summary.to_csv(
        OUTPUT_DIR / "analysis_summary.csv",
        index=False,
    )

    print("\nAnalysis complete.")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()