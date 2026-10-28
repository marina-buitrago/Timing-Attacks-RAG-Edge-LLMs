from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classify_frequency_regimes import (
    FORCE_FOUR_SEGMENTS,
    MAX_FIT_POINTS,
    MAX_SEGMENTS,
    MIN_LOG_WIDTH,
    RANDOM_SEED,
    PiecewiseFit,
    fit_piecewise_model,
    load_and_prepare,
    logarithmic_rank_sample,
    predict_from_model,
    select_number_of_segments,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "frequency_regime_results"
OUTPUT_PATH = OUTPUT_DIR / "pg19_pubmed_segmented_fit.png"

CORPORA = [
    (
        "PG19",
        PROJECT_DIR / "term_frequencies-pg19" / "noun_frequencies_pg19.csv",
        "#2878B5",
    ),
    (
        "PubMed",
        PROJECT_DIR / "term_frequencies-pubmed" / "noun_frequencies_pubmed.csv",
        "#E87524",
    ),
]


def fit_corpus(
    corpus_name: str,
    input_path: Path,
) -> tuple[pd.DataFrame, PiecewiseFit]:
    full_df = load_and_prepare(input_path, corpus_name)
    fit_df = logarithmic_rank_sample(full_df, max_points=MAX_FIT_POINTS)

    x = np.log10(fit_df["rank"].to_numpy(dtype=float))
    y = np.log10(fit_df["frequency_per_million"].to_numpy(dtype=float))

    selected_model, _, fitted_models = select_number_of_segments(
        x=x,
        y=y,
        max_segments=MAX_SEGMENTS,
        min_log_width=MIN_LOG_WIDTH,
        seed=RANDOM_SEED,
    )

    model = (
        fitted_models[4]
        if FORCE_FOUR_SEGMENTS
        else selected_model
    )
    return fit_df, model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))

    for corpus_name, input_path, color in CORPORA:
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        fit_df, model = fit_corpus(corpus_name, input_path)
        x = np.log10(fit_df["rank"].to_numpy(dtype=float))
        y = np.log10(fit_df["frequency_per_million"].to_numpy(dtype=float))

        grid_x = np.linspace(x.min(), x.max(), 1500)
        grid_y = predict_from_model(grid_x, model)

        ax.scatter(
            10**x,
            10**y,
            s=8,
            alpha=0.25,
            color=color,
            label=f"{corpus_name}: observed terms",
        )
        ax.plot(
            10**grid_x,
            10**grid_y,
            linewidth=2.5,
            color=color,
            label=f"{corpus_name}: segmented fit ({model.n_segments} segments)",
        )

        for index, breakpoint in enumerate(model.breakpoints_x):
            ax.axvline(
                10**breakpoint,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color=color,
                label=f"{corpus_name}: breakpoints" if index == 0 else None,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noun rank, r (log scale)", fontsize=14)
    ax.set_ylabel(
        "Frequency per million noun tokens (log scale)",
        fontsize=14,
    )
    ax.set_title("PG19 and PubMed Segmented Rank Frequency Model")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()