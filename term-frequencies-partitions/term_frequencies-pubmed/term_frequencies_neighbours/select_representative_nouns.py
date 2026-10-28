from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    SCRIPT_DIR
    / "output_neighbours"
    / "pubmed_terms_with_neighbour_scores.csv"
)

OUTPUT_ALL_PATH = (
    SCRIPT_DIR
    / "pubmed_terms_with_relaxed_representativeness.csv"
)

OUTPUT_REPRESENTATIVE_PATH = (
    SCRIPT_DIR
    / "pubmed_representative_terms_relaxed.csv"
)


# ============================================================
# RELAXED REPRESENTATIVENESS CRITERIA
# ============================================================

# The weighted median partition may differ by no more than one.
MAX_PARTITION_GAP = 1

# At least 30% of the weighted neighbours must belong
# to exactly the same partition.
MIN_SAME_PARTITION_RATIO = 0.6

# At least 60% must belong to the same or an adjacent partition.
MIN_ADJACENT_PARTITION_RATIO = 0.8

# No more than 10% may belong to clearly distant partitions.
MAX_DISTANT_PARTITION_RATIO = 0.15

# Minimum amount of usable neighbour evidence.
MIN_VALID_NEIGHBOURS = 25

# Minimum number of successfully retrieved chunks.
MIN_RETRIEVED_CHUNKS = 15

# Minimum continuous representativeness score.
MIN_REPRESENTATIVENESS_SCORE = 0.55


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Checks that the input CSV contains the required columns."""

    required_columns = {
        "term",
        "frequency",
        "partition",
        "neighbour_score",
        "same_partition_ratio",
        "adjacent_partition_ratio",
        "distant_partition_ratio",
        "n_retrieved_chunks",
        "n_valid_neighbour_types",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "The input CSV is missing the following columns: "
            f"{sorted(missing_columns)}"
        )


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalizes a numeric series to the interval [0, 1].

    Returns zero when all valid values are identical.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.nan,
            index=series.index,
        )

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        numeric_series - minimum
    ) / (
        maximum - minimum
    )


def calculate_representativeness_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Creates a score between approximately 0 and 1.

    Higher values indicate a more representative semantic
    neighbourhood.

    Components:
    - 30% same-partition concentration
    - 35% same-or-adjacent concentration
    - 20% absence of distant neighbours
    - 10% amount of neighbour evidence
    - 5% number of retrieved chunks

    A penalty is applied when neighbour_score differs from partition.
    """

    evidence_score = min_max_normalize(
        np.log1p(df["n_valid_neighbour_types"])
    )

    retrieved_chunks_score = (
        df["n_retrieved_chunks"]
        .clip(lower=0, upper=20)
        / 20
    )

    partition_gap = (
        df["neighbour_score"] - df["partition"]
    ).abs()

    partition_agreement_score = (
        1 - partition_gap / 5
    ).clip(lower=0, upper=1)

    score = (
        0.30 * df["same_partition_ratio"]
        + 0.35 * df["adjacent_partition_ratio"]
        + 0.20 * (1 - df["distant_partition_ratio"])
        + 0.10 * evidence_score
        + 0.05 * retrieved_chunks_score
    )

    # Terms whose neighbour median is farther from their own
    # partition receive a moderate penalty.
    score = score * (
        0.75 + 0.25 * partition_agreement_score
    )

    return score.clip(lower=0, upper=1)


def arrange_output_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Places term and original position at the beginning of the output.

    Any additional columns are retained after the preferred columns.
    """

    preferred_columns = [
        "term",
        "position",
        "frequency",
        "partition",
        "neighbour_score",
        "partition_gap",
        "same_partition_ratio",
        "adjacent_partition_ratio",
        "distant_partition_ratio",
        "neighbour_frequency_weighted_geometric_mean",
        "neighbour_frequency_median",
        "neighbour_frequency_p95",
        "mean_chunk_distance",
        "minimum_chunk_distance",
        "n_retrieved_chunks",
        "n_valid_neighbour_types",
        "partition_1_ratio",
        "partition_2_ratio",
        "partition_3_ratio",
        "partition_4_ratio",
        "partition_5_ratio",
        "partition_6_ratio",
        "representativeness_score",
        "passes_partition_gap",
        "passes_same_partition",
        "passes_adjacent_partition",
        "passes_distant_partition",
        "passes_neighbour_evidence",
        "passes_retrieved_chunks",
        "representative_relaxed",
    ]

    existing_preferred_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_preferred_columns
    ]

    return df[
        existing_preferred_columns + remaining_columns
    ]


# ============================================================
# MAIN PROCESS
# ============================================================

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "The input CSV was not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    validate_columns(df)

    # Preserve the original row position from the input CSV.
    #
    # This must be done before filtering or sorting so that:
    # datum = position 1
    # expression = position 2
    # activity = position 3
    # etc.
    #
    # If a later row is removed, the original positions are
    # preserved and are not recalculated.
    if "position" in df.columns:
        df = df.drop(columns=["position"])

    df.insert(
        1,
        "position",
        np.arange(
            1,
            len(df) + 1,
            dtype=np.int64,
        ),
    )

    numeric_columns = [
        "position",
        "frequency",
        "partition",
        "neighbour_score",
        "same_partition_ratio",
        "adjacent_partition_ratio",
        "distant_partition_ratio",
        "n_retrieved_chunks",
        "n_valid_neighbour_types",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows that cannot be evaluated.
    #
    # Position is deliberately not recalculated after this step.
    df = df.dropna(
        subset=[
            "term",
            "position",
            "partition",
            "neighbour_score",
            "same_partition_ratio",
            "adjacent_partition_ratio",
            "distant_partition_ratio",
            "n_retrieved_chunks",
            "n_valid_neighbour_types",
        ]
    ).copy()

    df["position"] = df["position"].astype(int)
    df["partition"] = df["partition"].astype(int)

    # Distance between the term partition and the weighted
    # median neighbour partition.
    df["partition_gap"] = (
        df["neighbour_score"] - df["partition"]
    ).abs()

    # Continuous ranking score.
    df["representativeness_score"] = (
        calculate_representativeness_score(df)
    )

    # Individual criteria are retained so you can see why
    # a term was accepted or rejected.
    df["passes_partition_gap"] = (
        df["partition_gap"] <= MAX_PARTITION_GAP
    )

    df["passes_same_partition"] = (
        df["same_partition_ratio"]
        >= MIN_SAME_PARTITION_RATIO
    )

    df["passes_adjacent_partition"] = (
        df["adjacent_partition_ratio"]
        >= MIN_ADJACENT_PARTITION_RATIO
    )

    df["passes_distant_partition"] = (
        df["distant_partition_ratio"]
        <= MAX_DISTANT_PARTITION_RATIO
    )

    df["passes_neighbour_evidence"] = (
        df["n_valid_neighbour_types"]
        >= MIN_VALID_NEIGHBOURS
    )

    df["passes_retrieved_chunks"] = (
        df["n_retrieved_chunks"]
        >= MIN_RETRIEVED_CHUNKS
    )

    # A term is representative only when it passes every
    # moderate minimum criterion.
    df["representative_relaxed"] = (
        df["passes_partition_gap"]
        & df["passes_same_partition"]
        & df["passes_adjacent_partition"]
        & df["passes_distant_partition"]
        & df["passes_neighbour_evidence"]
        & df["passes_retrieved_chunks"]
        & (
            df["representativeness_score"]
            >= MIN_REPRESENTATIVENESS_SCORE
        )
    )

    # Preserve the original frequency ranking in the complete output.
    df = df.sort_values(
        by="position",
        ascending=True,
    ).reset_index(drop=True)

    # Select only representative terms.
    representatives = df[
        df["representative_relaxed"]
    ].copy()

    # Preserve the original frequency ranking among selected terms.
    #
    # Positions may contain gaps because rejected terms are not
    # included in this file.
    representatives = representatives.sort_values(
        by="position",
        ascending=True,
    ).reset_index(drop=True)

    # Place term, position, frequency and partition first.
    df = arrange_output_columns(df)
    representatives = arrange_output_columns(
        representatives
    )

    df.to_csv(
        OUTPUT_ALL_PATH,
        index=False,
    )

    representatives.to_csv(
        OUTPUT_REPRESENTATIVE_PATH,
        index=False,
    )

    print("[+] Analysis completed.")
    print(f"[+] Total terms analysed: {len(df):,}")

    print(
        "[+] Relaxed representative terms: "
        f"{len(representatives):,}"
    )

    if len(df) > 0:
        representative_percentage = (
            len(representatives) / len(df)
        ) * 100

        print(
            "[+] Percentage selected: "
            f"{representative_percentage:.2f}%"
        )

    print(
        "\n[+] Representative terms by partition:"
    )

    if representatives.empty:
        print(
            "No terms passed the relaxed criteria."
        )
    else:
        print(
            representatives["partition"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print(
        "\n[+] Complete output:\n"
        f"{OUTPUT_ALL_PATH}"
    )

    print(
        "\n[+] Representative terms only:\n"
        f"{OUTPUT_REPRESENTATIVE_PATH}"
    )

    if not representatives.empty:
        preview_columns = [
            "term",
            "position",
            "frequency",
            "partition",
            "neighbour_score",
            "same_partition_ratio",
            "adjacent_partition_ratio",
            "distant_partition_ratio",
            "n_valid_neighbour_types",
            "representativeness_score",
        ]

        print(
            "\n[+] First representative terms "
            "according to their original positions:"
        )

        preview = (
            representatives
            .groupby(
                "partition",
                group_keys=False,
            )
            .head(10)
        )

        print(
            preview[preview_columns]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

