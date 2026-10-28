import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the two CSV files
# ---------------------------------------------------------
pg19 = pd.read_csv(
    "term_frequencies-pg19/noun_frequencies_pg19.csv"
)

pubmed = pd.read_csv(
    "term_frequencies/noun_frequencies_pubmed.csv"
)

# ---------------------------------------------------------
# 2. Clean the frequency column
# ---------------------------------------------------------
def clean_frequency_data(df):
    df = df.copy()

    df["frequency"] = pd.to_numeric(
        df["frequency"],
        errors="coerce"
    )

    df = df.dropna(subset=["frequency"])
    df = df[df["frequency"] > 0]

    return df


pg19 = clean_frequency_data(pg19)
pubmed = clean_frequency_data(pubmed)

# ---------------------------------------------------------
# 3. Sort nouns by decreasing frequency and calculate rank
# ---------------------------------------------------------
pg19 = pg19.sort_values(
    "frequency",
    ascending=False
).reset_index(drop=True)

pubmed = pubmed.sort_values(
    "frequency",
    ascending=False
).reset_index(drop=True)

pg19["rank"] = pg19.index + 1
pubmed["rank"] = pubmed.index + 1


# ---------------------------------------------------------
# 4. Rank-frequency plot: LINEAR AXES
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

plt.plot(
    pg19["rank"],
    pg19["frequency"],
    label="PG19",
    linewidth=1.8,
    alpha=0.85
)

plt.plot(
    pubmed["rank"],
    pubmed["frequency"],
    label="PubMed",
    linewidth=1.8,
    alpha=0.85
)

plt.xlabel("Noun rank, $r$ (linear scale)")
plt.ylabel("Absolute noun frequency, $f(r)$ (linear scale)")
plt.title(
    "Rank–frequency distribution of nouns in PG19 and PubMed\n"
    "Linear axes"
)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "pg19_pubmed_rank_frequency_linear.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 5. Rank-frequency plot: LOG-LOG AXES
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

plt.plot(
    pg19["rank"],
    pg19["frequency"],
    label="PG19",
    linewidth=1.8,
    alpha=0.85
)

plt.plot(
    pubmed["rank"],
    pubmed["frequency"],
    label="PubMed",
    linewidth=1.8,
    alpha=0.85
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Noun rank, $r$ (log scale)")
plt.ylabel("Absolute noun frequency, $f(r)$ (log scale)")
plt.title(
    "Rank–frequency distribution of nouns in PG19 and PubMed\n"
    "Log–log axes"
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "pg19_pubmed_rank_frequency_loglog.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 6. Calculate the empirical CCDF
# ---------------------------------------------------------
def calculate_ccdf(df):
    """
    Calculate the empirical complementary cumulative distribution:

        P(F >= x)

    where F is noun frequency.
    """

    frequencies = np.sort(
        df["frequency"].to_numpy()
    )

    unique_frequencies, counts = np.unique(
        frequencies,
        return_counts=True
    )

    cumulative_counts = np.cumsum(
        counts[::-1]
    )[::-1]

    ccdf = cumulative_counts / cumulative_counts[0]

    return unique_frequencies, ccdf


pg19_x, pg19_ccdf = calculate_ccdf(pg19)
pubmed_x, pubmed_ccdf = calculate_ccdf(pubmed)


# ---------------------------------------------------------
# 7. CCDF plot: LINEAR AXES
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

plt.plot(
    pg19_x,
    pg19_ccdf,
    label="PG19",
    linewidth=1.8
)

plt.plot(
    pubmed_x,
    pubmed_ccdf,
    label="PubMed",
    linewidth=1.8
)

plt.xlabel("Absolute noun frequency, $x$ (linear scale)")
plt.ylabel(
    r"Proportion of noun types with frequency $\geq x$"
)
plt.title(
    "Empirical frequency distribution of nouns in PG19 and PubMed\n"
    "Linear axes"
)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "pg19_pubmed_frequency_ccdf_linear.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 8. CCDF plot: LOG-LOG AXES
# ---------------------------------------------------------
plt.figure(figsize=(11, 7))

plt.plot(
    pg19_x,
    pg19_ccdf,
    label="PG19",
    linewidth=1.8
)

plt.plot(
    pubmed_x,
    pubmed_ccdf,
    label="PubMed",
    linewidth=1.8
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Absolute noun frequency, $x$ (log scale)")
plt.ylabel(
    r"Proportion of noun types with frequency $\geq x$ (log scale)"
)
plt.title(
    "Empirical frequency distribution of nouns in PG19 and PubMed\n"
    "Log–log axes"
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "pg19_pubmed_frequency_ccdf_loglog.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()