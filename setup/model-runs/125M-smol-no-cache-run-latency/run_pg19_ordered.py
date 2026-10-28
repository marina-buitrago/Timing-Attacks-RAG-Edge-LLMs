#!/usr/bin/env python3
"""
PG19 RAG timing experiment.

This script runs the PG19 experiment with per-query RAG,
retrieval, and LLM component latencies. Queries are grouped by semantic
frequency, query length, and biomedical category.

It runs two separate benchmarks:

1. ChromaDB/HNSW retrieval-only latency
   - Query embeddings are computed before timing starts.
   - Only collection.query(...) is timed.
   - Outputs are saved under: chroma_hnsw_retrieval_only/

2. Full RAG end-to-end latency with component timing
   - Raw query text -> embedding -> retrieval -> prompt construction -> LLM generation.
   - The end-to-end RAG latency is timed.
   - The retrieval component and LLM component are also timed separately.
   - Outputs are saved under: full_rag_end_to_end/

The query order is preserved exactly as loaded for every iteration.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


DEFAULT_K_VALUES = [5, 20, 50, 100]
DEFAULT_RETRIEVAL_ITERATIONS = 30
DEFAULT_RAG_ITERATIONS = 2
DEFAULT_SEED = 42

DEFAULT_QUERY_FILE = "pg19-queries.json"
DEFAULT_DATABASE_DIR = "workspace/pg19"
DEFAULT_COLLECTION_NAME = "pg19"
DEFAULT_OUTPUT_DIR = "experiment_4/results_pg19_combined"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

DEFAULT_FREQUENCY_ORDER = [
    "very high frequency",
    "high frequency",
    "moderate frequency",
    "low frequency",
    "very low frequency",
    "term not present",
]
DEFAULT_LENGTH_ORDER = ["short", "medium", "long"]
DEFAULT_CATEGORY_ORDER = [
    "Person or social role",
    "Object or artifact",
    "Place or setting",
    "Institution or collective",
    "Abstract concept",
]

FREQUENCY_LABELS = {
    "very high frequency": "Very high",
    "high frequency": "High",
    "moderate frecuency": "Moderate",
    "moderate frequency": "Moderate",
    "low frequency": "Low",
    "very low frequency": "Very low",
    "term not present": "Term not present",
}

LENGTH_LABELS = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
    "corta": "Short",
    "media": "Medium",
    "larga": "Long",
}

PROMPT_TEMPLATE = """You are an analytical assistant. Answer the question using only the retrieved context.

If there is no relevant context, explain that there is not enough relevant information in the retrieved context to answer this question.
If the context is limited, briefly present the related information and state what cannot be answered.
If it is sufficient, give a clear evidence-based answer.
If it is abundant, give a detailed answer without repetition.
Do not use external knowledge or invent missing information.

Context:
{context}

Question:
{input}

Answer:"""


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    frequency: str
    length: str
    category: str
    text: str


def parse_csv_values(value: str, cast=str) -> List[Any]:
    if value is None:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_") or "value"


def label_frequency(key: str) -> str:
    return FREQUENCY_LABELS.get(key, key.replace("_", " ").title())


def label_length(key: str) -> str:
    return LENGTH_LABELS.get(key, key.replace("_", " ").title())


def label_category(key: str) -> str:
    return str(key).strip().capitalize()


def ordered_keys(available: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
    available_set = set(available)
    ordered = [key for key in preferred_order if key in available_set]
    ordered.extend(sorted(key for key in available_set if key not in ordered))
    return ordered


def load_query_dataset(
    query_file: Path,
    frequency_keys_arg: str,
    length_keys_arg: str,
    category_keys_arg: str,
    max_queries_per_cell: int,
    seed: Optional[int],
) -> Tuple[List[QueryItem], List[str], List[str], List[str], Dict[str, Any]]:
    with query_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("The query dataset must be a JSON object at the top level.")

    if frequency_keys_arg.lower() == "auto":
        frequency_keys = ordered_keys(
            [key for key, value in data.items() if isinstance(value, dict)],
            DEFAULT_FREQUENCY_ORDER,
        )
    else:
        frequency_keys = parse_csv_values(frequency_keys_arg, str)

    available_lengths = set()
    for frequency in frequency_keys:
        if frequency in data and isinstance(data[frequency], dict):
            available_lengths.update(data[frequency].keys())

    if length_keys_arg.lower() == "auto":
        length_keys = ordered_keys(available_lengths, DEFAULT_LENGTH_ORDER)
    else:
        length_keys = parse_csv_values(length_keys_arg, str)

    available_categories = set()
    for frequency in frequency_keys:
        for length in length_keys:
            cell = data.get(frequency, {}).get(length, {})
            if isinstance(cell, dict):
                available_categories.update(cell.keys())

    if category_keys_arg.lower() == "auto":
        category_keys = ordered_keys(available_categories, DEFAULT_CATEGORY_ORDER)
    else:
        category_keys = parse_csv_values(category_keys_arg, str)

    if not category_keys:
        raise ValueError(
            "No query categories were found. Expected each frequency/length cell "
            "to contain category objects."
        )

    rng = random.Random(seed)
    items: List[QueryItem] = []
    counts: Dict[str, Any] = {}

    for frequency in frequency_keys:
        if frequency not in data:
            raise KeyError(f"Frequency key not found in query dataset: {frequency}")
        if not isinstance(data[frequency], dict):
            raise ValueError(f"Frequency key {frequency} must contain an object of length groups.")

        counts[frequency] = {}
        for length in length_keys:
            if length not in data[frequency]:
                raise KeyError(f"Length key not found for {frequency}: {length}")
            category_groups = data[frequency][length]
            if not isinstance(category_groups, dict):
                raise ValueError(
                    f"Queries for {frequency}/{length} must be an object grouped by category."
                )

            counts[frequency][length] = {}
            for category in category_keys:
                if category not in category_groups:
                    raise KeyError(f"Category key not found for {frequency}/{length}: {category}")
                queries = category_groups[category]
                if not isinstance(queries, list):
                    raise ValueError(
                        f"Queries for {frequency}/{length}/{category} must be a list."
                    )

                selected_queries = list(queries)
                if max_queries_per_cell and max_queries_per_cell > 0:
                    if len(selected_queries) > max_queries_per_cell:
                        selected_queries = rng.sample(selected_queries, max_queries_per_cell)

                counts[frequency][length][category] = len(selected_queries)
                for index, text in enumerate(selected_queries):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    query_id = (
                        f"{safe_name(frequency)}__{safe_name(length)}__"
                        f"{safe_name(category)}__{index:04d}"
                    )
                    items.append(
                        QueryItem(
                            query_id=query_id,
                            frequency=frequency,
                            length=length,
                            category=category,
                            text=text,
                        )
                    )

    if not items:
        raise ValueError("No valid queries were loaded from the dataset.")

    return items, frequency_keys, length_keys, category_keys, counts


def resolve_chroma_collection(client: chromadb.PersistentClient, preferred_name: str):
    try:
        return client.get_collection(name=preferred_name), preferred_name
    except Exception as original_error:
        collections = client.list_collections()
        names = []
        for collection in collections:
            if hasattr(collection, "name"):
                names.append(collection.name)
            else:
                names.append(str(collection))

        if len(names) == 1:
            fallback_name = names[0]
            print(
                f"[!] Collection '{preferred_name}' was not found. "
                f"Using the only available collection instead: '{fallback_name}'"
            )
            return client.get_collection(name=fallback_name), fallback_name

        raise RuntimeError(
            f"Could not load Chroma collection '{preferred_name}'. "
            f"Available collections: {names}"
        ) from original_error


def initialize_times(frequency_keys: Sequence[str], length_keys: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    return {frequency: {length: [] for length in length_keys} for frequency in frequency_keys}


def compute_results(
    times_by_config: Dict[str, Dict[str, List[float]]],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for frequency in frequency_keys:
        results[frequency] = {}
        for length in length_keys:
            raw_times = list(times_by_config[frequency][length])
            if raw_times:
                outlier_limit = float(np.percentile(raw_times, 99))
                clean_times = [value for value in raw_times if value <= outlier_limit]
            else:
                outlier_limit = float("nan")
                clean_times = []

            if clean_times:
                mean_value = float(np.mean(clean_times))
                median_value = float(np.median(clean_times))
                std_value = float(np.std(clean_times))
            else:
                mean_value = float("nan")
                median_value = float("nan")
                std_value = float("nan")

            results[frequency][length] = {
                "raw_times": raw_times,
                "clean_times": clean_times,
                "tiempos_sucios": raw_times,
                "tiempos_limpios": clean_times,
                "mean": mean_value,
                "median": median_value,
                "std": std_value,
                "media": mean_value,
                "mediana": median_value,
                "outlier_limit_p99": outlier_limit,
                "outliers": len(raw_times) - len(clean_times),
                "sample_size": len(clean_times),
            }

    return results


def save_summary_csv(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    benchmark_name: str,
    latency_column_label: str = "latency_ms",
) -> pd.DataFrame:
    rows = []
    for k, results in results_by_k.items():
        for frequency in frequency_keys:
            for length in length_keys:
                result = results[frequency][length]
                rows.append(
                    {
                        "benchmark": benchmark_name,
                        "k": k,
                        "frequency_key": frequency,
                        "frequency": label_frequency(frequency),
                        "length_key": length,
                        "length": label_length(length),
                        "latency_metric": latency_column_label,
                        "mean_latency_ms": result["mean"],
                        "median_latency_ms": result["median"],
                        "std_latency_ms": result["std"],
                        "filtered_outliers": result["outliers"],
                        "outlier_limit_p99_ms": result["outlier_limit_p99"],
                        "sample_size": result["sample_size"],
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Summary CSV saved: {csv_path}")
    print(df.to_string(index=False))
    return df


def save_raw_csv(raw_rows: List[Dict[str, Any]], output_dir: Path, filename: str = "raw_latencies.csv") -> None:
    raw_path = output_dir / filename
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    print(f"[+] Raw latency CSV saved: {raw_path}")


def save_ttest_csv(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    benchmark_name: str,
) -> None:
    high_key = "high_frequency" if "high_frequency" in frequency_keys else ("alta_frecuencia" if "alta_frecuencia" in frequency_keys else frequency_keys[0])
    low_key = "low_frequency" if "low_frequency" in frequency_keys else ("baja_frecuencia" if "baja_frecuencia" in frequency_keys else frequency_keys[-1])
    short_key = "short" if "short" in length_keys else ("corta" if "corta" in length_keys else length_keys[0])
    long_key = "long" if "long" in length_keys else ("larga" if "larga" in length_keys else length_keys[-1])

    rows = []
    for k, results in results_by_k.items():
        high_short = results[high_key][short_key]["clean_times"]
        low_long = results[low_key][long_key]["clean_times"]

        if len(high_short) >= 2 and len(low_long) >= 2:
            t_stat, p_value = stats.ttest_ind(high_short, low_long, equal_var=False)
        else:
            t_stat, p_value = float("nan"), float("nan")

        mean_high_short = float(np.mean(high_short)) if high_short else float("nan")
        mean_low_long = float(np.mean(low_long)) if low_long else float("nan")
        diff = abs(mean_high_short - mean_low_long)
        denominator = min(mean_high_short, mean_low_long)
        diff_pct = (diff / denominator * 100.0) if denominator and not np.isnan(denominator) else float("nan")

        rows.append(
            {
                "benchmark": benchmark_name,
                "k": k,
                "comparison_a": f"{high_key}/{short_key}",
                "comparison_b": f"{low_key}/{long_key}",
                "mean_a_ms": mean_high_short,
                "mean_b_ms": mean_low_long,
                "absolute_difference_ms": diff,
                "relative_difference_pct": diff_pct,
                "welch_t_statistic": float(t_stat),
                "p_value": float(p_value),
            }
        )

    path = output_dir / "extreme_comparison_ttest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[+] Extreme comparison t-test CSV saved: {path}")


def maybe_kdeplot(data: Sequence[float], ax, label: Optional[str] = None, fill: bool = True, **kwargs) -> None:
    values = [float(value) for value in data if value is not None and not np.isnan(value)]
    if len(values) < 2 or len(set(values)) < 2:
        ax.hist(values, bins=min(10, max(1, len(values))), alpha=0.5, label=label)
        return
    sns.kdeplot(data=values, ax=ax, fill=fill, label=label, **kwargs)


def generate_density_plots(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    title_prefix: str,
) -> None:
    for k, results in results_by_k.items():
        fig, axes = plt.subplots(
            len(frequency_keys),
            len(length_keys),
            figsize=(6 * len(length_keys), 4.5 * len(frequency_keys)),
            squeeze=False,
        )
        fig.suptitle(f"{title_prefix} Latency Distribution - k={k}", fontsize=16, fontweight="bold")

        for i, frequency in enumerate(frequency_keys):
            for j, length in enumerate(length_keys):
                ax = axes[i][j]
                data = results[frequency][length]["clean_times"]
                maybe_kdeplot(data, ax=ax, fill=True, color="blue", alpha=0.6)

                mean_value = results[frequency][length]["mean"]
                if not np.isnan(mean_value):
                    ax.axvline(
                        mean_value,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean: {mean_value:.2f} ms",
                    )

                ax.set_title(f"{label_frequency(frequency)} Frequency - {label_length(length)}")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"density_latencies_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Density plot saved: {path}")


def generate_overlaid_density_plots(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    title_prefix: str,
) -> None:
    for k in k_values:
        fig, axes = plt.subplots(1, len(length_keys), figsize=(6 * len(length_keys), 6), squeeze=False)
        fig.suptitle(f"{title_prefix} Overlaid Latency Density - Top-k = {k}", fontsize=16, fontweight="bold")

        for j, length in enumerate(length_keys):
            ax = axes[0][j]
            for frequency in frequency_keys:
                data = results_by_k[k][frequency][length]["clean_times"]
                maybe_kdeplot(data, ax=ax, fill=True, alpha=0.35, label=label_frequency(frequency))

            ax.set_title(f"Query length: {label_length(length)}", fontsize=14)
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Density")
            ax.legend(title="Frequency", loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"density_overlaid_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Overlaid density plot saved: {path}")


def generate_boxplots(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    title_prefix: str,
) -> None:
    for k in k_values:
        data = []
        labels = []
        for frequency in frequency_keys:
            for length in length_keys:
                data.append(results_by_k[k][frequency][length]["clean_times"])
                labels.append(f"{label_frequency(frequency)[:4]}\n{label_length(length)[:4]}")

        fig, ax = plt.subplots(figsize=(max(14, len(data) * 1.2), 7))
        sns.boxplot(data=data, ax=ax, palette="Set3", showfliers=True)
        ax.set_xticklabels(labels)
        ax.set_title(f"{title_prefix} Latency Dispersion - Top-k = {k}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Configuration (Frequency - Length)", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"box_whisker_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Boxplot saved: {path}")


def generate_comparative_bar_chart(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    title_prefix: str,
) -> None:
    rows = []
    configs = []
    for frequency in frequency_keys:
        for length in length_keys:
            configs.append((frequency, length))

    for k in k_values:
        for frequency, length in configs:
            rows.append(
                {
                    "k": k,
                    "frequency": label_frequency(frequency),
                    "length": label_length(length),
                    "latency_ms": results_by_k[k][frequency][length]["mean"],
                    "configuration": f"{label_frequency(frequency)[:3]}-{label_length(length)[:3]}",
                }
            )

    df = pd.DataFrame(rows)
    x = np.arange(len(configs))
    width = 0.8 / len(k_values)

    fig, ax = plt.subplots(figsize=(max(16, len(configs) * 1.2), 8))
    for index, k in enumerate(k_values):
        k_data = df[df["k"] == k]
        means = k_data["latency_ms"].values
        offset = (index - (len(k_values) - 1) / 2.0) * width
        bars = ax.bar(x + offset, means, width, label=f"k={k}")
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    labels = [f"{label_frequency(frequency)[:4]}\n{label_length(length)[:4]}" for frequency, length in configs]
    ax.set_xlabel("Configuration (Frequency - Length)", fontsize=12)
    ax.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax.set_title(f"{title_prefix} Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Search depth (k)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "comparative_all_configurations.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparative bar chart saved: {path}")


def generate_heatmaps(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    title_prefix: str,
) -> None:
    for k in k_values:
        matrix = np.zeros((len(frequency_keys), len(length_keys)), dtype=float)
        for i, frequency in enumerate(frequency_keys):
            for j, length in enumerate(length_keys):
                matrix[i, j] = results_by_k[k][frequency][length]["mean"]

        fig, ax = plt.subplots(figsize=(10, max(8, len(frequency_keys) * 1.2)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=[label_length(length) for length in length_keys],
            yticklabels=[label_frequency(frequency) for frequency in frequency_keys],
            ax=ax,
            cbar_kws={"label": "Latency (ms)"},
        )
        ax.set_title(f"{title_prefix} Latency Heatmap - k={k}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Query Length", fontsize=12)
        ax.set_ylabel("Semantic Frequency", fontsize=12)

        plt.tight_layout()
        path = output_dir / f"heatmap_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Heatmap saved: {path}")


def _summary_from_raw(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    latency_column: str,
) -> pd.DataFrame:
    """Summarize raw measurements after applying a p99 filter inside each group."""
    rows: List[Dict[str, Any]] = []
    for group_values, group in df.groupby(list(group_cols), sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        values = pd.to_numeric(group[latency_column], errors="coerce").dropna()
        limit = float(values.quantile(0.99)) if not values.empty else float("nan")
        clean = values[values <= limit] if not values.empty else values
        row = dict(zip(group_cols, group_values))
        row.update(
            {
                "mean_latency_ms": float(clean.mean()) if not clean.empty else float("nan"),
                "median_latency_ms": float(clean.median()) if not clean.empty else float("nan"),
                "std_latency_ms": float(clean.std(ddof=0)) if not clean.empty else float("nan"),
                "sample_size": int(clean.size),
                "filtered_outliers": int(values.size - clean.size),
                "outlier_limit_p99_ms": limit,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_category_outputs(
    raw_rows: List[Dict[str, Any]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    category_keys: Sequence[str],
    title_prefix: str,
    latency_column: str,
) -> None:
    """Create category-only and frequency × length × category summaries/plots."""
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return
    required = {"k", "frequency_key", "length_key", "category_key", latency_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Cannot generate category outputs; missing raw columns: {sorted(missing)}")

    df = df.copy()
    df[latency_column] = pd.to_numeric(df[latency_column], errors="coerce")
    df = df.dropna(subset=[latency_column])

    category_summary = _summary_from_raw(
        df,
        ["k", "category_key", "category"],
        latency_column,
    )
    category_summary.to_csv(output_dir / "summary_by_category.csv", index=False)

    detailed_summary = _summary_from_raw(
        df,
        [
            "k",
            "frequency_key",
            "frequency",
            "length_key",
            "length",
            "category_key",
            "category",
        ],
        latency_column,
    )
    detailed_summary.to_csv(
        output_dir / "summary_by_frequency_length_category.csv",
        index=False,
    )

    detail_dir = output_dir / "category_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        k_df = df[df["k"] == k]
        if k_df.empty:
            continue

        # Overall latency distributions split by biomedical category.
        fig, ax = plt.subplots(figsize=(13, 7))
        sns.boxplot(
            data=k_df,
            x="category",
            y=latency_column,
            hue="category",
            order=[label_category(category) for category in category_keys],
            hue_order=[label_category(category) for category in category_keys],
            showfliers=True,
            legend=False,
            ax=ax,
            palette="Set2",
        )
        ax.set_title(f"{title_prefix} Latency by Category - k={k}", fontweight="bold")
        ax.set_xlabel("Query category")
        ax.set_ylabel("Latency (ms)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = output_dir / f"boxplot_by_category_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Category boxplot saved: {path}")

        # Mean latency by category.
        means = (
            category_summary[category_summary["k"] == k]
            .set_index("category_key")
            .reindex(category_keys)
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(category_keys))
        bars = ax.bar(
            x,
            means["mean_latency_ms"].to_numpy(dtype=float),
            color=sns.color_palette("Set2", len(category_keys)),
        )
        ax.errorbar(
            x,
            means["mean_latency_ms"].to_numpy(dtype=float),
            yerr=means["std_latency_ms"].fillna(0).to_numpy(dtype=float),
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label_category(category) for category in category_keys], rotation=20)
        ax.set_title(f"{title_prefix} Mean Latency by Category - k={k}", fontweight="bold")
        ax.set_xlabel("Query category")
        ax.set_ylabel("Mean latency (ms)")
        ax.grid(True, axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        plt.tight_layout()
        path = output_dir / f"mean_latency_by_category_k{k}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Category mean chart saved: {path}")

        # One heatmap and one distribution plot per category. Each cell is an
        # exact frequency × length × category combination, e.g.
        # Very high × Short × Anatomy.
        for category in category_keys:
            category_label = label_category(category)
            category_df = k_df[k_df["category_key"] == category].copy()
            category_summary_k = detailed_summary[
                (detailed_summary["k"] == k)
                & (detailed_summary["category_key"] == category)
            ]
            matrix = np.full((len(frequency_keys), len(length_keys)), np.nan)
            for i, frequency in enumerate(frequency_keys):
                for j, length in enumerate(length_keys):
                    cell = category_summary_k[
                        (category_summary_k["frequency_key"] == frequency)
                        & (category_summary_k["length_key"] == length)
                    ]
                    if not cell.empty:
                        matrix[i, j] = float(cell.iloc[0]["mean_latency_ms"])

            fig, ax = plt.subplots(figsize=(10, max(7, len(frequency_keys) * 1.0)))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=[label_length(length) for length in length_keys],
                yticklabels=[label_frequency(frequency) for frequency in frequency_keys],
                cbar_kws={"label": "Mean latency (ms)"},
                ax=ax,
            )
            ax.set_title(
                f"{title_prefix}: {category_label} - k={k}",
                fontweight="bold",
            )
            ax.set_xlabel("Query length")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            path = detail_dir / f"heatmap_{safe_name(category)}_k{k}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[+] Category-detail heatmap saved: {path}")

            category_df["configuration"] = (
                category_df["frequency"].astype(str)
                + "\n"
                + category_df["length"].astype(str)
            )
            config_order = [
                f"{label_frequency(frequency)}\n{label_length(length)}"
                for frequency in frequency_keys
                for length in length_keys
            ]
            fig, ax = plt.subplots(figsize=(max(16, len(config_order) * 0.85), 7))
            sns.boxplot(
                data=category_df,
                x="configuration",
                y=latency_column,
                hue="configuration",
                order=config_order,
                hue_order=config_order,
                showfliers=True,
                legend=False,
                ax=ax,
                palette="Set3",
            )
            ax.set_title(
                f"{title_prefix}: Frequency × Length for {category_label} - k={k}",
                fontweight="bold",
            )
            ax.set_xlabel("Frequency and query length")
            ax.set_ylabel("Latency (ms)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            path = detail_dir / f"boxplot_{safe_name(category)}_k{k}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[+] Category-detail boxplot saved: {path}")


def generate_all_outputs(
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    raw_rows: List[Dict[str, Any]],
    output_dir: Path,
    k_values: Sequence[int],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    category_keys: Sequence[str],
    title_prefix: str,
    benchmark_name: str,
    raw_filename: str = "raw_latencies.csv",
    latency_column_label: str = "latency_ms",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_raw_csv(raw_rows, output_dir, filename=raw_filename)
    save_summary_csv(
        results_by_k,
        output_dir,
        frequency_keys,
        length_keys,
        benchmark_name,
        latency_column_label=latency_column_label,
    )
    save_ttest_csv(results_by_k, output_dir, frequency_keys, length_keys, benchmark_name)
    generate_density_plots(results_by_k, output_dir, frequency_keys, length_keys, title_prefix)
    generate_overlaid_density_plots(results_by_k, output_dir, k_values, frequency_keys, length_keys, title_prefix)
    generate_boxplots(results_by_k, output_dir, k_values, frequency_keys, length_keys, title_prefix)
    generate_comparative_bar_chart(results_by_k, output_dir, k_values, frequency_keys, length_keys, title_prefix)
    generate_heatmaps(results_by_k, output_dir, k_values, frequency_keys, length_keys, title_prefix)
    generate_category_outputs(
        raw_rows,
        output_dir,
        k_values,
        frequency_keys,
        length_keys,
        category_keys,
        title_prefix,
        latency_column_label,
    )


def parse_retrieval_include(value: str) -> Optional[List[str]]:
    normalized = value.strip().lower()
    if normalized == "default":
        return None
    if normalized in {"none", "empty"}:
        return []
    return parse_csv_values(value, str)


def warmup_retrieval(collection, embedding: Sequence[float], k: int, include: Optional[List[str]]) -> None:
    kwargs = {"query_embeddings": [list(embedding)], "n_results": k}
    if include is not None:
        kwargs["include"] = include
    collection.query(**kwargs)


def run_retrieval_only_benchmark(
    query_items: Sequence[QueryItem],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    database_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    k_values: Sequence[int],
    num_iterations: int,
    seed: Optional[int],
    include: Optional[List[str]],
    batch_size: int,
) -> Tuple[Dict[int, Dict[str, Dict[str, Dict[str, Any]]]], List[Dict[str, Any]], str]:
    print("\n" + "=" * 70)
    print("STARTING CHROMADB/HNSW RETRIEVAL-ONLY BENCHMARK")
    print("=" * 70)

    client = chromadb.PersistentClient(path=str(database_dir))
    collection, actual_collection_name = resolve_chroma_collection(client, collection_name)
    print(f"[+] Loaded collection: {actual_collection_name}")
    print(f"[+] Total vectors in collection: {collection.count()}")

    print(f"[*] Loading embedding model for pre-encoding: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    texts = [item.text for item in query_items]
    print(f"[*] Pre-encoding {len(texts)} queries before timing retrieval...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    print("[*] Warming up Chroma/HNSW retrieval...")
    warmup_retrieval(collection, embeddings[0].tolist(), k=1, include=include)
    print("[+] Retrieval warm-up complete")

    rng = random.Random(seed)
    results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    raw_rows: List[Dict[str, Any]] = []

    gc.disable()
    try:
        for k in k_values:
            print("\n" + "-" * 70)
            print(f"Retrieval-only benchmark for k={k}")
            print("-" * 70)
            gc.collect()
            time.sleep(2)

            times_by_config = initialize_times(frequency_keys, length_keys)

            for iteration in range(num_iterations):
                # Preserve the exact query order produced by load_query_dataset.
                # No shuffling is performed.
                order = list(range(len(query_items)))
                print(f"    -> Iteration {iteration + 1}/{num_iterations} in original query order")

                for position, query_index in enumerate(order):
                    item = query_items[query_index]
                    embedding = embeddings[query_index].tolist()
                    kwargs = {"query_embeddings": [embedding], "n_results": k}
                    if include is not None:
                        kwargs["include"] = include

                    start = time.perf_counter()
                    collection.query(**kwargs)
                    end = time.perf_counter()

                    retrieval_latency_ms = (end - start) * 1000.0
                    times_by_config[item.frequency][item.length].append(retrieval_latency_ms)
                    raw_rows.append(
                        {
                            "benchmark": "chroma_hnsw_retrieval_only",
                            "k": k,
                            "iteration": iteration + 1,
                            "position_in_shuffled_iteration": position + 1,
                            "query_id": item.query_id,
                            "frequency_key": item.frequency,
                            "frequency": label_frequency(item.frequency),
                            "length_key": item.length,
                            "length": label_length(item.length),
                            "category_key": item.category,
                            "category": label_category(item.category),
                            "retrieval_latency_ms": retrieval_latency_ms,
                            "latency_ms": retrieval_latency_ms,
                            "query_text": item.text,
                        }
                    )

                    if (position + 1) % 50 == 0:
                        print(f"        Progress: {position + 1}/{len(query_items)} queries", flush=True)

            results_by_k[k] = compute_results(times_by_config, frequency_keys, length_keys)
    finally:
        gc.enable()
        gc.collect()

    return results_by_k, raw_rows, actual_collection_name


def configure_llm(model_id: str, max_new_tokens: int) -> HuggingFacePipeline:
    print(f"[*] Loading LLM: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    text_generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        return_full_text=False,
        truncation=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipe)
    print("[+] LLM loaded")
    return llm


def format_documents(docs: Sequence[Any]) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        content = getattr(doc, "page_content", "")
        parts.append(f"[Document {index}]\n{content}")
    return "\n\n".join(parts)


def run_single_rag_with_component_timing(
    retriever: Any,
    llm: HuggingFacePipeline,
    prompt: PromptTemplate,
    query_text: str,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query_text)
    else:
        docs = retriever.get_relevant_documents(query_text)
    retrieval_end = time.perf_counter()

    prompt_start = time.perf_counter()
    context = format_documents(docs)
    formatted_prompt = prompt.format(context=context, input=query_text)
    prompt_end = time.perf_counter()

    llm_start = time.perf_counter()
    answer = llm.invoke(formatted_prompt)
    llm_end = time.perf_counter()

    total_end = time.perf_counter()

    retrieval_latency_ms = (retrieval_end - retrieval_start) * 1000.0
    prompt_formatting_latency_ms = (prompt_end - prompt_start) * 1000.0
    llm_latency_ms = (llm_end - llm_start) * 1000.0
    rag_end_to_end_latency_ms = (total_end - total_start) * 1000.0
    measured_component_sum_ms = retrieval_latency_ms + prompt_formatting_latency_ms + llm_latency_ms
    overhead_latency_ms = rag_end_to_end_latency_ms - measured_component_sum_ms

    return {
        "answer": answer,
        "num_retrieved_docs": len(docs),
        "context_num_chars": len(context),
        "prompt_num_chars": len(formatted_prompt),
        "retrieval_latency_ms": retrieval_latency_ms,
        "prompt_formatting_latency_ms": prompt_formatting_latency_ms,
        "llm_latency_ms": llm_latency_ms,
        "rag_end_to_end_latency_ms": rag_end_to_end_latency_ms,
        "component_sum_latency_ms": measured_component_sum_ms,
        "overhead_latency_ms": overhead_latency_ms,
    }


def run_full_rag_benchmark(
    query_items: Sequence[QueryItem],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    database_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    llm_model_name: str,
    k_values: Sequence[int],
    num_iterations: int,
    seed: Optional[int],
    max_new_tokens: int,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    List[Dict[str, Any]],
]:
    print("\n" + "=" * 70)
    print("STARTING FULL RAG END-TO-END BENCHMARK")
    print("=" * 70)

    llm = configure_llm(llm_model_name, max_new_tokens=max_new_tokens)

    print(f"[*] Connecting LangChain Chroma to: {database_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(database_dir),
        embedding_function=embeddings,
    )

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    rng = random.Random(seed)

    rag_results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    retrieval_component_results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    llm_component_results_by_k: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    raw_rows: List[Dict[str, Any]] = []

    gc.disable()
    try:
        for k in k_values:
            print("\n" + "-" * 70)
            print(f"Full RAG benchmark for k={k}")
            print("-" * 70)

            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

            print("[*] Warming up full RAG components...")
            run_single_rag_with_component_timing(
                retriever=retriever,
                llm=llm,
                prompt=prompt,
                query_text="warmup query to load the RAG pipeline",
            )
            print("[+] Full RAG warm-up complete")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)

            rag_times_by_config = initialize_times(frequency_keys, length_keys)
            retrieval_times_by_config = initialize_times(frequency_keys, length_keys)
            llm_times_by_config = initialize_times(frequency_keys, length_keys)

            for iteration in range(num_iterations):
                # Preserve the exact query order produced by load_query_dataset.
                # No shuffling is performed.
                order = list(range(len(query_items)))
                print(f"    -> Iteration {iteration + 1}/{num_iterations} in original query order")

                for position, query_index in enumerate(order):
                    item = query_items[query_index]
                    metrics = run_single_rag_with_component_timing(
                        retriever=retriever,
                        llm=llm,
                        prompt=prompt,
                        query_text=item.text,
                    )

                    rag_latency_ms = metrics["rag_end_to_end_latency_ms"]
                    retrieval_latency_ms = metrics["retrieval_latency_ms"]
                    llm_latency_ms = metrics["llm_latency_ms"]

                    rag_times_by_config[item.frequency][item.length].append(rag_latency_ms)
                    retrieval_times_by_config[item.frequency][item.length].append(retrieval_latency_ms)
                    llm_times_by_config[item.frequency][item.length].append(llm_latency_ms)

                    answer = metrics.get("answer", "")

                    raw_rows.append(
                        {
                            "benchmark": "full_rag_end_to_end",
                            "k": k,
                            "iteration": iteration + 1,
                            "position_in_shuffled_iteration": position + 1,
                            "query_id": item.query_id,
                            "frequency_key": item.frequency,
                            "frequency": label_frequency(item.frequency),
                            "length_key": item.length,
                            "length": label_length(item.length),
                            "category_key": item.category,
                            "category": label_category(item.category),
                            "query_text": item.text,
                            "rag_end_to_end_latency_ms": rag_latency_ms,
                            "retrieval_latency_ms": retrieval_latency_ms,
                            "llm_latency_ms": llm_latency_ms,
                            "prompt_formatting_latency_ms": metrics["prompt_formatting_latency_ms"],
                            "component_sum_latency_ms": metrics["component_sum_latency_ms"],
                            "overhead_latency_ms": metrics["overhead_latency_ms"],
                            "latency_ms": rag_latency_ms,
                            "num_retrieved_docs": metrics["num_retrieved_docs"],
                            "context_num_chars": metrics["context_num_chars"],
                            "prompt_num_chars": metrics["prompt_num_chars"],
                            "answer_num_chars": len(str(answer)),
                            "answer_preview": str(answer).replace("\n", " ")[:250],
                        }
                    )

                    if (position + 1) % 50 == 0:
                        print(f"        Progress: {position + 1}/{len(query_items)} queries", flush=True)

            rag_results_by_k[k] = compute_results(rag_times_by_config, frequency_keys, length_keys)
            retrieval_component_results_by_k[k] = compute_results(retrieval_times_by_config, frequency_keys, length_keys)
            llm_component_results_by_k[k] = compute_results(llm_times_by_config, frequency_keys, length_keys)
    finally:
        gc.enable()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rag_results_by_k, retrieval_component_results_by_k, llm_component_results_by_k, raw_rows


def save_rag_component_summary(
    raw_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return

    group_cols = [
        "k",
        "frequency_key",
        "frequency",
        "length_key",
        "length",
        "category_key",
        "category",
    ]
    metrics = [
        "rag_end_to_end_latency_ms",
        "retrieval_latency_ms",
        "llm_latency_ms",
        "prompt_formatting_latency_ms",
        "overhead_latency_ms",
    ]

    summary = (
        df.groupby(group_cols)[metrics]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    path = output_dir / "rag_component_summary_by_configuration.csv"
    summary.to_csv(path, index=False)
    print(f"[+] RAG component summary CSV saved: {path}")


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    query_counts: Dict[str, Any],
    frequency_keys: Sequence[str],
    length_keys: Sequence[str],
    category_keys: Sequence[str],
    actual_collection_name: Optional[str] = None,
) -> None:
    config = {
        "arguments": vars(args),
        "frequency_keys_used": list(frequency_keys),
        "length_keys_used": list(length_keys),
        "category_keys_used": list(category_keys),
        "query_counts": query_counts,
        "actual_collection_name": actual_collection_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"[+] Run configuration saved: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PG19 RAG timing benchmarks.")
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to the parent of this script's directory.")
    parser.add_argument("--queries", default=DEFAULT_QUERY_FILE, help="Path to the grouped query JSON relative to project root.")
    parser.add_argument("--database", default=DEFAULT_DATABASE_DIR, help="Path to Chroma database directory relative to project root.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_DIR, help="Output directory relative to project root.")
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)), help="Comma-separated k values.")
    parser.add_argument("--retrieval-iterations", type=int, default=DEFAULT_RETRIEVAL_ITERATIONS)
    parser.add_argument("--rag-iterations", type=int, default=DEFAULT_RAG_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used only for optional query subsampling. Query execution order is preserved.")
    parser.add_argument("--mode", choices=["both", "retrieval", "rag"], default="both")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--frequency-keys", default="auto", help="Comma-separated keys or 'auto'.")
    parser.add_argument("--length-keys", default="auto", help="Comma-separated keys or 'auto'.")
    parser.add_argument("--category-keys", default="auto", help="Comma-separated category keys or 'auto'.")
    parser.add_argument(
        "--max-queries-per-cell",
        type=int,
        default=0,
        help="Optional cap per frequency/length/category cell. 0 means use all queries.",
    )
    parser.add_argument(
        "--retrieval-include",
        default="distances",
        help="Chroma include list for retrieval-only queries. Use 'default' to reproduce Chroma defaults.",
    )
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = build_arg_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    if args.project_root is None:
        project_root = script_path.parents[1]
    else:
        project_root = Path(args.project_root).expanduser().resolve()

    query_file = resolve_path(project_root, args.queries)
    database_dir = resolve_path(project_root, args.database)
    output_root = resolve_path(project_root, args.output_root)
    k_values = parse_csv_values(args.k_values, int)
    seed = None if args.seed == -1 else args.seed
    retrieval_include = parse_retrieval_include(args.retrieval_include)

    print("=" * 70)
    print("PG19 RAG TIMING EXPERIMENT")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Query file: {query_file}")
    print(f"Database directory: {database_dir}")
    print(f"Preferred collection name: {args.collection_name}")
    print(f"Output root: {output_root}")
    print(f"K values: {k_values}")
    print(f"Mode: {args.mode}")
    print(f"Random seed: {'non-deterministic' if seed is None else seed}")
    print(f"Retrieval-only iterations: {args.retrieval_iterations}")
    print(f"Full RAG iterations: {args.rag_iterations}")

    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    if not database_dir.exists():
        raise FileNotFoundError(f"Chroma database directory not found: {database_dir}")

    query_items, frequency_keys, length_keys, category_keys, query_counts = load_query_dataset(
        query_file=query_file,
        frequency_keys_arg=args.frequency_keys,
        length_keys_arg=args.length_keys,
        category_keys_arg=args.category_keys,
        max_queries_per_cell=args.max_queries_per_cell,
        seed=seed,
    )

    print("\nLoaded query dataset:")
    print(f"  Frequency keys: {frequency_keys}")
    print(f"  Length keys: {length_keys}")
    print(f"  Category keys: {category_keys}")
    print(f"  Total queries used: {len(query_items)}")
    for frequency in frequency_keys:
        for length in length_keys:
            for category in category_keys:
                print(
                    f"    {frequency}/{length}/{category}: "
                    f"{query_counts[frequency][length][category]}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    actual_collection_name = args.collection_name

    if args.mode in {"both", "retrieval"}:
        retrieval_output_dir = output_root / "chroma_hnsw_retrieval_only"
        retrieval_results, retrieval_raw_rows, actual_collection_name = run_retrieval_only_benchmark(
            query_items=query_items,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            database_dir=database_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            k_values=k_values,
            num_iterations=args.retrieval_iterations,
            seed=seed,
            include=retrieval_include,
            batch_size=args.embedding_batch_size,
        )
        generate_all_outputs(
            results_by_k=retrieval_results,
            raw_rows=retrieval_raw_rows,
            output_dir=retrieval_output_dir,
            k_values=k_values,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            category_keys=category_keys,
            title_prefix="ChromaDB/HNSW Retrieval-only",
            benchmark_name="chroma_hnsw_retrieval_only",
            raw_filename="raw_retrieval_latencies.csv",
            latency_column_label="retrieval_latency_ms",
        )

    if args.mode in {"both", "rag"}:
        rag_output_dir = output_root / "full_rag_end_to_end"
        (
            rag_results,
            rag_retrieval_component_results,
            rag_llm_component_results,
            rag_raw_rows,
        ) = run_full_rag_benchmark(
            query_items=query_items,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            database_dir=database_dir,
            collection_name=actual_collection_name,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            k_values=k_values,
            num_iterations=args.rag_iterations,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
        )

        generate_all_outputs(
            results_by_k=rag_results,
            raw_rows=rag_raw_rows,
            output_dir=rag_output_dir,
            k_values=k_values,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            category_keys=category_keys,
            title_prefix="Full RAG End-to-end",
            benchmark_name="full_rag_end_to_end",
            raw_filename="per_query_rag_llm_latencies.csv",
            latency_column_label="rag_end_to_end_latency_ms",
        )

        # Additional summaries and graphs specifically for the component timings.
        save_rag_component_summary(rag_raw_rows, rag_output_dir)

        rag_retrieval_component_dir = output_root / "full_rag_retrieval_component"
        generate_all_outputs(
            results_by_k=rag_retrieval_component_results,
            raw_rows=rag_raw_rows,
            output_dir=rag_retrieval_component_dir,
            k_values=k_values,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            category_keys=category_keys,
            title_prefix="Full RAG Retrieval Component",
            benchmark_name="full_rag_retrieval_component",
            raw_filename="per_query_rag_components.csv",
            latency_column_label="retrieval_latency_ms",
        )

        rag_llm_component_dir = output_root / "full_rag_llm_component"
        generate_all_outputs(
            results_by_k=rag_llm_component_results,
            raw_rows=rag_raw_rows,
            output_dir=rag_llm_component_dir,
            k_values=k_values,
            frequency_keys=frequency_keys,
            length_keys=length_keys,
            category_keys=category_keys,
            title_prefix="Full RAG LLM Component",
            benchmark_name="full_rag_llm_component",
            raw_filename="per_query_rag_components.csv",
            latency_column_label="llm_latency_ms",
        )

    write_run_config(
        output_root / "run_config.json",
        args=args,
        query_counts=query_counts,
        frequency_keys=frequency_keys,
        length_keys=length_keys,
        category_keys=category_keys,
        actual_collection_name=actual_collection_name,
    )

    print("\n" + "=" * 70)
    print("PG19 EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results saved under: {output_root}")


if __name__ == "__main__":
    main()
