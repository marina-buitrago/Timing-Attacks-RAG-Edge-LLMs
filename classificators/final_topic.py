"""Leak-free topic inference shared by the *-box-cat-final notebooks.

Identical to revised_topic.py except that every tree ensemble uses 500 trees,
matching the tree count of the frequency experiments.

Why a revision was needed
-------------------------
The original topic notebooks could read the answer from the questions: every
topic is asked with its own templates ("Who is X?" for people, "Where is X?"
for places, ...), so features derived from the query text, such as query length
and its per-noun mean/std, identify the topic without using timing at all.

What this module does instead
-----------------------------
* Predictors come only from what the RAG system reveals. Black-box: end-to-end
  latency and answer length. White-box: additionally the internal latencies and
  the retrieved context size. Query text, query length, prompt size (it contains
  the query) and send position are never used.
* One prediction per target noun: the attacker probes the same target many times
  and combines everything it observes (mean, spread, median, 10th/90th
  percentile per k, the trend over k and the repeat effect).
* Queries were sent in a fixed, topic-blocked order, so each signal is first
  divided by the rolling median of the queries sent around it (a window that
  spans every topic of a block). This removes slow drift of the machine.
* Nested cross-validation over nouns: the model (a fixed grid of four) is chosen
  inside the training nouns only; every noun is tested once; repeated over seeds.
* A permutation test re-runs the whole pipeline on shuffled topics.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

CLASSIFICATORS = Path(__file__).resolve().parent

K_VALUES = (5, 20, 50, 100)
OUTER_SPLITS = 5
INNER_SPLITS = 3
SEEDS = (0, 1, 2)
N_PERMUTATIONS = 50
N_TREES = 500
DRIFT_WINDOW = 101

CATEGORIES = {
    "pg19": (
        "Person or social role",
        "Object or artifact",
        "Place or setting",
        "Institution or collective",
        "Abstract concept",
    ),
    "pubmed": (
        "Disease or condition",
        "Treatment or intervention",
        "Biological entity",
        "Diagnostic method",
        "Anatomy",
    ),
}
DATASET_LABELS = {"pg19": "PG19", "pubmed": "PubMed"}
TITLE_DATASET = {"pg19": "PG19", "pubmed": "Pubmed"}

BLACK_BOX_SOURCES = ("rag_end_to_end_latency_ms", "answer_num_chars")
WHITE_BOX_SOURCES = BLACK_BOX_SOURCES + (
    "retrieval_latency_ms",
    "llm_latency_ms",
    "prompt_formatting_latency_ms",
    "component_sum_latency_ms",
    "overhead_latency_ms",
    "latency_ms",
    "context_num_chars",
)
NEVER_USED = (
    "query_text",
    "query length (any form)",
    "prompt_num_chars (contains the query)",
    "position_in_shuffled_iteration (send order)",
    "query_id / category / frequency fields",
    "cache telemetry columns",
)


# ============================================================
# DATA
# ============================================================
def csv_path(box: str, cache: bool, dataset: str, model: str) -> Path:
    """Each final folder holds its own copy of the per-query CSVs."""
    folder = CLASSIFICATORS / f"{box}-box-cat-final" / ("cache" if cache else "no-cache")
    kind = "cache" if cache else "ordered"
    return folder / f"per_query_rag_llm_latencies_{kind}_{model}_{dataset}.csv"


def sources_for(box: str) -> tuple[str, ...]:
    return WHITE_BOX_SOURCES if box == "white" else BLACK_BOX_SOURCES


def load_source(box: str, cache: bool, dataset: str, model: str) -> pd.DataFrame:
    columns = ["query_id", "category", "k", "iteration", "position_in_shuffled_iteration", *sources_for(box)]
    raw = pd.read_csv(csv_path(box, cache, dataset, model), usecols=columns)
    if set(raw["category"].unique()) != set(CATEGORIES[dataset]):
        raise ValueError(f"Unexpected topics: {sorted(raw['category'].unique())}")
    return raw


def drift_corrected(raw: pd.DataFrame, sources: tuple[str, ...]) -> pd.DataFrame:
    """Divide each signal by the rolling median of the queries sent around it."""
    corrected = raw.copy()
    for source in sources:
        corrected[source] = corrected[source].astype(float)
    for _, block in raw.groupby(["k", "iteration"]):
        block = block.sort_values("position_in_shuffled_iteration")
        for source in sources:
            values = block[source].astype(float)
            trend = values.rolling(DRIFT_WINDOW, center=True, min_periods=20).median()
            corrected.loc[block.index, source] = values / trend
    return corrected


def noun_of(query_ids: pd.Series) -> pd.Series:
    parts = query_ids.str.split("__", expand=True)
    return parts[0] + "__" + parts[2] + "__" + (parts[3].astype(int) // 10).astype(str)


def noun_features(raw: pd.DataFrame, sources: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    """Summarise every probe of a noun into one feature vector."""
    raw = raw.assign(noun=noun_of(raw["query_id"]))
    if raw.groupby("noun")["category"].nunique().max() != 1:
        raise ValueError("A noun has more than one topic label")
    labels = raw.groupby("noun")["category"].first()

    blocks = []
    log_k = np.log(np.asarray(K_VALUES, dtype=float))
    centred_k = log_k - log_k.mean()
    for source in sources:
        values = raw[["noun", "query_id", "k", "iteration"]].assign(v=np.log1p(raw[source].clip(lower=0)))
        grouped = values.groupby(["noun", "k"])["v"]
        stats = pd.concat(
            {
                "mean": grouped.mean(),
                "std": grouped.std(),
                "median": grouped.median(),
                "q10": grouped.quantile(0.10),
                "q90": grouped.quantile(0.90),
            },
            axis=1,
        ).unstack("k")
        stats.columns = [f"{source}__{stat}__k{k}" for stat, k in stats.columns]
        blocks.append(stats)

        means = grouped.mean().unstack("k")[list(K_VALUES)]
        blocks.append(
            pd.Series(
                means.to_numpy() @ centred_k / np.sum(centred_k**2),
                index=means.index,
                name=f"{source}__k_slope",
            )
        )

        repeats = values.pivot_table(index=["noun", "query_id", "k"], columns="iteration", values="v")
        repeat_effect = (repeats[2] - repeats[1]).groupby(["noun", "k"]).mean().unstack("k")
        repeat_effect.columns = [f"{source}__repeat_effect__k{k}" for k in repeat_effect.columns]
        blocks.append(repeat_effect)

    features = pd.concat(blocks, axis=1).sort_index()
    return features, labels.loc[features.index].to_numpy()


# ============================================================
# MODEL
# ============================================================
def candidate_models() -> dict[str, Pipeline]:
    def trees(estimator):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])

    return {
        "extra_trees_narrow": trees(
            ExtraTreesClassifier(N_TREES, max_features=0.1, min_samples_leaf=1, class_weight="balanced", random_state=0)
        ),
        "extra_trees_wide": trees(
            ExtraTreesClassifier(N_TREES, max_features=0.3, min_samples_leaf=3, class_weight="balanced", random_state=0)
        ),
        "random_forest": trees(
            RandomForestClassifier(N_TREES, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=0)
        ),
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
            ]
        ),
    }


def nested_cv(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    """Outer folds score the pipeline; inner folds pick the model (training nouns only)."""
    predictions = np.empty(len(y), dtype=object)
    chosen = []
    outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
    for train, test in outer.split(X, y):
        X_train, y_train = X.iloc[train], y[train]
        inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
        best_name, best_score = None, -1.0
        for name, candidate in candidate_models().items():
            inner_pred = np.empty(len(y_train), dtype=object)
            for fit, val in inner.split(X_train, y_train):
                inner_pred[val] = clone(candidate).fit(X_train.iloc[fit], y_train[fit]).predict(X_train.iloc[val])
            score = f1_score(y_train, inner_pred, average="macro")
            if score > best_score:
                best_name, best_score = name, score
        model = clone(candidate_models()[best_name]).fit(X_train, y_train)
        predictions[test] = model.predict(X.iloc[test])
        chosen.append(best_name)
    return predictions, chosen


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(raw: pd.DataFrame, box: str, seeds=SEEDS) -> dict:
    """Drift-correct, build noun features and run nested CV for every seed."""
    sources = sources_for(box)
    X, y = noun_features(drift_corrected(raw, sources), sources)
    per_seed, chosen, first_predictions = [], [], None
    for seed in seeds:
        predictions, names = nested_cv(X, y, seed)
        per_seed.append(score(y, predictions))
        chosen += names
        if first_predictions is None:
            first_predictions = predictions
    summary = {
        metric: float(np.mean([s[metric] for s in per_seed])) for metric in per_seed[0]
    } | {
        f"{metric}_std": float(np.std([s[metric] for s in per_seed])) for metric in per_seed[0]
    }
    return {
        "X": X,
        "y": y,
        "predictions": first_predictions,
        "per_seed": per_seed,
        "summary": summary,
        "selected_models": pd.Series(chosen).value_counts().to_dict(),
    }


# ============================================================
# FULL-DATA EXPERIMENT
# ============================================================
def plot_confusion_matrix(matrix: np.ndarray, labels, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(labels)).plot(
        ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False, values_format="d"
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted topic")
    ax.set_ylabel("True topic")
    for text in ax.texts:
        text.set_fontsize(24)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix_noun_level.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "confusion_matrix_noun_level.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_experiment(
    box: str,
    cache: bool,
    dataset: str,
    model: str,
    output_directory: Path,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw = load_source(box, cache, dataset, model)
    result = evaluate(raw, box)
    X, y, predictions = result["X"], result["y"], result["predictions"]
    labels = CATEGORIES[dataset]

    rng = np.random.default_rng(12345)
    null = [
        f1_score(shuffled, nested_cv(X, shuffled, 0)[0], average="macro")
        for shuffled in (rng.permutation(y) for _ in range(n_permutations))
    ]
    observed = result["summary"]["macro_f1"]
    p_value = (1 + sum(value >= observed for value in null)) / (1 + len(null))
    pd.DataFrame({"permuted_macro_f1": null}).to_csv(output_directory / "permutation_null.csv", index=False)

    matrix = confusion_matrix(y, predictions, labels=list(labels))
    pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="True topic"),
        columns=pd.Index(labels, name="Predicted topic"),
    ).to_csv(output_directory / "confusion_matrix_noun_level.csv")
    pd.DataFrame(
        classification_report(y, predictions, labels=list(labels), output_dict=True, zero_division=0)
    ).T.to_csv(output_directory / "classification_report_noun_level.csv")
    pd.DataFrame({"noun": X.index, "true_topic": y, "predicted_topic": predictions}).to_csv(
        output_directory / "noun_predictions_seed0.csv", index=False
    )
    pd.DataFrame(result["per_seed"]).assign(seed=list(SEEDS)).to_csv(
        output_directory / "scores_per_seed.csv", index=False
    )

    metrics = {
        "corpus": DATASET_LABELS[dataset],
        "model": model,
        "box": box,
        "cache": cache,
        "prediction_unit": "target noun (all its probes combined)",
        "nouns": int(len(y)),
        "probes": int(raw["query_id"].nunique()),
        "feature_count": int(X.shape[1]),
        "signals_used": list(sources_for(box)),
        "never_used": list(NEVER_USED),
        "drift_correction_window": DRIFT_WINDOW,
        "evaluation": f"nested {OUTER_SPLITS}x{INNER_SPLITS} noun cross-validation, seeds {list(SEEDS)}",
        "random_baseline": 1 / len(labels),
        "held_out_test": result["summary"],
        "permutation_test": {
            "permutations": n_permutations,
            "null_mean": float(np.mean(null)),
            "null_95th_percentile": float(np.quantile(null, 0.95)),
            "p_value": float(p_value),
        },
        "selected_models": result["selected_models"],
    }
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    title = (
        f"{TITLE_DATASET[dataset]} {model} {'cache ' if cache else ''}"
        f"{box}-box topic inference (per noun)"
    )
    print(f"{title}\n")
    print(f"Macro-F1 {observed:.3f} ± {result['summary']['macro_f1_std']:.3f} "
          f"| accuracy {result['summary']['accuracy']:.3f} ± {result['summary']['accuracy_std']:.3f} "
          f"| random {1 / len(labels):.2f}")
    print(f"Permutation test: shuffled mean {np.mean(null):.3f}, 95th pct {np.quantile(null, 0.95):.3f}, "
          f"p = {p_value:.3f}")
    print("Selected models:", result["selected_models"])
    print(f"Results written to: {output_directory.resolve()}")
    plot_confusion_matrix(matrix, labels, title, output_directory)
    return metrics


# ============================================================
# QUERY BUDGETS
# ============================================================
BUDGETS_PCT = (100, 75, 50, 25, 10, 5)
TEMPLATE_RULES = ("random", "shared")
BUDGET_SEED = 42
LENGTH_CYCLE = ("short", "medium", "long")
TEMPLATES_PER_NOUN = 30
MODELS = ("smol", "qwen", "granite")


def select_budget_queries(query_ids, budget_pct: int, template_rule: str, seed: int = BUDGET_SEED) -> set[str]:
    """Same per-noun, length-balanced selection as the original budget notebooks."""
    rows = []
    for query_id in sorted(set(query_ids)):
        frequency_key, length_key, category_key, suffix = str(query_id).split("__")
        suffix = int(suffix)
        rows.append(
            {
                "query_id": query_id,
                "frequency_key": frequency_key,
                "length_key": length_key,
                "noun": f"{frequency_key}__{category_key}__target_{suffix // 10}",
                "template": suffix % 10,
            }
        )
    queries = pd.DataFrame(rows)
    if budget_pct == 100:
        return set(queries["query_id"])

    rng = np.random.default_rng(seed)
    if template_rule == "shared":
        shared_order = {length_key: rng.permutation(10) for length_key in LENGTH_CYCLE}
        order = {
            (noun, length_key): [t for t in shared_order[length_key] if t in set(block["template"])]
            for (noun, length_key), block in queries.groupby(["noun", "length_key"], sort=True)
        }
    elif template_rule == "random":
        order = {
            key: rng.permutation(sorted(block["template"]))
            for key, block in queries.groupby(["noun", "length_key"], sort=True)
        }
    else:
        raise ValueError(f"Unknown template rule: {template_rule!r}")

    per_noun = TEMPLATES_PER_NOUN * budget_pct / 100
    kept = set()
    for _, class_block in queries.groupby("frequency_key", sort=True):
        cycle_position = 0
        for index, noun in enumerate(sorted(class_block["noun"].unique())):
            n_keep = math.ceil(per_noun) if index % 2 == 0 else math.floor(per_noun)
            per_length = dict.fromkeys(LENGTH_CYCLE, 0)
            for _ in range(n_keep):
                per_length[LENGTH_CYCLE[cycle_position % 3]] += 1
                cycle_position += 1
            for length_key, n_length in per_length.items():
                chosen = set(order.get((noun, length_key), [])[:n_length])
                block = queries[(queries["noun"] == noun) & (queries["length_key"] == length_key)]
                kept |= set(block.loc[block["template"].isin(chosen), "query_id"])
    return kept


def experiment_label(box: str) -> str:
    return f"{box.capitalize()}-box. Topic of use"


def run_budget(box, cache, dataset, model, raw, budget_pct, template_rule, seed, output_directory) -> dict:
    kept = select_budget_queries(raw["query_id"], budget_pct, template_rule, seed)
    metrics_path = output_directory / "metrics.json"
    retained_path = output_directory / "retained_queries.csv"
    if (
        metrics_path.exists()
        and retained_path.exists()
        and set(pd.read_csv(retained_path)["query_id"]) == kept
    ):
        # Same queries already evaluated: reuse the stored result.
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))["held_out_test"]
    else:
        result = evaluate(raw[raw["query_id"].isin(kept)], box)
        summary = result["summary"]
        output_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"query_id": sorted(kept)}).to_csv(retained_path, index=False)
        metrics_path.write_text(
            json.dumps({"held_out_test": summary, "selected_models": result["selected_models"]}, indent=2),
            encoding="utf-8",
        )
    n_classes = len(CATEGORIES[dataset])
    macro_f1 = summary["macro_f1"]
    print(f"{template_rule:>6} templates | seed {seed} | {model:<7} | {budget_pct:>3}% | "
          f"{len(kept):>4} queries | Macro-F1 {macro_f1:.3f}")
    return {
        "dataset": DATASET_LABELS[dataset],
        "cache": cache,
        "experiment": experiment_label(box),
        "template_rule": template_rule,
        "model": model,
        "budget_pct": budget_pct,
        "retained_queries": len(kept),
        "total_queries": int(raw["query_id"].nunique()),
        "n_classes": n_classes,
        "random_baseline": 1 / n_classes,
        "macro_f1": macro_f1,
        "gain_over_random_pct": 100 * (macro_f1 * n_classes - 1),
        "ow_accuracy": np.nan,
        "accuracy": summary["accuracy"],
        "balanced_accuracy": summary["balanced_accuracy"],
        "results_folder": str(output_directory),
    }


def run_budget_model(box, cache, dataset, model, root: Path) -> pd.DataFrame:
    """All budgets for one model, for both template rules."""
    raw = load_source(box, cache, dataset, model)
    full = run_budget(box, cache, dataset, model, raw, 100, "random", BUDGET_SEED,
                      root / "random_templates" / model / "100pct")
    # 100% keeps every query, so it is identical under both template rules.
    rows = [full, full | {"template_rule": "shared"}]
    for budget_pct in BUDGETS_PCT[1:]:
        for template_rule in TEMPLATE_RULES:
            rows.append(
                run_budget(box, cache, dataset, model, raw, budget_pct, template_rule, BUDGET_SEED,
                           root / f"{template_rule}_templates" / model / f"{budget_pct:03d}pct")
            )
    return pd.DataFrame(rows)


def write_budget_summary(rows: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Same file and columns as the original budget notebooks."""
    summary = (
        rows.assign(
            rule_order=rows["template_rule"].map({rule: i for i, rule in enumerate(TEMPLATE_RULES)}),
            model_order=rows["model"].map({model: i for i, model in enumerate(MODELS)}),
        )
        .sort_values(["rule_order", "model_order", "budget_pct"], ascending=[True, True, False])
        .drop(columns=["rule_order", "model_order"])
        .reset_index(drop=True)
    )
    summary.to_csv(root / "budget_summary.csv", index=False)
    return summary


MODEL_STYLES = {"smol": ("#2a78d6", "o"), "qwen": ("#eb6834", "s"), "granite": ("#1baf7a", "^")}
INK_SECONDARY = "#52514e"
GRID_COLOUR = "#e4e3df"
AXIS_COLOUR = "#b5b4ad"


def plot_budget_curves(curves: pd.DataFrame, title: str, output: Path) -> None:
    """Macro-F1 against query budget, one line per model."""
    n_classes = int(round(1 / curves["random_baseline"].iloc[0]))
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    for model, (colour, marker) in MODEL_STYLES.items():
        curve = curves[curves["model"] == model].sort_values("budget_pct")
        ax.plot(curve["budget_pct"], curve["macro_f1"], color=colour, marker=marker, linewidth=2,
                markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=model)
    ax.axhline(1 / n_classes, color=INK_SECONDARY, linestyle="--", linewidth=1.2, label=f"Random (1/{n_classes})")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS_PCT)
    ax.set_xticklabels([f"{b}%" for b in BUDGETS_PCT])
    ax.minorticks_off()
    ax.set_xlim(4, 125)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Query budget (retained queries)")
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOUR)
    ax.tick_params(colors=INK_SECONDARY)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=13)
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_all_budget_curves(box, cache, dataset, root: Path, summary: pd.DataFrame) -> None:
    head = f"{DATASET_LABELS[dataset]} {'cache ' if cache else ''}{box}-box topic inference (per noun)"
    for rule in TEMPLATE_RULES:
        plot_budget_curves(summary[summary["template_rule"] == rule],
                           f"{head}:\nquery budget ({rule} templates)",
                           root / f"budget_curves_{rule}_templates")


def budget_root(box: str, cache: bool, dataset: str) -> Path:
    return Path(f"{dataset}_{'cache_' if cache else ''}{box}_box_topic_budget_results")
