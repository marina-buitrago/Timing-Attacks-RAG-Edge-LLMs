from __future__ import annotations

import gc
import signal
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import spacy
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# RUTAS DE PG19
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# CSV que debe contener:
# term, frequency, partition
TERMS_PATH = (
    PROJECT_DIR
    / "term_frequencies"
    / "pg19_terms_with_partitions.csv"
)

# IMPORTANTE:
# chroma.sqlite3 está directamente dentro de workspace,
# por lo que PersistentClient debe apuntar a workspace.
DB_PATH = (
    PROJECT_DIR
    / "workspace"
    / "pg19"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "output_neighbours_pg19"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "pg19_terms_with_neighbour_scores.csv"
)

REPRESENTATIVE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pg19_representative_terms.csv"
)

PARTIAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pg19_terms_with_neighbour_scores_partial.csv"
)

COLLECTION_NAME = "pg19"


# ============================================================
# CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

N_CHUNKS = 20
MIN_VALID_NEIGHBOURS = 20

MIN_SAME_PARTITION_RATIO = 0.70
MIN_ADJACENT_PARTITION_RATIO = 0.90
MAX_DISTANT_PARTITION_RATIO = 0.05

# None = procesar todos los términos.
# Para una prueba inicial, por ejemplo:
# MAX_TERMS = 100
MAX_TERMS = None

# Número máximo de resultados mantenidos en memoria.
SAVE_EVERY = 500

SPACY_BATCH_SIZE = 64


# ============================================================
# ESTADO GLOBAL PARA GUARDADO DE EMERGENCIA
# ============================================================

pending_batch: list[dict[str, Any]] = []
write_header = True
stop_requested = False


# ============================================================
# GPU / CPU
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(
        "[+] GPU disponible para SentenceTransformer: "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )
else:
    DEVICE = "cpu"
    print(
        "[-] CUDA no disponible. SentenceTransformer usará CPU.",
        flush=True,
    )

SPACY_GPU_ACTIVE = spacy.prefer_gpu()

if SPACY_GPU_ACTIVE:
    print(
        "[+] spaCy ha podido activar GPU para los componentes "
        "compatibles.",
        flush=True,
    )
else:
    print(
        "[-] spaCy usará CPU. Esto es normal con en_core_web_sm.",
        flush=True,
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_term(term: str) -> str:
    return str(term).strip().lower()


def distance_to_weight(distance: float) -> float:
    return float(np.exp(-float(distance)))


def weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> int | float:
    values = np.asarray(values, dtype=int)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2.0

    position = np.searchsorted(
        cumulative_weights,
        cutoff,
        side="left",
    )

    return int(values[position])


def get_distant_partitions(partition: int) -> set[int]:
    if partition in {1, 2}:
        return {5, 6}

    if partition in {5, 6}:
        return {1, 2}

    if partition == 3:
        return {6}

    if partition == 4:
        return {1}

    return set()


def extract_nouns_by_chunk(
    texts: list[str],
    nlp,
) -> list[list[str]]:
    nouns_by_chunk: list[list[str]] = []

    for doc in nlp.pipe(
        texts,
        batch_size=SPACY_BATCH_SIZE,
    ):
        nouns: list[str] = []

        for token in doc:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue

            lemma = normalize_term(token.lemma_)

            if not lemma:
                continue

            if not any(character.isalpha() for character in lemma):
                continue

            nouns.append(lemma)

        nouns_by_chunk.append(nouns)

    return nouns_by_chunk


def calculate_weighted_partition_distribution(
    partitions: np.ndarray,
    weights: np.ndarray,
) -> dict[int, float]:
    total_weight = float(weights.sum())
    distribution: dict[int, float] = {}

    for partition in range(1, 7):
        partition_weight = float(
            weights[partitions == partition].sum()
        )

        if total_weight > 0:
            distribution[partition] = (
                partition_weight / total_weight
            )
        else:
            distribution[partition] = np.nan

    return distribution


def empty_metrics() -> dict[str, Any]:
    return {
        "neighbour_score": np.nan,
        "same_partition_ratio": np.nan,
        "adjacent_partition_ratio": np.nan,
        "distant_partition_ratio": np.nan,
        "neighbour_frequency_weighted_geometric_mean": np.nan,
        "neighbour_frequency_median": np.nan,
        "neighbour_frequency_p95": np.nan,
        "mean_chunk_distance": np.nan,
        "minimum_chunk_distance": np.nan,
        "n_retrieved_chunks": 0,
        "n_valid_neighbour_types": 0,
        "partition_1_ratio": np.nan,
        "partition_2_ratio": np.nan,
        "partition_3_ratio": np.nan,
        "partition_4_ratio": np.nan,
        "partition_5_ratio": np.nan,
        "partition_6_ratio": np.nan,
    }


def flush_batch() -> None:
    """
    Escribe en el CSV parcial únicamente el lote actualmente
    almacenado en memoria y lo libera después.
    """
    global pending_batch
    global write_header

    if not pending_batch:
        return

    batch_df = pd.DataFrame(pending_batch)

    batch_df.to_csv(
        PARTIAL_OUTPUT_PATH,
        mode="a",
        header=write_header,
        index=False,
    )

    write_header = False
    pending_batch.clear()

    del batch_df
    gc.collect()


def handle_termination_signal(
    signum: int,
    frame: Any,
) -> None:
    """
    Guarda el lote pendiente cuando Slurm envía SIGTERM.
    """
    global stop_requested

    print(
        f"\n[!] Señal {signum} recibida. "
        "Guardando el lote pendiente...",
        flush=True,
    )

    stop_requested = True
    flush_batch()


# ============================================================
# ANÁLISIS DE UN TÉRMINO
# ============================================================

def analyse_term_neighbourhood(
    term: str,
    term_partition: int,
    collection,
    embedding_model,
    nlp,
    frequency_map: dict[str, float],
    partition_map: dict[str, int],
) -> dict[str, Any]:
    with torch.inference_mode():
        query_embedding = embedding_model.encode(
            [term],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0].tolist()

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_CHUNKS,
        include=[
            "documents",
            "distances",
            "metadatas",
        ],
    )

    documents = query_result.get("documents", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    valid_documents: list[str] = []
    valid_distances: list[float] = []

    for document, distance in zip(
        documents,
        distances,
    ):
        if document is None or distance is None:
            continue

        valid_documents.append(str(document))
        valid_distances.append(float(distance))

    if not valid_documents:
        return empty_metrics()

    nouns_by_chunk = extract_nouns_by_chunk(
        valid_documents,
        nlp,
    )

    neighbour_partitions: list[int] = []
    neighbour_frequencies: list[float] = []
    neighbour_weights: list[float] = []

    normalized_query = normalize_term(term)

    for nouns, distance in zip(
        nouns_by_chunk,
        valid_distances,
    ):
        chunk_weight = distance_to_weight(distance)
        noun_counts = Counter(nouns)

        for noun, noun_count in noun_counts.items():
            if noun == normalized_query:
                continue

            if noun not in partition_map:
                continue

            neighbour_partitions.append(
                int(partition_map[noun])
            )

            neighbour_frequencies.append(
                float(frequency_map[noun])
            )

            neighbour_weights.append(
                chunk_weight * noun_count
            )

    if not neighbour_partitions:
        return empty_metrics()

    partitions = np.asarray(
        neighbour_partitions,
        dtype=int,
    )

    frequencies = np.asarray(
        neighbour_frequencies,
        dtype=float,
    )

    weights = np.asarray(
        neighbour_weights,
        dtype=float,
    )

    total_weight = float(weights.sum())

    if total_weight <= 0:
        return empty_metrics()

    partition_distribution = (
        calculate_weighted_partition_distribution(
            partitions,
            weights,
        )
    )

    neighbour_score = weighted_median(
        partitions,
        weights,
    )

    same_partition_ratio = float(
        weights[
            partitions == term_partition
        ].sum()
        / total_weight
    )

    adjacent_partition_ratio = float(
        weights[
            np.abs(partitions - term_partition) <= 1
        ].sum()
        / total_weight
    )

    distant_partitions = get_distant_partitions(
        term_partition
    )

    if distant_partitions:
        distant_mask = np.isin(
            partitions,
            list(distant_partitions),
        )

        distant_partition_ratio = float(
            weights[distant_mask].sum()
            / total_weight
        )
    else:
        distant_partition_ratio = 0.0

    weighted_log_frequency = float(
        np.average(
            np.log1p(frequencies),
            weights=weights,
        )
    )

    weighted_geometric_frequency = float(
        np.expm1(weighted_log_frequency)
    )

    return {
        "neighbour_score": neighbour_score,
        "same_partition_ratio": same_partition_ratio,
        "adjacent_partition_ratio": adjacent_partition_ratio,
        "distant_partition_ratio": distant_partition_ratio,
        "neighbour_frequency_weighted_geometric_mean": (
            weighted_geometric_frequency
        ),
        "neighbour_frequency_median": float(
            np.median(frequencies)
        ),
        "neighbour_frequency_p95": float(
            np.quantile(frequencies, 0.95)
        ),
        "mean_chunk_distance": float(
            np.mean(valid_distances)
        ),
        "minimum_chunk_distance": float(
            np.min(valid_distances)
        ),
        "n_retrieved_chunks": len(valid_documents),
        "n_valid_neighbour_types": len(partitions),
        "partition_1_ratio": partition_distribution[1],
        "partition_2_ratio": partition_distribution[2],
        "partition_3_ratio": partition_distribution[3],
        "partition_4_ratio": partition_distribution[4],
        "partition_5_ratio": partition_distribution[5],
        "partition_6_ratio": partition_distribution[6],
    }


# ============================================================
# GENERACIÓN DE RESULTADOS FINALES
# ============================================================

def create_final_outputs() -> None:
    if not PARTIAL_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "No existe el archivo parcial necesario para "
            "generar los resultados finales:\n"
            f"{PARTIAL_OUTPUT_PATH}"
        )

    print(
        "[*] Generando archivos finales desde el CSV parcial...",
        flush=True,
    )

    result = pd.read_csv(PARTIAL_OUTPUT_PATH)

    result["term"] = (
        result["term"]
        .astype(str)
        .map(normalize_term)
    )

    # Si un término llegó a escribirse más de una vez por una
    # interrupción, se conserva únicamente la última aparición.
    result = (
        result.drop_duplicates(
            subset=["term"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    result["representative"] = (
        result["neighbour_score"].eq(
            result["partition"]
        )
        & result["same_partition_ratio"].ge(
            MIN_SAME_PARTITION_RATIO
        )
        & result["adjacent_partition_ratio"].ge(
            MIN_ADJACENT_PARTITION_RATIO
        )
        & result["distant_partition_ratio"].le(
            MAX_DISTANT_PARTITION_RATIO
        )
        & result["n_valid_neighbour_types"].ge(
            MIN_VALID_NEIGHBOURS
        )
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    representative_terms = result[
        result["representative"]
    ].copy()

    representative_terms.to_csv(
        REPRESENTATIVE_OUTPUT_PATH,
        index=False,
    )

    print("\n[+] Análisis terminado.", flush=True)
    print(
        f"[+] Resultado completo:\n{OUTPUT_PATH}",
        flush=True,
    )
    print(
        "[+] Términos representativos:\n"
        f"{REPRESENTATIVE_OUTPUT_PATH}",
        flush=True,
    )

    print(
        "\nTérminos representativos por partición:",
        flush=True,
    )

    print(
        representative_terms[
            "partition"
        ]
        .value_counts()
        .sort_index(),
        flush=True,
    )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main() -> None:
    global write_header

    signal.signal(
        signal.SIGTERM,
        handle_termination_signal,
    )

    signal.signal(
        signal.SIGINT,
        handle_termination_signal,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TERMS_PATH.exists():
        raise FileNotFoundError(
            "No se ha encontrado el CSV de PG19:\n"
            f"{TERMS_PATH}\n\n"
            "Debe contener las columnas term, frequency y partition."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "No se ha encontrado el directorio de Chroma:\n"
            f"{DB_PATH}"
        )

    sqlite_path = DB_PATH / "chroma.sqlite3"

    if not sqlite_path.exists():
        raise FileNotFoundError(
            "No se ha encontrado chroma.sqlite3 en:\n"
            f"{sqlite_path}\n\n"
            "DB_PATH debe apuntar al directorio que contiene "
            "directamente chroma.sqlite3."
        )

    terms = pd.read_csv(TERMS_PATH)

    required_columns = {
        "term",
        "frequency",
        "partition",
    }

    missing_columns = (
        required_columns - set(terms.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {TERMS_PATH}: {missing_columns}. "
            f"Columnas disponibles: {list(terms.columns)}"
        )

    terms = terms.copy()

    terms["term"] = (
        terms["term"]
        .astype(str)
        .map(normalize_term)
    )

    terms["frequency"] = pd.to_numeric(
        terms["frequency"],
        errors="coerce",
    )

    terms["partition"] = pd.to_numeric(
        terms["partition"],
        errors="coerce",
    )

    terms = terms.dropna(
        subset=[
            "term",
            "frequency",
            "partition",
        ]
    )

    terms = terms[
        terms["frequency"] > 0
    ].copy()

    terms["partition"] = (
        terms["partition"]
        .astype(int)
    )

    terms = (
        terms.sort_values(
            "frequency",
            ascending=False,
        )
        .drop_duplicates(
            subset=["term"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    frequency_map = dict(
        zip(
            terms["term"],
            terms["frequency"],
        )
    )

    partition_map = dict(
        zip(
            terms["term"],
            terms["partition"],
        )
    )

    if MAX_TERMS is None:
        terms_to_process = terms.copy()
    else:
        terms_to_process = (
            terms.groupby(
                "partition",
                group_keys=False,
            )
            .head(MAX_TERMS)
            .reset_index(drop=True)
        )

    total_terms = len(terms_to_process)

    print(
        f"[+] Términos disponibles: {len(terms):,}",
        flush=True,
    )

    print(
        "[+] Términos previstos para el análisis: "
        f"{total_terms:,}",
        flush=True,
    )

    processed_terms: set[str] = set()

    if PARTIAL_OUTPUT_PATH.exists():
        try:
            partial_terms = pd.read_csv(
                PARTIAL_OUTPUT_PATH,
                usecols=["term"],
            )

            processed_terms = set(
                partial_terms["term"]
                .astype(str)
                .map(normalize_term)
            )

            print(
                "[+] Términos recuperados del CSV parcial: "
                f"{len(processed_terms):,}",
                flush=True,
            )

            write_header = False

        except pd.errors.EmptyDataError:
            print(
                "[!] El CSV parcial existe pero está vacío. "
                "Se comenzará desde cero.",
                flush=True,
            )

            PARTIAL_OUTPUT_PATH.unlink()
            write_header = True
    else:
        write_header = True

    terms_pending = terms_to_process[
        ~terms_to_process["term"].isin(processed_terms)
    ].copy()

    print(
        f"[+] Términos pendientes: {len(terms_pending):,}",
        flush=True,
    )

    if terms_pending.empty:
        print(
            "[+] No quedan términos pendientes.",
            flush=True,
        )

        create_final_outputs()
        return

    print(
        "[*] Cargando SentenceTransformer: "
        f"{EMBEDDING_MODEL}",
        flush=True,
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE,
    )

    print(
        f"[*] Cargando spaCy: {SPACY_MODEL}",
        flush=True,
    )

    nlp = spacy.load(
        SPACY_MODEL,
        disable=[
            "ner",
            "parser",
        ],
    )

    print(
        f"[*] Conectando con Chroma: {DB_PATH}",
        flush=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    available_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME not in available_collections:
        raise ValueError(
            f"No existe la colección '{COLLECTION_NAME}'. "
            f"Colecciones disponibles: {available_collections}"
        )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
    f"[+] Colección Chroma abierta correctamente: {COLLECTION_NAME}",
    flush=True,
    )

    pending_count = len(terms_pending)

    for iteration, (_, row) in enumerate(
        terms_pending.iterrows(),
        start=1,
    ):
        if stop_requested:
            print(
                "[!] Detención solicitada. Finalizando.",
                flush=True,
            )
            break

        term = row["term"]

        metrics = analyse_term_neighbourhood(
            term=term,
            term_partition=int(row["partition"]),
            collection=collection,
            embedding_model=embedding_model,
            nlp=nlp,
            frequency_map=frequency_map,
            partition_map=partition_map,
        )

        pending_batch.append({
            "term": term,
            "frequency": float(row["frequency"]),
            "partition": int(row["partition"]),
            **metrics,
        })

        is_batch_complete = (
            len(pending_batch) >= SAVE_EVERY
        )

        is_last_term = (
            iteration == pending_count
        )

        if is_batch_complete or is_last_term:
            flush_batch()

            total_completed = (
                len(processed_terms) + iteration
            )

            print(
                f"[*] Procesados {total_completed:,} "
                f"de {total_terms:,}",
                flush=True,
            )

    flush_batch()

    if stop_requested:
        print(
            "[!] Ejecución interrumpida. "
            "El progreso quedó guardado en:\n"
            f"{PARTIAL_OUTPUT_PATH}",
            flush=True,
        )
        return

    create_final_outputs()


if __name__ == "__main__":
    main()
