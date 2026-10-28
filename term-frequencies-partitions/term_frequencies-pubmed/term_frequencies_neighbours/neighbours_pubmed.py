from __future__ import annotations

import gc
import os
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
# RUTAS DE PUBMED
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# CSV que debe contener las columnas:
# term, frequency, partition
TERMS_PATH = (
    PROJECT_DIR
    / "term_frequencies"
    / "pubmed_terms_with_partitions.csv"
)

# Directorio que contiene chroma.sqlite3.
DB_PATH = (
    PROJECT_DIR
    / "workspace"
    / "pubmed_central"
)

OUTPUT_DIR = SCRIPT_DIR / "output_neighbours"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "pubmed_terms_with_neighbour_scores.csv"
)

REPRESENTATIVE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pubmed_representative_terms.csv"
)

PARTIAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "pubmed_terms_with_neighbour_scores_partial.csv"
)

COLLECTION_NAME = "pubmed_central"


# ============================================================
# CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Número de chunks cercanos recuperados para cada término.
N_CHUNKS = 20

# Un término necesita al menos este número de sustantivos
# clasificados en sus chunks recuperados.
MIN_VALID_NEIGHBOURS = 20

# Criterios de representatividad.
MIN_SAME_PARTITION_RATIO = 0.70
MIN_ADJACENT_PARTITION_RATIO = 0.90
MAX_DISTANT_PARTITION_RATIO = 0.05

# Para una prueba inicial, indica un número, por ejemplo 100.
# Para procesar todos los términos, usa None.
MAX_TERMS: int | None = None

# Número máximo de resultados conservados en RAM antes de escribirlos.
SAVE_EVERY = 500

# Batch de spaCy para los chunks recuperados de cada consulta.
SPACY_BATCH_SIZE = 64

# Cada cuántos lotes se fuerza la recolección de basura.
GC_EVERY_N_BATCHES = 10


# ============================================================
# ESTADO GLOBAL MÍNIMO PARA GUARDADO DE EMERGENCIA
# ============================================================

_PENDING_BATCH: list[dict[str, Any]] = []
_WRITE_HEADER = True
_STOP_REQUESTED = False


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

# Debe ejecutarse antes de spacy.load().
SPACY_GPU_ACTIVE = spacy.prefer_gpu()

if SPACY_GPU_ACTIVE:
    print(
        "[+] spaCy ha podido activar GPU para componentes compatibles.",
        flush=True,
    )
else:
    print(
        "[-] spaCy usará CPU. Es normal con en_core_web_sm.",
        flush=True,
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_term(term: str) -> str:
    """Normaliza un término para compararlo con lemas de spaCy."""
    return str(term).strip().lower()


def distance_to_weight(distance: float) -> float:
    """Convierte una distancia de Chroma en un peso positivo."""
    return float(np.exp(-float(distance)))


def weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> int | float:
    """Calcula la mediana ponderada de las particiones."""
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
    """Devuelve las particiones consideradas claramente alejadas."""
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
    """Extrae sustantivos comunes y propios de cada chunk."""
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
    """Calcula la proporción ponderada de cada partición."""
    total_weight = float(weights.sum())
    distribution: dict[int, float] = {}

    for partition in range(1, 7):
        partition_weight = float(
            weights[partitions == partition].sum()
        )

        distribution[partition] = (
            partition_weight / total_weight
            if total_weight > 0
            else np.nan
        )

    return distribution


def empty_metrics() -> dict[str, Any]:
    """Resultado utilizado cuando no hay vecinos válidos."""
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


def write_batch_to_partial(
    batch_rows: list[dict[str, Any]],
    partial_path: Path,
    write_header: bool,
) -> bool:
    """
    Añade un lote al CSV parcial y devuelve False para indicar
    que la cabecera ya ha sido escrita.
    """
    if not batch_rows:
        return write_header

    batch_df = pd.DataFrame.from_records(batch_rows)

    batch_df.to_csv(
        partial_path,
        mode="a",
        header=write_header,
        index=False,
    )

    batch_rows.clear()
    del batch_df

    return False


def load_processed_terms(partial_path: Path) -> set[str]:
    """Carga únicamente la columna term del CSV parcial."""
    if not partial_path.exists() or partial_path.stat().st_size == 0:
        return set()

    try:
        partial_terms = pd.read_csv(
            partial_path,
            usecols=["term"],
            dtype={"term": "string"},
        )
    except pd.errors.EmptyDataError:
        return set()

    processed_terms = set(
        partial_terms["term"]
        .dropna()
        .astype(str)
        .map(normalize_term)
    )

    del partial_terms
    gc.collect()

    return processed_terms


def handle_stop_signal(signum, frame) -> None:
    """Solicita una parada limpia ante SIGTERM o SIGINT."""
    del frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"\n[!] Señal {signum} recibida. Se guardará el lote actual.",
        flush=True,
    )


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
    """Calcula las métricas del vecindario semántico de un término."""

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

    for document, distance in zip(documents, distances):
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
        dtype=np.int8,
    )
    frequencies = np.asarray(
        neighbour_frequencies,
        dtype=np.float64,
    )
    weights = np.asarray(
        neighbour_weights,
        dtype=np.float64,
    )

    total_weight = float(weights.sum())

    if total_weight <= 0 or not np.isfinite(total_weight):
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
        weights[partitions == term_partition].sum()
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
    """Genera los dos CSV finales a partir del CSV parcial."""
    if not PARTIAL_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "No existe el archivo parcial necesario para generar "
            f"los resultados finales: {PARTIAL_OUTPUT_PATH}"
        )

    print(
        "[*] Cargando el resultado parcial para generar los CSV finales...",
        flush=True,
    )

    result = pd.read_csv(PARTIAL_OUTPUT_PATH)

    # Evita duplicados si un job fue relanzado tras una escritura parcial.
    result["term"] = (
        result["term"]
        .astype(str)
        .map(normalize_term)
    )

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

    # Escritura atómica: primero temporal y luego sustitución.
    temp_output_path = OUTPUT_PATH.with_suffix(".tmp.csv")
    temp_representative_path = (
        REPRESENTATIVE_OUTPUT_PATH.with_suffix(".tmp.csv")
    )

    result.to_csv(
        temp_output_path,
        index=False,
    )
    os.replace(temp_output_path, OUTPUT_PATH)

    representative_terms = result[
        result["representative"]
    ].copy()

    representative_terms.to_csv(
        temp_representative_path,
        index=False,
    )
    os.replace(
        temp_representative_path,
        REPRESENTATIVE_OUTPUT_PATH,
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
        representative_terms["partition"]
        .value_counts()
        .sort_index(),
        flush=True,
    )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main() -> None:
    global _PENDING_BATCH
    global _WRITE_HEADER

    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TERMS_PATH.exists():
        raise FileNotFoundError(
            "No se ha encontrado el CSV de PubMed:\n"
            f"{TERMS_PATH}\n\n"
            "Debe contener las columnas term, frequency y partition."
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "No se ha encontrado la base persistente de Chroma:\n"
            f"{DB_PATH}"
        )

    chroma_sqlite_path = DB_PATH / "chroma.sqlite3"
    if not chroma_sqlite_path.exists():
        raise FileNotFoundError(
            "El directorio existe, pero no contiene chroma.sqlite3:\n"
            f"{chroma_sqlite_path}"
        )

    terms = pd.read_csv(TERMS_PATH)

    required_columns = {
        "term",
        "frequency",
        "partition",
    }

    missing_columns = required_columns - set(terms.columns)

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

    processed_terms = load_processed_terms(
        PARTIAL_OUTPUT_PATH
    )

    if processed_terms:
        print(
            "[+] Términos recuperados del CSV parcial: "
            f"{len(processed_terms):,}",
            flush=True,
        )

    terms_pending = terms_to_process[
        ~terms_to_process["term"].isin(processed_terms)
    ].copy()

    pending_count = len(terms_pending)

    print(
        f"[+] Términos pendientes: {pending_count:,}",
        flush=True,
    )

    # Si no queda nada pendiente, solo genera los archivos finales.
    if pending_count == 0:
        print(
            "[+] No quedan términos pendientes. "
            "Se generarán los archivos finales.",
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
        "[+] Vectores almacenados en Chroma: "
        f"{collection.count():,}",
        flush=True,
    )

    _PENDING_BATCH = []
    _WRITE_HEADER = not (
        PARTIAL_OUTPUT_PATH.exists()
        and PARTIAL_OUTPUT_PATH.stat().st_size > 0
    )

    batches_written = 0
    completed_before_run = len(processed_terms)

    try:
        for iteration, (_, row) in enumerate(
            terms_pending.iterrows(),
            start=1,
        ):
            if _STOP_REQUESTED:
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

            _PENDING_BATCH.append({
                "term": term,
                "frequency": float(row["frequency"]),
                "partition": int(row["partition"]),
                **metrics,
            })

            is_batch_complete = (
                len(_PENDING_BATCH) >= SAVE_EVERY
            )
            is_last_term = iteration == pending_count

            if is_batch_complete or is_last_term:
                _WRITE_HEADER = write_batch_to_partial(
                    _PENDING_BATCH,
                    PARTIAL_OUTPUT_PATH,
                    _WRITE_HEADER,
                )

                batches_written += 1
                total_completed = (
                    completed_before_run + iteration
                )

                print(
                    f"[*] Procesados {total_completed:,} "
                    f"de {total_terms:,}",
                    flush=True,
                )

                if (
                    batches_written
                    % GC_EVERY_N_BATCHES
                    == 0
                ):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Guarda el lote incompleto si Slurm envía SIGTERM o el usuario interrumpe.
        if _PENDING_BATCH:
            _WRITE_HEADER = write_batch_to_partial(
                _PENDING_BATCH,
                PARTIAL_OUTPUT_PATH,
                _WRITE_HEADER,
            )
            print(
                "[+] Lote pendiente guardado en el CSV parcial.",
                flush=True,
            )

    finally:
        # Intento final de guardado para excepciones Python normales.
        # Un SIGSEGV nativo no puede ser capturado de forma fiable por Python.
        if _PENDING_BATCH:
            _WRITE_HEADER = write_batch_to_partial(
                _PENDING_BATCH,
                PARTIAL_OUTPUT_PATH,
                _WRITE_HEADER,
            )

    if _STOP_REQUESTED:
        print(
            "[!] Ejecución detenida antes de completar todos los términos. "
            "Vuelve a ejecutar el script para reanudar.",
            flush=True,
        )
        return

    create_final_outputs()


if __name__ == "__main__":
    main()
