
import os
import csv
from collections import Counter

import chromadb
import spacy
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

WORKSPACE_DIR = "./workspace"
DB_PATH = os.path.join(WORKSPACE_DIR, "pg19")
COLLECTION_NAME = "pg19"

TERM_FREQUENCIES_DIR = "./term_frequencies"

OUTPUT_CSV = os.path.join(TERM_FREQUENCIES_DIR, "noun_frequencies_pg19.csv")
OUTPUT_PNG = os.path.join(TERM_FREQUENCIES_DIR, "top50_nouns_pg19.png")
OUTPUT_ZIPF = os.path.join(TERM_FREQUENCIES_DIR,"zipf_distribution_pg19.png")
OUTPUT_HIST = os.path.join(TERM_FREQUENCIES_DIR,"frequency_histogram_pg19.png")


# Número máximo de chunks que se usarán para calcular frecuencias.
# Para una primera prueba, usa 50_000 o 100_000.
# Para una extracción más fuerte, puedes subirlo.
MAX_DOCS_TO_ANALYZE = None

# Tamaño de lote para extraer documentos desde ChromaDB.
CHROMA_BATCH_SIZE = 5000

# Tamaño de lote para spaCy.
SPACY_BATCH_SIZE = 500

# Frecuencia mínima para guardar un término en el CSV.
MIN_FREQUENCY_TO_SAVE = 2

# Top términos que aparecerán en el histograma.
TOP_N_PLOT = 50


def is_valid_term(token):
    """
    #Filtra tokens para quedarnos con sustantivos útiles.
    """

    lemma = token.lemma_.lower().strip()

    if token.pos_ not in {"NOUN", "PROPN"}:
        return False

    if not token.is_alpha:
        return False

    if len(lemma) < 4:
        return False

    if token.is_stop:
        return False

    return True


def main():

    os.makedirs(TERM_FREQUENCIES_DIR, exist_ok=True)

    spacy.require_gpu()
    nlp = spacy.load(
        "en_core_web_trf",
        disable=["parser", "ner"]
    )

    print("[*] Conectando a ChromaDB")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    total_docs = collection.count()

    print(f"[+] Total de chunks en ChromaDB: {total_docs}")

    if total_docs == 0:
        print("[!] La colección está vacía. Ejecuta primero la ingesta.")
        return

    if MAX_DOCS_TO_ANALYZE is None:
    	docs_to_analyze = total_docs 
    
    else: 
        docs_to_analyze = min(MAX_DOCS_TO_ANALYZE, total_docs)

    print(f"[*] Analizando {docs_to_analyze} chunks")

    noun_counter = Counter()

    processed_docs = 0

    for offset in range(0, docs_to_analyze, CHROMA_BATCH_SIZE):

        current_limit = min(CHROMA_BATCH_SIZE, docs_to_analyze - offset)

        print(
            f"[*] Extrayendo documentos de ChromaDB: "
            f"{offset} - {offset + current_limit}"
        )

        batch = collection.get(
            limit=current_limit,
            offset=offset,
            include=["documents"]
        )

        documents = batch["documents"]

        print(f"[*] Procesando lote con spaCy: {len(documents)} textos")

        for doc in nlp.pipe(documents, batch_size=SPACY_BATCH_SIZE):

            terms = [
                token.lemma_.lower()
                for token in doc
                if is_valid_term(token)
            ]

            noun_counter.update(terms)

        processed_docs += len(documents)

        print(f"    -> Procesados {processed_docs} / {docs_to_analyze}")

    print(f"\n[+] Términos únicos encontrados: {len(noun_counter)}")

    # Guardar CSV
    print(f"[*] Guardando frecuencias en: {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "term",
            "frequency"
        ])

        for term, freq in noun_counter.most_common():
            if freq >= MIN_FREQUENCY_TO_SAVE:
                writer.writerow([term, freq])

    # Histograma
    # ============================================================
    # Top 50 términos
    # ============================================================

    top_terms = noun_counter.most_common(TOP_N_PLOT)

    if top_terms:

        words = [item[0] for item in top_terms]
        freqs = [item[1] for item in top_terms]

        plt.figure(figsize=(16, 7))
        plt.bar(words, freqs)
        plt.title(f"Top {TOP_N_PLOT} sustantivos en PG-19")
        plt.xlabel("Término lematizado")
        plt.ylabel("Frecuencia absoluta")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_PNG, dpi=300)
        plt.close()
        print(f"[+] Histograma Top-{TOP_N_PLOT} guardado en: {OUTPUT_PNG}")

    # ============================================================
    # Distribución completa (Zipf)
    # ============================================================

    all_freqs = sorted(noun_counter.values(), reverse=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_freqs) + 1), all_freqs)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Rank")
    plt.ylabel("Frecuencia")
    plt.title("Distribución completa de frecuencias (PG-19)")

    plt.tight_layout()
    plt.savefig(OUTPUT_ZIPF, dpi=300)
    plt.close()

    print(f"[+] Distribución Zipf guardada en: {OUTPUT_ZIPF}")


    # ============================================================
    # Histograma de TODAS las frecuencias
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.hist(all_freqs, bins=100)

    plt.xlabel("Frecuencia")
    plt.ylabel("Número de términos")
    plt.title("Histograma de frecuencias de todos los términos (PG-19)")

    plt.tight_layout()
    plt.savefig(OUTPUT_HIST, dpi=300)
    plt.close()

    print(f"[+] Histograma completo guardado en: {OUTPUT_HIST}")


    print("\n--- Top términos ---")
    for i, (term, freq) in enumerate(noun_counter.most_common(30), start=1):
    	print(f"{i}. {term}: {freq}")

print("\n[+] Extracción completada.")

if __name__ == "__main__":
    main()
