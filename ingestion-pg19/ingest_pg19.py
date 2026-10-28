import os
import time
import torch
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURACIÓN GENERAL DEL EXPERIMENTO
# ============================================================

DATASET_NAME = "emozilla/pg19"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"

# Parámetros comparables con el experimento anterior
BATCH_SIZE = 5000
ENCODE_BATCH_SIZE = 512
CHUNK_SIZE = 1000
MIN_CHUNK_LENGTH = 50

# Para pruebas iniciales:
# MAX_DOCUMENTS = 100
# MAX_DOCUMENTS = 1000
# MAX_DOCUMENTS = None  -> ingesta completa
MAX_DOCUMENTS = None

WORKSPACE_DIR = "./workspace"
DB_PATH = os.path.join(WORKSPACE_DIR, "pg19")
CHECKPOINT_FILE = os.path.join(WORKSPACE_DIR, "checkpoint_pg19.txt")

COLLECTION_NAME = "pg19"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# FUNCIÓN DE CHUNKING
# ============================================================

def split_text_into_chunks(text, chunk_size=1000, min_chunk_length=50):
    """
    Divide el texto en chunks fijos de `chunk_size` caracteres, sin overlap.
    Devuelve una lista de tuplas: (chunk_text, chunk_start_position).
    """
    if text is None:
        return []

    text = str(text)

    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]

        if len(chunk.strip()) >= min_chunk_length:
            chunks.append((chunk, start))

    return chunks


# ============================================================
# PREPARAR GPU / CPU
# ============================================================

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[+] GPU detectada, usando: {gpu_name}")
    DEVICE = "cuda"
else:
    print("[-] No se detecta GPU. Usando CPU.")
    DEVICE = "cpu"


# ============================================================
# PREPARAR CARPETAS Y BASE DE DATOS
# ============================================================

os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(f"[+] Base de datos conectada en: {DB_PATH}")
print(f"[*] Documentos actuales en la colección: {collection.count()}")


# ============================================================
# CARGAR MODELO DE EMBEDDINGS
# ============================================================

print(f"[*] Cargando modelo de embeddings: {EMBEDDING_MODEL}")

model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=DEVICE
)

print("[+] Modelo de embeddings cargado")


# ============================================================
# CARGAR DATASET
# ============================================================

print(f"[*] Cargando dataset: {DATASET_NAME} | split={DATASET_SPLIT}")

dataset = load_dataset(
    DATASET_NAME,
    split=DATASET_SPLIT,
    streaming=True
)

print("[+] Dataset cargado en modo streaming")


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

last_processed_document = -1

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        content = f.read().strip()

        if content.isdigit():
            last_processed_document = int(content)
            print(
                f"[+] Checkpoint encontrado. "
                f"Retomando a partir del documento: {last_processed_document + 1}"
            )
        else:
            print("[*] Checkpoint encontrado, pero no contiene un número válido.")
else:
    print("[*] No hay checkpoint previo. Empezando desde cero.")


# ============================================================
# INGESTA
# ============================================================

chunks_buffer = []
ids_buffer = []
metadata_buffer = []

global_id_counter = collection.count()

start_total = time.perf_counter()

try:
    for document_index, document in enumerate(dataset):

        if MAX_DOCUMENTS is not None and document_index >= MAX_DOCUMENTS:
            print(f"\n[*] Límite de prueba alcanzado: {MAX_DOCUMENTS} documentos.")
            break

        if document_index <= last_processed_document:
            continue

        if TEXT_FIELD not in document:
            raise ValueError(
                f"No existe el campo '{TEXT_FIELD}' en el dataset. "
                f"Campos disponibles: {list(document.keys())}"
            )

        full_text = document[TEXT_FIELD]

        chunks = split_text_into_chunks(
            full_text,
            chunk_size=CHUNK_SIZE,
            min_chunk_length=MIN_CHUNK_LENGTH
        )

        for chunk_id, (chunk, chunk_start) in enumerate(chunks):

            chunks_buffer.append(chunk)

            ids_buffer.append(
                f"pg19_{global_id_counter:09d}"
            )

            metadata_buffer.append({
                "dataset": DATASET_NAME,
                "document_index": document_index,
                "chunk_id": chunk_id,
                "chunk_start": chunk_start
            })

            global_id_counter += 1

            if len(chunks_buffer) >= BATCH_SIZE:

                start_batch = time.perf_counter()

                embeddings = model.encode(
                    chunks_buffer,
                    batch_size=ENCODE_BATCH_SIZE,
                    show_progress_bar=False
                ).tolist()

                embedding_time = time.perf_counter() - start_batch

                start_db = time.perf_counter()

                collection.add(
                    documents=chunks_buffer,
                    embeddings=embeddings,
                    metadatas=metadata_buffer,
                    ids=ids_buffer
                )

                db_time = time.perf_counter() - start_db
                total_batch_time = time.perf_counter() - start_batch

                print(
                    f"    -> [Documento {document_index}] "
                    f"Insertados {len(chunks_buffer)} vectores | "
                    f"Embeddings: {embedding_time:.2f}s | "
                    f"BD: {db_time:.2f}s | "
                    f"Batch total: {total_batch_time:.2f}s | "
                    f"Total BD: {collection.count()}",
                    flush=True
                )

                chunks_buffer = []
                ids_buffer = []
                metadata_buffer = []

        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(document_index))

    if chunks_buffer:

        start_final_batch = time.perf_counter()

        embeddings = model.encode(
            chunks_buffer,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False
        ).tolist()

        embedding_time = time.perf_counter() - start_final_batch

        start_db = time.perf_counter()

        collection.add(
            documents=chunks_buffer,
            embeddings=embeddings,
            metadatas=metadata_buffer,
            ids=ids_buffer
        )

        db_time = time.perf_counter() - start_db

        print(
            f"    -> [Batch final] "
            f"Insertados {len(chunks_buffer)} vectores | "
            f"Embeddings: {embedding_time:.2f}s | "
            f"BD: {db_time:.2f}s | "
            f"Total BD: {collection.count()}",
            flush=True
        )

    end_total = time.perf_counter()

    print("\n[+] INGESTA COMPLETADA")
    print(f"[+] Tiempo total: {(end_total - start_total) / 60:.2f} minutos")
    print(f"[+] Total de vectores en disco: {collection.count()}")
    print(f"[+] Base de datos guardada en: {DB_PATH}")

except KeyboardInterrupt:
    print("\n[-] Proceso pausado manualmente por el usuario.")
    print("[*] El checkpoint permite retomar la ingesta al volver a ejecutar el script.")

except Exception as e:
    print(f"\n[!] ERROR CRÍTICO: {e}")
    print("[*] Revisa el error. Si había checkpoint, la ingesta podrá retomarse.")
