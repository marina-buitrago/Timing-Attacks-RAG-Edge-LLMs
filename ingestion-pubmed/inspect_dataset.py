from datasets import load_dataset

DATASET_NAME = "timaeus/pile-pubmed_central"
DATASET_SPLIT = "train"

print(f"[*] Cargando dataset: {DATASET_NAME}")
print(f"[*] Split: {DATASET_SPLIT}")

dataset = load_dataset(
    DATASET_NAME,
    split=DATASET_SPLIT,
    streaming=True
)

print("[*] Leyendo primer registro...")

first_row = next(iter(dataset))

print("\n[+] Columnas disponibles:")
print(list(first_row.keys()))

print("\n[+] Tipo de cada campo:")
for key, value in first_row.items():
    print(f"- {key}: {type(value)}")

print("\n[+] Vista previa del campo 'text':")
text = first_row.get("text", "")

print(text[:2000])

print("\n[+] Metadata:")
print(first_row.get("meta", None))