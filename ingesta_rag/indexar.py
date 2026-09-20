from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingesta_rag.lector import leer_documentos
from ingesta_rag.chunker import dividir_texto


# 1. Cargar modelo de embeddings
print("Cargando modelo de embeddings...")

modelo = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 2. Leer documentos
carpeta = Path("documentos")

documentos = leer_documentos(carpeta)

print(f"\nDocumentos encontrados: {len(documentos)}")


# 3. Crear o abrir Chroma persistente
cliente = chromadb.PersistentClient(
    path="chroma_db"
)

coleccion = cliente.get_or_create_collection(
    name="nuevamente"
)


# 4. Preparar chunks
ids = []
textos = []
metadatos = []

contador = 0

for documento in documentos:

    chunks = dividir_texto(documento["texto"])

    print(
        f"{documento['nombre']} -> "
        f"{len(chunks)} chunks"
    )

    for chunk in chunks:

        ids.append(f"chunk_{contador}")

        textos.append(chunk)

        metadatos.append(
            {
                "fuente": documento["nombre"],
                "tipo": "pdf",
                "idioma": "es",
            }
        )

        contador += 1


print(f"\nTotal de chunks: {len(textos)}")


# 5. Generar embeddings
print("\nGenerando embeddings...")

embeddings = modelo.encode(
    textos,
    show_progress_bar=True
).tolist()

print("Embeddings generados.")


# 6. Guardar en Chroma
print("\nGuardando información en Chroma...")

coleccion.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=textos,
    metadatas=metadatos,
)

print("Indexación completada.")

print(
    f"Elementos almacenados en Chroma: "
    f"{coleccion.count()}"
)
