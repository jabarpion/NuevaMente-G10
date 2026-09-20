from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingesta_rag.lector import leer_documentos
from ingesta_rag.chunker import dividir_texto


# 1. Cargar modelo de embeddings
modelo = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Leer todos los documentos
carpeta = Path("documentos")
documentos = leer_documentos(carpeta)

print(f"\nDocumentos encontrados: {len(documentos)}")

# 3. Crear Chroma en memoria
cliente = chromadb.Client()

coleccion = cliente.create_collection(
    name="prueba_rag_completa"
)

# 4. Preparar todos los chunks
ids = []
textos = []
metadatos = []

contador = 0

for documento in documentos:
    chunks = dividir_texto(documento["texto"])

    print(f"{documento['nombre']} -> {len(chunks)} chunks")

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

# 5. Crear embeddings de todos los chunks
print("\nGenerando embeddings...")

embeddings = modelo.encode(
    textos,
    show_progress_bar=True
).tolist()

print("Embeddings generados.")

# 6. Guardar los chunks en Chroma
print("\nGuardando chunks en Chroma...")

coleccion.add(
    ids=ids,
    embeddings=embeddings,
    documents=textos,
    metadatas=metadatos,
)

print("Chunks guardados correctamente.")

# 7. Preguntas de prueba
preguntas = [
    "¿Qué es la inteligencia artificial generativa?",
    "¿Qué competencias digitales debe desarrollar un docente?",
    "¿Qué es un agente de inteligencia artificial?",
]

# 8. Realizar las consultas
for pregunta in preguntas:

    print("\n" + "=" * 70)
    print("PREGUNTA:")
    print(pregunta)

    embedding_pregunta = modelo.encode(pregunta).tolist()

    resultado = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=3,
    )

    print("\nRESULTADOS:")

    for i, documento in enumerate(resultado["documents"][0]):
        print(f"\n--- Resultado {i + 1} ---")
        print(f"Fuente: {resultado['metadatas'][0][i]['fuente']}")
        print(documento[:500])
