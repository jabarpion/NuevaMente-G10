import chromadb
from sentence_transformers import SentenceTransformer


# 1. Cargar el modelo de embeddings
print("Cargando modelo de embeddings...")

modelo = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 2. Abrir Chroma persistente
cliente = chromadb.PersistentClient(
    path="chroma_db"
)

coleccion = cliente.get_collection(
    name="nuevamente"
)

print(f"\nChunks disponibles: {coleccion.count()}")


# 3. Preguntas de prueba
preguntas = [
    "¿Qué es la inteligencia artificial generativa?",
    "¿Qué competencias digitales debe desarrollar un docente?",
    "¿Qué es un agente de inteligencia artificial?",
]


# 4. Realizar las búsquedas
for pregunta in preguntas:

    print("\n" + "=" * 70)
    print("PREGUNTA:")
    print(pregunta)

    # Convertir pregunta en embedding
    embedding_pregunta = modelo.encode(
        pregunta
    ).tolist()

    # Buscar los 3 fragmentos más relevantes
    resultado = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=3,
    )

    print("\nRESULTADOS:")

    for i, documento in enumerate(
        resultado["documents"][0]
    ):

        fuente = resultado["metadatas"][0][i]["fuente"]

        print(f"\n--- Resultado {i + 1} ---")
        print(f"Fuente: {fuente}")
        print(documento[:700])
