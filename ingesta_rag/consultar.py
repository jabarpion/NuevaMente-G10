import chromadb
from sentence_transformers import SentenceTransformer


# 1. Cargar el modelo de embeddings
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


def consultar(pregunta, cantidad=5):
    """
    Busca los fragmentos más relevantes para una pregunta.

    Retorna una lista de diccionarios con:
    - texto
    - fuente
    """

    embedding_pregunta = modelo.encode(
        pregunta
    ).tolist()

    resultado = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=cantidad,
    )

    documentos = resultado["documents"][0]
    metadatos = resultado["metadatas"][0]

    resultados = []

    for documento, metadata in zip(
        documentos,
        metadatos
    ):
        resultados.append(
            {
                "texto": documento,
                "fuente": metadata["fuente"],
            }
        )

    return resultados


# 3. Prueba directa del módulo
if __name__ == "__main__":

    pregunta = "¿Qué es la inteligencia artificial generativa?"

    resultados = consultar(pregunta)

    print(f"Pregunta: {pregunta}")
    print(
        f"\nResultados encontrados: "
        f"{len(resultados)}"
    )

    for i, resultado in enumerate(
        resultados,
        start=1
    ):
        print(
            f"\n--- Resultado {i} ---"
        )
        print(
            f"Fuente: {resultado['fuente']}"
        )
        print(
            resultado["texto"][:700]
        )
