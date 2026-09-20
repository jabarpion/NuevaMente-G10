import chromadb


# Crear una base Chroma persistente
cliente = chromadb.PersistentClient(
    path="chroma_db_prueba"
)

# Crear o recuperar la colección
coleccion = cliente.get_or_create_collection(
    name="prueba_persistencia"
)

# Guardar un registro de prueba
coleccion.upsert(
    ids=["prueba_001"],
    documents=[
        "La inteligencia artificial generativa puede crear contenido nuevo."
    ],
    metadatas=[
        {"fuente": "prueba"}
    ],
)

print("Chroma persistente OK")
print("Colección:", coleccion.name)
print("Elementos:", coleccion.count())

# Volver a abrir la base para comprobar que quedó guardada
cliente2 = chromadb.PersistentClient(
    path="chroma_db_prueba"
)

coleccion2 = cliente2.get_collection(
    name="prueba_persistencia"
)

print("Elementos después de reabrir:", coleccion2.count())

resultado = coleccion2.get(
    ids=["prueba_001"]
)

print("Documento recuperado:")
print(resultado["documents"][0])
