import os

from dotenv import load_dotenv
from google import genai

from ingesta_rag.consultar import consultar


# Cargar variables de entorno
load_dotenv()


# Crear cliente de Gemini
cliente = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def preparar_contexto(resultados):
    """
    Convierte los resultados del RAG en un contexto
    que recibirá Gemini.
    """

    partes = []

    for i, resultado in enumerate(resultados, start=1):

        partes.append(
            f"[Fuente {i}: {resultado['fuente']}]\n"
            f"{resultado['texto']}"
        )

    return "\n\n".join(partes)


def generar_prompt(pregunta, resultados):
    """
    Construye el prompt que recibirá Gemini.
    """

    contexto = preparar_contexto(resultados)

    prompt = f"""
Responde la siguiente pregunta utilizando únicamente
la información proporcionada en el contexto.

No utilices información externa al contexto.

Si la información necesaria no aparece en el contexto,
indica claramente que no se encuentra en los documentos
disponibles.

Pregunta:
{pregunta}

Contexto:
{contexto}
""".strip()

    return prompt


def generar_respuesta(pregunta, resultados):
    """
    Envía el contexto recuperado por el RAG a Gemini
    y devuelve la respuesta generada.
    """

    prompt = generar_prompt(
        pregunta,
        resultados
    )

    respuesta = cliente.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return respuesta.text


if __name__ == "__main__":

    pregunta = "¿Qué es la inteligencia artificial generativa?"

    resultados = consultar(
        pregunta,
        cantidad=5
    )

    respuesta = generar_respuesta(
        pregunta,
        resultados
    )

    print("\n" + "=" * 70)
    print("RESPUESTA DE GEMINI")
    print("=" * 70)

    print(respuesta)
