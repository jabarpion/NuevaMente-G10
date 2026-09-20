from pathlib import Path
from pypdf import PdfReader


def extraer_texto_pdf(ruta_pdf):
    """
    Extrae el texto de todas las páginas de un archivo PDF.

    Retorna una cadena con el texto completo del documento.
    """
    reader = PdfReader(ruta_pdf)

    paginas = []

    for pagina in reader.pages:
        texto = pagina.extract_text()

        if texto:
            paginas.append(texto)

    return "\n\n".join(paginas)


def leer_documentos(carpeta):
    """
    Lee todos los archivos PDF de una carpeta.

    Retorna una lista de diccionarios con:
    - nombre
    - ruta
    - texto
    - paginas
    """
    carpeta = Path(carpeta)
    documentos = []

    for ruta_pdf in sorted(carpeta.glob("*.pdf")):
        reader = PdfReader(ruta_pdf)

        paginas = len(reader.pages)
        texto = extraer_texto_pdf(ruta_pdf)

        documentos.append(
            {
                "nombre": ruta_pdf.name,
                "ruta": str(ruta_pdf),
                "texto": texto,
                "paginas": paginas,
            }
        )

    return documentos
