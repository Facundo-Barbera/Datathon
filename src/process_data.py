import json

from openai import OpenAI
import pandas as pd

with open('../config.json') as f:
    config = json.load(f)

client = OpenAI(api_key=config["api_key"])

system_message = """
Eres un categorizador de tweets. 
El usuario escribirá un tweet y deberás clasificarlo según su contenido.
Los tweets son interacciones de una empresa llamada "Hey" con sus clientes.
También se incluyen mensajes de la empresa y de bots.

Tus respuestas *sin excepción* debe tener el siguiente formato:
{categoría}, {tono}

Donde {categoría} es una de las siguientes opciones:
"""

# Categories
categories = [
    "Agradecimiento",
    "Quejas",
    "Spam de la empresa",
    "Spam de bots",
    "Pedir ayuda",
    "Recomendaciones",
    "Experiencias de usuario",
    "Saludos"
]

# Add categories to system message
for category in categories:
    system_message += f"- {category}\n"

system_message += """
En caso de que no puedas clasificarlo en ninguna categoría, escoge una categoría arbitraria que consideres adecuada.
La sugerencia debe ser corta.
En caso de realizar una sugerencia, debe mantener el formato de respuesta.

Y {tono} debe describir el tono del mensaje. Algunos ejemplos son:
- Formal
- Sarcástico
- Amigable
- Serio
- Enojado
"""

# Cargar dataset
df = pd.read_csv("../data/raw.csv")

# Añadir columna de categorías
df["category"] = None
df["tone"] = None

# Escoger un solo mensaje para probar
# df = df.head(10)

# Iterar sobre los mensajes
for index, row in df.iterrows():
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": row["tweet"]},
                ]
            )

            category, tone = response.choices[0].message.content.split(", ")
            df.at[index, "category"] = category
            df.at[index, "tone"] = tone

            print("--------------------")
            print(f"Tweet: {row['tweet']}")
            print(f"Response: {response.choices[0].message.content}")
            print(f"Chosen Category: {category}")
            print(f"Chosen Tone: {tone}")
            print("--------------------")
            break
        except ValueError:
            print("Unable to unpack response into two variables. Retrying...")

# Guardar dataset
df.to_csv("../data/processed.csv", index=False)