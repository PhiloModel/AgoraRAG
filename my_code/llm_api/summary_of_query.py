import requests
import os
from dotenv import load_dotenv
import json

import os
from groq import Groq


# Wczytanie zmiennych z pliku .env
load_dotenv()

GROQ_API = os.getenv("GROQ_API")

def summarize_query(query: str, temperature=0.1):

    # Inicjalizacja klienta Groq
    client = Groq(api_key=GROQ_API)

    # Tworzenie zapytania do modelu
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "Streszczaj każde zapytanie użytkownika w jednym krótkim zdaniu, podając jego główny temat. Niech streszczenie przypomina nazwę artkykułu (w takim języku jak język zapytania). Zwracaj tylko ta nazwę artykułu. Odpowiedź/Streszczenie jest w takim samym języku jak zapytanie."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=temperature,
        max_tokens=600,
    )   

    return response.choices[0].message.content.strip().replace('"',"")