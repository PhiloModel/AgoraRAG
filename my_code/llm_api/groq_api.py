import requests
import os
from dotenv import load_dotenv
import json
import aiohttp
import asyncio

# Wczytanie zmiennych z pliku .env
load_dotenv()

# Pobranie wartości zmiennych środowiskowych
API_TOKEN = os.getenv("API_TOKEN")
GROQ_API = os.getenv("GROQ_API")

# Endpoint API
url = 'https://api.groq.com/openai/v1/chat/completions'

# Nagłówki żądania
headers = {
    'Authorization': f'Bearer {GROQ_API}',
    'Content-Type': 'application/json'
}

async def groq_query(query, model='llama3-70b-8192', conversation_history=None, temperature=0.2, max_tokens=512):
    try:
        if conversation_history is None:
            conversation_history = [
                {"role": "system", "content": "Jestem pomocnym asystentem. Moje imie to PhiloBot. "},
            ]
        
        prompt = f"""
         Answer in same language as question
         This is query: {query}
        """

        conversation_history.append(prompt)

        # Dane żądania
        data = {
            'model': model,
            'messages': conversation_history,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        conversation_history.pop()
        conversation_history.append({"role": "user", "content": query})
        
        # Wysłanie żądania
        response = requests.post(url, headers=headers, json=data)

        # Sprawdzenie odpowiedzi
        if response.status_code == 200:
            result = response.json()
            assistant_reply = result['choices'][0]['message']['content']
            # Dodanie odpowiedzi asystenta do historii
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply, conversation_history
        else:
            error_message = f'API Error: {response.status_code} - {response.text}'
            print(error_message)
            return f"Sorry, there was an error: {error_message}", conversation_history

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        print(error_message)
        return error_message, conversation_history
