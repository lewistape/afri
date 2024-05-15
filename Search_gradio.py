import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import gradio as gr

def translate(phrase, src_lang, dest_lang):
    valid_languages = {'fon', 'fr', 'yo', 'en'}
    if src_lang not in valid_languages or dest_lang not in valid_languages:
        return "Invalid language code"

    encoded_phrase = quote(phrase)
    url = f"https://translate.glosbe.com/{src_lang}-{dest_lang}/{encoded_phrase}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        translation_div = soup.find('div', class_='w-full h-full bg-gray-100 h-full border p-2 min-h-25vh sm:min-h-50vh whitespace-pre-wrap break-words')
        translation = translation_div.text if translation_div else "Translation not found"
        return translation
    else:
        return "Failed to translate"

def get_simplified_answer(query, input_lang):
    try:
        # Translate the query to English
        query_en = translate(query, input_lang, 'en')

        # Make the API call with the prompt
        endpoint = 'https://api.together.xyz/v1/chat/completions'
        prompt = f"Answer this question for a 5 year old child. Be quick, use easy vocabulary and short sentence. Question: {query_en}<|eot_id|>"
        res = requests.post(endpoint, json={
            "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "max_tokens": 2004,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": [
                "<|eot_id|>"
            ],
            "messages": [
                {
                    "content": prompt,
                    "role": "user"
                }
            ]
        }, headers={
            "Authorization": "Bearer 92180460a15a35e821f0490d8800f74809073d7659eb6899368fe874dd5338e4",
        })

        # Check if the API call was successful
        if res.status_code == 200:
            # Extract the content of the assistant's message from the API response
            content = res.json().get('choices')
            if content:
                content = content[0].get('message', {}).get('content')
                # Translate the answer back to the input language
                answer_translated = translate(content, 'en', input_lang)
                return answer_translated
            else:
                return "Failed to retrieve the assistant's answer"
        else:
            return "API call failed"
    except requests.RequestException as e:
        return f"An error occurred: {e}"

iface = gr.Interface(
    fn=get_simplified_answer,
    inputs=[
        gr.Textbox(lines=1, label="Enter your query"),
        gr.Dropdown(choices=['en', 'fr', 'fon', 'yo'], label="Input Language")
    ],
    outputs="text",
    title="Afri Search",
    description="Enter a query in your language, and this app will search on internet and answers you.",
)

iface.launch()
