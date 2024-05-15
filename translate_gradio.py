import gradio as gr
from urllib.parse import quote
from requests import get
from bs4 import BeautifulSoup
from gradio_client import Client

valid_languages = {'fon': 'Fon/Fongbé', 'fr': 'French/Francais', 'yo': 'Yoruba', 'en': 'English/Anglais'}
languages = {
    "fon": "fon (Fon)",
    "yo": "yo (Yoruba)",
    "fr": "fra (French)",
    "en": "eng (English)"
}

def translate_phrase(phrase, src_lang, dest_lang):
    encoded_phrase = quote(phrase)
    url = f"https://translate.glosbe.com/{src_lang}-{dest_lang}/{encoded_phrase}"
    response = get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        translation_div = soup.find('div', class_='w-full h-full bg-gray-100 h-full border p-2 min-h-25vh sm:min-h-50vh whitespace-pre-wrap break-words')
        translation = translation_div.text if translation_div else "Translation not found"
        return translation
    else:
        return "Failed to retrieve the webpage"

def transcribe_and_translate(text, audio, src_lang, dest_lang):
    if text:
        return translate_phrase(text, src_lang, dest_lang)
    else:
        language_name = languages[src_lang]
        client = Client("https://afrinetwork-speak.hf.space/")
        result = client.predict(
            "Record from Mic",  # str in 'Audio input' Radio component
            audio,
            audio,  # str (filepath or URL to file) in 'Use mic' Audio component
            language_name,
            api_name="/predict"
        )
        return translate_phrase(result, src_lang, dest_lang)

iface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Textbox(label="Entrez le texte (optionnel) / Enter text (optional)", lines=4),
        gr.Audio(type="filepath", label="Enregistrez ou téléchargez l'audio (optionnel) / Record or Upload Audio (optional)"),
        gr.Dropdown(choices=list(valid_languages.keys()), label="Langue source / Source Language"),
        gr.Dropdown(choices=list(valid_languages.keys()), label="Langue de destination / Destination Language"),
    ],
    outputs=gr.Textbox(label="Votre traduction / Your Translation"),
    title="Afri Translate",
    description="Entrez un texte ou enregistrez un message audio, et sélectionnez les langues source et destination pour traduire le texte. / Enter text or record an audio message, and select the source and destination languages to translate the text."
)

iface.launch()
