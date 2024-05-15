import gradio as gr
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
from urllib.parse import quote
from bs4 import BeautifulSoup

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def caption(img, min_len, max_len):
    raw_image = Image.open(img).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, min_length=min_len, max_length=max_len)
    return processor.decode(out[0], skip_special_tokens=True)

def translate(phrase, src_lang, dest_lang):
    valid_languages = {'fon', 'fr', 'yo', 'en', 'yor'}
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
        return "Translation API error"

def greet(img, min_len, max_len):
    start = time.time()
    caption_text = caption(img, min_len, max_len)
    fon_translation = translate(caption_text, 'en', 'fon')
    fr_translation = translate(caption_text, 'en', 'fr')
    yo_translation = translate(caption_text, 'en', 'yo')
    yor_translation = translate(caption_text, 'en', 'yor')
    end = time.time()
    total_time = str(end - start)
    result = f"English: {caption_text}\nFrench: {fr_translation}\nFon: {fon_translation}\nYoruba: {yo_translation}\nYoruba (yor): {yor_translation}\n\nTime taken: {total_time} seconds"
    return result

iface = gr.Interface(
    fn=greet,
    title='Afri Image Vision',
    description="",
    inputs=[
        gr.Image(type='filepath', label='Image'),
        gr.Slider(label='Minimum Length', minimum=1, maximum=1000, value=30),
        gr.Slider(label='Maximum Length', minimum=1, maximum=1000, value=100)
    ],
    outputs=gr.Textbox(label='Caption and Translations'),
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"),
)
iface.launch()
