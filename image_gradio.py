import os
import random
import uuid
from urllib.parse import quote
from requests import get
from bs4 import BeautifulSoup

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import DiffusionPipeline

DESCRIPTION = """Afri Ai Image"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1536"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_PER_PROMPT = 1

valid_languages = {'fon', 'fr', 'yo', 'en'}

if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    )
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print("Loaded on Device!")

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")


def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def translate_to_english(phrase, src_lang):
    if src_lang == 'en':
        return phrase

    dest_lang = 'en'
    encoded_phrase = quote(phrase)
    url = f"https://translate.glosbe.com/{src_lang}-{dest_lang}/{encoded_phrase}"
    response = get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        translation_div = soup.find('div', class_='w-full h-full bg-gray-100 h-full border p-2 min-h-25vh sm:min-h-50vh whitespace-pre-wrap break-words')
        translation = translation_div.text if translation_div else "Translation not found"
        return translation
    else:
        return "Error: Unable to translate"


@spaces.GPU(enable_queue=True)
def generate(
    phrase: str,
    input_lang: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    pipe.to(device)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if input_lang != 'en':
        prompt = translate_to_english(phrase, input_lang)
    else:
        prompt = phrase

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        use_resolution_binning=use_resolution_binning,
        output_type="pil",
    ).images

    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed


examples = [
    ["nyɔnu e nɔ sa tomati ɖo aximɛ", "fon"],
    ["ọba ilẹ̀ benin kan", "yo"],
    ["an astronaut riding a horse in space", "en"],
    ["a cartoon of a boy playing with a tiger", "en"],
    ["a cute robot artist painting on an easel, concept art", "en"],
    ["a close up of a woman wearing a transparent, prismatic, elaborate nemeses headdress, over the should pose, brown skin-tone", "en"]
]


css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            input_lang = gr.Dropdown(choices=list(valid_languages), value='en', label='Input Language')
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=NUM_IMAGES_PER_PROMPT, show_label=False)
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20,
                step=0.1,
                value=3.0,
            )

    gr.Examples(
        examples=examples,
        inputs=[prompt, input_lang],
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            input_lang,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
       ],
       outputs=[result, seed],
       api_name="run",
   )

if __name__ == "__main__":
   demo.queue(max_size=20).launch()
