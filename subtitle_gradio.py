import os
import requests
import json
import base64
import requests
import pandas as pd
from urllib.parse import quote
from bs4 import BeautifulSoup

os.system('git clone https://github.com/ggerganov/whisper.cpp.git')
os.system('make -C ./whisper.cpp')
os.system('bash ./whisper.cpp/models/download-ggml-model.sh small')
os.system('bash ./whisper.cpp/models/download-ggml-model.sh base')
os.system('bash ./whisper.cpp/models/download-ggml-model.sh medium')
os.system('bash ./whisper.cpp/models/download-ggml-model.sh large')
os.system('bash ./whisper.cpp/models/download-ggml-model.sh base.en')


import gradio as gr
from pathlib import Path
import pysrt
import pandas as pd
import re
import time

from pytube import YouTube


import torch

whisper_models = ["base"]

custom_models = ["belarus-small"]

combined_models = []
combined_models.extend(whisper_models)
combined_models.extend(custom_models)



DeepL_language_codes_for_translation = {
    "Fon": "fon",
    "French": "fr",  # Already exists in the dictionary
    "Yoruba": "yo",
    "English": "en" 
}

translation_models_list = [value[0] for value in DeepL_language_codes_for_translation.items()]



LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

# language code lookup by name, with a few language aliases
source_languages = {
    **{language: code for code, language in LANGUAGES.items()},
    "Burmese": "my",
    "Valencian": "ca",
    "Flemish": "nl",
    "Haitian": "ht",
    "Letzeburgesch": "lb",
    "Pushto": "ps",
    "Panjabi": "pa",
    "Moldavian": "ro",
    "Moldovan": "ro",
    "Sinhalese": "si",
    "Castilian": "es",
    "Let the model analyze": "Let the model analyze"
}


transcribe_options = dict(beam_size=3, best_of=3, without_timestamps=False)


source_language_list = [key[0] for key in source_languages.items()]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS: ")
print(device)
  
videos_out_path = Path("./videos_out")
videos_out_path.mkdir(parents=True, exist_ok=True)


def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("LADATATTU POLKUUN")
    print(abs_video_path)

    
    return abs_video_path

def speech_to_text(video_file_path, selected_source_lang, whisper_model):
    """
    # Youtube with translated subtitles using OpenAI Whisper and Opus-MT models.
    # Currently supports only English audio
    This space allows you to:
    1. Download youtube video with a given url
    2. Watch it in the first video component
    3. Run automatic speech recognition on the video using fast Whisper models
    4. Translate the recognized transcriptions to 26 languages supported by deepL (If free API usage for the month is not yet fully consumed)
    5. Download generated subtitles in .vtt and .srt formats
    6. Watch the the original video with generated subtitles
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    This space is using c++ implementation by https://github.com/ggerganov/whisper.cpp
    """
    
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)
    try:

        

        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file enging is {file_ending}')
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{video_file_path.replace(file_ending, ".wav")}"')
        print("conversion to wav ready")
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

    try:

        print("starting whisper c++")
        srt_path = str(video_file_path.replace(file_ending, ".wav")) + ".srt"
        os.system(f'rm -f {srt_path}')
        if selected_source_lang == "Let the model analyze":
            os.system(f'./whisper.cpp/main "{video_file_path.replace(file_ending, ".wav")}" -t 4 -l "auto" -m ./whisper.cpp/models/ggml-{whisper_model}.bin -osrt')
        else:
            if whisper_model in custom_models:
                os.system(f'./whisper.cpp/main "{video_file_path.replace(file_ending, ".wav")}" -t 4 -l {source_languages.get(selected_source_lang)} -m ./converted_models/ggml-{whisper_model}.bin -osrt')
            else:
                os.system(f'./whisper.cpp/main "{video_file_path.replace(file_ending, ".wav")}" -t 4 -l {source_languages.get(selected_source_lang)} -m ./whisper.cpp/models/ggml-{whisper_model}.bin -osrt')
        print("starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Error running Whisper cpp model")

    try:    

        df = pd.DataFrame(columns = ['start','end','text'])
        srt_path = str(video_file_path.replace(file_ending, ".wav")) + ".srt"
        subs = pysrt.open(srt_path)


        objects = []
        for sub in subs:
            
            
            start_hours = str(str(sub.start.hours) + "00")[0:2] if len(str(sub.start.hours)) == 2 else str("0" + str(sub.start.hours) + "00")[0:2]
            end_hours = str(str(sub.end.hours) + "00")[0:2] if len(str(sub.end.hours)) == 2 else str("0" + str(sub.end.hours) + "00")[0:2]
            
            start_minutes = str(str(sub.start.minutes) + "00")[0:2] if len(str(sub.start.minutes)) == 2 else str("0" + str(sub.start.minutes) + "00")[0:2]
            end_minutes = str(str(sub.end.minutes) + "00")[0:2] if len(str(sub.end.minutes)) == 2 else str("0" + str(sub.end.minutes) + "00")[0:2]
            
            start_seconds = str(str(sub.start.seconds) + "00")[0:2] if len(str(sub.start.seconds)) == 2 else str("0" + str(sub.start.seconds) + "00")[0:2]
            end_seconds = str(str(sub.end.seconds) + "00")[0:2] if len(str(sub.end.seconds)) == 2 else str("0" + str(sub.end.seconds) + "00")[0:2]
            
            start_millis = str(str(sub.start.milliseconds) + "000")[0:3]
            end_millis = str(str(sub.end.milliseconds) + "000")[0:3]
            objects.append([sub.text, f'{start_hours}:{start_minutes}:{start_seconds}.{start_millis}', f'{end_hours}:{end_minutes}:{end_seconds}.{end_millis}'])

        for object in objects:
            srt_to_df = {
            'start': [object[1]],
            'end': [object[2]], 
            'text': [object[0]] 
            }
    
            df = pd.concat([df, pd.DataFrame(srt_to_df)])
    except Exception as e:
        print("Error creating srt df")

        
    try:
        usage_response = requests.get('https://api-free.deepl.com/v2/usage', headers=headers)
        if usage_response.status_code == 200 and usage_response.text:
            usage = json.loads(usage_response.text)
            deepL_character_usage = str(usage['character_count'])
            print("deepL_character_usage")
            print(deepL_character_usage)
        else:
            print("Error: Unable to fetch DeepL API usage")
    except Exception as e:
        print("Error: Unable to fetch DeepL API usage")
        print(e)
    
                    
    return df
    


valid_languages = {'fon', 'fr', 'yo', 'en'}

def translate_phrase(phrase, src_lang, dest_lang):
    encoded_phrase = quote(phrase)
    url = f"https://translate.glosbe.com/{src_lang}-{dest_lang}/{encoded_phrase}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        translation_div = soup.find('div', class_='w-full h-full bg-gray-100 h-full border p-2 min-h-25vh sm:min-h-50vh whitespace-pre-wrap break-words')
        translation = translation_div.text if translation_div else "Translation not found"
        return translation
    else:
        return "Failed to retrieve the webpage"

def translate_transcriptions(df, selected_translation_lang_2):
    if selected_translation_lang_2 is None:
        selected_translation_lang_2 = 'English'
    df.reset_index(inplace=True)

    print("start_translation")
    translations = []

    text_combined = ""
    for i, sentence in enumerate(df['text']):
        if i == 0:
            text_combined = sentence
        else:
            text_combined = text_combined + '\n' + sentence

    src_lang = 'en' # assuming the source language is English
    dest_lang = selected_translation_lang_2.lower()
    if dest_lang not in valid_languages:
        dest_lang = 'en' # default to English if the selected language is not supported

    sentences = text_combined.split('\n')
    for sentence in sentences:
        translation = translate_phrase(sentence, src_lang, dest_lang)
        translations.append(translation)

    df['translation'] = translations

    print("translations done")

    print("Starting WEBVTT-file creation")
    with open('subtitles.vtt','w', encoding="utf-8") as file:
        file.write('WEBVTT\n')
        for i in range(len(df)):
            file.write(str(i+1))
            file.write('\n')
            start = df.iloc[i]['start']
            file.write(f"{start.strip()}")
            file.write(' --> ')
            stop = df.iloc[i]['end']
            file.write(f"{stop.strip()}\n")
            file.writelines(df.iloc[i]['translation'])
            if int(i) != len(df)-1:
                file.write('\n\n')

    print("WEBVTT DONE")

    print("Starting SRT-file creation")
    with open('subtitles.srt','w', encoding="utf-8") as file:
        for i in range(len(df)):
            file.write(str(i+1))
            file.write('\n')
            start = df.iloc[i]['start']
            file.write(f"{start.strip()}")
            file.write(' --> ')
            stop = df.iloc[i]['end']
            file.write(f"{stop.strip()}\n")
            file.writelines(df.iloc[i]['translation'])
            if int(i) != len(df)-1:
                file.write('\n\n')

    print("SRT DONE")
    subtitle_files = ['subtitles.vtt','subtitles.srt']

    return df, subtitle_files

# def burn_srt_to_video(srt_file, video_in):
    
#     print("Starting creation of video wit srt")
    
#     try:
#         video_out = video_in.replace('.mp4', '_out.mp4')
#         print(os.system('ls -lrth'))
#         print(video_in)
#         print(video_out)
#         command = 'ffmpeg -i "{}" -y -vf subtitles=./subtitles.srt "{}"'.format(video_in, video_out)
#         os.system(command)
        
#         return video_out
        
#     except Exception as e:
#         print(e)
#         return video_out

def create_video_player(subtitle_files, video_in):

    with open(video_in, "rb") as file:
        video_base64 = base64.b64encode(file.read())
    with open('./subtitles.vtt', "rb") as file:
        subtitle_base64 = base64.b64encode(file.read())

    video_player = f'''<video id="video" controls preload="metadata">
      <source src="data:video/mp4;base64,{str(video_base64)[2:-1]}" type="video/mp4" />
      <track
        label="English"
        kind="subtitles"
        srclang="en"
        src="data:text/vtt;base64,{str(subtitle_base64)[2:-1]}"
        default />
    </video>
    '''
    #video_player = gr.HTML(video_player)
    return video_player




# ---- Gradio Layout -----
video_in = gr.Video(label="Video file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
video_out = gr.Video(label="Video Out", mirror_webcam=False)



df_init = pd.DataFrame(columns=['start','end','text', 'translation'])

selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="Let the model analyze", label="Spoken language in video", interactive=True)
selected_translation_lang_2 = gr.Dropdown(choices=translation_models_list, type="value", value="English", label="In which language you want the transcriptions?", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="base", label="Selected Whisper model", interactive=True)

transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
transcription_and_translation_df = gr.DataFrame(value=df_init,label="Transcription and translation dataframe", max_rows = 10, wrap=True, overflow_row_behaviour='paginate')

subtitle_files = gr.File(
                label="Download srt-file",
                file_count="multiple",
                type="file",
                interactive=False,
            )

video_player = gr.HTML('<p>video will be played here after you press the button at step 4')


demo = gr.Blocks(css='''
#cut_btn, #reset_btn { align-self:stretch; }
#\\31 3 { max-width: 540px; }
.output-markdown {max-width: 65ch !important;}
''')
demo.encrypt = False




with demo:
    transcription_var = gr.Variable()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown('''
            Vous pouvez / You can: 
            1. Download youtube video with a given url / Télecharger une video Youtube
            2. Watch it in the first video component / Regarder la vidéo
            3. Run automatic speech recognition / Transcrire La video
            4. Traduire les sous titres En fon Yoruba Francais ou Anglais
            5. Download generated subtitles in .vtt and .srt formats/ Télecharger les sous titres
            6. Watch the the original video with generated subtitles/ Regarder la Video sous titré
            ''')
            
        with gr.Column():
            gr.Markdown('''
             ### 1. Copy any non-private Youtube video URL/ Copier l'url de la video.
            ''')
            examples = gr.Examples(examples=
                [ "https://www.youtube.com/watch?v=fLeJJPxua3E&ab_channel=Motiversity", 
                  "https://www.youtube.com/watch?v=3RDaPV_rJ1Y", 
                  "https://www.youtube.com/watch?v=B5-thhkuaYI"],
               label="Examples", inputs=[youtube_url_in])
            
    with gr.Row():
        with gr.Column():
            youtube_url_in.render()
            download_youtube_btn = gr.Button("Step 1. Download Youtube video")
            download_youtube_btn.click(get_youtube, [youtube_url_in], [
                video_in])
            print(video_in)
            

    with gr.Row():
        with gr.Column():
            video_in.render()
            with gr.Column():
                gr.Markdown('''
                ##### Here you can start the transcription and translation process.
                ''')
            selected_source_lang.render()
            selected_whisper_model.render()
            transcribe_btn = gr.Button("Step 2. Transcribe audio")
            transcribe_btn.click(speech_to_text, [video_in, selected_source_lang, selected_whisper_model], [transcription_df])



            
    with gr.Row():
        gr.Markdown('''
        ##### Here you will get transcription  output
        ##### ''')

    with gr.Row():
        with gr.Column():
            transcription_df.render()
            
    with gr.Row():
        with gr.Column():
            language_list = gr.Markdown("---\n".join([f"- {language}" for language in valid_languages]))
            selected_translation_lang_2.render()
            translate_transcriptions_button = gr.Button("Step 3. Translate transcription")
            translate_transcriptions_button.click(translate_transcriptions, [transcription_df, selected_translation_lang_2], [transcription_and_translation_df, subtitle_files])
            transcription_and_translation_df.render()

    with gr.Row():
        with gr.Column():
            gr.Markdown('''##### From here you can download subtitles in .srt or .vtt format''')
            subtitle_files.render()
            
    with gr.Row():
        with gr.Column():
            gr.Markdown('''
            ##### Now press the Step 4. Button to create output video with translated transcriptions
            ##### ''')
            create_video_button = gr.Button("Step 4. Create and add subtitles to video")
            print(video_in)
            create_video_button.click(create_video_player, [subtitle_files,video_in], [
                video_player])
            video_player.render()

# Launch both the web interface and the API interface
demo.launch()
