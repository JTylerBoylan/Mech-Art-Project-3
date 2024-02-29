import threading
import time
import requests
from flask import Flask, Response, render_template_string
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI

RECORDING_DURATION = 15
RECORDING_RATE = 44100

GPT_MODEL = "gpt-4"
GPT_AUDIO_PROMPT_FILE = "gpt-audio-prompt.txt"
GPT_IMAGE_PROMPT_FILE = "gpt-image-prompt.txt"

DALLE_MODEL = "dall-e-3"
DALLE_SIZE = "1792x1024"
DALLE_QUALITY = "standard"

DEFAULT_FOREST_URL = "https://raw.githubusercontent.com/JTylerBoylan/Mech-Art-Project-3/main/default_forest.webp"

app = Flask(__name__)
client = OpenAI()

with open(GPT_AUDIO_PROMPT_FILE, "r") as file:
    gpt_audio_prompt = file.read()

with open(GPT_IMAGE_PROMPT_FILE, "r") as file:
    gpt_image_prompt = file.read()
    
quit_app = False

def record_audio(audio_lock : threading.Lock):
    while not quit_app:
        audio_lock.acquire()
        print(f"Recording for {RECORDING_DURATION} seconds...")
        recording = sd.rec(int(RECORDING_DURATION * RECORDING_RATE), samplerate=RECORDING_RATE, channels=2, dtype='int16')
        sd.wait()
        print("Recording complete.")
        write('output.wav', RECORDING_RATE, recording)
        audio_lock.release()
        time.sleep(0.1)

transcript = ""
def transcribe_audio(audio_lock : threading.Lock, transcribe_lock : threading.Lock):
    global transcript
    while not quit_app:
        audio_lock.acquire()
        audio_file = open('output.wav', "rb")
        audio_lock.release()

        transcribe_lock.acquire()
        print("Transcribing audio...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
        print("Transcription:", transcript)
        transcribe_lock.release()
        time.sleep(0.1)

wish = ""
def get_wish_from_transcript(transcribe_lock : threading.Lock, wish_lock : threading.Lock):
  global transcript, wish
  while not quit_app:
    transcribe_lock.acquire()
    print("Getting wish from transcript...")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
        {
            "role": "system",
            "content": gpt_audio_prompt
        },
        {
            "role": "user",
            "content": transcript
        }
        ],
        max_tokens=1000
    )
    transcribe_lock.release()

    wish_lock.acquire()
    wish = response.choices[0].message.content
    wish = wish.rstrip().lower().replace("\"", "")
    print(f"Wish: {wish}")
    wish_lock.release()
    time.sleep(0.1)

wish_list = []
def add_wish_to_list(wish_lock : threading.Lock, wish_list_lock : threading.Lock):
    global wish, wish_list
    while not quit_app:
        wish_lock.acquire()
        wish_list_lock.acquire()
        if (wish.find("none") != -1) or (len(wish) == 0):
            print(f"Skipping empty wish...")
        elif wish.find("reset") != -1:
            print(f"Resetting wish list...")
            wish_list.clear()
            wish = ""
        else:
            wish_list.append(wish)
            wish_list_reverse = wish_list.copy()
            wish_list_reverse.reverse()
            print(f"Wish list: {wish_list_reverse}")
        wish_lock.release()
        wish_list_lock.release()
        time.sleep(0.1)

prompt = ""
def generate_image_prompt(wish_list_lock : threading.Lock, prompt_lock : threading.Lock):
    global wish_list, prompt
    while not quit_app:
        wish_list_lock.acquire()
        print(f"Generating prompt...")
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
            {
                "role": "system",
                "content": gpt_image_prompt
            },
            {
                "role": "user",
                "content": ", ".join(wish_list)
            }
            ],
            max_tokens=1000
        )
        wish_list_lock.release()

        prompt_lock.acquire()
        prompt = response.choices[0].message.content
        print(f"Prompt: {prompt}")
        prompt_lock.release()
        time.sleep(0.1)

image_url = DEFAULT_FOREST_URL
def generate_image(prompt_lock : threading.Lock, image_url_lock : threading.Lock):
    global prompt, image_url
    while not quit_app:
        prompt_lock.acquire()
        print(f"Generating image...")
        response = client.images.create(
            model=DALLE_MODEL,
            prompt=prompt,
            size=DALLE_SIZE,
            quality=DALLE_QUALITY
        )
        prompt_lock.release()

        image_url_lock.acquire()
        image_url = response.data[0].url
        print(f"Image URL: {image_url}")
        image_url_lock.release()
        time.sleep(0.1)

if __name__ == '__main__':
    
    audio_lock = threading.Lock()
    transcribe_lock = threading.Lock()
    wish_lock = threading.Lock()
    wish_list_lock = threading.Lock()
    prompt_lock = threading.Lock()
    image_url_lock = threading.Lock()
    
    audio_thread = threading.Thread(target=record_audio, args=(audio_lock,))
    transcribe_thread = threading.Thread(target=transcribe_audio, args=(audio_lock, transcribe_lock))
    wish_thread = threading.Thread(target=get_wish_from_transcript, args=(transcribe_lock, wish_lock))
    wish_list_thread = threading.Thread(target=add_wish_to_list, args=(wish_lock, wish_list_lock))
    prompt_thread = threading.Thread(target=generate_image_prompt, args=(wish_list_lock, prompt_lock))
    image_thread = threading.Thread(target=generate_image, args=(prompt_lock, image_url_lock))
    
    audio_thread.start()
    transcribe_thread.start()
    wish_thread.start()
    wish_list_thread.start()
    prompt_thread.start()
    image_thread.start()
    
    try:
        audio_thread.join()
        transcribe_thread.join()
        wish_thread.join()
        wish_list_thread.join()
        prompt_thread.join()
        image_thread.join()
    except KeyboardInterrupt:
        quit_app = True
        print("Quitting...")
    finally:
        audio_thread.join()
        transcribe_thread.join()
        wish_thread.join()
        wish_list_thread.join()
        prompt_thread.join()
        image_thread.join()

    print("Done.")