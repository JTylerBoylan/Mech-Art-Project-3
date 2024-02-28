import threading
import time
import requests
from flask import Flask, Response, render_template_string
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI

RECORDING_DURATION = 30
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
    
def record_audio(audio_lock : threading.Lock):
    while True:
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
    while True:
        audio_lock.acquire()
        audio_file = open('output.wav', "rb")
        audio_lock.release()

        transcribe_lock.acquire()
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
        print("Transcription:", transcript)
        transcribe_lock.release()
        time.sleep(0.1)
        
if __name__ == '__main__':
    
    audio_lock = threading.Lock()
    transcribe_lock = threading.Lock()
    
    audio_thread = threading.Thread(target=record_audio, args=(audio_lock,))
    transcribe_thread = threading.Thread(target=transcribe_audio, args=(audio_lock, transcribe_lock))
    
    audio_thread.start()
    transcribe_thread.start()
    
    try:
        audio_thread.join()
        transcribe_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")