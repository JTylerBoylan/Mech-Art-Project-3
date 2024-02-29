import threading
import queue
import time
import requests
from flask import Flask, Response, render_template_string
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI

RECORDING_DURATION = 10
RECORDING_SAMPLE_RATE = 44100

GPT_MODEL = "gpt-4"
GPT_AUDIO_PROMPT_FILE = "gpt-audio-prompt.txt"
GPT_IMAGE_PROMPT_FILE = "gpt-image-prompt.txt"

DALLE_MODEL = "dall-e-3"
DALLE_SIZE = "1792x1024"
DALLE_QUALITY = "standard"

DEFAULT_FOREST_URL = "https://raw.githubusercontent.com/JTylerBoylan/Mech-Art-Project-3/main/default_forest.webp"

app = Flask(__name__)

client = OpenAI()

image_url_queue = queue.Queue(maxsize=1)
image_url_queue.put(DEFAULT_FOREST_URL)

with open(GPT_AUDIO_PROMPT_FILE, "r") as file:
    gpt_audio_prompt = file.read()

with open(GPT_IMAGE_PROMPT_FILE, "r") as file:
    gpt_image_prompt = file.read()

def gen_frames():
    while True:

      image_url = image_url_queue.get_nowait()
      image_url_queue.put(image_url)

      response = requests.get(image_url)
      if response.status_code == 200:
          image_data = response.content 
          yield (b'--frame\r\n'
                  b'Content-Type: image/png\r\n\r\n' + image_data + b'\r\n')
      else:
          print(f"Failed to fetch image from {image_url}, status code: {response.status_code}")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
        </style>
    </head>
    <body>
        <img src="{{ url_for('video_feed') }}">
    </body>
    </html>
    """)

def input_loop():
    wish_list = []
    time.sleep(2.0)
    while True:
        
        input("Press Enter to record audio...")

        print(f"Recording audio for {RECORDING_DURATION} seconds...")
        transcript = get_audio_transcript()
        print(f"Transcript: {transcript}")

        print(f"Getting wishes...")
        wish = get_wishes_from_transcript(transcript)
        print(f"Wish: {wish}")

        wish = wish.rstrip().lower().replace("\"", "")

        if (wish.find("none") != -1) or (len(wish) == 0):
            print(f"Skipping empty wish...")
            continue

        if wish.find("reset") != -1:
            wish_list.clear()
            image_url_queue.get_nowait()
            image_url_queue.put(DEFAULT_FOREST_URL)
            continue
        
        wish_list.append(wish)
        wish_list_reverse = wish_list.copy()
        wish_list_reverse.reverse()
        print(f"Wish list: {wish_list_reverse}")

        try:
          print(f"Generating prompt...")
          prompt = generate_image_prompt(wish_list_reverse)
          print(f"Prompt: {prompt}")

          print(f"Generating image...")
          image_url = generate_image(prompt)       
          print(f"Image URL: {image_url}")

          if not image_url_queue.empty():
              image_url_queue.get_nowait()
          image_url_queue.put(image_url)

        except Exception as e:
          print(f"Error generating image: {e}")
          wish_list.pop()
          time.sleep(1.0)

        print("\n")

def get_audio_transcript():
    recording = sd.rec(int(RECORDING_DURATION * RECORDING_SAMPLE_RATE), samplerate=RECORDING_SAMPLE_RATE, channels=2, dtype='int16')
    sd.wait()
    write('output.wav', RECORDING_SAMPLE_RATE, recording)
    audio_file= open('output.wav', "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text"
    )
    return transcript

def get_wishes_from_transcript(audio: str):
  response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
      {
        "role": "system",
        "content": gpt_audio_prompt
      },
      {
        "role": "user",
        "content": audio
      }
    ],
    max_tokens=1000
  )
  return response.choices[0].message.content

def generate_image_prompt(wish_list: list[str]):
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
  return response.choices[0].message.content

def generate_image(prompt: str):
  response = client.images.generate(
    model=DALLE_MODEL,
    prompt=prompt,
    size=DALLE_SIZE,
    quality=DALLE_QUALITY,
    n=1,
  )
  return response.data[0].url

def save_image_url(image_url: str):
  with open('url-log.txt', 'a') as f:
    f.write(image_url + '\n')

if __name__ == '__main__':
    threading.Thread(target=input_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
