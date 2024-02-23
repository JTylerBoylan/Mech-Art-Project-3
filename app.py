import threading
import queue
import time
import requests
from flask import Flask, Response, render_template_string
from openai import OpenAI, BadRequestError

GPT_MODEL = "gpt-4"
GPT_PROMPT_FILE = "gpt-prompt.txt"

DALLE_MODEL = "dall-e-3"
DALLE_SIZE = "1792x1024"
DALLE_QUALITY = "standard"

DEFAULT_FOREST_URL = "https://raw.githubusercontent.com/JTylerBoylan/Mech-Art-Project-3/main/default_forest.webp"

app = Flask(__name__)

client = OpenAI()

image_url_queue = queue.Queue(maxsize=1)
image_url_queue.put(DEFAULT_FOREST_URL)

with open(GPT_PROMPT_FILE, "r") as file:
    gpt_prompt = file.read()

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
        
        wish = input("Enter your wish: ")

        if wish == 'reset':
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

          break
        except Exception as e:
          print(f"Error generating image: {e}")
          wish_list.pop()
          time.sleep(1.0)

        print("\n")

def generate_image_prompt(wish_list: list[str]):
  response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
      {
        "role": "system",
        "content": gpt_prompt
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

if __name__ == '__main__':
    threading.Thread(target=input_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
