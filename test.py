from openai import OpenAI
client = OpenAI()

def generate_image_prompt(wish_list: list[str]):
  response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {
        "role": "system",
        "content": "The Mystical Wishes Image Prompt Generator is a specialized GPT assistant designed to craft detailed prompts for DALL·E image generation. It combines elements of mysticism and dreamlike vividness to create unique and captivating images. The system interprets user-provided \"wishes\" and translates them into visual elements, integrating them into a mystical forest background setting. The assistant ensures that the final output is a concise, stand-alone prompt ready for image generation, adhering to a set of predefined rules to ensure consistency and creativity in the output."
      },
      {
        "role": "user",
        "content": ", ".join(wish_list)
      }
    ],
    temperature=1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].message.content

def generate_image(prompt: str):
  response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
  )
  return response.data[0].url

wish_list = ["pony", "yacht", "world peace"]

prompt = generate_image_prompt(wish_list)
print(prompt)

image_url = generate_image(prompt)
print(image_url)