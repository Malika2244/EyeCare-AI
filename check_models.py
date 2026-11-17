import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key missing in .env")

client = genai.Client(api_key=api_key)

print("\nChecking available Gemini models...\n")

try:
    count = 0

    for model in client.models.list():
        name = model.name or ""
        actions = model.supported_actions or []

        if (
            "gemini" in name.lower()
            and "generateContent" in actions
        ):
            print(name)
            count += 1

    print("\nTotal compatible models:", count)

except Exception as error:
    print("Error:", type(error).__name__)
    print(str(error))