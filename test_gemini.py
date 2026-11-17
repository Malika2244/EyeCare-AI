import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not api_key:
    raise ValueError("API key not found in .env file")

client = genai.Client(api_key=api_key)

print("Using model:", model_name)
print("Connecting to Google Gemini API...")

for attempt in range(1, 4):
    try:
        print(f"\nAttempt {attempt}/3")

        response = client.models.generate_content(
            model=model_name,
            contents="Explain cataract in 3 simple sentences."
        )

        print("\nGEMINI RESPONSE")
        print(response.text)
        print("\nGemini API connected successfully!")
        break

    except ServerError as error:
        if error.code != 503:
            print("Server error:", error)
            break

        if attempt == 3:
            print("Gemini server is still busy.")
            print("Please try again later.")
            break

        delay = 2 ** attempt
        print(f"Server busy. Retrying in {delay} seconds...")
        time.sleep(delay)

    except Exception as error:
        print("API Error:", type(error).__name__)
        print(str(error))
        break