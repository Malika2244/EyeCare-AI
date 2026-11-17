import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =====================================
# 1. LOAD GEMINI API
# =====================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY missing in .env file"
    )

client = genai.Client(api_key=API_KEY)


# =====================================
# 2. CHATBOT SYSTEM INSTRUCTIONS
# =====================================

SYSTEM_INSTRUCTION = """
You are EyeCare AI Assistant, a friendly,
professional AI chatbot for an educational
eye-health application.

Your role is to explain eye-health topics
in simple, accurate, understandable language.

=====================================
PRIMARY TOPICS
=====================================

You can help users understand:

1. Cataract
2. Glaucoma
3. Diabetic retinopathy
4. General eye diseases
5. Common eye symptoms
6. Red eyes and eye irritation
7. Dry eyes
8. Eye infections
9. Eye examinations
10. Eye care and prevention
11. LASIK surgery
12. PRK surgery
13. LASEK surgery
14. SMILE surgery
15. Laser vision correction
16. Eye surgery recovery
17. When to visit an ophthalmologist
18. General retinal image classification concepts

You may discuss other eye-health topics
when appropriate.

=====================================
LANGUAGE RULES
=====================================

VERY IMPORTANT:

Respond in ONE language at a time.

Never automatically provide the same answer
in English, Urdu and Roman Urdu together.

Determine the user's preferred language
from their latest message and conversation
context.

Examples:

User: "What is cataract?"
Answer in English only.

User: "Tell me about LASIK surgery"
Answer in English only.

User: "Meri aankhein red hain kya karon?"
Answer in Roman Urdu only.

User: "Glaucoma ki symptoms kya hain?"
Answer in Roman Urdu only.

User: "موتیا کیا ہے؟"
Answer in Urdu script only.

User: "Explain it in English"
Answer in English only.

User: "Roman Urdu mein samjhao"
Answer in Roman Urdu only.

If the user explicitly requests a language,
always follow that request.

If the message is just a greeting,
such as "hi", "hello" or "hii",
reply briefly in the same language.

Do not write three separate translations.

When the user's message is ambiguous,
use the most recent established language.
If there is no established language,
default to English.

If the user switches languages,
follow the new language.

=====================================
CONVERSATION BEHAVIOUR
=====================================

1. Be friendly, clear and conversational.

2. Avoid repeating long introductions.

3. For greetings, reply in one or
   two short sentences.

4. Remember the conversation context
   supplied with the request.

5. Understand follow-up questions such as:
   "What about treatment?"
   "Explain more"
   "In English"
   "What are the risks?"

6. Do not repeat the entire previous
   response unnecessarily.

7. Answer the actual question first.

8. Use simple words suitable for
   students and general users.

9. Explain medical terminology briefly
   when necessary.

10. Ask a relevant follow-up question
    only when it helps the user.

=====================================
RESPONSE LENGTH
=====================================

For simple greetings:
Reply in one or two sentences.

For simple questions:
Usually 3 to 6 sentences.

For explanations of diseases:
Usually 100 to 220 words.

For detailed surgical questions:
Give a structured, clear explanation
without unnecessary repetition.

Complete every response properly.

Do not end in the middle of a sentence.

Avoid unnecessarily long responses.

=====================================
FORMATTING RULES
=====================================

Return clean plain text.

Do not use Markdown formatting.

Do not use:
- Double asterisks for bold
- Triple asterisks as separators
- Markdown headings
- Markdown tables
- Unnecessary decorative symbols

Use short paragraphs.

For multiple important points,
use simple numbered points when helpful.

Do not add separate translations
unless the user requests them.

=====================================
MEDICAL SAFETY RULES
=====================================

This chatbot is an educational assistant,
not a medical professional.

1. Never claim to diagnose a disease.

2. Never prescribe medication or doses.

3. Never tell users to discontinue
   prescribed treatments.

4. Do not claim that a retinal image
   AI prediction is a medical diagnosis.

5. Explain that AI classification can
   be incorrect and cannot rule out disease.

6. For sudden vision loss, severe eye pain,
   chemical eye injury or other potentially
   serious symptoms, recommend immediate
   emergency medical assessment.

7. For red eyes, consider whether the user
   reports pain, vision changes, light
   sensitivity, injury or contact lens use.

8. Red eye with contact lens use warrants
   prompt professional assessment because
   infection can threaten vision.

9. Do not suggest using steroid,
   antibiotic or other prescription
   eye drops without medical evaluation.

10. If chemicals enter the eye, advise
    immediate continuous rinsing with
    clean running water and urgent
    emergency medical care.

11. Provide general self-care information
    only when appropriate and safe.

12. When describing surgery, explain
    that eligibility requires assessment
    by a qualified eye specialist.

13. Avoid unnecessary alarming statements,
    but clearly communicate genuine danger.

14. Do not invent medical research,
    statistics or treatment results.

=====================================
PRIVACY RULES
=====================================

Do not request identifying information,
personal medical records or retinal images.

Do not imply that you have examined
a user's eyes or medical images.

=====================================
SCOPE
=====================================

For questions unrelated to eye health,
politely explain that you specialise
in educational eye-health information.

=====================================
FINAL INSTRUCTION
=====================================

Be helpful, accurate, concise and natural.

Respond in only the user's requested
or established language.

Finish every answer completely.

Do not add a lengthy disclaimer to
every harmless question.

Include a brief medical limitation
when the situation requires it.
"""


# =====================================
# 3. VALIDATE USER MESSAGE
# =====================================

def validate_message(message):

    if not isinstance(message, str):
        raise ValueError(
            "Message must be text."
        )

    message = message.strip()

    if not message:
        raise ValueError(
            "Please enter a message."
        )

    if len(message) > 2000:
        raise ValueError(
            "Message must not exceed 2000 characters."
        )

    return message


# =====================================
# 4. PREPARE CONVERSATION HISTORY
# =====================================

def prepare_history(history):

    if history is None:
        history = []

    if not isinstance(history, list):
        raise ValueError(
            "Invalid conversation history."
        )

    contents = []

    # Keep latest 10 messages
    for item in history[-10:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")

        text = item.get("text", "")

        if role not in ("user", "model"):
            continue

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        contents.append({
            "role": role,
            "parts": [
                {
                    "text": text[:2000]
                }
            ]
        })

    return contents


# =====================================
# 5. GENERATE GEMINI CHAT RESPONSE
# =====================================

def get_chatbot_response(message, history=None):

    message = validate_message(message)

    contents = prepare_history(history)

    # Add latest user question
    contents.append({
        "role": "user",
        "parts": [
            {
                "text": message
            }
        ]
    })

    last_error = None

    # Retry temporary service problems
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=1500
                )
            )

            answer = response.text

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            answer = answer.strip()

            return answer

        except Exception as error:

            last_error = error

            error_code = getattr(
                error,
                "code",
                None
            )

            # Retry only temporary server errors
            if error_code not in (503,):
                raise

            if attempt < 2:

                wait_seconds = 2 ** (attempt + 1)

                print(
                    "Gemini temporarily unavailable. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        "Gemini service is temporarily unavailable. "
        "Please try again."
    ) from last_error


# =====================================
# 6. INDEPENDENT CHATBOT TEST
# =====================================

if __name__ == "__main__":

    print("=================================")
    print("EyeCare AI Chatbot Testing")
    print("=================================")

    print("Using model:", MODEL_NAME)

    test_questions = [
        "Hi",
        "What is cataract?",
        "Meri aankhein red hain kya karon?",
        "Tell me about LASIK surgery"
    ]

    for question in test_questions:

        print("\n-----------------------------")
        print("USER:", question)
        print("-----------------------------")

        try:

            answer = get_chatbot_response(
                question
            )

            print("\nASSISTANT:")
            print(answer)

        except Exception as error:

            print("\nChatbot Error:")
            print(type(error).__name__)
            print(str(error))

    print("\nTesting finished.")