import json
import os
import re
import requests
from typing import Dict, Text, Any, List
from pathlib import Path
from dotenv import load_dotenv

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# =========================
# LOAD ENV
# =========================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

# =========================
# IMPORT AI SYSTEMS
# =========================
from .snake_bot_api import SnakeQABot

try:
    from .local_llm import LocalLLM
    LOCAL_LLM_AVAILABLE = True
    local_llm = LocalLLM()
except Exception as e:
    print(f"[ERROR] Local LLM failed: {e}")
    LOCAL_LLM_AVAILABLE = False

snake_qa_bot = SnakeQABot()

# =========================
# LOAD DB
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAKE_DB_PATH = os.path.join(BASE_DIR, "data", "snake_knowledge.json")

with open(SNAKE_DB_PATH, "r", encoding="utf-8") as f:
    SNAKES = json.load(f)

# =========================
# UTILS
# =========================
def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s.,%-]", "", text or "").strip()

def normalize_snake(name: str) -> str:
    return name.lower().replace(" ", "_") if name else None

def get_local_snake_info(name: str):
    if not name:
        return None
    return SNAKES.get(normalize_snake(name))

def is_snake_related(text: str) -> bool:
    keywords = ["snake", "cobra", "krait", "viper", "python", "venom", "bite"]
    return any(k in text.lower() for k in keywords)

def get_user_text(tracker):
    return tracker.latest_message.get("text", "").lower()

def extract_snake_from_text(text: str):
    for snake in SNAKES.keys():
        readable = snake.replace("_", " ")
        if readable in text.lower():
            return snake
    return None

def contains_any(text: str, keywords: list):
    text = text.lower()
    return any(k in text for k in keywords)

# =========================
# 🧠 MEMORY SEARCH
# =========================
def search_qa_store(question: str, threshold: float = 0.6):
    try:
        if not snake_qa_bot.qa_store:
            return None

        q = question.lower()

        for entry in reversed(snake_qa_bot.qa_store):
            stored_q = entry["question"].lower()

            # simple similarity check
            if len(q) > 10 and q in stored_q:
                print("🧠 Memory hit (direct)")
                return entry["answer"]

        return None

    except Exception as e:
        print(f"[Memory Error] {e}")
        return None

# =========================
# 🤖 HYBRID AI CORE
# =========================
def get_ai_answer(question: str) -> str:
    print(f"\n[AI] Question: {question}")

    # =========================
    # 🧠 STEP 1: MEMORY FIRST
    # =========================
    memory_answer = search_qa_store(question)

    if memory_answer:
        print("[AI] ✅ Answer from memory")
        return memory_answer

    # =========================
    # 🧠 STEP 2: GROQ
    # =========================
    try:
        answer = snake_qa_bot.get_answer(question)
        if answer:
            print("[AI] ✅ Groq used")
            return answer
    except Exception as e:
        print(f"[AI] Groq failed: {e}")

    # =========================
    # 🧠 STEP 3: LOCAL LLM
    # =========================
    if LOCAL_LLM_AVAILABLE:
        try:
            answer = local_llm.generate(question)
            if answer:
                print("[AI] ✅ Local LLM used")
                return answer
        except Exception as e:
            print(f"[AI] Local LLM failed: {e}")

    # =========================
    # 🛡️ FINAL FALLBACK
    # =========================
    return "I can help with snake identification, venom, habitat, and safety."

# =========================
# 🧠 LLM INTENT CLASSIFIER
# =========================
def classify_intent_with_llm(text: str) -> str:
    try:
        response = snake_qa_bot.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY one word: snake_question, identify_snake, or not_snake"},
                {"role": "user", "content": text}
            ],
            max_tokens=5
        )

        result = response.choices[0].message.content.lower().strip()
        print(f"[LLM RAW] {result}")

        if result in ["snake_question", "identify_snake", "not_snake"]:
            return result
        return "snake_question"

    except Exception as e:
        print(f"[LLM INTENT ERROR] {e}")
        return "snake_question"
    

# =========================
# ACTIONS
# =========================

class ActionIdentifySnake(Action):
    def name(self) -> Text:
        return "action_identify_snake"

    def run(self, dispatcher, tracker, domain):

        metadata = tracker.latest_message.get("metadata", {}) or {}
        image_b64 = metadata.get("image_base64")

        if not image_b64:
            dispatcher.utter_message(text="Please upload a snake image.")
            return [SlotSet("awaiting_image", True)]

        dispatcher.utter_message(text="Analyzing image...")

        try:
            # VALIDATION
            val_res = requests.post(
                "https://inputvalid.onrender.com/classify-image",
                json={"image_base64": image_b64},
                timeout=60
            )

            if val_res.status_code != 200:
                dispatcher.utter_message(text="Validation failed.")
                return []

            result = val_res.json().get("final", "").lower()
            print("VALIDATION RESULT:", result)

            # 🚨 FIXED LOGIC
            if "non snake" in result:
                dispatcher.utter_message(text="This does not appear to be a snake.")
                return []

            elif "snake" in result:
            # Snake detected (valid OR invalid)
                dispatcher.utter_message(text=result.title())
            # continue to identification
            else:
                dispatcher.utter_message(text="Could not analyze the image.")
                return []
            
            # IDENTIFICATION
            id_res = requests.post(
                "https://identification-sccm.onrender.com/identify",
                json={"image_base64": image_b64},
                timeout=60
            )
            print("IDENT URL:", os.getenv("IDENTIFICATION_API_URL"))

            print("IDENT STATUS:", id_res.status_code)
            print("IDENT RESPONSE:", id_res.text)

            if id_res.status_code != 200:
                dispatcher.utter_message(text="Could not identify the snake.")
                return []

            data = id_res.json()
            snake_name = data.get("snake_name")
            confidence = data.get("confidence", 0)

            print("Parsed:", snake_name, confidence)

            if not snake_name:
                dispatcher.utter_message(text="Could not identify the snake.")
                return []

            normalized = normalize_snake(snake_name)

            dispatcher.utter_message(
                text=f"Identified: {normalized.replace('_',' ').title()}\n"
                     f"Confidence: {confidence:.2%}"
            )

            snake_data = get_local_snake_info(normalized)

            if snake_data:
                dispatcher.utter_message(
                    text=f"Danger: {snake_data['danger_level']}/10\n"
                         f"Venom: {snake_data['venom']}"
                )

            dispatcher.utter_message(
                text="What would you like to know?\n- Habitat\n- Appearance\n- First Aid\n- General Info"
            )

            return [
                SlotSet("current_snake", normalized),
                SlotSet("snake", normalized)
            ]

        except Exception as e:
            print(f"[ERROR] Image processing: {e}")
            dispatcher.utter_message(text="Error processing image.")
            return []


class ActionAnswerSnakeQuestion(Action):
    def name(self) -> Text:
        return "action_answer_snake_question"

    def run(self, dispatcher, tracker, domain):
        user_msg = get_user_text(tracker)
        current_snake = tracker.get_slot("current_snake")
        detected_snake = extract_snake_from_text(user_msg)

        if detected_snake:
            current_snake = detected_snake

        if not current_snake:
            dispatcher.utter_message(text="Please specify the snake first.")
            return []

        data = get_local_snake_info(current_snake)

        if data:
            if contains_any(user_msg, ["appearance", "look", "color", "size", "describe", "looks like"]):
                dispatcher.utter_message(text=f"Appearance: {data['appearance']}")
                return []

            if contains_any(user_msg, ["habitat", "live", "found"]):
                dispatcher.utter_message(text=f"Habitat: {data['habitat']}")
                return []

            if contains_any(user_msg, ["venom", "poison", "danger"]):
                dispatcher.utter_message(text=f"Venom: {data['venom']}")
                return []

            if contains_any(user_msg, ["first aid", "bite", "treatment"]):
                dispatcher.utter_message(text=f"First Aid: {data['first_aid']}")
                return []
            
            if contains_any(user_msg, ["identify", "recognize"]):
                dispatcher.utter_message(
                    text=f"Identification: {data.get('appearance', 'Look for body color, patterns, and head shape.')}"
                )
                return []
            
            if contains_any(user_msg, ["eat", "food", "diet", "prey"]):
                dispatcher.utter_message(
                    text=f"Diet: {data.get('diet', 'Feeds on small animals like rodents, frogs, and other snakes.')}"
                )
                return []

            # =========================
            # 🤖 AI FALLBACK (FIXED)
            # =========================
            snake = current_snake.replace("_", " ")

            if user_msg in ["explain", "explain briefly", "brief", "in brief"]:
                user_msg = f"What does {snake} snake eat? Explain briefly."

            answer = get_ai_answer(
                f"""
                    You are a snake expert. Answer clearly and briefly.

                    Snake: {snake}
                    Question: {user_msg}

                    Give a direct answer only.
                """
            )

            if not answer:
                answer = "I can help with snake-related questions like habitat, venom, or safety."

                dispatcher.utter_message(text=clean_text(answer))
            return []


class ActionGetSnakeInfo(Action):
    def name(self) -> Text:
        return "action_get_snake_info"

    def run(self, dispatcher, tracker, domain):

        snake = tracker.get_slot("snake") or tracker.get_slot("current_snake")

        if not snake:
            dispatcher.utter_message(text="Which snake?")
            return []

        normalized = normalize_snake(snake)
        data = get_local_snake_info(normalized)

        if data:
            dispatcher.utter_message(
                text=f"{normalized.replace('_',' ').title()}\n"
                     f"Scientific: {data['scientific_name']}\n"
                     f"Danger: {data['danger_level']}/10\n"
                     f"Venom: {data['venom']}\n"
                     f"Habitat: {data['habitat']}"
            )
        else:
            answer = get_ai_answer(f"{snake} snake info")
            dispatcher.utter_message(text=clean_text(answer))

        return [
            SlotSet("current_snake", normalized),
            SlotSet("snake", normalized)
        ]


class ActionGeneralSafety(Action):
    def name(self) -> Text:
        return "action_general_safety"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(
            text="Avoid disturbing snakes, keep distance, and call 1990 for help."
        )
        return []


class ActionBiteEmergency(Action):
    def name(self) -> Text:
        return "action_bite_emergency"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(
            text="Snake bite emergency:\n- Stay calm\n- Immobilize limb\n- Go hospital immediately\n- Call 1990"
        )
        return []


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher, tracker, domain):

        user_msg = tracker.latest_message.get("text", "")
        confidence = tracker.latest_message.get("intent", {}).get("confidence", 0)

        print(f"[Fallback Triggered] confidence={confidence}")

        llm_intent = classify_intent_with_llm(user_msg)

        if llm_intent == "identify_snake":
            if "image" in user_msg or "photo" in user_msg or "identify" in user_msg:
                dispatcher.utter_message(text="Please upload the snake image.")
                return [SlotSet("awaiting_image", True)]
            else:
                # prevent false trigger
                llm_intent = "snake_question"

        elif llm_intent == "snake_question":
            answer = get_ai_answer(user_msg)
            if not answer:
                answer = "I can help with snake-related questions like habitat, venom, or safety."
            dispatcher.utter_message(text=clean_text(answer))
            # ✅ STORE FALLBACK QA
            try:
                if len(answer.split()) > 3:  # avoid junk answers
                    snake_qa_bot._store_qa(user_msg, answer)
                    print("💾 Stored fallback QA")
            except Exception as e:
                print(f"[QA STORE ERROR] {e}")
            return []
        
        if llm_intent == "not_snake":
            if is_snake_related(user_msg):
                llm_intent = "snake_question"

        else:
            dispatcher.utter_message(
                text="I specialize in snake-related questions."
            )
            return []