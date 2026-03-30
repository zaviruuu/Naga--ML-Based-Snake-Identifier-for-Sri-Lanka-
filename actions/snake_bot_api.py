import os
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class SnakeQABot:
    def __init__(self, groq_api_key=None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.qa_store = []

        base_dir = Path(__file__).resolve().parent.parent
        self.qa_file = base_dir / "data" / "qa_store.json"

        self.load_qa_store()

        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                print("Groq initialized")
            except Exception as e:
                print(f"Groq init failed: {e}")
        else:
            print("Groq not available")

    def get_answer(self, question):
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a snake expert."},
                    {"role": "user", "content": question}
                ],
                max_tokens=400
            )

            answer = response.choices[0].message.content.strip()

            if not answer:
                return None

            if any(x in answer.lower() for x in ["error", "quota", "failed"]):
                return None

            self._store_qa(question, answer)

            return answer

        except Exception as e:
            print(f"Groq error: {e}")
            return None

    def _store_qa(self, question, answer):
        if not answer or len(answer.split()) < 5:
            return  # skip low-quality answers

        entry = {
            "question": question,
            "answer": answer,
            "source": "fallback",
            "timestamp": datetime.now().isoformat()
        }

        if len(answer.split()) < 3:
            return  # ignore low-quality answers

        # avoid duplicates
        for qa in self.qa_store:
            if question.lower() in qa["question"].lower():
                return

        self.qa_store.append(entry)

        self._save_qa_store()

    def _save_qa_store(self):
        try:
            os.makedirs(self.qa_file.parent, exist_ok=True)
            with open(self.qa_file, "w", encoding="utf-8") as f:
                json.dump(self.qa_store, f, indent=2)
        except Exception as e:
            print(f"Save error: {e}")

    def load_qa_store(self):
        try:
            if self.qa_file.exists():
                with open(self.qa_file, "r", encoding="utf-8") as f:
                    self.qa_store = json.load(f)
        except Exception as e:
            print(f"Load error: {e}")