from transformers import pipeline

class LocalLLM:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

    def generate(self, question):
        try:
            out = self.pipe(question, max_new_tokens=150)
            return out[0]["generated_text"]
        except:
            return None