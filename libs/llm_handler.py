from ollama import chat
from ollama import ChatResponse


class LLMHandler:
    MODEL_NAME = "gemma3:1b"
    DUMMY_QUESTION = "is the creator of dragon ball z alive"

    def __init__(self):
        pass

    def make_a_dummy_question(self) -> str:
        response: ChatResponse = chat(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": self.DUMMY_QUESTION,
                },
            ],
        )
        return response["message"]["content"]
