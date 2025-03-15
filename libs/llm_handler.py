from ollama import chat
from ollama import ChatResponse


class LLMHandler:
    """
    A handler for interacting with the Ollama chat model.

    Attributes:
        MODEL_NAME (str): The name of the model to be used.
        DUMMY_QUESTION (str): A predefined question to test the model.
    """

    MODEL_NAME: str = "gemma3:1b"
    DUMMY_QUESTION: str = "is the creator of dragon ball z alive"

    def make_a_dummy_question(self) -> str:
        """
        Sends a predefined question to the chat model and returns the response.

        Returns:
            str: The content of the model's response.
        """
        response: ChatResponse = chat(
            model=self.MODEL_NAME,
            messages=[
                {"role": "user", "content": self.DUMMY_QUESTION},
            ],
        )
        return response["message"]["content"]
