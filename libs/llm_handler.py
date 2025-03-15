from ollama import chat
from ollama import ChatResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


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

    def make_a_dummy_question_langchain(self) -> str:
        """
        Sends a predefined dummy question to the chat model and returns
        the response.

        This function initializes a chat model using the specified model
        name and provider, then invokes the model with a predefined dummy
        question.

        Returns:
            str: The content of the model's response.
        """
        model = init_chat_model(self.MODEL_NAME, model_provider="ollama")
        return model.invoke(self.DUMMY_QUESTION).content

    def make_a_dummy_question_langchain_messages(self) -> str:
        """
        Sends a predefined dummy question to the chat model using a system
        and human message structure, then returns the model's response.

        The system message sets the model's role as a comics expert, and
        the human message contains the dummy question.

        Returns:
            str: The content of the model's response.
        """
        model = init_chat_model(self.MODEL_NAME, model_provider="ollama")

        system_prompt = (
            "You are a comics expert. Please answer the user's question."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.DUMMY_QUESTION),
        ]

        return model.invoke(messages).content

    def make_a_dummy_question_langchain_messages_streaming(self) -> str:
        """
        Sends a predefined dummy question to the chat model using a system
        and human message structure, then streams and returns the response.

        The system message sets the model's role as a comics expert, and
        the human message contains the dummy question. The response is
        streamed and concatenated into a single string.

        Returns:
            str: The complete content of the model's streamed response.
        """
        model = init_chat_model(self.MODEL_NAME, model_provider="ollama")

        system_prompt = (
            "You are a comics expert. Please answer the user's question."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.DUMMY_QUESTION),
        ]

        output = []

        for token in model.stream(messages):
            content = token.content
            print(content)
            output.append(content)

        print("-" * 50)
        return "".join(output)
