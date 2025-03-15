from pathlib import Path

from ollama import chat
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class LLMHandler:
    """
    A handler for interacting with the Ollama chat model.

    Attributes:
        MODEL_NAME (str): The name of the model to be used.
        DUMMY_QUESTION (str): A predefined question to test the model.
    """

    MODEL_NAME: str = "gemma3:1b"
    EMBEDDING_MODEL_NAME: str = "mxbai-embed-large"
    DUMMY_QUESTION: str = "is the creator of dragon ball z alive"

    @property
    def pdf_path(self) -> Path:
        """
        Returns the path to the PDF file.

        The file is expected to be located in the 'files/input' directory
        relative to the project's base directory.

        Returns:
            Path: The absolute path to the PDF.
        """
        input_files_dir = self.base_dir / "files" / "input"
        return input_files_dir / "song.pdf"

    @property
    def base_dir(self) -> Path:
        """
        Returns the base directory of the project.

        The base directory is determined as the parent of the directory
        containing this file.

        Returns:
            Path: The absolute path to the project's base directory.
        """
        return Path(__file__).resolve().parent.parent

    def make_a_dummy_question(self) -> str:
        """
        Sends a predefined question to the chat model and returns the response.

        Returns:
            str: The content of the model's response.
        """
        response = chat(
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

    def make_a_dummy_question_langchain_messages_template(self) -> str:
        """
        Sends a predefined dummy question to the chat model using a system
        and human message structure with a templated system message.

        The system message dynamically sets the model's role as a comics
        expert, and the human message contains the dummy question.

        Returns:
            str: The content of the model's response.
        """
        model = init_chat_model(self.MODEL_NAME, model_provider="ollama")

        system_template = (
            "You are a {target} expert. Please answer the user's question."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", "{question}"),
            ]
        )

        prompt = prompt_template.invoke(
            {
                "target": "comics",
                "question": self.DUMMY_QUESTION,
            }
        )

        # messages = prompt.to_messages()
        #
        # return model.invoke(messages).content

        return model.invoke(prompt).content

    def query_document(self, query: str) -> str:
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model=self.EMBEDDING_MODEL_NAME)

        persist_directory = self.base_dir / "files" / "output"

        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory.as_posix(),
        )
        _ = vector_store.add_documents(documents=all_splits)

        return f"{len(all_splits)}"
