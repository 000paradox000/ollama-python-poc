from libs.llm_handler import LLMHandler


def main():
    handler = LLMHandler()

    # print(handler.make_a_dummy_question())
    # print(handler.make_a_dummy_question_langchain())
    # print(handler.make_a_dummy_question_langchain_messages())
    # print(handler.make_a_dummy_question_langchain_messages_streaming())
    print(handler.make_a_dummy_question_langchain_messages_template())


if __name__ == "__main__":
    main()
