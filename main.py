from libs.llm_handler import LLMHandler


def main():
    handler = LLMHandler()

    print(handler.make_a_dummy_question())


if __name__ == "__main__":
    main()
