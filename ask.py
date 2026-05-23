from rag import answer_question


def main() -> None:
    print("\nRAG Support Agent ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer, sources = answer_question(question)
        except RuntimeError as error:
            print(f"\nError: {error}\n")
            break

        print(f"\nAgent: {answer}")
        if sources:
            print("Sources: " + ", ".join(sources))
        print()


if __name__ == "__main__":
    main()
