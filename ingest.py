from rag import DOCS_DIR, ingest_documents, list_document_files


def main() -> None:
    files = list_document_files(DOCS_DIR)
    if not files:
        raise SystemExit(f"No .txt, .md, or .pdf files found in {DOCS_DIR}.")

    print("Indexing documents:")
    for path in files:
        print(f"- {path}")

    chunk_count = ingest_documents()
    print(f"\nDone. Saved {chunk_count} chunks to ChromaDB.")


if __name__ == "__main__":
    main()
