import os
import uuid
import traceback

from ingestion.scan import extract_text_from_pdf
from ingestion.chunking import chunk_markdown_document
from ingestion.final_ingestion import (
    embed_and_insert,
    insert_document,
    compute_fingerprint,
    update_document_status,
    document_exists_by_fingerprint
)


def ingest_folder(pdf_folder: str):

    if not os.path.isdir(pdf_folder):
        raise ValueError(f"Not a directory: {pdf_folder}")

    files = sorted(os.listdir(pdf_folder))

    if not files:
        print("No files found. Nothing to ingest.")
        return

    print(f"Starting ingestion of {len(files)} files...\n")

    for filename in files:
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder, filename)
        document_name = os.path.splitext(filename)[0]
        document_id = str(uuid.uuid4())

        print("=" * 80)
        print(f"Ingesting: {filename}")
        print(f"Document ID: {document_id}")

        try:
            # 1. Compute fingerprint
            fingerprint = compute_fingerprint(pdf_path)

            if document_exists_by_fingerprint(fingerprint):
                 print(f"⏭️ Skipping (already ingested): {filename}")
                 continue


            # 2. Insert document row
            insert_document(
                document_id=document_id,
                document_name=document_name,
                fingerprint=fingerprint,
                status="processing",
            )

            # 3. Extract text
            markdown = extract_text_from_pdf(pdf_path)

            if not markdown or len(markdown.strip()) < 500:
                raise ValueError("Extracted text too small or empty")

            # 4. Chunk document
            chunks = chunk_markdown_document(
                markdown=markdown,
                document_name=document_name,
            )

            if not chunks:
                raise ValueError("No chunks produced")

            # 5. Embed + insert
            embed_and_insert(
                chunks=chunks,
                document_id=document_id,
            )

            # 6. Mark success
            update_document_status(
                document_id=document_id,
                status="done",
            )

            print(f"✅ Success: {filename} ({len(chunks)} chunks)")

        except Exception as e:
            update_document_status(
                document_id=document_id,
                status="failed",
                error_message=str(e),
            )

            print(f"❌ Failed: {filename}")
            traceback.print_exc()

    print("\nIngestion completed.")
