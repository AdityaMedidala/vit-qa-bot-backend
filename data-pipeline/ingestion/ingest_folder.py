import os
import uuid
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from .scan import extract_text_from_pdf
from .chunking import chunk_markdown_document
from .final_ingestion import (
    embed_and_insert,
    insert_document,
    compute_fingerprint,
    update_document_status,
    document_exists_by_fingerprint
)

MAX_WORKERS = 3  # Keep low — marker is RAM/VRAM heavy. Tune between 2–4.


def ingest_single(pdf_path: str, filename: str) -> str:
    """
    Processes a single PDF end-to-end.
    Must be a top-level function so ProcessPoolExecutor can pickle it.
    Returns a status string for logging.
    """
    document_name = os.path.splitext(filename)[0]
    document_id = str(uuid.uuid4())

    try:
        # 1. Compute fingerprint
        fingerprint = compute_fingerprint(pdf_path)
        if document_exists_by_fingerprint(fingerprint):
            return f"⏭️  Skipped (already ingested): {filename}"

        # 2. Insert document row
        document_id = insert_document(
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
        update_document_status(document_id=document_id, status="done")

        return f"✅ Success: {filename} ({len(chunks)} chunks)"

    except Exception as e:
        update_document_status(
            document_id=document_id,
            status="failed",
            error_message=str(e),
        )
        tb = traceback.format_exc()
        return f"❌ Failed: {filename} — {type(e).__name__}: {e}\n{tb}"


def ingest_folder(pdf_folder: str):
    if not os.path.isdir(pdf_folder):
        raise ValueError(f"Not a directory: {pdf_folder}")

    pdf_files = [
        f for f in sorted(os.listdir(pdf_folder))
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found. Nothing to ingest.")
        return

    print(f"Starting ingestion of {len(pdf_files)} PDFs with {MAX_WORKERS} workers...\n")

    futures = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for filename in pdf_files:
            pdf_path = os.path.join(pdf_folder, filename)
            future = executor.submit(ingest_single, pdf_path, filename)
            futures[future] = filename

        for future in as_completed(futures):
            filename = futures[future]
            print("=" * 80)
            print(f"Finished: {filename}")
            try:
                result = future.result()
                print(result)
            except Exception as e:
                # Shouldn't happen since ingest_single catches internally,
                # but guard anyway
                print(f"❌ Unhandled error for {filename}: {e}")

    print("\nIngestion completed.")