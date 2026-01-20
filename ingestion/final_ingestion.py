import json
from openai import OpenAI
import psycopg2 
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
load_dotenv()
import os
import hashlib
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def batch_chunks_by_tokens(chunks, max_tokens=80000):
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        tokens = count_tokens(chunk["text"])

        if tokens > max_tokens:
            # Extremely large chunk → still embed alone
            batches.append([chunk])
            continue

        if current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches



client=OpenAI()


def compute_fingerprint(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def insert_document(document_id: str, document_name: str,fingerprint: str,status: str="processing"):
    db_url = os.getenv("SUPABASE_URL")
    if not db_url:
        raise ValueError("SUPABASE_URL not set")

    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO documents (document_id, document_name,fingerprint,status)
        VALUES (%s, %s,%s,%s)
        ON CONFLICT (fingerprint) DO NOTHING
        """,
        (document_id, document_name,fingerprint,status),
    )

    conn.commit()
    cursor.close()
    conn.close()


def embed_and_insert (chunks,document_id):
    batches = batch_chunks_by_tokens(chunks)
    total_inserted = 0

    for batch in batches:
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]

        rows = []
        for chunk, emb in zip(batch, embeddings):
            rows.append((
                chunk["chunk_id"],
                document_id,
                chunk["metadata"]["document"],
                chunk["text"],
                emb,
                json.dumps(chunk["metadata"]),
                chunk["metadata"]["char_count"]
            ))

        conn = psycopg2.connect(os.getenv("SUPABASE_URL"))
        cursor = conn.cursor()

        query = """
            INSERT INTO public.document_chunks
            (chunk_id, document_id, document_text, text, embedding, metadata, char_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
        """

        execute_batch(cursor, query, rows)
        conn.commit()
        cursor.close()
        conn.close()

        total_inserted += len(rows)

    print(f"Ingested {total_inserted} chunks in {len(batches)} batches")


def update_document_status(document_id: str,status: str,error_message: str | None = None,):
    conn = psycopg2.connect(os.getenv("SUPABASE_URL"))
    cursor = conn.cursor()
    cursor.execute(
    """
    UPDATE documents
    SET status = %s,
    error_message = %s,
    updated_at = now()
    WHERE document_id = %s
    """,
    (status, error_message, document_id),)
    conn.commit()
    cursor.close()
    conn.close()

def document_exists_by_fingerprint(fingerprint: str) -> bool:
    conn = psycopg2.connect(os.getenv("SUPABASE_URL"))
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1
        FROM documents
        WHERE fingerprint = %s
        AND status = 'done'
        LIMIT 1
        """,
        (fingerprint,),
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists

