import json
from openai import OpenAI
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
load_dotenv()
import os
import hashlib
import tiktoken

from app.db import get_db_connection, release_db_connection

MAX_EMBED_TOKENS = 7500

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


client = OpenAI()


def compute_fingerprint(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def truncate_to_token_limit(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def insert_document(document_id: str, document_name: str, fingerprint: str, status: str = "processing") -> str:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Use UPSERT — on conflict, overwrite with new document_id and reset status
        cursor.execute(
            """
            INSERT INTO documents (document_id, document_name, fingerprint, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (fingerprint) 
            DO UPDATE SET 
                document_id = EXCLUDED.document_id,
                status = EXCLUDED.status,
                error_message = NULL,
                updated_at = now()
            RETURNING document_id
            """,
            (document_id, document_name, fingerprint, status),
        )
        actual_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return actual_id
    finally:
        release_db_connection(conn)

def embed_and_insert(chunks, document_id):
    batches = batch_chunks_by_tokens(chunks)
    total_inserted = 0

    for batch in batches:
        texts = [truncate_to_token_limit(c["text"]) for c in batch]

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

        conn = get_db_connection()
        try:
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
        finally:
            release_db_connection(conn)

        total_inserted += len(rows)

    print(f"Ingested {total_inserted} chunks in {len(batches)} batches")


def update_document_status(document_id: str, status: str, error_message: str | None = None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE documents
            SET status = %s,
            error_message = %s,
            updated_at = now()
            WHERE document_id = %s
            """,
            (status, error_message, document_id),
        )
        conn.commit()
        cursor.close()
    finally:
        release_db_connection(conn)

def document_exists_by_fingerprint(fingerprint: str) -> bool:
    conn = get_db_connection()
    try:
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
        return exists
    finally:
        release_db_connection(conn)