"""
NOVA Admin Router
=================
Mounts at /admin — protected by ADMIN_SECRET env var.
All endpoints require ?secret=<ADMIN_SECRET> in the query string
or X-Admin-Secret header.

Endpoints:
  GET  /admin/stats                  — total docs, chunks, last ingestion
  GET  /admin/documents              — list all documents with status + chunk count
  DELETE /admin/documents/{doc_id}   — delete document + all its chunks
  POST /admin/documents/{doc_id}/reingest  — mark failed doc for re-ingestion (resets status)
"""

import os
from fastapi import APIRouter, HTTPException, Query, Header
from typing import Optional

from app.db import get_db_connection, release_db_connection

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "nova-admin-2025")


# ── Auth helper ───────────────────────────────────────────────────────────────

def verify_auth(
    secret: Optional[str] = Query(None),
    x_admin_secret: Optional[str] = Header(None),
):
    token = secret or x_admin_secret
    if token != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── GET /admin/stats ──────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(
    secret: Optional[str] = Query(None),
    x_admin_secret: Optional[str] = Header(None),
):
    verify_auth(secret, x_admin_secret)
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents WHERE status = 'done'")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'processing'"
        )
        processing = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'failed'"
        )
        failed = cur.fetchone()[0]

        cur.execute(
            "SELECT MAX(updated_at) FROM documents WHERE status = 'done'"
        )
        last_ingestion = cur.fetchone()[0]

        cur.close()
        return {
            "total_docs":      total_docs,
            "total_chunks":    total_chunks,
            "processing":      processing,
            "failed":          failed,
            "last_ingestion":  last_ingestion.isoformat() if last_ingestion else None,
        }
    finally:
        release_db_connection(conn)


# ── GET /admin/documents ──────────────────────────────────────────────────────

@router.get("/documents")
def list_documents(
    secret: Optional[str] = Query(None),
    x_admin_secret: Optional[str] = Header(None),
):
    verify_auth(secret, x_admin_secret)
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                d.document_id,
                d.document_name,
                d.status,
                d.error_message,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON c.document_id = d.document_id
            GROUP BY d.document_id
            ORDER BY d.updated_at DESC
        """)
        rows = cur.fetchall()
        cur.close()

        return [
            {
                "document_id":   str(row[0]),
                "document_name": row[1],
                "status":        row[2],
                "error_message": row[3],
                "created_at":    row[4].isoformat() if row[4] else None,
                "updated_at":    row[5].isoformat() if row[5] else None,
                "chunk_count":   row[6],
            }
            for row in rows
        ]
    finally:
        release_db_connection(conn)


# ── DELETE /admin/documents/{doc_id} ─────────────────────────────────────────

@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str,
    secret: Optional[str] = Query(None),
    x_admin_secret: Optional[str] = Header(None),
):
    verify_auth(secret, x_admin_secret)
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Check it exists
        cur.execute(
            "SELECT document_name FROM documents WHERE document_id = %s",
            (doc_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_name = row[0]

        # Delete chunks first (FK constraint), then the document row
        cur.execute(
            "DELETE FROM document_chunks WHERE document_id = %s", (doc_id,)
        )
        chunks_deleted = cur.rowcount

        cur.execute(
            "DELETE FROM documents WHERE document_id = %s", (doc_id,)
        )
        conn.commit()
        cur.close()

        return {
            "deleted":        True,
            "document_name":  doc_name,
            "chunks_deleted": chunks_deleted,
        }
    finally:
        release_db_connection(conn)


# ── POST /admin/documents/{doc_id}/reingest ───────────────────────────────────

@router.post("/documents/{doc_id}/reingest")
def mark_for_reingest(
    doc_id: str,
    secret: Optional[str] = Query(None),
    x_admin_secret: Optional[str] = Header(None),
):
    """
    Resets a document's status to 'pending_reingest' and clears its chunks
    so the Colab ingestion notebook can pick it up on the next run.
    The fingerprint is also cleared so the duplicate-check doesn't skip it.
    """
    verify_auth(secret, x_admin_secret)
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT document_name, status FROM documents WHERE document_id = %s",
            (doc_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_name = row[0]

        # Clear chunks so re-ingestion inserts fresh ones
        cur.execute(
            "DELETE FROM document_chunks WHERE document_id = %s", (doc_id,)
        )
        chunks_cleared = cur.rowcount

        # Reset status + fingerprint so Colab notebook doesn't skip it
        cur.execute(
            """
            UPDATE documents
            SET status        = 'pending_reingest',
                fingerprint   = fingerprint || '_reset',
                error_message = NULL,
                updated_at    = now()
            WHERE document_id = %s
            """,
            (doc_id,)
        )
        conn.commit()
        cur.close()

        return {
            "document_name":  doc_name,
            "chunks_cleared": chunks_cleared,
            "status":         "pending_reingest",
            "message": (
                "Document marked for re-ingestion. "
                "Run the Colab ingestion notebook to re-process it."
            ),
        }
    finally:
        release_db_connection(conn)