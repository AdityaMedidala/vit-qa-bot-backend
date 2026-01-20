from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2 

from app.retrieval_core import (
    retrieve_from_scored_chunks,
    build_context,
    embed_texts
)
#limit controls number of candidates
def retrieval_raw(query: str,limit:int=10):
    query=query.strip()
    query_emb=embed_texts([query])[0]
    conn=psycopg2.connect(os.getenv("SUPABASE_URL"))
    cursor=conn.cursor()

    sql='''
    SELECT chunk_id,
    text,
    metadata,
    1-(embedding <=> %s::vector) as similarity 
    from document_chunks
    order by similarity desc 
    limit %s;
    '''

    cursor.execute(sql,(query_emb,limit))
    rows=cursor.fetchall()

    cursor.close()
    conn.close()

    result=[]

    for chunk_id,text,metadata,similarity in rows:
        result.append((
            similarity,
            {
                "chunk_id": chunk_id,
                "text":text,
                "metadata":metadata
            }
        ))
    return result

def retrieve_sql(query: str):
    scored = retrieval_raw(query, limit=10)
    return retrieve_from_scored_chunks(scored)


if __name__ =="__main__":
    query_text=input("\n Ask a question \n")
    result = retrieve_sql(query_text)