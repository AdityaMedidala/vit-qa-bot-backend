from langchain_text_splitters import MarkdownHeaderTextSplitter
import re
import json
from collections import defaultdict

MAX_CHARS = 2000
FALLBACK_CHUNK_SIZE = 1800

#Checks so tables dont get cut in the middle 
def safe_character_split(text: str, max_chars: int):
    chunks = []
    current = []

    in_table = False
    current_len = 0

    for line in text.splitlines(keepends=True):
        if "|" in line:
            in_table = True
        elif in_table and line.strip() == "":
            in_table = False

        #splitting logic checks for if it is outside table and if the current chunk is too long 
        if current_len + len(line) > max_chars and not in_table:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)

    #for final lines
    if current:
        chunks.append("".join(current))

    return chunks


def group_by_level_1(docs):
    grouped = defaultdict(list)

    for doc in docs:
        level_1 = doc.metadata.get("level_1", "UNKNOWN")
        grouped[level_1].append(doc)

    return grouped

#for splitting by subsections until it fits charecter limits
def recursive_split(docs, levels=("level_2", "level_3", "level_4")):
    #one big string for checking size
    combined_text = "\n".join(doc.page_content for doc in docs)

    if len(combined_text) <= MAX_CHARS:
        return [combined_text]
    #fall back
    if not levels:
        return safe_character_split(combined_text, FALLBACK_CHUNK_SIZE)
    #alwas picks highest level
    level = levels[0]
    buckets = defaultdict(list)

    for doc in docs:
        key = doc.metadata.get(level)
        buckets[key].append(doc)

    chunks = []
    #recursion to send smaller parts
    for bucket_docs in buckets.values():
        chunks.extend(recursive_split(bucket_docs, levels[1:]))

    return chunks

#create slug to make a URL-friendly version of the string by removeing special charecters
def slugify(text: str) -> str:
    text = text.lower()
    # to swap out bad charecters
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_final_chunks(docs, document_name: str):
    grouped = group_by_level_1(docs)
    final_chunks = []
    
    empty_count = 0
    tiny_count = 0

    for level_1, section_docs in grouped.items():
        chunks = recursive_split(section_docs)

        chunk_idx = 0  # Separate counter for non-empty chunks only
        for chunk_text in chunks:
            # Filter empty chunks
            if len(chunk_text.strip()) == 0:
                empty_count += 1
                continue
            
            # Tiny chunks which are insufficient for RAG
            if len(chunk_text) < 100:
                tiny_count += 1
            
            chunk_id = (
                f"{slugify(document_name)}__"
                f"{slugify(level_1)}__chunk_{chunk_idx:03d}"
            )
            chunk_idx += 1

            final_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document": document_name,
                    "level_1": level_1,
                    "char_count": len(chunk_text),
                }
            })
    
    # info for report
    if empty_count > 0:
        print(f"filtered out {empty_count} empty chunks")
    if tiny_count > 0:
        print(f"{tiny_count} chunks are <100 chars")

    return final_chunks



# Header splitting
def headers_splitter(markdown: str):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "level_1"),
            ("##", "level_2"),
            ("###", "level_3"),
            ("####", "level_4"),
        ]
    )
    return splitter.split_text(markdown)


#Logic to catch and find TOC(Tables of contents) which can confuse LLM
INSTRUCTIONAL_KEYWORDS = {
    "credit", "credits", "course", "courses", "semester",
    "lecture", "practical", "lab", "hours", "category",
    "programme", "program", "curriculum", "structure"
}

TOC_KEYWORDS = {
    "table of contents", "contents", "index"
}


def contains_markdown_table(text: str) -> bool:
    return "|---" in text or re.search(r"\|\s*[-:]+\s*\|", text) is not None


def contains_instructions(text: str) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in INSTRUCTIONAL_KEYWORDS)


def toc_header(metadata: dict) -> bool:
    level_1 = (metadata.get("level_1") or "").lower()
    return any(k in level_1 for k in TOC_KEYWORDS)


def obvious_toc(text: str, section_index: int) -> bool:
    text_lower = text.lower()
    
    # Must be in first 3 sections
    if section_index > 2:
        return False
    
    # Must have markdown table
    if not contains_markdown_table(text):
        return False
    
    # Must have TOC keywords
    has_contents = "contents" in text_lower and "page" in text_lower
    
    if not has_contents:
        return False
    
    # Looking for patterns TOC like patterns
    section_pattern = re.compile(r'\|\s*\d+\.[\d\.]*\s*\|')
    section_refs = len(section_pattern.findall(text))
    
    # Must have at least 10 section references
    if section_refs < 10:
        return False
    
    return True


def classify_section_as_toc_candidate(
    text: str,
    metadata: dict,
    section_index: int,
    total_sections: int
) -> bool:
 
    # first check
    if obvious_toc(text, section_index):
        return True
    
    return False

def chunk_markdown_document(
    markdown: str,
    document_name: str,
):
    # Header split
    docs = headers_splitter(markdown)

    total_sections = len(docs)

    #Drop TOC sections
    instructional_docs = [
        doc
        for idx, doc in enumerate(docs)
        if not classify_section_as_toc_candidate(
            text=doc.page_content,
            metadata=doc.metadata,
            section_index=idx,
            total_sections=total_sections,
        )
    ]

    #Build final chunks
    final_chunks = build_final_chunks(
        docs=instructional_docs,
        document_name=document_name,
    )

    return final_chunks



# Inspection pipeline for debugging
def inspect_and_classify_sections(markdown: str, output_path: str):
    docs = headers_splitter(markdown)
    total_sections = len(docs)

    inspection_output = []

    for idx, doc in enumerate(docs):
        text = doc.page_content
        metadata = doc.metadata

        is_toc = classify_section_as_toc_candidate(
            text=text,
            metadata=metadata,
            section_index=idx,
            total_sections=total_sections
        )

        entry = {
            "section_index": idx,
            "section_type": "toc_candidate" if is_toc else "instructional",
            "metadata": metadata,
            "text": text,
            "debug": {
                "has_markdown_table": contains_markdown_table(text),
                "has_instructional_language": contains_instructions(text),
                "is_toc_header": toc_header(metadata),
                "is_extremely_obvious_toc": obvious_toc(text, idx),
                "early_section": idx / max(total_sections, 1) < 0.15,
                "line_count": len([l for l in text.splitlines() if l.strip()])
            }
        }

        inspection_output.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inspection_output, f, indent=2, ensure_ascii=False)