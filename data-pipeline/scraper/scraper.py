import os
import re
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
from playwright.async_api import async_playwright

# Dynamically set OUTPUT_DIR to point to the root 'data/' folder
# This file is in: VIT-RAG-Project/data_pipeline/scraper/scraper.py
# So we need to go up three levels to get to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

DELAY = 1.5
TIMEOUT = 15
MAX_PDF_MB = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; VIT-Research-Bot/1.0)"}

SEED_URLS = [
    "https://vit.ac.in/academics",
    "https://vit.ac.in/admissions",
    "https://vit.ac.in/research",
    "https://vit.ac.in/campus-life/hostels",
    "https://vit.ac.in/placements",
    "https://vit.ac.in/schools/sense",
    "https://vit.ac.in/schools/scope",
    "https://vit.ac.in/schools/site",
    "https://vit.ac.in/schools/select",
    "https://vit.ac.in/schools/smec",
    "https://vit.ac.in/schools/sbst",
    "https://vit.ac.in/schools/ssl",
    "https://vit.ac.in/schools/design",
    "https://vit.ac.in/schools/sense/academics",
    "https://vit.ac.in/schools/scope/academics",
    "https://vit.ac.in/schools/site/academics",
]


def is_vit_domain(url):
    return "vit.ac.in" in urlparse(url).netloc.lower()


def url_to_filename(url):
    path = urlparse(url).path
    name = os.path.basename(path)
    if not name.lower().endswith(".pdf"):
        name = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
    return re.sub(r"[^\w\-.]", "_", name)


def get_pdf_size(url):
    try:
        r = requests.head(url, headers=HEADERS, timeout=8, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        if cl:
            kb = int(cl) / 1024
            return f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
    except Exception:
        pass
    return "unknown"


async def get_links_async(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=20000, wait_until="networkidle")
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        pdfs, subpages = [], []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith("javascript"):
                continue
            full = urljoin(url, href)
            if full.lower().endswith(".pdf") and is_vit_domain(full):
                pdfs.append(full)
            elif (is_vit_domain(full)
                  and urlparse(full).scheme in ("http", "https")
                  and "#" not in full and full != url):
                subpages.append(full)
        return list(set(pdfs)), list(set(subpages))
    except Exception as e:
        print(f"  ⚠️  {url}: {e}")
        return [], []


def get_links(url):
    # Standard Python async wrapper instead of Colab's nest_asyncio hack
    return asyncio.run(get_links_async(url))


def preview_pdfs(seed_urls, max_pages=120):
    visited, queue = set(), list(seed_urls)
    found = {}

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"🔍 [{len(visited)}/{max_pages}] {url}")

        pdfs, subpages = get_links(url)

        for pdf_url in pdfs:
            if pdf_url not in found:
                found[pdf_url] = {
                    "filename": url_to_filename(pdf_url),
                    "source_page": url,
                    "url": pdf_url,
                }
                print(f"  📄 {url_to_filename(pdf_url)}")

        if url in seed_urls:
            for link in subpages:
                if link not in visited:
                    queue.append(link)

        time.sleep(DELAY)

    print(f"\n{'=' * 60}")
    print(f"Pages crawled: {len(visited)} | PDFs found: {len(found)}")
    print("Checking sizes...")

    rows, known_kb = [], []
    for url, meta in found.items():
        size = get_pdf_size(url)
        rows.append({"filename": meta["filename"], "size": size,
                     "source_page": meta["source_page"], "url": url})
        if "KB" in size:
            known_kb.append(float(size.replace(" KB", "")))
        elif "MB" in size:
            known_kb.append(float(size.replace(" MB", "")) * 1024)

    if known_kb:
        print(f"Estimated total: ~{sum(known_kb) / 1024:.1f} MB\n")

    if rows:
        df = pd.DataFrame(rows)[["filename", "size", "source_page", "url"]]
        print("\n--- Summary of Found PDFs ---")
        print(df.head(20).to_string())  # Print top 20 instead of crashing terminal
        if len(df) > 20:
            print(f"... and {len(df) - 20} more.")

    return found


def download_all(found, skip_keywords=None):
    skip_keywords = skip_keywords or []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = set(os.listdir(OUTPUT_DIR))

    total = len(found)
    downloaded = skipped = failed = 0

    for i, (url, meta) in enumerate(found.items(), 1):
        filename = meta["filename"]

        if any(kw.lower() in filename.lower() for kw in skip_keywords):
            print(f"[{i}/{total}] ⏭️  Filtered: {filename}")
            skipped += 1
            continue

        dest = os.path.join(OUTPUT_DIR, filename)
        if filename in existing or (os.path.exists(dest) and os.path.getsize(dest) > 1024):
            print(f"[{i}/{total}] ⏭️  Already have: {filename}")
            skipped += 1
            continue

        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            cl = int(r.headers.get("Content-Length", 0))
            if cl > MAX_PDF_MB * 1024 * 1024:
                print(f"[{i}/{total}] ⏭️  Too large ({cl // 1024 // 1024}MB): {filename}")
                skipped += 1
                continue

            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            size_kb = os.path.getsize(dest) // 1024
            print(f"[{i}/{total}] ✅ {filename} ({size_kb} KB)")
            downloaded += 1
        except Exception as e:
            print(f"[{i}/{total}] ❌ {filename}: {e}")
            failed += 1
            if os.path.exists(dest):
                os.remove(dest)
        time.sleep(0.3)

    total_bytes = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    )

    print(f"\n{'=' * 60}")
    print(f"Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    print(f"Folder size: {total_bytes / 1024 / 1024:.1f} MB")


# Standard Python Execution Block
if __name__ == "__main__":
    print(f"✅ Config ready. Output directory set to: {OUTPUT_DIR}")

    # 1. Crawl and find PDFs
    found_pdfs = preview_pdfs(SEED_URLS)

    # 2. Download them with filters
    download_all(found_pdfs, skip_keywords=[
        "Minutes-of-the", "Phy-Edu-Achievements", "Sports-Achievements",
        "EventAchievements", "sports_achievements", "Events-20", "Events-AY",
        "Newletter", "Newsletter", "ssl-conference", "SSL-International",
        "IEA-conference", "SSl-Conference", "SENSE-HACK", "Engineers-day",
        "DHR-ICMR", "Affidavit", "SponsorshipLetter", "EmployerCertificate",
        "Undertaking", "PhysicalFitness", "Country-Codes", "SMEC-Type",
        "Declaration-best", "Acad-Calen-for-Trimester",
        "Acad-Calen-for-Fall-Semester-2022", "Acad-Calen-for-Fall-Semester-2023",
        "Acad-Calen-for-Winter-Semester-2022", "Acad-Calen-for-Winter-Semester-2023",
        "VITREE-January-2025", "selected-candidates-instructions-july-2024",
        "Personal-Interview-Details", "VITREE-2026-Hall_Ticket",
        "Instructions-to-the-selected", "Awarness-Circular",
        "template-for-two-page", "Template-Deep-Tech",
    ])