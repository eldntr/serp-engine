import argparse
import asyncio
from src.crawler.engine.static import StaticCrawler
from tests.utils import save_parsed_markdown
from src.crawler.parsers.cleaner import ContentCleaner
from src.crawler.parsers.deduplicator import ContentDeduplicator


async def main(debug: bool = False):
    crawler = StaticCrawler()
    cleaner = ContentCleaner()
    dedup = ContentDeduplicator(threshold=0.80)

    # 1. Ambil artikel nyata dari web
    target_url = "https://en.wikipedia.org/wiki/Information_retrieval"
    print(f"Fetching: {target_url} ...")
    raw_res = await crawler.fetch(target_url)

    # 2. Uji Cleansing & Semantic Markdown Extraction
    print("\n=== 1. TESTING SEMANTIC PARSER & MARKDOWN CONVERSION ===")
    cleaned_doc = cleaner.clean(html=raw_res.html, url=target_url)

    if cleaned_doc:
        print(f"Title       : {cleaned_doc.title}")
        print(f"Word Count  : {cleaned_doc.word_count} words")
        print(f"Metadata    : {cleaned_doc.metadata}")
        print("\n--- SAMPLE MARKDOWN OUTPUT (FIRST 500 CHARACTERS) ---")
        print(cleaned_doc.content_markdown[:500])
        print("...\n-----------------------------------------------------")
        
        save_parsed_markdown(cleaned_doc, debug=debug)
    else:
        print("Failed to extract clean text.")
        return

    # 3. Uji Deduplikasi (MinHash LSH)
    print("\n=== 2. TESTING MINHASH LSH DEDUPLICATION ===")

    # Dokumen asli
    doc_1_id = "wiki-ir-original"
    doc_1_text = cleaned_doc.content_markdown

    # Dokumen duplikat sintetis (artikel yang sama dengan sedikit perubahan kalimat di awal)
    doc_2_id = "wiki-ir-repost"
    doc_2_text = (
        "This is a syndicated repost of the article. "
        + cleaned_doc.content_markdown
    )

    # Dokumen unik lain
    doc_3_id = "unrelated-doc"
    doc_3_text = (
        "Deep Learning and Neural Networks have transformed "
        "the landscape of Artificial Intelligence and Computer Vision."
    )

    # Insert Doc 1 (Original)
    is_saved_1 = dedup.insert(doc_1_id, doc_1_text)
    print(
        f"Insert Doc 1 ({doc_1_id}): {'SUCCESS (Unique)' if is_saved_1 else 'REJECTED (Duplicate)'}"
    )

    # Insert Doc 2 (Near-duplicate)
    is_saved_2 = dedup.insert(doc_2_id, doc_2_text)
    print(
        f"Insert Doc 2 ({doc_2_id}): {'SUCCESS (Unique)' if is_saved_2 else 'REJECTED (Duplicate)'}"
    )

    # Insert Doc 3 (Unrelated unique)
    is_saved_3 = dedup.insert(doc_3_id, doc_3_text)
    print(
        f"Insert Doc 3 ({doc_3_id}): {'SUCCESS (Unique)' if is_saved_3 else 'REJECTED (Duplicate)'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parser Script")
    parser.add_argument("--debug", action="store_true", help="Print debug information and save markdown results")
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug))