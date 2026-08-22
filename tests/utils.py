from pathlib import Path

def save_crawl_result_json(res, folder="tests/results", debug=False):
    """Menyimpan hasil crawling ke dalam format JSON jika mode debug aktif."""
    if not debug:
        return
    Path(folder).mkdir(parents=True, exist_ok=True)
    safe_name = "".join([c if c.isalnum() else "_" for c in res.url])
    filepath = Path(folder) / f"{res.engine}_{safe_name}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(res.model_dump_json(indent=2))
    print(f"  -> Saved JSON result to: {filepath}")

def save_parsed_markdown(cleaned_doc, folder="tests/results", debug=False):
    """Menyimpan hasil parsing teks ke dalam format Markdown jika mode debug aktif."""
    if not debug:
        return
    Path(folder).mkdir(parents=True, exist_ok=True)
    safe_name = "".join([c if c.isalnum() else "_" for c in cleaned_doc.url])
    filepath = Path(folder) / f"parser_{safe_name}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_doc.content_markdown)
    print(f"  -> Saved parsed markdown to: {filepath}")
