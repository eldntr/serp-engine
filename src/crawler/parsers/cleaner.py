import json
import trafilatura
from readability import Document
from typing import Optional, Dict, Any
from src.crawler.parsers.schemas import CleanedDocument


class ContentCleaner:
    def __init__(self, min_output_length: int = 100):
        self.min_output_length = min_output_length

    def clean(self, html: str, url: str = "") -> Optional[CleanedDocument]:
        if not html or len(html.strip()) == 0:
            return None

        extracted_json = trafilatura.extract(
            html,
            url=url,
            output_format="json",
            include_links=True,
            include_images=False,
            include_tables=True,
            include_formatting=True,
            favor_precision=True,
            favor_recall=False,
        )

        if extracted_json:
            data = json.loads(extracted_json)
            content_md = data.get("text", "")
            title = data.get("title")
            author = data.get("author")
            date = data.get("date")
            raw_text = data.get("raw_text", content_md)

            meta = {
                "sitename": data.get("sitename"),
                "description": data.get("description"),
                "categories": data.get("categories"),
                "tags": data.get("tags"),
                "fingerprint": data.get("fingerprint"),
            }

            if len(content_md.strip()) >= self.min_output_length:
                return CleanedDocument(
                    url=url,
                    title=title,
                    author=author,
                    date=date,
                    content_markdown=content_md,
                    raw_text=raw_text,
                    word_count=len(content_md.split()),
                    metadata=meta,
                )

        try:
            doc = Document(html)
            summary_html = doc.summary()
            raw_text = trafilatura.html2txt(summary_html)
            title = doc.short_title()

            if len(raw_text.strip()) >= self.min_output_length:
                return CleanedDocument(
                    url=url,
                    title=title,
                    content_markdown=raw_text,
                    raw_text=raw_text,
                    word_count=len(raw_text.split()),
                    metadata={"extractor": "readability_fallback"},
                )
        except Exception:
            pass

        return None