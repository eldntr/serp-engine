import json
import re
from typing import Any, Dict, List, Optional
from src.client.factory import get_llm_client
from src.core.config import settings
from src.core.logger import logger

class QueryGeneratorNode:
    """LangGraph node to convert a user prompt into a list of web search queries using an LLM."""
    
    def __init__(
        self, 
        provider: Optional[str] = None, 
        api_base: Optional[str] = None, 
        model: Optional[str] = None,
        max_queries: int = 4
    ):
        # Gunakan settings dari config jika tidak disediakan
        self.provider = provider or settings.llm.llm_provider
        self.api_base = api_base or settings.llm.llm_api_base
        self.model = model or settings.llm.llm_model
        self.max_queries = max_queries
        
        # Inisialisasi LLM client
        self.client = get_llm_client(
            provider=self.provider,
            api_base=self.api_base,
            model=self.model,
            api_key=settings.llm.llm_api_key,
            timeout=settings.llm.llm_timeout
        )

    def _parse_queries(self, response_text: str) -> List[str]:
        """Secara tangguh mem-parse respons LLM menjadi list of string query."""
        text = response_text.strip()
        
        # 1. Coba ekstrak dari JSON code blocks jika ada markdown
        json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if json_match:
            try:
                queries = json.loads(json_match.group(1))
                if isinstance(queries, list):
                    return [str(q).strip() for q in queries if q]
            except Exception as e:
                logger.warning(f"Gagal mem-parse JSON dari code block: {e}")

        # 2. Coba mem-parse seluruh respons langsung sebagai JSON list
        try:
            queries = json.loads(text)
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q]
        except Exception:
            pass

        # 3. Fallback: Parse baris per baris (mendukung list angka maupun bullet point)
        queries = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Hapus bullet points atau nomor baris seperti: "1. query", "- query", "* query"
            cleaned = re.sub(r"^(?:\d+[\.\)]|[-*+])\s*", "", line).strip()
            # Hapus kutipan pembungkus jika ada
            cleaned = re.sub(r"^['\"]|['\"]$", "", cleaned).strip()
            if cleaned and cleaned not in ["", "json"]:
                queries.append(cleaned)

        return queries

    async def generate_search_queries(self, prompt: str) -> List[str]:
        """Menerima prompt dan mengubahnya menjadi daftar kata kunci pencarian yang optimal."""
        system_prompt = (
            "You are an expert search query planner. Your task is to analyze a user prompt "
            "and generate a list of search queries that will retrieve the most relevant "
            "and factual information from the web to answer the prompt.\n\n"
            "Guidelines:\n"
            "1. Generate search queries that are keyword-focused, concise, and diverse.\n"
            f"2. Return between 1 and {self.max_queries} queries.\n"
            "3. Format your response strictly as a JSON list of strings. Do not add markdown code blocks or explanations.\n"
            "Example output format:\n"
            '["query number one", "query number two"]'
        )

        user_prompt = f"User Prompt: {prompt}\n\nSearch Queries:"
        
        logger.info(f"Mulai membuat search query untuk prompt: '{prompt}' menggunakan model '{self.model}'")
        try:
            response_text = await self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2  # Gunakan temperatur rendah untuk hasil terstruktur yang konsisten
            )
            logger.debug(f"LLM Raw response: {response_text}")
            queries = self._parse_queries(response_text)
            
            # Batasi jumlah query yang dikembalikan
            queries = queries[:self.max_queries]
            
            logger.success(f"Berhasil menghasilkan {len(queries)} search queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Gagal memproses search queries: {e}")
            # Fallback menggunakan prompt asli jika terjadi kesalahan fatal
            return [prompt]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node signature untuk integrasi StateGraph LangGraph.
        
        Menerima Graph State (dict) yang berisi key 'prompt' dan mengembalikan
        pembaruan state berupa dict dengan key 'queries'.
        """
        prompt = state.get("prompt", "")
        if not prompt:
            logger.warning("Node 'QueryGeneratorNode' menerima prompt kosong di State.")
            return {"queries": []}
            
        queries = await self.generate_search_queries(prompt)
        return {"queries": queries}
