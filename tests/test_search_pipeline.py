import asyncio
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.search.rankers.rrf import ReciprocalRankFusion
from src.search.rankers.cross_encoder import NeuralReranker
from src.search.query.snippets import SnippetExtractor


async def main():
    print("=== 1. INISIALISASI ENGINE & MODEL SEARCH ===")
    dense_enc = DenseEncoder()
    sparse_enc = BM25SparseEncoder()
    # Menggunakan mode in-memory Qdrant untuk pengetesan langsung
    indexer = QdrantHybridIndexer(collection_name="serp_demo", host=":memory:")
    await indexer.init_collection()

    rrf = ReciprocalRankFusion(k=60)
    reranker = NeuralReranker()
    snippet_extractor = SnippetExtractor()

    # Dokumen sampel hasil crawling web
    docs = [
        {
            "id": "doc-1",
            "url": "https://example.com/vector-search",
            "title": "Memahami Vector Database dan Semantic Search",
            "content": "Vector database seperti Qdrant memungkinkan penyimpanan embedding dense berdimensi tinggi untuk mencari kesamaan semantik dokumen secara efisien pada skala miliaran data.",
        },
        {
            "id": "doc-2",
            "url": "https://example.com/bm25-explained",
            "title": "Algoritma BM25 untuk Lexical Search",
            "content": "BM25 adalah algoritma perangkingan informasi berbasis frekuensi kata kunci sparse matching yang sangat akurat untuk menemukan kata kunci eksak, kode, dan identitas dokumen.",
        },
        {
            "id": "doc-3",
            "url": "https://example.com/hybrid-search-guide",
            "title": "Panduan Hybrid Search: BM25 + Dense Vectors",
            "content": "Hybrid Search menggabungkan keunggulan lexical matching BM25 dan dense neural search menggunakan algoritma Reciprocal Rank Fusion RRF untuk hasil SERP yang optimal dan presisi.",
        },
    ]

    print("\n=== 2. MENGINDEKS DOKUMEN KE QDRANT ===")
    for doc in docs:
        d_vec = dense_enc.encode(doc["content"])[0]
        s_vec = sparse_enc.encode(doc["content"])
        await indexer.upsert_document(
            doc_id=doc["id"],
            dense_vector=d_vec,
            sparse_vector=s_vec,
            payload={"url": doc["url"], "title": doc["title"], "text": doc["content"]},
        )
        print(f"Indeks berhasil: [{doc['id']}] - {doc['title']}")

    # 3. Eksekusi Pencarian
    user_query = "bagaimana arsitektur hybrid search menggabungkan bm25 dan vector"
    print(f"\n=== 3. EKSEKUSI PENCARIAN HYBRID: '{user_query}' ===")

    q_dense = dense_enc.encode(user_query)[0]
    q_sparse = sparse_enc.encode(user_query)

    dense_hits = await indexer.search_dense(q_dense, limit=5)
    sparse_hits = await indexer.search_sparse(q_sparse, limit=5)

    # 4. Reciprocal Rank Fusion
    fused_candidates = rrf.fuse(dense_hits, sparse_hits, top_n=5)
    print(f"Total kandidat setelah RRF: {len(fused_candidates)}")

    candidate_dicts = [
        {"id": c.doc_id, "text": c.payload["text"], "title": c.payload["title"], "url": c.payload["url"]}
        for c in fused_candidates
    ]

    # 5. Cross-Encoder Re-Ranking
    print("\n=== 4. NEURAL CROSS-ENCODER RE-RANKING ===")
    final_results = reranker.rerank(query=user_query, candidates=candidate_dicts, top_k=3)

    # 6. Menampilkan Hasil SERP Akhir
    print("\n=== HASIL FINAL SERP (TOP-RANKED) ===")
    for rank, res in enumerate(final_results, start=1):
        snippet = snippet_extractor.extract(res["text"], user_query)
        print(f"\n[Rank #{rank}] Score: {res['cross_score']:.4f}")
        print(f"Title  : {res['title']}")
        print(f"URL    : {res['url']}")
        print(f"Snippet: {snippet}")


if __name__ == "__main__":
    asyncio.run(main())
