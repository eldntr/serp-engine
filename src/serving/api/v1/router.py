from fastapi import APIRouter
from src.serving.api.v1.endpoints import search, crawl

api_router = APIRouter()
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(crawl.router, prefix="/crawl", tags=["Crawl"])
