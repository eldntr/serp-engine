class SERPBaseException(Exception):
    """Base exception for SERP Engine."""
    pass

class SearchEngineError(SERPBaseException):
    """Raised when there is an error in the search engine pipeline."""
    pass

class CrawlerError(SERPBaseException):
    """Raised when there is an error during crawling."""
    pass
