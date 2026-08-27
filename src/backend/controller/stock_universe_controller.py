from src.backend.core.config import settings
import requests
from datetime import datetime
from sqlalchemy.orm import Session

ZAPI_URL = settings.BASE_URL

def fetch_listing_date():
    """Fetch listing date for all ticker from 1977 to now using ZPI API."""
    
    
def fetch_delisting_date():
    """Fetch delisting date for all ticker from 1977 to now using ZPI API."""
    
def fetch_relisting_date():
    """Fetch relisting date for all ticker from 1977 to now using ZPI API."""