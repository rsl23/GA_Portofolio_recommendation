from pydantic import BaseModel
from typing import List


# Payload data untuk endpoint /market/filter-stocks
class MarketFilterResponse(BaseModel):
    total_saham: int
    # List[dict] agar fleksibel terhadap kolom dataframe hasil preprocessing
    data: List[dict]
