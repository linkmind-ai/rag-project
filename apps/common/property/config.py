from elasticsearch import Elasticsearch

# Elastic search 연결
ES_HOST = "https://es.nabee.ai.kr"
ES_API_KEY = "EfCAHLahUiiJuB4jvu3_tA"
ES_ID = "mLnV0pkBm3_E9PSPSOOM"
VEC_DIMS = 384

es = Elasticsearch(
    ES_HOST,
    api_key = (ES_ID, ES_API_KEY)
)

NOTION_TOKEN = "ntn_S49134845636QN7OizYlyythCTORUXOCvYcp2U19S0P6dy"
NOTION_VERSION = "2022-06-28"