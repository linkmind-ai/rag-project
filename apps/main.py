import uvicorn
from common.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )

    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload
