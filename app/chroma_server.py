import chromadb
from chromadb.server.fastapi import FastAPI
import uvicorn


def run_chroma_server():
    settings = chromadb.config.Settings(
        persist_directory="chroma_db",
        allow_reset=True,
        is_persistent=True
    )

    server = FastAPI(settings)
    app = server.app

    print("Starting ChromaDB server on http://0.0.0.0:8000")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        run_chroma_server()
    except KeyboardInterrupt:
        print("\nShutting down ChromaDB server.")