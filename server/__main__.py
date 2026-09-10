"""Run the FastAPI server with uvicorn: python -m server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="127.0.0.1", port=8877, reload=True)
