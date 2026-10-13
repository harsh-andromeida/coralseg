#!/usr/bin/env python3
"""
Andromeida Reef Vision Studio - Standalone FastAPI Server Launcher
"""

import sys
import argparse
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Launch ReefVision FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"  Starting Andromeida Reef Vision Studio Backend  ")
    print(f"  URL: http://{args.host}:{args.port}              ")
    print(f"  Docs: http://{args.host}:{args.port}/docs        ")
    print(f"==================================================")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
