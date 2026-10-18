"""Source and PyInstaller entry point."""
from imageboost_pro.application import safe_main

if __name__ == "__main__":
    raise SystemExit(safe_main())
