from pathlib import Path


LLAMA_BIN_FILE = str(
    Path(__file__).resolve().parent.parent / "data" / "llama-2-7b.ggmlv3.q4_0.bin"
)
