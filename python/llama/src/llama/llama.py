from pathlib import Path
from langchain.llms import LlamaCpp

from src.config import LLAMA_BIN_FILE


def llama_cpp_prompt_response(prompt: str) -> str:
    llm = LlamaCpp(
        model_path=str(LLAMA_BIN_FILE),
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        verbose=False,
    )
    llm.client.verbose = False

    return llm(prompt)
