from langchain.llms import LlamaCpp

from src.config import get_llama_config


def llama_cpp_prompt_response(prompt: str) -> str:
    llama_config = get_llama_config()
    llm = LlamaCpp(
        model_path=str(llama_config.llam_bin_file),
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},  # type: ignore
        verbose=False,
    )
    llm.client.verbose = False

    return llm(prompt)
