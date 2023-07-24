from fastapi import APIRouter, status

from src.llama.llama import llama_cpp_prompt_response
from src.llama.model import Prompt

llama_router = APIRouter(prefix="/llama2", tags=["Invoke LLAMA 2"])


@llama_router.post("/prompt", status_code=status.HTTP_202_ACCEPTED)
async def prompt_response(prompt: Prompt) -> str:
    """
    Calling this is slow, so ideally we should have asynchronous calls,
    meaning we should put the prompt in a Database, and trigger
    a queue with this. Once the queue is done processing,
    it puts the response in another table with the same id.
    We can poll to see if this is done or not!

    MVP: Create the API, host on ECS
    Then, put the prompt and response on Postgres
    Make sure to have a test for each endpoint
    Make sure to log everything and add cloudwatch

    Then we start integrating queueing for asynchronous calls
    """
    return llama_cpp_prompt_response(prompt.prompt)
