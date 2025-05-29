from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.pregel import Pregel
from pydantic import BaseModel, Field

from cli import chat_cli


class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""

    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")


@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    return a * b


def create_agent() -> Pregel:
    tools = [multiply]

    llm = ChatOllama(model="qwen3:4b", num_ctx=16384)

    return create_react_agent(llm, tools=tools, checkpointer=MemorySaver())


if __name__ == "__main__":
    agent = create_agent()

    # We need a thread_id for the chat to retain state between messages
    chat_config = {"configurable": {"thread_id": "1"}, "recursion_limit": 200}
    chat_cli(agent, chat_config=chat_config)
