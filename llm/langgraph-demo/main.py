from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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

    # TODO: Test out with a smaller model for all integrations. Then use the larger model.
    llm = ChatOllama(model="qwen3:0.6b", num_ctx=16384)
    # llm = ChatOllama(model="qwen3:4b", num_ctx=16384)

    return create_react_agent(llm, tools=tools, checkpointer=MemorySaver())


class ChatAgent(BaseModel):
    """
    We are creating our own model because we want the CLI/FrontEnd to be decoupled from figuring out
    how an agent works. We want the frontend to only deal with the concept of either Thinking or Response.
    """

    agent: Pregel
    chat_config: dict[str, Any] = Field(
        default_factory=lambda: {"configurable": {"thread_id": "1"}, "recursion_limit": 200}
    )

    def invoke(self, messages: list[BaseMessage]) -> str:
        AIMessage(content="")
        HumanMessage(content="")

        return ""


if __name__ == "__main__":
    agent = ChatAgent(agent=create_agent())

    # We need a thread_id for the chat to retain state between messages
    chat_config = {"configurable": {"thread_id": "1"}, "recursion_limit": 200}
    chat_cli(agent, chat_config=chat_config)
