from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""

    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")


@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    return a * b


tools = [multiply]

llm = ChatOllama(model="qwen3:4b")

agent = create_react_agent(
    llm,
    tools=tools,
)

output = agent.invoke({"messages": [{"role": "user", "content": "what's 42 x 7?"}]})
print(output)
