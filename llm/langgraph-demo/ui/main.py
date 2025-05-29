import asyncio
import re

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent
from nicegui import ui
from pydantic import BaseModel, Field


class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""

    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")


@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    writer = get_stream_writer()
    # Stream custom progress data
    writer(f"Multiplying {a} × {b} using Python...")
    result = a * b
    writer(f"Result: {result}")
    return result


def parse_thinking_content(content: str):
    """Parse thinking tags from qwen model output"""
    thinking_pattern = r"<think>(.*?)</think>"
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)

    # Remove thinking tags from content to get clean response
    clean_content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()

    return thinking_matches, clean_content


async def simulate_char_streaming(content: str, thinking_section, response_content, current_thinking, current_response):
    """Simulate character-by-character streaming for smooth UI effect"""
    accumulated = ""

    for char in content:
        accumulated += char

        # Parse thinking vs response content as we build up
        thinking_matches, clean_content = parse_thinking_content(accumulated)

        # Check if thinking is complete
        thinking_complete = "</think>" in accumulated

        # Update thinking section if we found thinking content
        if thinking_matches:
            new_thinking = "\n\n".join(thinking_matches)
            if new_thinking != current_thinking[0]:
                current_thinking[0] = new_thinking
                thinking_section.clear()
                with thinking_section:
                    ui.markdown(current_thinking[0]).classes("text-sm text-gray-600 whitespace-pre-wrap")
                if not thinking_section.value:  # Auto-open if closed
                    thinking_section.open()

        # Auto-collapse thinking section when thinking is complete and we have response content
        if thinking_complete and clean_content.strip() and thinking_section.value:
            thinking_section.close()

        # Update response section with clean content
        if clean_content.strip() != current_response[0]:
            current_response[0] = clean_content.strip()
            response_content.content = current_response[0]

        # Auto-scroll and allow UI to update
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.02)  # Character streaming speed


@ui.page("/")
def main():
    tools = [multiply]
    llm = ChatOllama(model="qwen3:4b")

    # Create streaming LangGraph agent
    agent = create_react_agent(llm, tools=tools)

    async def send() -> None:
        question = text.value
        text.value = ""

        with message_container:
            ui.chat_message(text=question, name="You", sent=True)

            # Create message container for bot response
            bot_message = ui.chat_message(name="Bot", sent=False)

            thinking_section = None
            response_content = None

            with bot_message:
                # Create thinking section (expandable)
                thinking_section = ui.expansion("🤔 Thinking Process", icon="psychology").classes("w-full mb-4")
                thinking_section.close()  # Start closed

                # Create response section
                response_section = ui.card().classes("w-full")
                with response_section:
                    response_content = ui.html("")

            spinner = ui.spinner(type="dots")

        current_thinking = [""]
        current_response = [""]
        accumulated_content = ""

        try:
            # Use LangGraph with multiple stream modes for comprehensive streaming
            print("Starting LangGraph multi-mode streaming...")

            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": question}]}, stream_mode=["updates", "messages", "custom"]
            ):
                stream_mode, data = chunk
                print(f"Stream mode: {stream_mode}")

                if stream_mode == "custom":
                    # Custom data from tools (e.g., progress updates)
                    print(f"Custom data: {data}")
                    if isinstance(data, str):
                        current_response[0] = f"🔧 {data}"
                        response_content.content = current_response[0]
                        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(0.5)  # Show tool progress briefly

                elif stream_mode == "updates":
                    # State updates after each node
                    print(f"Update: {data}")
                    # Could show step progress here if needed

                elif stream_mode == "messages":
                    message_chunk, metadata = data
                    print(f"Message from {metadata.get('langgraph_node', 'unknown')}")

                    # Handle different message types
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        content = str(message_chunk.content)
                        print(f"Content length: {len(content)} chars")

                        # Simulate character-by-character streaming for this response
                        await simulate_char_streaming(
                            content, thinking_section, response_content, current_thinking, current_response
                        )

                    elif hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                        # Tool call decision
                        tool_call = message_chunk.tool_calls[0]
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        current_response[0] = f"🔧 Calling {tool_name} with {tool_args}"
                        response_content.content = current_response[0]
                        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(0.3)

        except Exception as e:
            # Handle any streaming errors
            print(f"Error: {e}")
            with bot_message:
                ui.html(f"<p style='color: red;'>Error: {str(e)}</p>")

        finally:
            message_container.remove(spinner)

    ui.add_css(r"a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}")
    ui.add_css(".q-expansion__content { max-height: 300px; overflow-y: auto; }")

    # the queries below are used to expand the content down to the footer (content can then use flex-grow to expand)
    ui.query(".q-page").classes("flex")
    ui.query(".nicegui-content").classes("w-full")

    with ui.tabs().classes("w-full") as tabs:
        chat_tab = ui.tab("Chat")
    with ui.tab_panels(tabs, value=chat_tab).classes("w-full max-w-2xl mx-auto flex-grow items-stretch"):
        message_container = ui.tab_panel(chat_tab).classes("items-stretch")

    with ui.footer().classes("bg-white"), ui.column().classes("w-full max-w-3xl mx-auto my-6"):
        with ui.row().classes("w-full no-wrap items-center"):
            placeholder = "Ask a question (e.g., 'What's 42 x 7?')"
            text = (
                ui.input(placeholder=placeholder)
                .props("rounded outlined input-class=mx-3")
                .classes("w-full self-center")
                .on("keydown.enter", send)
            )
        ui.markdown(
            "Chat app with LangGraph agent and thinking model - built with [NiceGUI](https://nicegui.io)"
        ).classes("text-xs self-end mr-8 m-[-1em] text-primary")


ui.run(title="LangGraph Chat with Thinking", port=8080)
