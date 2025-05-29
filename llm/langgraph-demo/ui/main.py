import asyncio
import re

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from nicegui import ui
from pydantic import BaseModel, Field


class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""

    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")


@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


def parse_thinking_content(content: str):
    """Parse thinking tags from qwen model output"""
    thinking_pattern = r"<think>(.*?)</think>"
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)

    # Remove thinking tags from content to get clean response
    clean_content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()

    return thinking_matches, clean_content


@ui.page("/")
def main():
    # tools = [multiply]
    llm = ChatOllama(model="qwen3:4b")

    # FIXME: We are directly using the llm instead of the agent.
    # agent = create_react_agent(llm, tools=tools)

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

        current_thinking = ""
        current_response = ""

        try:
            # Stream from the LLM directly for better control
            print("Starting real-time streaming...")
            accumulated_content = ""
            current_thinking = ""
            current_response = ""

            async for chunk in llm.astream(question):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                accumulated_content += content

                # Parse the accumulated content so far
                thinking_matches, clean_content = parse_thinking_content(accumulated_content)

                # Check if thinking is complete (has closing tag)
                thinking_complete = "</think>" in accumulated_content

                # Update thinking section if we found thinking content
                if thinking_matches:
                    new_thinking = "\n\n".join(thinking_matches)
                    if new_thinking != current_thinking:
                        current_thinking = new_thinking
                        thinking_section.clear()
                        with thinking_section:
                            ui.markdown(current_thinking).classes("text-sm text-gray-600 whitespace-pre-wrap")
                        if not thinking_section.value:  # Auto-open if closed
                            thinking_section.open()

                # Auto-collapse thinking section when thinking is complete and we have response content
                if thinking_complete and clean_content.strip() and thinking_section.value:
                    thinking_section.close()

                # Update response section with clean content
                if clean_content.strip() != current_response:
                    current_response = clean_content.strip()
                    response_content.content = current_response

                # Auto-scroll and allow UI to update
                ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(0.03)  # Small delay for smooth streaming effect

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
