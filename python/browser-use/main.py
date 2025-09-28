import asyncio

from browser_use import Agent, Browser
from browser_use.llm import ChatGoogle
from dotenv import load_dotenv


async def main(query: str) -> None:
    load_dotenv()

    browser = Browser(
        headless=False,  # Show browser window
        window_size={"width": 1000, "height": 700},  # Set window size
    )

    llm = ChatGoogle(model="gemini-2.5-flash-preview-09-2025")

    agent = Agent(task=query, llm=llm, browser=browser)

    await agent.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", help="Query to pass to agent.", required=True)

    args = parser.parse_args()
    asyncio.run(main(query=args.query))
