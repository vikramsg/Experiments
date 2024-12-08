import time

from playwright.sync_api import sync_playwright
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    datev_username: str
    datev_password: SecretStr

    model_config = SettingsConfigDict(env_file=".env")


def login_to_datev(settings: Settings) -> None:
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=False)  # Set headless=True in production
        context = browser.new_context()
        page = context.new_page()

        # Navigate to DATEV login page
        page.goto("https://www.datev.de/ano/")

        # Wait for login form to be visible
        page.wait_for_selector("#username")

        # Fill in login credentials
        page.fill("#username", settings.datev_username)
        page.fill("#password", settings.datev_password)

        # Click login button
        page.click('button[type="submit"]')

        # Wait for navigation after login
        page.wait_for_load_state("networkidle")

        # Add a small delay to ensure everything is loaded
        time.sleep(2)

        # From here, we'll need to add the specific steps to navigate to payslips
        # This will depend on the exact structure of the DATEV interface

        # Keep browser open for debugging
        browser.close()


if __name__ == "__main__":
    settings = Settings()

    login_to_datev(settings=settings)
