# Playwright Python: Recording and Playing Back Website Interactions

## Summary of Findings

Playwright Python offers several powerful approaches to record interactions with websites and replay them:

1. **Codegen Tool**: Playwright will generate the code for the user interactions which you can see in the Playwright Inspector window. When running the codegen command two windows will be opened, a browser window where you interact with the website you wish to test and the Playwright Inspector window where you can record your tests and then copy them into your editor.

2. **Trace Recording**: Playwright Trace Viewer is a GUI tool that lets you explore recorded Playwright traces of your tests meaning you can go back and forward though each action of your test and visually see what was happening during each action

3. **Video Recording**: With Playwright you can record videos for your tests. Videos are saved upon browser context closure at the end of a test.

## Code Examples

### 1. Basic Codegen Recording

To start recording interactions, use the `playwright codegen` command:

```bash
# Basic usage
playwright codegen https://example.com

# With specific browser
playwright codegen --browser=chromium https://example.com

# Save to file
playwright codegen https://example.com > recorded_script.py
```

Use the codegen command to run the test generator followed by the URL of the website you want to generate tests for. The URL is optional and you can always run the command without it and then add the URL directly into the browser window instead.

**Generated code example:**
```python
from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    # Go to the website
    page.goto("https://example.com")
    
    # Recorded interactions
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("user@example.com")
    page.get_by_role("button", name="Submit").click()
    
    # Close browser
    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
```

### 2. Advanced Codegen Options

**Recording with authentication state:**
```bash
# Save authentication state
playwright codegen --save-storage=auth.json https://example.com

# Load authentication state for recording
playwright codegen --load-storage=auth.json https://example.com
```

Run codegen with --save-storage to save cookies, localStorage and IndexedDB data at the end of the session. This is useful to separately record an authentication step and reuse it later when recording more tests. After performing authentication and closing the browser, auth.json will contain the storage state which you can then reuse in your tests.

**Recording with device emulation:**
```bash
# Emulate mobile device
playwright codegen --device="iPhone 13" https://example.com

# Custom viewport
playwright codegen --viewport-size="800,600" https://example.com
```

### 3. Programmatic Recording with page.pause()

For custom setups, you can use `page.pause()` to start recording within your script:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Make sure to run headed
    browser = p.chromium.launch(headless=False)
    # Setup context however you like
    context = browser.new_context()
    # Pass any options
    context.route('**/*', lambda route: route.continue_())
    # Pause the page, and start recording manually
    page = context.new_page()
    page.pause()  # This opens codegen controls
```

If you would like to use codegen in some non-standard setup (for example, use browser_context.route()), it is possible to call page.pause() that will open a separate window with codegen controls.

### 4. Trace Recording and Replay

**Recording traces:**
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context()
    
    # Start tracing before creating/navigating a page
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    
    page = context.new_page()
    page.goto("https://playwright.dev")
    
    # Your test actions here
    page.get_by_text("Get Started").click()
    
    # Stop tracing and export it into a zip archive
    context.tracing.stop(path="trace.zip")
    
    context.close()
    browser.close()
```

Start tracing before creating / navigating a page. context.tracing.start(screenshots=True, snapshots=True, sources=True) page = context.new_page() page.goto("https://playwright.dev") # Stop tracing and export it into a zip archive.

**Viewing traces:**
```bash
# View trace with Playwright CLI
playwright show-trace trace.zip

# Or open online at trace.playwright.dev
```

You can open the saved trace using the Playwright CLI or in your browser on trace.playwright.dev.

### 5. Video Recording

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    # Enable video recording
    context = browser.new_context(record_video_dir="videos/")
    
    page = context.new_page()
    page.goto("https://example.com")
    
    # Your interactions here
    page.get_by_text("Click me").click()
    
    # Make sure to close, so that videos are saved
    context.close()
    browser.close()
```

context = browser.new_context(record_video_dir="videos/")

### 6. Complete Recording and Playback Workflow

```python
from playwright.sync_api import sync_playwright

def record_and_replay():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            record_video_dir="videos/",
            # Enable all recording features
        )
        
        # Start tracing
        context.tracing.start(
            screenshots=True,
            snapshots=True,
            sources=True
        )
        
        page = context.new_page()
        
        # Navigate and perform actions
        page.goto("https://example.com")
        page.get_by_label("Search").fill("playwright")
        page.get_by_role("button", name="Search").click()
        
        # Wait for results and take screenshot
        page.wait_for_selector(".results")
        page.screenshot(path="search_results.png")
        
        # Stop tracing
        context.tracing.stop(path="search_trace.zip")
        
        context.close()
        browser.close()

if __name__ == "__main__":
    record_and_replay()
```

## Common Pitfalls and Solutions

### 1. Security Settings Issues

**Problem**: we changed our application in a way that it only works in Chrome / chromium when chrome://flags/#block-insecure-private-network-requests is disabled. When I start again manually with "playwright codegen" in terminal, codegen chromium starts again with this setting enabled by default.

**Solution**: Use custom browser args with codegen:
```bash
playwright codegen --browser=chromium --args="--disable-features=BlockInsecurePrivateNetworkRequests"
```

### 2. Docker Container Issues

**Problem**: I'm trying to run python -m playwright codegen inside the container. The goal is for the container to open my chrome browser so I can start writing a script.

**Solution**: Use xvfb for headless recording in containers:
```bash
xvfb-run -a --server-args="-screen 0 1280x800x24 -ac -nolisten tcp -dpi 96 +extension RANDR" playwright codegen
```

### 3. HTTPS Certificate Issues

**Problem**: my project have self-signed certificate and because of this prefer to run codegen with ignore_https_errors flag but not saw this option in help.

**Solution**: Use `--ignore-https-errors` flag:
```bash
playwright codegen --ignore-https-errors https://your-site.com
```

### 4. Codegen Generated Code Issues

**Problem**: Codegen generates python code with a dictionary record_har entry instead of record_har_path and record_har_mode. Thus, running python har.py will fail with TypeError: new_context() got an unexpected keyword argument 'record_har'.

**Solution**: Manually fix generated code by replacing dictionary format with proper parameters:
```python
# Generated (incorrect)
context = browser.new_context(record_har={"path":"test.har","mode":"minimal"})

# Fixed (correct)
context = browser.new_context(record_har_path="test.har", record_har_mode="minimal")
```

### 5. Strict Mode Violations

**Problem**: The test generated by codegen gives the below error, when I used first with the locator, it went into a loop, opening the dropdown and scrolling but never completing the steps. Error: strict mode violation: locator("#client-table-1078866").get_by_text("Applied") resolved to 2 elements

**Solution**: Add `.first` or use more specific locators:
```python
# Generated (problematic)
page.get_by_text("Applied").click()

# Fixed
page.get_by_text("Applied").first.click()
# Or use more specific locator
page.locator("#specific-id").get_by_text("Applied").click()
```

## Best Practices

1. **Always run codegen in headed mode** for better visibility and interaction
2. **Use `--save-storage` for authentication** workflows to avoid repeated logins
3. **Enable tracing for debugging** complex interactions
4. **Combine video recording with traces** for comprehensive debugging
5. **Clean up generated code** by adding proper assertions and error handling
6. **Use specific locators** instead of relying solely on auto-generated ones

## Links to Relevant Resources

- [Official Playwright Python Codegen Documentation](https://playwright.dev/python/docs/codegen) Playwright will generate the code for the user interactions which you can see in the Playwright Inspector window.
- [Trace Viewer Documentation](https://playwright.dev/python/docs/trace-viewer) Playwright Trace Viewer is a GUI tool that helps you explore recorded Playwright traces after the script has run. Traces are a great way for debugging your tests when they fail on CI.
- [Video Recording Guide](https://playwright.dev/python/docs/videos) With Playwright you can record videos for your tests
- [Playwright Python GitHub Issues](https://github.com/microsoft/playwright-python/issues) for troubleshooting common problems
- [Online Trace Viewer](https://trace.playwright.dev) for viewing traces without installing Playwright locally

This comprehensive approach to recording and playing back website interactions with Playwright Python provides multiple methods to capture user behavior and replay it for testing, debugging, and automation purposes. The combination of codegen, tracing, and video recording creates a powerful toolkit for understanding and reproducing web application interactions.

## Note

This doc was completely generated by running `dzai api-research-agent -q "Find me how to use playwright python to record interactions with a website so that I can play it back again."` 
