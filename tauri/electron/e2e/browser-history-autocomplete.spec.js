const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { tmpdir } = require('node:os')

const { _electron: electron, expect, test } = require('@playwright/test')

async function launchApp(userDataDir) {
  return electron.launch({
    executablePath: path.resolve(
      __dirname,
      '../out/electron-workspace-darwin-arm64/electron-workspace.app/Contents/MacOS/electron-workspace',
    ),
    env: {
      ...process.env,
      ELECTRON_USER_DATA_DIR: userDataDir,
      ELECTRON_OPENCODE_MOCK: '1',
    },
  })
}

async function waitForPageByUrlPart(electronApp, urlPart) {
  const existing = electronApp.context().pages().find((page) => page.url().includes(urlPart))
  if (existing) {
    return existing
  }

  return electronApp.context().waitForEvent('page', {
    predicate: (page) => page.url().includes(urlPart),
  })
}

function getRemoteBrowserPage(electronApp) {
  return electronApp
    .context()
    .pages()
    .find((page) => !page.url().includes('launcher.html') && !page.url().includes('browser-chrome.html') && !page.url().includes('notes.html'))
}

async function openBrowserNotesAndNavigate(electronApp, url) {
  const launcher = await electronApp.firstWindow()
  await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

  const browserChromePage = await waitForPageByUrlPart(electronApp, 'browser-chrome.html')
  const browserPage = getRemoteBrowserPage(electronApp) ?? (await electronApp.context().waitForEvent('page'))
  await browserChromePage.getByRole('combobox', { name: /browser url/i }).fill(url)
  await browserChromePage.getByRole('button', { name: /^go$/i }).click()

  await expect.poll(async () => browserPage.url(), { timeout: 15000 }).toContain(url)
}

test('browser url autocomplete persists the last visited urls', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-browser-history-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    await openBrowserNotesAndNavigate(electronApp, 'https://example.org')
    await openBrowserNotesAndNavigate(electronApp, 'https://example.net/app')
    await electronApp.close()

    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()
    await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()
    const browserChromePage = await waitForPageByUrlPart(electronApp, 'browser-chrome.html')

    const historyValues = await browserChromePage.locator('datalist option').evaluateAll((options) =>
      options.map((option) => option.getAttribute('value')),
    )

    expect(historyValues).toContain('https://example.org/')
    expect(historyValues).toContain('https://example.net/app')
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
