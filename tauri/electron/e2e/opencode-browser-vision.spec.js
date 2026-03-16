const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { tmpdir } = require('node:os')

const { _electron: electron, expect, test } = require('@playwright/test')

const repoRoot = path.resolve(__dirname, '../..')

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
      ELECTRON_OPENCODE_REPO_ROOT: repoRoot,
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

test('OpenCode can explain what it sees in the browser', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-opencode-browser-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()

    await launcher.getByRole('button', { name: /launch opencode/i }).click()
    const openCodePage = await waitForPageByUrlPart(electronApp, 'opencode.html')
    const browserChromePage = await waitForPageByUrlPart(electronApp, 'browser-chrome.html')
    const browserPage = await waitForPageByUrlPart(electronApp, 'example.com')

    await expect.poll(async () => browserPage.url(), { timeout: 15000 }).toContain('https://example.com')
    await browserChromePage.getByRole('combobox', { name: /browser url/i }).fill('https://example.com/docs')
    await browserChromePage.getByRole('button', { name: /^go$/i }).click()
    await expect.poll(async () => browserPage.url(), { timeout: 15000 }).toContain('https://example.com/docs')

    const prompt = openCodePage.getByRole('textbox', { name: /ask opencode/i })
    await prompt.fill('What do you see in the browser?')
    await prompt.press('Enter')

    await expect(openCodePage.getByText(/i can see the browser is currently at/i)).toBeVisible()
    await expect(openCodePage.getByText(/example.com/i)).toBeVisible()
    await expect(openCodePage.getByText(/based on that screenshot/i)).toBeVisible()
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
