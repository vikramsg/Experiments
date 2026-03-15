const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { tmpdir } = require('node:os')

const { _electron: electron, expect, test } = require('@playwright/test')
const { closeElectronApp } = require('./helpers')

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

test('launcher opens OpenCode and the chat responds', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-opencode-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()

    await expect(launcher.getByRole('heading', { name: /choose an app/i })).toBeVisible()
    await expect(launcher.getByRole('button', { name: /launch opencode/i })).toBeVisible()
    await launcher.getByRole('button', { name: /launch opencode/i }).click()

    const openCodePage = await waitForPageByUrlPart(electronApp, 'opencode.html')

    await expect(openCodePage.getByRole('heading', { name: /opencode/i })).toBeVisible()
    await expect(openCodePage.getByText(/local opencode server behind a narrow electron bridge/i)).toBeVisible()

    const prompt = openCodePage.getByRole('textbox', { name: /ask opencode/i })
    await prompt.fill('Where does the launcher live?')
    await prompt.press('Enter')

    await expect(openCodePage.getByText('Where does the launcher live?', { exact: true })).toBeVisible()
    await expect(openCodePage.getByText(/mock opencode reply/i)).toBeVisible()
    await expect(openCodePage.getByText(repoRoot, { exact: true })).toBeVisible()

    await prompt.fill('Line one')
    await prompt.press('Shift+Enter')
    await expect(prompt).toHaveValue('Line one\n')
    await prompt.fill('Overflow check')
    await prompt.press('Enter')

    for (let index = 0; index < 5; index += 1) {
      await prompt.fill(`Prompt ${index}`)
      await prompt.press('Enter')
    }

    await expect(openCodePage.getByRole('heading', { name: /opencode/i })).toBeVisible()
    await expect(openCodePage.getByRole('button', { name: /send prompt/i })).toBeVisible()
  } finally {
    await closeElectronApp(electronApp)
    await rm(userDataDir, { recursive: true, force: true })
  }
})
