const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { tmpdir } = require('node:os')

const { _electron: electron, expect, test } = require('@playwright/test')
const { closeElectronApp } = require('./helpers')

async function launchApp(userDataDir) {
  return electron.launch({
    executablePath: path.resolve(
      __dirname,
      '../out/electron-workspace-darwin-arm64/electron-workspace.app/Contents/MacOS/electron-workspace',
    ),
    env: {
      ...process.env,
      ELECTRON_USER_DATA_DIR: userDataDir,
      ELECTRON_TERMINAL_MOCK: '1',
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

test('launcher opens Terminal and shows terminal state', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-terminal-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()

    await expect(launcher.getByRole('heading', { name: /choose an app/i })).toBeVisible()
    await expect(launcher.getByRole('button', { name: /launch terminal/i })).toBeVisible()
    await launcher.getByRole('button', { name: /launch terminal/i }).click()

    const terminalPage = await waitForPageByUrlPart(electronApp, 'terminal.html')

    await expect(terminalPage.getByLabel('Terminal surface')).toBeVisible()
    await expect(terminalPage.getByRole('heading', { name: /terminal/i })).toHaveCount(0)
    await expect(terminalPage.getByRole('button', { name: /restart terminal/i })).toHaveCount(0)
    await expect(terminalPage.getByTestId('terminal-status')).toHaveText(/local shell is ready|mock terminal session ready/i)
  } finally {
    await closeElectronApp(electronApp)
    await rm(userDataDir, { recursive: true, force: true })
  }
})
