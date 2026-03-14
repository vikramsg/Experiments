const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { tmpdir } = require('node:os')

const { _electron: electron, expect, test } = require('@playwright/test')

async function launchApp(userDataDir) {
  return electron.launch({
    args: ['.'],
    cwd: path.resolve(__dirname, '..'),
    env: {
      ...process.env,
      ELECTRON_USER_DATA_DIR: userDataDir,
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

test('launcher opens the split workspace', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-launcher-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()

    await expect(launcher.getByRole('heading', { name: /choose an app/i })).toBeVisible()
    await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

    const notesPage = await waitForPageByUrlPart(electronApp, 'notes.html')
    const splitterPage = await waitForPageByUrlPart(electronApp, 'splitter.html')
    const browserPage = await waitForPageByUrlPart(electronApp, 'example.com')

    await expect(notesPage.getByRole('textbox', { name: /notes editor/i })).toBeVisible()
    await expect(splitterPage.getByRole('separator', { name: /resize panes/i })).toBeVisible()
    await expect
      .poll(async () => browserPage.url(), { timeout: 15000 })
      .toContain('https://example.com')
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
