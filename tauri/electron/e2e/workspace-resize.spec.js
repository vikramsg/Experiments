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

test('window resize keeps notes and browser panes within safe widths', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-resize-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()
    await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

    const notesPage = await waitForPageByUrlPart(electronApp, 'notes.html')
    const browserPage = await waitForPageByUrlPart(electronApp, 'example.com')

    await electronApp.evaluate(async ({ BaseWindow }) => {
      const workspaceWindow = BaseWindow.getAllWindows().find((window) => window.getTitle() === 'Browser + Notes')
      if (!workspaceWindow) {
        throw new Error('Workspace window not found')
      }

      const currentBounds = workspaceWindow.getBounds()
      workspaceWindow.setContentBounds({
        x: currentBounds.x,
        y: currentBounds.y,
        width: 760,
        height: 640,
      })
    })

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeGreaterThanOrEqual(280)
    await expect.poll(async () => browserPage.evaluate(() => window.innerWidth)).toBeGreaterThanOrEqual(360)

    await electronApp.evaluate(async ({ BaseWindow }) => {
      const workspaceWindow = BaseWindow.getAllWindows().find((window) => window.getTitle() === 'Browser + Notes')
      if (!workspaceWindow) {
        throw new Error('Workspace window not found')
      }

      const currentBounds = workspaceWindow.getBounds()
      workspaceWindow.setContentBounds({
        x: currentBounds.x,
        y: currentBounds.y,
        width: 1280,
        height: 820,
      })
    })

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeGreaterThan(280)
    await expect.poll(async () => browserPage.evaluate(() => window.innerWidth)).toBeGreaterThan(360)
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
