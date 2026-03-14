import { mkdtemp, rm } from 'node:fs/promises'
import path from 'node:path'
import { tmpdir } from 'node:os'

import { _electron as electron, expect, test, type ElectronApplication, type Page } from '@playwright/test'

async function launchApp(userDataDir: string) {
  return electron.launch({
    args: ['.'],
    cwd: path.resolve(import.meta.dirname, '..'),
    env: {
      ...process.env,
      ELECTRON_USER_DATA_DIR: userDataDir,
    },
  })
}

async function waitForPageByUrlPart(electronApp: ElectronApplication, urlPart: string) {
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
  let electronApp: ElectronApplication | undefined

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
      .poll(async () => browserPage.url(), { timeout: 15_000 })
      .toContain('https://example.com')
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
