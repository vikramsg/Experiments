import { mkdtemp, rm } from 'node:fs/promises'
import path from 'node:path'
import { tmpdir } from 'node:os'

import { _electron as electron, expect, test, type ElectronApplication } from '@playwright/test'

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

async function openWorkspace(electronApp: ElectronApplication) {
  const launcher = await electronApp.firstWindow()
  await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

  const notesPage = await waitForPageByUrlPart(electronApp, 'notes.html')
  const splitterPage = await waitForPageByUrlPart(electronApp, 'splitter.html')

  return { notesPage, splitterPage }
}

test('notes, splitter width, and browser url persist across relaunch', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-persist-'))
  let electronApp: ElectronApplication | undefined

  try {
    electronApp = await launchApp(userDataDir)
    const firstRun = await openWorkspace(electronApp)

    await firstRun.notesPage.getByRole('textbox', { name: /notes editor/i }).fill('Persist this note')
    await firstRun.notesPage.getByRole('textbox', { name: /browser url/i }).fill('https://example.org/')
    await firstRun.notesPage.getByRole('button', { name: /^go$/i }).click()

    const separator = firstRun.splitterPage.getByRole('separator', { name: /resize panes/i })
    const box = await separator.boundingBox()
    expect(box).not.toBeNull()

    await firstRun.splitterPage.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2)
    await firstRun.splitterPage.mouse.down()
    await firstRun.splitterPage.mouse.move(box!.x + box!.width / 2 + 90, box!.y + box!.height / 2, { steps: 8 })
    await firstRun.splitterPage.mouse.up()

    const savedWidth = await firstRun.notesPage.evaluate(() => window.innerWidth)

    await expect.poll(async () => firstRun.notesPage.getByRole('textbox', { name: /notes editor/i }).inputValue()).toBe(
      'Persist this note',
    )

    await electronApp.close()
    electronApp = await launchApp(userDataDir)

    const secondRun = await openWorkspace(electronApp)

    await expect(secondRun.notesPage.getByRole('textbox', { name: /notes editor/i })).toHaveValue('Persist this note')
    await expect(secondRun.notesPage.getByRole('textbox', { name: /browser url/i })).toHaveValue('https://example.org/')
    await expect.poll(async () => secondRun.notesPage.evaluate(() => window.innerWidth)).toBe(savedWidth)
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
