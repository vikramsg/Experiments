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

test('dragging the splitter changes pane widths in both directions', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-splitter-'))
  let electronApp: ElectronApplication | undefined

  try {
    electronApp = await launchApp(userDataDir)
    const launcher = await electronApp.firstWindow()
    await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

    const notesPage = await waitForPageByUrlPart(electronApp, 'notes.html')
    const splitterPage = await waitForPageByUrlPart(electronApp, 'splitter.html')

    const beforeWidth = await notesPage.evaluate(() => window.innerWidth)
    const separator = splitterPage.getByRole('separator', { name: /resize panes/i })
    const box = await separator.boundingBox()

    expect(box).not.toBeNull()

    await splitterPage.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2)
    await splitterPage.mouse.down()
    await splitterPage.mouse.move(box!.x + box!.width / 2 + 120, box!.y + box!.height / 2, { steps: 8 })
    await splitterPage.mouse.up()

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeGreaterThan(beforeWidth)

    const largerWidth = await notesPage.evaluate(() => window.innerWidth)

    await splitterPage.mouse.move(box!.x + box!.width / 2 + 120, box!.y + box!.height / 2)
    await splitterPage.mouse.down()
    await splitterPage.mouse.move(box!.x + box!.width / 2 + 40, box!.y + box!.height / 2, { steps: 8 })
    await splitterPage.mouse.up()

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeLessThan(largerWidth)
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
