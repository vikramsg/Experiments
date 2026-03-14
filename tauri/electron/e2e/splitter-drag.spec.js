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

test('dragging the splitter changes pane widths in both directions', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-splitter-'))
  let electronApp

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
    if (!box) {
      throw new Error('Splitter handle did not render')
    }

    await splitterPage.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await splitterPage.mouse.down()
    await splitterPage.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2, { steps: 8 })
    await splitterPage.mouse.up()

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeGreaterThan(beforeWidth)

    const largerWidth = await notesPage.evaluate(() => window.innerWidth)
    const movedBox = await separator.boundingBox()

    expect(movedBox).not.toBeNull()
    if (!movedBox) {
      throw new Error('Splitter handle moved out of view')
    }

    await splitterPage.mouse.move(movedBox.x + movedBox.width / 2, movedBox.y + movedBox.height / 2)
    await splitterPage.mouse.down()
    await splitterPage.mouse.move(movedBox.x + movedBox.width / 2 - 80, movedBox.y + movedBox.height / 2, { steps: 8 })
    await splitterPage.mouse.up()

    await expect.poll(async () => notesPage.evaluate(() => window.innerWidth)).toBeLessThan(largerWidth)
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
