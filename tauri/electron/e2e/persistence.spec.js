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

async function openWorkspace(electronApp, browserUrlPart = null) {
  const launcher = await electronApp.firstWindow()
  await launcher.getByRole('button', { name: /launch browser \+ notes/i }).click()

  const notesPage = await waitForPageByUrlPart(electronApp, 'notes.html')
  const splitterPage = await waitForPageByUrlPart(electronApp, 'splitter.html')
  const browserChromePage = await waitForPageByUrlPart(electronApp, 'browser-chrome.html')
  const browserPage = browserUrlPart ? await waitForPageByUrlPart(electronApp, browserUrlPart) : null

  return { notesPage, splitterPage, browserChromePage, browserPage }
}

test('notes, splitter width, and browser url persist across relaunch', async () => {
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'electron-e2e-persist-'))
  let electronApp

  try {
    electronApp = await launchApp(userDataDir)
    const firstRun = await openWorkspace(electronApp, 'example.com')

    await firstRun.notesPage.getByRole('textbox', { name: /notes editor/i }).fill('Persist this note')
    await firstRun.browserChromePage.getByRole('textbox', { name: /browser url/i }).fill('https://example.org/')
    await firstRun.browserChromePage.getByRole('button', { name: /^go$/i }).click()

    await expect
      .poll(async () => firstRun.browserPage.url(), { timeout: 15000 })
      .toContain('https://example.org/')

    const separator = firstRun.splitterPage.getByRole('separator', { name: /resize panes/i })
    const box = await separator.boundingBox()
    expect(box).not.toBeNull()
    if (!box) {
      throw new Error('Splitter handle did not render')
    }

    await firstRun.splitterPage.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await firstRun.splitterPage.mouse.down()
    await firstRun.splitterPage.mouse.move(box.x + box.width / 2 + 90, box.y + box.height / 2, { steps: 8 })
    await firstRun.splitterPage.mouse.up()

    const savedWidth = await firstRun.notesPage.evaluate(() => window.innerWidth)

    await expect.poll(async () => firstRun.notesPage.getByRole('textbox', { name: /notes editor/i }).inputValue()).toBe(
      'Persist this note',
    )

    await electronApp.close()
    electronApp = await launchApp(userDataDir)

    const secondRun = await openWorkspace(electronApp, 'example.org')

    await expect(secondRun.notesPage.getByRole('textbox', { name: /notes editor/i })).toHaveValue('Persist this note')
    await expect(secondRun.browserChromePage.getByRole('textbox', { name: /browser url/i })).toHaveValue('https://example.org/')
    await expect.poll(async () => secondRun.notesPage.evaluate(() => window.innerWidth)).toBe(savedWidth)
    await expect.poll(async () => secondRun.browserPage.url(), { timeout: 15000 }).toContain('https://example.org/')
  } finally {
    await electronApp?.close()
    await rm(userDataDir, { recursive: true, force: true })
  }
})
