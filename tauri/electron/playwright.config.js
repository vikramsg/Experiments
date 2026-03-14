/** @type {import('@playwright/test').PlaywrightTestConfig} */
module.exports = {
  testDir: './e2e',
  timeout: 120000,
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    trace: 'on-first-retry',
  },
}
