async function closeElectronApp(electronApp) {
  if (!electronApp) {
    return
  }

  const process = electronApp.process()

  try {
    await Promise.race([
      electronApp.close(),
      new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error('Electron app close timed out'))
        }, 5000)
      }),
    ])
  } catch {
    if (!process.killed) {
      process.kill('SIGKILL')
    }
  }
}

module.exports = {
  closeElectronApp,
}
