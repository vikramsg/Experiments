import pino from 'pino'

export const logger = pino({
  name: 'electron-workspace',
  level: process.env.ELECTRON_LOG_LEVEL || 'info',
})
