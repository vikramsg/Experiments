import pino from 'pino'
import pretty from 'pino-pretty'

export function shouldUsePrettyLogs(environment: NodeJS.ProcessEnv): boolean {
  return environment.NODE_ENV === 'development'
}

export function createLoggerDestination(environment: NodeJS.ProcessEnv) {
  if (!shouldUsePrettyLogs(environment)) {
    return undefined
  }

  return pretty({
    colorize: true,
    translateTime: 'HH:MM:ss',
    ignore: 'pid,hostname',
  })
}

export const logger = pino(
  {
    name: 'electron-workspace',
    level: process.env.ELECTRON_LOG_LEVEL || 'info',
  },
  createLoggerDestination(process.env),
)
