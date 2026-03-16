import { createLoggerDestination, shouldUsePrettyLogs } from './logger'

describe('logger', () => {
  it('uses pretty logs in development', () => {
    expect(shouldUsePrettyLogs({ NODE_ENV: 'development' })).toBe(true)
  })

  it('does not use pretty logs in production or test', () => {
    expect(shouldUsePrettyLogs({ NODE_ENV: 'production' })).toBe(false)
    expect(shouldUsePrettyLogs({ NODE_ENV: 'test' })).toBe(false)
  })

  it('uses a direct pretty stream instead of a worker transport in development', () => {
    const destination = createLoggerDestination({ NODE_ENV: 'development' })

    expect(destination).toBeDefined()
    expect(destination?.constructor?.name).not.toBe('ThreadStream')
  })
})
