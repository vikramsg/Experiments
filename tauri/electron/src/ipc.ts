/**
 * Central IPC channel definitions for the Electron workspace app.
 *
 * This file exists at the root of `src/` because these channel names are a
 * transport boundary shared by app composition code, preload bridges, and
 * isolated feature code. Keeping the constants here makes the ownership clear
 * and avoids turning a generic `src/shared/` folder into a catch-all bucket.
 *
 * Import guidance:
 * - `src/app/*` may import this file freely.
 * - `src/features/*` may import this file when they need channel names.
 * - features must still not import each other directly.
 */
export const IPC_CHANNELS = {
  launcherOpenWorkspace: 'launcher:open-workspace',
  launcherOpenOpenCode: 'launcher:open-opencode',
  launcherOpenTerminal: 'launcher:open-terminal',
  workspaceGetState: 'workspace:get-state',
  workspaceSaveNotes: 'workspace:save-notes',
  workspaceSetBrowserUrl: 'workspace:set-browser-url',
  workspaceGoBack: 'workspace:go-back',
  workspaceGoForward: 'workspace:go-forward',
  workspaceAdjustSplitter: 'workspace:adjust-splitter',
  workspaceState: 'workspace:state',
  opencodeGetState: 'opencode:get-state',
  opencodeSendPrompt: 'opencode:send-prompt',
  opencodeState: 'opencode:state',
  terminalGetState: 'terminal:get-state',
  terminalConnect: 'terminal:connect',
  terminalWrite: 'terminal:write',
  terminalResize: 'terminal:resize',
  terminalRestart: 'terminal:restart',
  terminalState: 'terminal:state',
  terminalData: 'terminal:data',
} as const
