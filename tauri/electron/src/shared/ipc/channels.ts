export const IPC_CHANNELS = {
  launcherOpenWorkspace: 'launcher:open-workspace',
  workspaceGetState: 'workspace:get-state',
  workspaceSaveNotes: 'workspace:save-notes',
  workspaceSetBrowserUrl: 'workspace:set-browser-url',
  workspaceGoBack: 'workspace:go-back',
  workspaceGoForward: 'workspace:go-forward',
  workspaceAdjustSplitter: 'workspace:adjust-splitter',
  workspaceState: 'workspace:state',
} as const
