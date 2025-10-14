# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a TypeScript project for experimentation and development.

## Development Commands

### Building
```bash
npm run build          # Compile TypeScript to JavaScript
npm run build:watch    # Watch mode for development
tsc                    # Direct TypeScript compilation
```

### Testing
```bash
npm test               # Run all tests
npm test -- <file>     # Run specific test file
npm run test:watch     # Watch mode for tests
npm run test:coverage  # Run tests with coverage report
```

### Linting and Formatting
```bash
npm run lint           # Run ESLint
npm run lint:fix       # Auto-fix linting issues
npm run format         # Format code with Prettier
npm run type-check     # Run TypeScript type checking without emitting
```

### Development
```bash
npm run dev            # Start development server
npm start              # Run the application
ts-node src/index.ts   # Run TypeScript file directly
```

## Code Architecture

### Project Structure
- `src/` - Source TypeScript files
- `dist/` or `build/` - Compiled JavaScript output
- `tests/` or `__tests__/` - Test files
- `types/` - TypeScript type definitions

### TypeScript Configuration
- Check `tsconfig.json` for compiler options and module resolution
- May have separate configs for build (`tsconfig.build.json`) and tests
- Look for path aliases defined in `compilerOptions.paths`

### Module System
- Verify whether the project uses CommonJS (`require`) or ES Modules (`import/export`)
- Check `package.json` for `"type": "module"` field
- TypeScript `module` setting in `tsconfig.json` determines output format

## Development Notes

### Type Checking
Always run `tsc --noEmit` or `npm run type-check` before committing to catch type errors without generating output files.

### Dependencies
- Check `package.json` for scripts and available commands
- Use `npm ci` for clean installs in CI/CD environments
