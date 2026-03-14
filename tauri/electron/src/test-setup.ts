/**
 * Shared Vitest test environment setup for renderer and unit tests.
 *
 * This file stays shallow at the root of `src/` because it is test-only support
 * code used across the codebase. Naming it explicitly makes its purpose clear
 * without suggesting that arbitrary production runtime code should accumulate
 * in a generic `src/shared/` directory.
 */
import '@testing-library/jest-dom/vitest'
