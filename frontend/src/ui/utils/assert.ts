/**
 * @fileoverview Assertion utilities for runtime validation.
 */

/**
 * Asserts that a condition is true.
 *
 * @param condition - The condition to validate.
 * @param message - Optional error message.
 * @throws {Error} If the condition is false.
 */
export const assert = (condition: boolean, message: string | null = null) => {
    if (!condition) throw new Error(message || 'Assertion failed')
}
