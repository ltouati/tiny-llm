---
name: Linter
description: Automatically runs cargo fmt and cargo clippy to enforce code style and fix common Rust anti-patterns.
---

# Rust Linter & Formatter

This skill automates the process of grooming the codebase by running the standard Rust formatting tools and the Clippy linter with automatic fixes enabled.

## How to use this skill

1.  **Run the Formatter/Linter**:
    -   Execute the helper script to scan the workspace and automatically apply fixes where possible.
    -   Example: `bash .agent/skills/linting/scripts/run_lint.sh`
2.  **What it Does**:
    -   First, it runs `cargo fmt` to reformat all Rust source files according to standard style guidelines.
    -   Then, it runs `cargo clippy --fix --allow-dirty --allow-no-vcs` to analyze the code for lints, dead code, and unused imports, and applies suggested fixes automatically without requiring a git commit boundary.
