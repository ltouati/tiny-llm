#!/bin/bash
# Automates the Rust fmt and clippy fixes for the workspace.

echo "Formatting codebase with cargo fmt..."
cargo fmt

if [ $? -ne 0 ]; then
    echo "Warning: cargo fmt encountered an issue."
fi

echo "Running cargo clippy and applying automatic fixes..."
# We use --allow-dirty and --allow-no-vcs so clippy can operate freely outside of hard git commits
cargo clippy --fix --allow-dirty --allow-no-vcs

echo "=========================================="
echo "Linting and formatting complete!"
echo "=========================================="
