#!/bin/bash
# Automates the Rust fmt and clippy fixes for the workspace.

echo "Formatting codebase with cargo fmt..."
cargo fmt

if [ $? -ne 0 ]; then
    echo "Warning: cargo fmt encountered an issue."
fi

echo "Running cargo clippy and applying automatic fixes..."
#!/bin/bash
set -eu


echo "clippy"

if result=$(cargo clippy --all-targets --all --all-features -- -D warnings); then
    echo " OK"
else
    echo " FAIL"
    echo " $result"
    exit 1
fi

echo "fmt"

if result=$(cargo fmt --check); then
    echo " OK"
else
    echo " FAIL"
    echo " $result"
    exit 1
fi

echo "shear"

if result=$(cargo  shear); then
    echo " OK"
else
    echo " FAIL"
    echo " $result"
    exit 1
fi

echo "=========================================="
echo "Linting and formatting complete!"
echo "=========================================="
