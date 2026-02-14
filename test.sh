#!/bin/bash
set -e

# Detect cargo
if command -v cargo &> /dev/null; then
    CARGO_CMD="cargo"
elif [ -f "$HOME/.cargo/bin/cargo" ]; then
    CARGO_CMD="$HOME/.cargo/bin/cargo"
else
    echo "Error: cargo not found."
    exit 1
fi

echo "=========================================="
echo "Running All Unit Tests..."
echo "=========================================="
$CARGO_CMD test --verbose

echo "=========================================="
echo "Tests Completed."
echo "=========================================="
