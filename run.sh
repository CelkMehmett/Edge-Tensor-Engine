#!/bin/bash
set -e

# Detect cargo
if command -v cargo &> /dev/null; then
    CARGO_CMD="cargo"
elif [ -f "$HOME/.cargo/bin/cargo" ]; then
    CARGO_CMD="$HOME/.cargo/bin/cargo"
else
    echo "Error: cargo not found. Please install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Detect dart
if command -v dart &> /dev/null; then
    DART_CMD="dart"
else
    echo "Error: dart not found. Please install Dart SDK."
    exit 1
fi

echo "=========================================="
echo "Building Edge Tensor Engine (Rust)..."
echo "Using cargo: $CARGO_CMD"
echo "=========================================="
$CARGO_CMD build --release

echo ""
echo "=========================================="
echo "Running Dart Verification..."
echo "Using dart: $DART_CMD"
echo "=========================================="
cd dart_example
$DART_CMD pub get
$DART_CMD run bin/main.dart

echo ""
echo "=========================================="
echo "ALL DONE! Project is ready."
echo "=========================================="
