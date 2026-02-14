# Edge Tensor Engine

A high-performance Tensor and Autograd engine written in Rust, optimized for edge inference (ARM NEON / x86 AVX2). Designed for integration with Dart/Flutter via FFI.

## Features

- **Core**: Thread-safe `Tensor` struct with zero-copy slicing and Arcs.
- **Autograd**: Reverse-mode generic autodiff engine.
- **Optimized Ops**: 
  - Cache-blocked Matrix Multiplication (GEMM).
  - INT4 Quantized Linear Layer support.
  - SIMD optimizations (AVX2/NEON).
- **Transformer Ready**: Includes RoPE, KV-Cache, and Attention mechanisms.
- **Dart Integration**: Full FFI bindings and example usage.

## Prerequisities

- **Rust**: Install via `rustup` (recommended).
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Dart**: Install Dart SDK (or Flutter).

## Building

1.  **Build Rust Library**:
    ```bash
    cargo build --release
    ```
    This will produce `target/release/libedge_tensor_engine.so` (Linux), `.dylib` (macOS), or `.dll` (Windows).

2.  **Run Dart Example**:
    ```bash
    cd dart_example
    dart pub get
    dart run bin/main.dart
    ```

## Testing

To run the comprehensive unit test suite (Core, Autograd, Ops, Quantization):
```bash
cargo test
# OR use the helper script
bash test.sh
```

## Project Structure

- `src/tensor`: Core struct, storage, and DType.
- `src/autograd`: Backward pass engine and Node trait.
- `src/ops`: Mathematical operations (Matmul, Add, etc.).
- `src/nn`: Neural network layers (LinearInt4, Attention).
- `src/ffi`: C ABI exports for Dart.
- `dart_example`: Dart implementation using `dart:ffi`.
