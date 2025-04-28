# NeuraRust 🦀🧠

**Goal**: Create a performant, safe, and ergonomic deep learning framework in Rust, inspired by PyTorch but leveraging Rust's unique advantages.

[![Rust](https://github.com/Paqquito/NeuraRust/actions/workflows/rust.yml/badge.svg)](https://github.com/Paqquito/NeuraRust/actions/workflows/rust.yml)

---

## ✨ Core Features (Based on Goals.md)

NeuraRust aims to provide a PyTorch-like experience while harnessing the power of Rust:

*   **Tensors (`neurarust-core::Tensor`)**: A performant multi-dimensional `Tensor` structure with explicit memory management and Rust-guaranteed safety. Includes broadcasting capabilities (initially for Add).
*   **Automatic Differentiation (`neurarust-core::Autograd`)**: Dynamic autograd engine to automatically compute gradients via `.backward()`. Currently supports basic operations including Add with broadcasting.
*   **Neural Network Modules (`neurarust-nn`)** *(Future)*: Building blocks (Linear, Conv layers, etc.) and activation/loss functions.
*   **Optimizers (`neurarust-optim`)** *(Partially Implemented)*: Standard optimization algorithms (SGD implemented, Adam...). 
*   **Data Handling (`neurarust-data`)** *(Future)*: Tools for loading and preprocessing data (`Dataset`, `DataLoader`).
*   **Accelerator Support** *(Future)*: GPU integration (CUDA, etc.) for fast computations.
*   **Interoperability & Deployment** *(Future)*: ONNX export, Python bindings (PyO3), WASM and native binary compilation.

## 🎯 Key Rust Advantages

*   **Performance:** Native speed close to C/C++, fine-grained memory control.
*   **Safety:** Guaranteed absence of data races and many memory errors thanks to the compiler.
*   **Concurrency:** "Fearless" parallelism for multi-core acceleration (e.g., data loading, autograd).
*   **Deployment:** Compilation to WASM, lightweight and standalone native binaries.

## 🚧 Current Status (According to Roadmap)

The project is currently in **early development (Phase 0 & 1)**:

*   ✅ **Phase 0: Foundations & Basic CPU Tensor [Completed]**
    *   Project structure (Cargo workspace, CI).
    *   Initial `Tensor` implementation (data, shape, Rc<RefCell>).
    *   Fundamental CPU operations (element-wise arithmetic).
    *   Broadcasting utilities and integration into Add (forward/backward).
    *   Unit tests for `Tensor`, basic ops, and broadcasting.
*   🚧 **Phase 1: Autograd & NN Building Blocks [In Progress]**
    *   Basic Autograd engine foundations (`BackwardOp` trait, dynamic graph).
    *   Backward pass implemented for Add with broadcasting.
    *   *Next steps: Implement broadcasting for other ops (Sub, Mul, Div), implement more core Tensor ops (MatMul, reductions...), start basic `nn` modules (Linear, losses).* 

See [`Goals.md`](Goals.md) for the complete roadmap.

## 🚀 Getting Started

1.  **Prerequisites:** Ensure you have [Rust installed](https://www.rust-lang.org/tools/install).
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/Paqquito/NeuraRust.git # TODO: Use correct URL
    cd NeuraRust
    ```
3.  **Build:**
    ```bash
    cargo build
    ```
4.  **Run tests:**
    ```bash
    cargo test
    ```

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) (to be created) for guidelines.

## 📜 License

This project is licensed under the [MIT License](LICENSE). 