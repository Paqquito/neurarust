# ✨ NeuraRust 🦀🧠

[![CI](https://github.com/nebulyts/neurarust/actions/workflows/ci.yml/badge.svg)](https://github.com/nebulyts/neurarust/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://opensource.org/licenses/MIT)
[![Docs](https://docs.rs/neurarust-core/badge.svg)](https://docs.rs/neurarust-core) <!-- TODO: Update link when published -->

**NeuraRust** aims to become a leading **Deep Learning framework in Rust**, combining the flexibility and ergonomics of PyTorch with the **raw performance**, **memory safety**, and **portability** offered by Rust.

---

## 🚀 Current Status (Milestone 1: Core Ops & Autograd)

Milestone 1, focusing on core CPU tensor operations (F32) and a functional autograd system, is largely **complete**.

**Key Features Implemented:**

*   **Tensor Core (`neurarust-core`):**
    *   CPU Tensor implementation using `Arc<RwLock<TensorData>>` for thread-safety.
    *   Supports F32 data type (`DType::F32`).
    *   Explicit strides for efficient memory layout control.
    *   Creation ops: `new`, `zeros`, `ones`, `full`, `rand`, `randn`, `eye`.
*   **View Operations (Non-copying):**
    *   `slice`, `transpose`, `permute`, `reshape`/`view`.
    *   `contiguous()`, `is_contiguous()`.
*   **Automatic Differentiation (Autograd):**
    *   Dynamic computation graph tracking (`grad_fn`, `BackwardOp`).
    *   `.backward()` method for gradient calculation.
    *   Backward passes implemented for most core operations.
    *   Gradient checking utility (`check_grad`) is functional but has known precision limitations (F32/F64) for view ops (`permute`, `transpose`) and `matmul`. Related tests are currently ignored.
*   **Core Operations (CPU/F32, with Autograd):**
    *   **Element-wise Arithmetic:** `add`, `sub`, `mul`, `div`, `neg`, `pow`.
    *   **Reductions:** `sum`, `mean`, `max` (along axes or all).
    *   **Linear Algebra:** `matmul` (2D).
    *   **Activations:** `relu`.

**Next Steps:** Focus on Milestone 1 completion (MLP Layer, Loss, Training Loop) and documentation improvements.

---

## 🎯 Core Pillars & Vision

*(Keep existing Pillars & Vision sections)*
*   🚀 **Exceptional Performance**
*   🤝 **Intuitive Ergonomics**
*   🔄 **Seamless Interoperability**
*   🔒 **Safety & Easy Deployment**

---

## 🛠️ Core Features (PyTorch Inspired, Rust Superpowered)

*(Keep existing Core Features sections, potentially update status markers if needed)*
1.  Multi-Dimensional Tensors (`neurarust-core::Tensor`) 📐
2.  Automatic Differentiation (`neurarust-core::Autograd`) 📈
3.  Neural Network Modules (`neurarust-nn`) 🧩 *(Future)*
4.  Optimizers (`neurarust-optim`) ⚙️ *(Future)*
5.  Data Loading (`neurarust-data`) 💾 *(Future)*
6.  Accelerator Support (GPU & Beyond) 🔥 *(Future)*
7.  Interoperability & Deployment (`neurarust-deploy`) 🌍 *(Future)*

---

## 💎 Our Differentiators: The Unique Rust Advantage

*(Keep existing Differentiators section)*
*   First-Class WASM Support 🕸️
*   Enhanced Safety Guarantees ✅
*   Advanced Static Optimizations 🚀
*   Simplified & Safe Parallelism ⛓️

---

## 🗺️ Roadmap Summary

(See [Goals.md](./Goals.md) for the highly detailed roadmap)

*   ✅ **Milestone 1:** Core Ops & Basic Autograd (CPU/F32) - *Largely Complete*
*   ⏳ **Milestone 2:** Optimizers & Advanced Features
*   ⏳ **Milestone 3:** Ecosystem & Polish

---

*(Keep existing Contributing, License sections)*

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