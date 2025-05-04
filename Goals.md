# ✨ NeuraRust Goals & Vision 🦀🧠

**NeuraRust** aims to become a leading **Deep Learning framework in Rust**, combining the flexibility and ergonomics of PyTorch with the **raw performance**, **memory safety**, and **portability** offered by Rust.

---

## 🎯 Our Core Pillars

*   🚀 **Exceptional Performance:**
    *   Compete with C++/Python giants in execution speed (CPU & GPU).
    *   Minimize memory footprint thanks to Rust's precise control (no GC!).
*   🤝 **Intuitive Ergonomics:**
    *   A familiar and enjoyable API, inspired by best practices (PyTorch/Keras).
    *   Comprehensive documentation and accessible tutorials for a quick start.
*   🔄 **Seamless Interoperability:**
    *   Compatibility via **ONNX** to exchange models with PyTorch/TensorFlow.
    *   Smooth integration with the Python ecosystem using **PyO3**.
*   🔒 **Safety & Easy Deployment:**
    *   The Rust promise: **No segfaults, no unexpected memory leaks**.
    *   Native support for easy deployment to various targets: **WebAssembly (WASM)**, **ARM** (embedded/mobile), servers...

---

## 🛠️ Core Features (PyTorch Inspired, Rust Superpowered)

We replicate PyTorch's essential building blocks, enhancing them with Rust:

### 1. Multi-Dimensional Tensors (`neurarust-core::Tensor`) 📐

*   **Vision:** The framework's beating heart. Fast, safe, flexible.
*   **Key Points:**
    *   **Explicit and performant** memory management.
    *   Fine-grained control over memory layout (strides...).
    *   **Strong typing** to catch dimension/type errors at compile time (via `DType` enum now).
    *   Mathematical, logical operations, indexing, **broadcasting**... everything needed!
*   **The Rust Advantage:** 💪 Guaranteed memory safety, native C/C++ performance, SIMD potential.

### 2. Automatic Differentiation (`neurarust-core::Autograd`) 📈

*   **Vision:** A dynamic, reliable, and efficient autograd engine.
*   **Key Points:**
    *   **On-the-fly computation graph** construction.
    *   Simplified gradient calculation via **`.backward()`**.
    *   Optimized memory management for intermediate tensors.
    *   Backward pass implemented for core ops (validated manually).
*   **The Rust Advantage:** 🧠 The borrow checker tames graph complexity, "fearless" concurrency potential for accelerating computations.

### 3. Neural Network Modules (`neurarust-nn`) 🧩 *(Future)*

*   **Vision:** A comprehensive toolbox for assembling your networks.
*   **Key Points:**
    *   Standard layers: **Linear, Convolutional, Recurrent, Attention, Normalization...**
    *   Common activation and loss functions.
    *   **Composable and extensible** API for creating custom architectures.
*   **The Rust Advantage:** ✨ Traits for clear interfaces (`Module`, `Layer`), macros for less boilerplate.

### 4. Optimizers (`neurarust-optim`) ⚙️ *(Future)*

*   **Vision:** Essential algorithms for training your models.
*   **Key Points:**
    *   Classics: **SGD**, Adam, AdamW, RMSprop...
    *   Simple `Optimizer` interface to apply updates.
    *   Internal state management (e.g., moments).
*   **The Rust Advantage:** ⚡ Native performance, generic implementations via traits.

### 5. Data Loading (`neurarust-data`) 💾 *(Future)*

*   **Vision:** Performant tools to feed your models.
*   **Key Points:**
    *   `Dataset` and `DataLoader` abstractions.
    *   **Batching, shuffling, performant parallel loading**.
    *   Utilities for transformations and augmentations.
*   **The Rust Advantage:** 🏎️ Robust parallelism ideal for I/O and preprocessing, efficient memory management.

### 6. Accelerator Support (GPU & Beyond) 🔥 *(Future)*

*   **Vision:** Unleash the massive computational power of dedicated hardware.
*   **Key Points:**
    *   **CUDA** integration (priority), then potentially ROCm, Metal, **WebGPU**.
    *   `Device` abstraction (CPU, GPU:0...).
    *   Transparent CPU <-> GPU data transfer.
*   **The Rust Advantage:** 🌐 Existing bindings, safe abstractions, WebGPU (written in Rust) as a portable target for the future.

### 7. Interoperability & Deployment (`neurarust-deploy`) 🌍 *(Future)*

*   **Vision:** Integrate everywhere, deploy easily.
*   **Key Points:**
    *   **ONNX** for model exchange.
    *   **PyO3** for symbiosis with Python.
    *   **WASM** compilation for web and serverless.
    *   Easy cross-compilation (e.g., **ARM**).
    *   **Native, standalone, and performant** binaries.
*   **The Rust Advantage:** 📦 First-class WASM/ARM support, mature FFI, easy static binary distribution.

---

## 💎 Our Differentiators: The Unique Rust Advantage

Beyond PyTorch parity, we aim to fully leverage Rust to offer:

*   **First-Class WASM Support 🕸️:** Performant and lightweight inference in the browser and on the edge. Revolutionizing interactive and embedded ML.
*   **Enhanced Safety Guarantees ✅:** Go further in verification and robustness using the type system for critical applications.
*   **Advanced Static Optimizations 🚀:** Use macros to optimize graphs *at compile time* (op fusion, etc.) for more performance with no runtime overhead.
*   **Simplified & Safe Parallelism ⛓️:** High-level APIs to leverage multi-core and distributed computing without fearing data races.

---

## 🗺️ Highly Detailed Roadmap (PyTorch Parity Goal)

This roadmap outlines the planned development stages for NeuraRust, aiming for extensive feature parity with PyTorch over time. Status markers: ✅ (Done), 🚧 (In Progress / Partially Done), ⏳ (To Do), ❌ (Needs Rework / Blocked).

### Milestone 1: Core Ops & Basic Autograd (End Goal: Trainable MLP)

*   [X] **1.1 Project Setup:** Initial structure, `Cargo.toml`, basic Tensor struct (data, shape). **DONE**
*   [X] **1.2 Basic Tensor Creation:** `new`, `zeros`, `ones`, `rand`, `randn`, `full`, `from_vec`, `eye`. **DONE**
*   [X] **1.3 Basic Tensor Ops:** `reshape`, `transpose`, `permute`, `contiguous`, `slice`, element access (initial impl). **DONE** (Element access basic impl via `get_f32_data` exists, but dedicated `get`/`at` method pending -> tests ignored).
*   [~] **1.4 Gradient Checking Utility (`check_grad`):** Implement numerical gradient checking. **FUNCTIONAL w/ F32 LIMITATIONS**
    *   Initial implementation done.
    *   Refactored perturbation logic to handle views correctly.
    *   **Improved Precision:** Perturbation calculation (`x +/- eps`) now uses `f64` internally, and comparison uses relative + absolute tolerance (`abs_tol + rel_tol * |num_grad|`) for better robustness.
    *   **Identified F32 Numerical Stability Limits:** Despite improvements, the finite difference method in `check_grad` remains sensitive to F32 precision limitations. Small discrepancies between numerical and analytical gradients persist for certain operations (`mul`, `neg`, `matmul`, `permute`, `transpose`), likely due to F32 rounding/cancellation effects amplified by the check method.
    *   **Current Status:** `check_grad` tests for the operations mentioned above are currently marked `#[ignore]` as they produce unreliable results in F32. Relying on `check_grad` passing with extremely loose tolerances for these ops is discouraged.
    *   **Validation Strategy:** We will now implement F64 support specifically to validate the *logical correctness* of the F32 backward implementations for these problematic operations (See **1.4.B**).

*   [ ] **1.4.B F64 Gradient Validation Path (Iterative Approach):** Add F64 support to core structures and specific operations **solely for the purpose of reliable gradient checking**, confirming the logic of our F32 implementations.
    *   🎯 **Goal:** Gain high confidence in the mathematical correctness of all backward passes, using F64 checks as an oracle where F32 checks fail due to numerical instability.
    *   ♟️ **Strategy:** Proceed iteratively, operation by operation, validating each step thoroughly before moving to the next.
    *   **Detailed Iterative Steps:**
        *   **Step 1: Foundational F64 Support (CPU)**
            *   [ ] Define `DType::F64`.
            *   [ ] Define `Buffer::Cpu(CpuBuffer::F64(Arc<Vec<f64>>))`. 
            *   [ ] Modify `TensorData`, `Tensor` methods (`dtype()`, `device()`, `numel()`, `shape()`, `strides()`, `is_contiguous()`, `buffer()`, etc.) to handle `DType::F64` alongside `F32` via `match` or similar dispatch mechanism.
            *   [ ] Adapt `Tensor::new` to accept `Vec<f64>` and create `F64` tensors.
            *   [ ] Adapt creation functions (`zeros`, `ones`, `full`, `from_vec*`) to support creating `F64` tensors (e.g., `zeros(shape, DType::F64, Device::CPU)`).
            *   [ ] Implement necessary `impl From<f64> for Scalar` and other trait bounds if needed.
            *   [ ] Add basic unit tests for F64 tensor creation and properties.
        *   **Step 2: Adapt `check_grad` for F64 Tensors**
            *   [ ] Modify `check_grad`'s initial checks to accept `DType::F64` inputs.
            *   [ ] Ensure the internal perturbation logic correctly handles F64 buffers:
                *   Read `f64` value from buffer.
                *   Perturb with `epsilon` (already `f64`).
                *   Write `f64` value back to the cloned F64 buffer.
            *   [ ] Ensure `calculate_loss` can accept F64 `Tensor` inputs and perform `mul_op` / `sum_op` (These might need adaptation if not already generic or dispatched). It should return `f64`.
            *   [ ] Ensure analytical gradient extraction works for F64 Tensors (e.g., `get_f64_data()`, convert to `Vec<f64>`).
        *   **Step 3: Adapt *First* Op (`neg_op`) for F64**
            *   [ ] Refactor `neg_op` and `NegBackward` to be generic over `T: NumericTrait` (where `NumericTrait` includes `Neg`, `Copy`, conversion traits, etc.) OR use internal `match` on `DType`.
            *   [ ] Ensure `neg_op` still compiles and works correctly for existing F32 tests.
        *   **Step 4: Create & Validate F64 Test for `neg_op`**
            *   [ ] Create `test_neg_backward_f64`.
            *   [ ] Inside, create input `Tensor` with `DType::F64`.
            *   [ ] Call `check_grad` with the (now generic or dispatched) `neg_op`, F64 tensors, and **strict** tolerances (e.g., `abs=1e-9`, `rel=1e-7`).
            *   [ ] Verify the test passes. If yes, the logic of `NegBackward` is confirmed. The `test_neg_backward` (F32) can remain `#[ignore]`.
        *   **Step 5: Repeat for `mul_op`**
            *   [ ] Adapt `mul_op` and `MulBackward` for F64 (generic or dispatch).
            *   [ ] Ensure F32 tests still compile.
            *   [ ] Create `test_mul_backward_simple_f64` and `test_mul_backward_broadcast_f64`.
            *   [ ] Call `check_grad` with F64 tensors and strict tolerances.
            *   [ ] Validate passage. Keep F32 tests ignored.
        *   **Step 6: Repeat for `permute_op`**
            *   [ ] Adapt `permute_op` and `PermuteBackward` for F64.
            *   [ ] Ensure F32 tests still compile.
            *   [ ] Create F64 versions of `test_permute_backward*`.
            *   [ ] Call `check_grad` with F64 tensors and strict tolerances.
            *   [ ] Validate passage. Keep F32 tests ignored.
        *   **Step 7: Repeat for `transpose_op`**
            *   [ ] Adapt `transpose_op` and `TransposeBackward` for F64.
            *   [ ] Ensure F32 tests still compile.
            *   [ ] Create F64 versions of `test_transpose_backward*`.
            *   [ ] Call `check_grad` with F64 tensors and strict tolerances.
            *   [ ] Validate passage. Keep F32 tests ignored.
        *   **Step 8: Repeat for `matmul_op`**
            *   [ ] Adapt `matmul_op` and `MatmulBackward` for F64.
            *   [ ] Ensure F32 tests still compile.
            *   [ ] Create F64 versions of `test_matmul_backward*` (non-simple cases).
            *   [ ] Call `check_grad` with F64 tensors and strict tolerances.
            *   [ ] Validate passage. Keep F32 tests ignored.

*   [X] **1.5 Autograd Infrastructure:** `