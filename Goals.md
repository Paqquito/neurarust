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

**Phase 0: Foundations & Core Tensor [✅ Done]**
*   🎯 **Goal:** Establish project structure, implement basic CPU `Tensor` with core functionalities.
*   **0.1 Project Setup [✅ Done]**
    *   ✅ Workspace Setup: Defined workspace in root `Cargo.toml`, configured basic CI, added `rustfmt.toml` and standard `clippy` lints.
    *   ✅ Licensing: Added `LICENSE` file (MIT/Apache 2.0 chosen).
    *   ✅ Contribution Docs: Created `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
*   **0.2 Core `Tensor` Struct (`neurarust-core`) [✅ Done]**
    *   ✅ `Tensor` Struct Definition (`tensor::mod.rs`): Created the main user-facing `Tensor` struct.
    *   ✅ `TensorData` Struct Definition (`tensor_data.rs`): Internal struct holding core data.
    *   ✅ Data Storage: Transitioned from `Rc<RefCell<Vec<T>>>` to `Arc<RwLock<TensorData>>` holding an `Arc<Buffer>`. This resolves the thread-safety limitations of the previous approach, enabling future parallelism.
    *   ✅ Shape Representation: Implemented using `shape: Vec<usize>` field in `TensorData`.
    *   ✅ **Strides Representation:** Implemented `strides: Vec<usize>` field to `TensorData`.
    *   ✅ Basic Creation Methods: `Tensor::new`, `Tensor::zeros_like`, `Tensor::ones_like` adapted.
    *   ✅ **Standalone Creation Functions:** Implemented `zeros`, `ones`, `full` (for F32/CPU).
    *   ✅ Initial Data Type Support: Refactored to non-generic `Tensor`. Now uses `DType` enum, `Buffer` enum, `StorageDevice` enum. Currently focused on `DType::F32` on `StorageDevice::CPU`.
*   **0.3 Basic CPU Operations (`neurarust-core::ops` - Forward Pass Only) [✅ Done]**
    *   ✅ Element-wise Arithmetic (`ops::arithmetic`): Forward pass implemented for `add`, `sub`, `mul`, `div`, `neg`, `pow` (for F32/CPU).
    *   ✅ Broadcasting Utilities (`tensor::utils`): `broadcast_shapes` helper implemented.
    *   ✅ Add Operation with Broadcasting: Forward pass handles broadcasting.
    *   ✅ **Stride-Aware Indexing:** Added `TensorData::get_offset`. Forward passes use `get_offset` for data access.
    *   ✅ **Basic Backward Infrastructure:** `BackwardOp` trait exists (non-generic), graph structure defined.
*   **0.4 Initial Testing [✅ Done]**
    *   ✅ Basic Unit Tests: Added tests covering `Tensor` creation, shape validation, basic arithmetic operations (forward pass), broadcasting, and creation functions.
*   **0.5 Overall Status & Key Issues [✅ Done]**
    *   **Status:** Project structure and foundational `Tensor` struct are implemented with explicit stride support and thread-safety (`Arc<RwLock>`). Basic element-wise operations use strides. Standalone creation functions added. Core error handling implemented. Codebase cleaned, tests pass.
    *   ✅ **Strides Stored for Views:** Prerequisite met.
    *   ✅ **Error Handling Improvement:** Addressed. Core functions return `Result`.
    *   ✅ **Thread-Safety for Parallelism:** Addressed via `Arc<RwLock>`.

**Phase 1: Views, Autograd & Expanded CPU Ops [🚧 In Progress]**
*   🎯 **Goal:** Implement view semantics, establish and **validate** a functional dynamic autograd system on CPU, and implement backward passes for core CPU tensor operations & API, **ensuring compatibility with the `Arc<RwLock>`, `Buffer`, `DType` and `StorageDevice` structures for F32/CPU.**

*   **1.1 View Semantics & Core Shape Ops [✅ Done]**
    *   🎯 Goal: Implement non-copying views for shape manipulation.
    *   ✅ **Refine `TensorData::new_view`:** Works with `Arc<Buffer>`, etc.
    *   ✅ **Implement `slice` Operation.**
    *   ✅ **Implement `transpose` Operation.**
    *   ✅ **Implement `permute` Operation.**
    *   ✅ **Implement `reshape` / `view` Operation:** Done, but requires input/output to be representable by stride manipulation only. Often requires `.contiguous()` first.
    *   ✅ **Implement `contiguous()` Method.**
    *   ✅ **Implement `is_contiguous()` Check.**

*   **1.2 Basic Autograd Infrastructure [✅ Done]**
    *   🎯 Goal: Establish the foundational components for automatic differentiation.
    *   ✅ **Add `TensorData` Fields:** `requires_grad`, `grad`, `grad_fn`.
    *   ✅ **Define `BackwardOp` Trait:** Non-generic, `Send + Sync`.
    *   ✅ **Implement `Tensor` Autograd Accessors/Mutators.**
    *   ✅ **Implement Graph Traversal (`autograd::graph`).**
    *   ✅ **Implement `Tensor::backward()` Logic.**

*   **1.3 Autograd Integration for `Add` Op [✅ Done (Validated Manually)]**
    *   🎯 Goal: Implement the first end-to-end autograd path.
    *   ✅ **Define `AddBackward` Struct.**
    *   ✅ **Implement `BackwardOp` for `AddBackward`.**
    *   ✅ **Modify `add_op` Forward Pass:** Sets `grad_fn`.
    *   ✅ **Validation:** Gradient logic verified manually.

*   **1.4 Numerical Gradient Checking Utility [❌ Needs Complete Rework/Replacement]**
    *   🎯 Goal: Implement `check_grad` for verifying backward implementations.
    *   ❌ **Implementation Status:** Current `check_grad` is **unreliable** (f32 vs f64 issues, non-leaf tensor warnings). It does not provide trustworthy validation for backward passes. **Decision needed: Fix, replace, or rely solely on manual checks.**

*   **1.5 First Autograd Tests (`Add` Op) [✅ Done (Validated Manually)]**
    *   🎯 Goal: Test the first autograd path.
    *   ✅ **Test Cases:** `test_add_backward_simple`, `test_add_backward_broadcast` pass using **manual gradient checks**, not the unreliable `check_grad`.

*   **1.6 Autograd Integration for Basic Arithmetic Ops [✅ Done (Validated Manually)]**
    *   🎯 Goal: Extend autograd support to other basic arithmetic operations.
    *   ✅ **Implement Backward Structs:** `MulBackward`, `NegBackward`, `SubBackward`, `DivBackward`, `PowBackward` (base grad only) implemented.
    *   ✅ **Modify Forward Ops:** Integrated autograd logic.
    *   ✅ **Tests & Validation:** Tests exist and **pass using manual gradient checks**.

*   **1.7 Autograd Integration for View Ops [🚧 Partially Done]**
    *   🎯 Goal: Implement backward passes for view operations.
    *   ✅ **Implement Backward Structs:** `SliceBackward`, `TransposeBackward`, `ReshapeBackward` implemented.
    *   🚧 **`PermuteBackward` has `todo!` in backward method.**
    *   ✅ **Modify Forward Ops:** Integrated autograd logic.
    *   🚧 **Tests & Validation:** Tests for `Slice`, `Transpose`, `Reshape` pass **using manual gradient checks**. `Permute` backward is not tested due to `todo!`. High-dim `transpose` test ignored (f32 precision).

*   **1.8 Autograd Integration for Reduction Ops [🚧 Partially Done]**
    *   🎯 Goal: Implement backward passes for reduction operations.
    *   ✅ **Implement `SumAxesBackward`:** Defined and implemented.
    *   ❌ **Implement `MeanBackward`:** Struct defined, but `mean_op` is `dead_code`, **not integrated or tested.**
    *   ✅ **Implement `MaxBackward`:** Implementation exists, `#[allow(dead_code)]` added to op/helpers due to `#[path]` test structure.
    *   ✅ **Modify `sum_axes_op`:** Integrates autograd.
    *   ❌ **Modify `mean_op`:** Marked `dead_code`, **needs integration.**
    *   🚧 **Modify `max_op`:** Marked `dead_code` (allowed), but **integration confirmed via manual tests.**
    *   ✅ **Add Tests (`Sum`):** Backward tests pass **using manual checks**.
    *   ❌ **Add Tests (`Mean`):** Backward tests **missing/non-functional.**
    *   ✅ **Add Tests (`Max`):** Backward tests exist and pass **using manual checks**.

*   **1.9 Autograd Integration for Other Core Ops [✅ Done (Validated Manually)]**
    *   🎯 Goal: Implement backward passes for remaining essential ops.
    *   ✅ **Implement `ReluBackward`**.
    *   ✅ **Implement `LnBackward`**.
    *   ✅ **Implement `MatmulBackward` (2D)**.
    *   ✅ **Modify `relu_op`, `ln_op`, `matmul_op`:** Integrate autograd.
    *   ✅ **Tests & Validation:** `Ln` backward passes. `Relu` backward manually checked. `Matmul` backward manually checked (standard `check_grad` tests ignored due to f32 instability).

*   **1.10 Tensor API & Data Type Expansion [🚧 In Progress]**
    *   🎯 Goal: Enhance `Tensor` usability and type support, **currently focused on adapting all ops to the non-generic `Tensor` with `Buffer`/`DType` for F32/CPU.**
    *   ⏳ **Implement Creation Methods:** `arange`, `linspace`, `eye`, `rand`, `randn` defined but unused. **Need integration/usage in tests or examples.**
    *   **`DType` Handling (Decomposed):**
        *   **Phase 1.10.A: Foundations & F32 Adaptation [🚧 In Progress]**
            *   ✅ **1-6. Core Structures & Methods:** `DType`, `Buffer`, `TensorData`, `Tensor`, base methods, creation functions adapted for non-generic `F32/CPU`.
            *   🚧 **7. Incremental Adaptation of Operations:** Most ops adapted for F32/CPU forward/backward. **Remaining Ops:** `mean`, `max` adaptation/integration needed.
            *   ✅ **8. Continuous Integration & Commits:** Followed.
        *   **Phase 1.10.B: Add Second DType (e.g., I64) [⏳ To Do]**
        *   **Phase 1.10.C: Mixed Types & Conversion [⏳ To Do]**
        *   **Phase 1.10.D: Extend to Other Ops & DTypes [⏳ To Do]**
        *   **Phase 1.10.E: In-place Operations [⏳ To Do]**
    *   ✅ **Implement `Tensor::detach()`:** Functionality exists (implicitly).

*   **1.11 Testing & Documentation Consolidation [🚧 Partially Done]**
    *   🎯 Goal: Ensure comprehensive testing and documentation for Phase 1 features.
    *   🚧 Expand Unit Tests: Good forward coverage. **Backward validation relies heavily on manual checks due to unreliable `check_grad` (1.4).** Needs fixes/expansion for `mean`, `max`, ignored tests.
    *   ⏳ Consider Property-Based Testing (`proptest`).
    *   ❌ Documentation (`rustdoc`, Guides): **Significantly outdated.** Needs complete update for non-generic API, `DType`/`Buffer`/`Device` structure, current autograd status (**manual checks preferred!**), `check_grad` issues.

*   **Overall Status Phase 1:** Major refactoring (non-generic Tensor, Arc<RwLock>, DType/Buffer) largely complete for F32/CPU. Most core ops have manually validated backward passes. **Key Blockers/Issues: 1) Unreliable `check_grad` utility (1.4). 2) Integration/testing of `mean_op` and confirmation of `max_op` usage (1.8). 3) Resolving `PermuteBackward` `todo!` (1.7). 4) Complete documentation update (1.11).**

**Phase 2: Neural Network Primitives & Optimization [⏳ To Do]**
*   🎯 **Goal:** Build foundational `nn` modules, loss functions, and optimization algorithms to enable basic model definition and training, **integrating device management (`CPU`/`GPU` eventually) and leveraging the thread-safe `Tensor` structure.**
*   📝 **Note:** All components were removed during refactoring and need reimplementation with device awareness.
*   **2.1 NN Module System (`neurarust-core::nn` or new crate?) [⏳ To Do]**
    *   ⏳ `Module` Trait (with device handling)
    *   ⏳ `Parameter` Struct
    *   ⏳ Module Containers (`Sequential`, etc.)
    *   ⏳ Helper Methods (`named_parameters`, etc.)
*   **2.2 Core Layers (`neurarust-core::nn::layers`?) [⏳ To Do]**
    *   ⏳ Linear Layer (device aware)
    *   ⏳ Other Layers (Conv, Pool, etc.)
*   **2.3 Loss Functions (`neurarust-core::nn::losses`?) [⏳ To Do]**
    *   ⏳ MSE (device aware)
    *   ⏳ Other Losses (CrossEntropy, etc.)
*   **2.4 Weight Initialization (`neurarust-core::nn::init`?) [⏳ To Do]**
*   **2.5 Optimizers (`neurarust-optim` or `neurarust-core::optim`?) [⏳ To Do]**
    *   ⏳ `Optimizer` Trait (device aware)
    *   ⏳ SGD Implementation (device aware)
    *   ⏳ Adam Implementation (device aware state)
*   **2.6 Learning Rate Schedulers [⏳ To Do]**
*   **2.7 Integration & Training Loop Example [⏳ To Do]** (Must show device handling)
*   **2.8 Serialization [⏳ To Do]** (Must handle device mapping)
*   **2.9 Testing & Documentation [⏳ To Do]** (Must cover device scenarios)
*   **Overall Status Phase 2:** **Not started.**

**Phase 3: Data Loading & Handling (`neurarust-data` or `neurarust-core::data`?) [⏳ To Do]**
*   🎯 **Goal:** Develop robust and performant tools for data loading, preprocessing, and augmentation, **ensuring efficient batch creation (potentially on target device later) and leveraging thread-safe structures.**
*   📝 **Note:** All components were removed during refactoring and need reimplementation with device/collation awareness.
*   **3.1 Dataset Abstractions [⏳ To Do]**
*   **3.2 DataLoader [⏳ To Do]** (Focus on device-aware collation, parallelism, `pin_memory`)
*   **3.3 Data Preprocessing & Augmentation (`neurarust-vision`, `neurarust-text`?) [⏳ To Do]**
*   **3.4 Integration & Utilities [⏳ To Do]**
*   **3.5 Testing & Documentation [⏳ To Do]**
*   **Overall Status Phase 3:** **Not started.**

**Phase 4: GPU Acceleration (CUDA First, then Others) [⏳ To Do]**
*   🎯 **Goal:** Enable high-performance computation using accelerators (starting with NVIDIA GPUs via CUDA), **leveraging the `Buffer`/`StorageDevice` framework.**
*   **Overall Status Phase 4:** **To Do.**

---
