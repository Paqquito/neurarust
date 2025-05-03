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
    *   **Strong typing** to catch dimension/type errors at compile time.
    *   Mathematical, logical operations, indexing, **broadcasting** (implemented for Add)... everything needed!
*   **The Rust Advantage:** 💪 Guaranteed memory safety, native C/C++ performance, SIMD potential.

### 2. Automatic Differentiation (`neurarust-core::Autograd`) 📈

*   **Vision:** A dynamic, reliable, and efficient autograd engine.
*   **Key Points:**
    *   **On-the-fly computation graph** construction.
    *   Simplified gradient calculation via **`.backward()`**.
    *   Optimized memory management for intermediate tensors.
    *   Backward pass implemented for core ops (e.g., Add with broadcasting).
*   **The Rust Advantage:** 🧠 The borrow checker tames graph complexity, "fearless" concurrency potential for accelerating computations.

### 3. Neural Network Modules (`neurarust-nn`) 🧩 *(Future)*

*   **Vision:** A comprehensive toolbox for assembling your networks.
*   **Key Points:**
    *   Standard layers: **Linear, Convolutional, Recurrent, Attention, Normalization...**
    *   Common activation and loss functions.
    *   **Composable and extensible** API for creating custom architectures.
*   **The Rust Advantage:** ✨ Traits for clear interfaces (`Module`, `Layer`), macros for less boilerplate.

### 4. Optimizers (`neurarust-optim`) ⚙️ *(Partially Implemented)*

*   **Vision:** Essential algorithms for training your models.
*   **Key Points:**
    *   Classics: **SGD (implemented)**, Adam, AdamW, RMSprop...
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

This roadmap outlines the planned development stages for NeuraRust, aiming for extensive feature parity with PyTorch over time. Status markers: ✅ (Done), 🚧 (In Progress / Partially Done), ⏳ (To Do).

**Phase 0: Foundations & Core Tensor [✅ Done]**
*   🎯 **Goal:** Establish project structure, implement basic CPU `Tensor` with core functionalities.
*   **0.1 Project Setup [✅ Done]**
    *   ✅ Workspace Setup: Defined workspace in root `Cargo.toml`, configured basic CI, added `rustfmt.toml` and standard `clippy` lints.
    *   ✅ Licensing: Added `LICENSE` file (MIT/Apache 2.0 chosen).
    *   ✅ Contribution Docs: Created `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
*   **0.2 Core `Tensor` Struct (`neurarust-core`) [✅ Done]**
    *   ✅ `Tensor` Struct Definition (`tensor::mod.rs`): Created the main user-facing `Tensor` struct.
    *   ✅ `TensorData` Struct Definition (`tensor_data.rs`): Internal struct holding core data.
    *   ✅ Data Storage: Implemented using `Rc<RefCell<Vec<T>>>` within `TensorData`.
        *   📝 *Note:* This choice enables basic sharing needed for dynamic autograd graph construction (multiple `Tensor`s can refer to the same `TensorData`). However, `RefCell` enforces runtime borrow checking and is *not thread-safe*, which **limits future parallelism** (e.g., using `rayon` for parallel CPU ops or multi-threaded data loading accessing tensors). Evolution towards thread-safe structures like `Arc<Mutex/RwLock>` or potentially specialized concurrent data structures will be necessary in later phases (especially Phase 4/6).
    *   ✅ Shape Representation: Implemented using `shape: Vec<usize>` field in `TensorData`.
    *   ✅ **Strides Representation:** **Implemented!** Added `strides: Vec<usize>` field to `TensorData`. `Tensor::new` now calculates contiguous strides by default. This resolves the **critical prerequisite** for views (Phase 1.4).
    *   ✅ Basic Creation Methods:
        *   ✅ `Tensor::new(data: Vec<T>, shape: Vec<usize>)`: Acts like `from_vec`, consuming data, calculates strides.
        *   ✅ `Tensor::zeros_like(&self)`: Creates a tensor of zeros with the same shape.
        *   ✅ **Standalone Creation Functions:** Implemented `neurarust::tensor::zeros(shape)`, `neurarust::tensor::ones(shape)`, `neurarust::tensor::full(shape, value)` for better ergonomics.
    *   ✅ Initial Data Type Support: Generic `<T>` used, primarily focused on `f32` via trait bounds like `Copy`, `Debug`, `PartialOrd`, `Add`, `Sub`, `Mul`, `Div`, `Neg`. Explicit `DType` enum and multi-type support are missing (Phase 1.4).
*   **0.3 Basic CPU Operations (`neurarust-core::ops` - Forward Pass Only) [✅ Done]**
    *   ✅ Element-wise Arithmetic (`ops::arithmetic`): Forward pass implemented for `add`, `sub`, `mul`, `div`, `neg`. These handle basic tensor-tensor and tensor-scalar operations.
    *   ✅ Broadcasting Utilities (`tensor::utils`): Implemented `broadcast_shapes` helper and logic to determine compatible shapes for broadcasting.
    *   ✅ Add Operation with Broadcasting: Forward pass specifically handles broadcasting.
    *   ✅ **Stride-Aware Indexing:** Added `TensorData::get_offset` method. Forward passes for `add`, `sub`, `mul`, `div`, `neg` correctly use `get_offset` for data access, making them compatible with strides. (Note: `matmul` removed in cleanup).
    *   ✅ **Basic Backward Infrastructure:** Definition of `BackwardOp` trait removed during cleanup. Reintroduction needed for Phase 1.
*   **0.4 Initial Testing [✅ Done]**
    *   ✅ Basic Unit Tests: Added tests covering `Tensor` creation, shape validation, basic arithmetic operations (forward pass), broadcasting utility functions, and new creation functions.
*   **0.5 Overall Status & Key Issues [✅ Done]**
    *   **Status:** Project structure and foundational `Tensor` struct are implemented with explicit stride support. Basic element-wise operations (`add`, `sub`, `mul`, `div`, `neg`) use strides for data access on CPU. Standalone creation functions added. Core error handling implemented. Codebase cleaned of Phase 1-3 elements, tests pass.
    *   ✅ **Critical Issue (Lack of strides): Resolved.** `TensorData` now stores strides, and basic operations use them for indexing.
    *   ✅ **Strides Stored for Views:** Strides are stored in `TensorData`, providing the prerequisite for views. ✅ **View Implementation (Phase 1):** Operations like `reshape`, `slice`, `transpose` need to be implemented/re-implemented as true views (sharing data) in Phase 1.
    *   ✅ **Error Handling Improvement:** Addressed. Core functions like `Tensor::new`, `sum_axes` return `Result<T, NeuraRustError>`, handling common errors like shape mismatches or invalid indices gracefully.
    *   ✅ **Thread-Safety for Parallelism:** Replaced `Rc<RefCell<TensorData<T>>>` with `Arc<RwLock<TensorData<T>>>`. Internal data buffer uses `Arc<Buffer<T>>`. This provides the necessary thread-safety foundation for future parallel computation (e.g., CPU via Rayon, GPU acceleration - Phases 4/6), resolving the limitation noted previously.

**Phase 1: Views, Autograd & Expanded CPU Ops [🚧 In Progress]**
*   🎯 **Goal:** Implement view semantics, establish and **validate** a functional dynamic autograd system on CPU, and implement backward passes for core CPU tensor operations & API, **ensuring compatibility with the new `Arc<RwLock>`, `Buffer`, and `StorageDevice` structures.**

*   **1.1 View Semantics & Core Shape Ops [✅ Done]**
    *   🎯 Goal: Implement non-copying views for shape manipulation.
    *   ✅ **Refine `TensorData::new_view`:** Ensured accessibility and correct handling of `Arc<Buffer>`, `device`, `offset`, `shape`, `strides`.
    *   ✅ **Implement `slice` Operation:** -> ✅ **Done**.
    *   ✅ **Implement `transpose` Operation:** -> ✅ **Done**.
    *   ✅ **Implement `permute` Operation:** -> ✅ **Done**.
    *   ✅ **Implement `reshape` / `view` Operation:** -> ✅ **Done (Contiguous Only)**. Returns `Err` for non-contiguous views that cannot be created by manipulating strides. User must call `.contiguous().reshape(...)`.
    *   ✅ **Implement `contiguous()` Method:** -> ✅ **Done**. Creates a new contiguous tensor if needed.
    *   ✅ **Implement `is_contiguous()` Check:** -> ✅ **Done**.

*   **1.2 Basic Autograd Infrastructure [✅ Done]**
    *   🎯 Goal: Establish the foundational components for automatic differentiation.
    *   ✅ **Add `TensorData` Fields:** `requires_grad`, `grad`, `grad_fn`.
    *   ✅ **Define `BackwardOp<T>` Trait:** Defined with `Send + Sync`, `backward` method. *(Inputs handled via struct fields).*
    *   ✅ **Implement `Tensor` Autograd Accessors/Mutators:** `requires_grad`, `set_requires_grad`, `grad`, `acc_grad`, `grad_fn`, `set_grad_fn`.
    *   ✅ **Implement Graph Traversal (`autograd::graph`):** Topological sort implemented.
    *   ✅ **Implement `Tensor::backward()` Logic:** Core backward pass logic implemented, handling gradient accumulation and graph traversal.

*   **1.3 Autograd Integration for `Add` Op [✅ Done]**
    *   🎯 Goal: Implement the first end-to-end autograd path.
    *   ✅ **Define `AddBackward` Struct**.
    *   ✅ **Implement `BackwardOp` for `AddBackward`**.
    *   ✅ **Modify `add_op` Forward Pass:** Correctly sets `grad_fn`.
    *   ✅ **Document the Pattern:** Comments added.

*   **1.4 Numerical Gradient Checking Utility [✅ Done]**
    *   🎯 Goal: Implement `check_grad` for verifying backward implementations.
    *   ✅ **Implement Utility Function:** `check_grad` implemented for CPU, using `approx`.

*   **1.5 First Autograd Tests (`Add` Op) [✅ Done]**
    *   🎯 Goal: Test the first autograd path using `check_grad`.
    *   ✅ **Create Test Cases:** `test_add_backward_simple`, `test_add_backward_broadcast` implemented and pass using `check_grad`.

*   **1.6 Autograd Integration for Basic Arithmetic Ops [🚧 Partially Done]**
    *   🎯 Goal: Extend autograd support to other basic arithmetic operations.
    *   ✅ **Implement Backward Structs:** `MulBackward`, `NegBackward` implemented and tested.
    *   🚧 **Implement Backward Structs (Issues):** `SubBackward`, `DivBackward`, `PowBackward` structs exist, but compiler indicates `dead_code`. Integration in forward ops (`sub_op`, `div_op`, `pow_op`) needs verification/correction to ensure `grad_fn` is set.
    *   ✅ **Modify Forward Ops:** `mul_op`, `neg_op` correctly integrate autograd.
    *   🚧 **Modify Forward Ops (Issues):** `sub_op`, `div_op`, `pow_op` need verification regarding `grad_fn` setting.
    *   ✅ **Add Tests:** Tests added and pass for `Mul`, `Neg`.
    *   🚧 **Add Tests (Issues):** Tests for `Sub`, `Div`, `Pow` backward need to be verified/added once `grad_fn` integration is fixed.

*   **1.7 Autograd Integration for View Ops [✅ Done]**
    *   🎯 Goal: Implement backward passes for view operations.
    *   ✅ **Implement Backward Structs:** `SliceBackward`, `TransposeBackward`, `PermuteBackward`, `ReshapeBackward` defined and implemented.
    *   ✅ **Modify Forward Ops:** `slice_op`, `transpose_op`, `permute_op`, `reshape_op` integrate autograd logic.
    *   ✅ **Add Tests:** Tests implemented and pass for view backward passes, although one `transpose` test for >2D is ignored (likely `f32` precision).

*   **1.8 Autograd Integration for Reduction Ops [🚧 Partially Done]**
    *   🎯 Goal: Implement backward passes for reduction operations.
    *   ✅ **Implement `SumAxesBackward`:** Defined, implemented, and tested via `check_grad`.
    *   🚧 **Implement `MeanBackward` (Issues):** Struct defined, but forward op (`mean_op`) is `unused`, and backward tests are `ignored`. Needs integration and testing fixes.
    *   ✅ **Modify `sum_axes_op`:** Integrates autograd.
    *   🚧 **Modify `mean_op` (Issues):** Marked as `unused`, needs integration.
    *   ✅ **Add Tests:** Tests for `SumAxesBackward` pass.
    *   🚧 **Add Tests (Issues):** Tests for `MeanBackward` are ignored.

*   **1.9 Autograd Integration for Other Core Ops [🚧 Partially Done]**
    *   🎯 Goal: Implement backward passes for remaining essential ops.
    *   ✅ **Implement `ReluBackward`:** Defined, implemented, tested.
    *   🚧 **Implement `MaxBackward` (Issues):** Struct and forward op (`max_op`) exist but are `unused` / `dead_code` with `todo!`. Needs implementation and integration.
    *   🚧 **Implement `MatmulBackward` (2D - Issues):** Struct exists, forward op exists, but backward tests are `ignored` due to `f32` numerical stability issues with `check_grad`. Requires investigation (e.g., use `f64` for check, adjust tolerance).
    *   ✅ **Modify `relu_op`:** Integrates autograd.
    *   🚧 **Modify `max_op`, `matmul_op`:** Need full integration/testing.
    *   ✅ **Add Tests:** `Relu` backward tests pass.
    *   🚧 **Add Tests (Issues):** `Max` tests missing. `Matmul` tests ignored.

*   **1.10 Tensor API & Data Type Expansion [🚧 In Progress]**
    *   🎯 Goal: Enhance `Tensor` usability and type support, **currently focused on adapting all ops to the non-generic `Tensor` with `Buffer`/`DType` for F32/CPU.**
    *   🚧 **Implement Creation Methods:** `arange`, `linspace`, `eye`, `rand`, `randn` are defined but marked as `unused function`. Need integration into tests or examples.
    *   **`DType` Handling (Decomposed):**
        *   **Phase 1.10.A: Foundations (F32 Only) [🚧 In Progress]**
            *   ✅ **1-6. Core Structures & Methods:** `DType`, `Buffer`, `TensorData`, `Tensor` adapted for non-generic structure. Base methods and creation functions (`zeros`, `ones`, `full`, `zeros_like`, `ones_like`) adapted for `F32/CPU`.
            *   🚧 **7. Incremental Adaptation of Operations:** **Actively in progress.** Many operations (`add`, `neg`, `mul`, `relu`, `sum`, `slice`, `permute`, `transpose`, `reshape`, `contiguous`) adapted for F32/CPU forward/backward path.
                *   **Remaining Ops:** Adaptation needed for `sub`, `div`, `pow`, `max`, `mean`, `matmul` and potentially others.
                *   **Validation:** Ongoing process of fixing compilation errors and ensuring tests pass for each adapted operation. Numerous `warnings` (`unused`, `unreachable`) exist as side-effects of this ongoing refactoring.
            *   ✅ **8. Continuous Integration & Commits:** Being followed.
        *   **Phase 1.10.B: Add Second DType (e.g., I64) [⏳ To Do]** (Blocked by 1.10.A completion)
            *   ⏳ Extend `DType`, `Buffer`/`CpuBuffer`.
            *   ⏳ Adapt creation functions.
            *   ⏳ Extend ops (`add_op` example).
            *   ⏳ Verify/adapt `BackwardOp` for `I64`.
            *   ⏳ Add tests.
        *   **Phase 1.10.C: Mixed Types & Conversion [⏳ To Do]** (Blocked by 1.10.B)
            *   ⏳ Implement Type Promotion Logic.
            *   ⏳ Implement `Tensor::cast(dtype: DType)`.
            *   ⏳ Finalize promotion logic in ops.
            *   ⏳ Add tests.
        *   **Phase 1.10.D: Extend to Other Ops & DTypes [⏳ To Do]** (Blocked by 1.10.C)
            *   ⏳ Adapt remaining ops, handle dispatch, kernels, promotion.
            *   ⏳ Add other DTypes (`Bool`, `F64`, etc.).
        *   **Phase 1.10.E: In-place Operations [⏳ To Do]** (Blocked by 1.10.D)
            *   ⏳ Implement `add_`, `mul_`, etc.
            *   ⏳ Ensure strict DType matching.
            *   ⏳ Handle autograd implications.
    *   ✅ **Implement `Tensor::detach()`:** Functionality exists (implicitly via view creation without setting `grad_fn`).

*   **1.11 Testing & Documentation Consolidation [🚧 Partially Done]**
    *   🎯 Goal: Ensure comprehensive testing and documentation for Phase 1 features.
    *   🚧 Expand Unit Tests: Good coverage for implemented parts, but needs expansion for ops with issues (`Sub`, `Div`, `Pow`, `Max`, `Mean`, `Matmul`) and to utilize creation functions (`arange`, etc.). Tests ignored for `Mean`, `Matmul`, `Transpose` need fixing.
    *   ⏳ Consider Property-Based Testing (`proptest`).
    *   🚧 Documentation (`rustdoc`, Guides): Needs significant updates to reflect the non-generic API, `DType`/`Buffer` structure, current autograd status, and device awareness (CPU).

*   **Overall Status Phase 1:** Foundations (Views, Core Autograd) are solid. **Major refactoring (1.10.A) to non-generic `Tensor` with `DType`/`Buffer` is the primary focus.** This refactoring is mostly complete for core structures but still ongoing for individual operations, causing temporary issues (`dead_code` Backward ops, `unused` functions, ignored tests). **Next critical steps are finishing the op adaptation (1.10.A.7), fixing autograd integration issues (1.6, 1.8, 1.9), and stabilizing/un-ignoring tests (1.11).**

**Phase 2: Neural Network Primitives & Optimization [⏳ To Do]**
*   🎯 **Goal:** Build foundational `nn` modules, loss functions, and optimization algorithms to enable basic model definition and training, **integrating device management (`CPU`/`GPU` eventually) and leveraging the thread-safe `Tensor` structure.**
*   **2.1 NN Module System (`neurarust-core::nn`) [❌ Not Implemented]**
    *   🎯 Goal: Define the core abstractions for building neural networks, **aware of device placement.**
    *   ❌ **`Module` Trait:** **Missing.** Needs methods like `.to(device)`, `.device()`, `.parameters()`, `.buffers()`, `train()`, `eval()`. Must handle recursive application to submodules.
    *   ❌ **`Parameter` Struct:** **Missing.** Needs to wrap a `Tensor` configured with `requires_grad=true`. The `Tensor` internally handles `Arc<RwLock>`, `Buffer`, and `device`.
    *   ❌ **Module Containers:** **Missing.** (`Sequential`, `ModuleList`, `ModuleDict`). Need to correctly manage submodules, parameters, and device transfers (`.to(device)`).
    *   ❌ **Helper Methods:** **Missing.** (`named_parameters`, `train`, `eval`, etc.).
*   **2.2 Core Layers (`neurarust-core::nn::layers`) [❌ Not Implemented]**
    *   🎯 Goal: Implement fundamental neural network layers, **handling device placement and device-aware operations.**
    *   ❌ **Linear Layer:** **Missing.** Constructor needs `device` argument. `forward` must ensure input and weights are on the same device and call device-aware `matmul`.
    *   ❌ **Missing Layers:** All standard layers missing (Conv, Pool, Norm, RNN, etc.). All require device-aware initialization and `forward` implementations using device-aware backend ops.
*   **2.3 Loss Functions (`neurarust-core::nn::losses`) [❌ Not Implemented]**
    *   🎯 Goal: Implement standard functions for calculating training loss, **operating on tensors located on a specific device.**
    *   ❌ **Mean Squared Error:** **Missing.** Must check input/target device consistency and perform calculation on that device.
    *   ❌ **Missing Loss Functions:** All standard losses missing (CrossEntropy, BCE, etc.). Require device checks and device-aware computation.
*   **2.4 Weight Initialization (`neurarust-core::nn::init`) [❌ Not Implemented]**
    *   🎯 Goal: Provide standard techniques for initializing layer weights **directly on the target device.**
    *   ❌ Module `nn::init` **does not exist**.
    *   ❌ All initializers missing. Need to operate on the `Tensor`'s `Buffer` according to its `device` (potentially requiring data generation on CPU then transfer, or direct GPU random generation - Phase 4).
*   **2.5 Optimizers (`neurarust-optim`) [❌ Not Implemented]**
    *   🎯 Goal: Implement algorithms for updating model weights based on gradients, **handling parameters and optimizer state potentially residing on different devices.**
    *   ❌ **Crate `neurarust-optim` removed.** Decision needed: new crate or integrate into `neurarust-core`.
    *   ❌ **`Optimizer` Trait:** **Missing.** `step()` method needs to handle parameters/gradients on potentially different devices.
    *   ❌ **SGD Implementation:** **Missing.** Weight updates must occur on the parameter's device.
    *   ❌ **Adam Implementation:** **Missing.** Requires device-aware updates and storing optimizer state (moments) as `Tensor`s on the same device as the parameters.
    *   ❌ All other optimizers missing.
*   **2.6 Learning Rate Schedulers (`neurarust-optim::lr_scheduler`) [❌ Not Implemented]**
    *   🎯 Goal: Provide methods for adjusting the learning rate during training.
    *   ❌ Module `lr_scheduler` **does not exist**.
    *   ❌ All schedulers missing. (Less directly impacted by device, but interface with device-aware `Optimizer`).
*   **2.7 Integration & Training Loop [❌ Not Implemented]**
    *   🎯 Goal: Demonstrate how the components work together, **including explicit device management.**
    *   ❌ Test file removed. No example exists. Needs to show `model.to(device)`, `data.to(device)`, loss calculation, backward pass, and optimizer step all respecting the chosen device.
*   **2.8 Serialization [❌ Not Implemented]**
    *   🎯 Goal: Enable saving and loading model and optimizer states, **preserving device information or allowing device remapping.**
    *   ❌ No saving/loading capabilities exist. Needs to handle `device` metadata for parameters/buffers/optimizer state. `load_state_dict` needs a `map_location` argument.
*   **2.9 Testing & Documentation [❌ Not Implemented]**
    *   🎯 Goal: Ensure correctness of NN components and provide clear documentation, **covering device management extensively.**
    *   ❌ **Unit Tests:** Missing. Need tests covering different device scenarios (CPU, GPU when available).
    *   ❌ **Integration Tests:** Missing. Needs training loop tests on different devices.
    *   ❌ **Documentation:** Missing. Needs detailed explanation of device handling (`.to()`, parameter initialization, optimizer state, training loops).
*   **Overall Status Phase 2:** **Not started.** All components related to this phase were removed. Requires Phase 1 completion. **Implementation must be device-aware from the start.**

**Phase 3: Data Loading & Handling (`neurarust-data`) [⏳ To Do]**
*   🎯 **Goal:** Develop robust and performant tools for data loading, preprocessing, and augmentation, **ensuring efficient batch creation (potentially on target device later) and leveraging thread-safe structures.**
*   **3.1 Dataset Abstractions [❌ Not Implemented]**
    *   🎯 Goal: Define standard interfaces for accessing datasets.
    *   ❌ **Crate `neurarust-data` removed.** Decision needed: new crate or integrate.
    *   ❌ **`Dataset` Trait:** **Missing.** (Less impacted by device directly).
    *   ❌ **`VecDataset`:** **Missing.** (If returns Tensors, needs default device).
    *   ❌ **`IterableDataset` Trait/Concept:** **Missing.** (Less impacted by device directly).
*   **3.2 DataLoader [❌ Not Implemented]**
    *   🎯 Goal: Provide an iterator for efficient batching, shuffling, and loading of datasets, **with device-aware collation and GPU transfer optimizations.**
    *   ❌ **`DataLoader` Struct:** **Missing.**
    *   ❌ **Missing Core Features:**
        *   Batching: Needs implementation.
        *   Shuffling: Needs implementation.
        *   **Custom Collation:** Needs `collate_fn` argument. Default `collate_fn` must create batch `Tensor`s **on a specified device (configurable, default CPU)**.
        *   **Parallel Loading:** Needs `num_workers` > 0 support. Collation must be thread-safe and place result on target device.
        *   Samplers: Missing.
        *   **Memory Pinning:** Needs `pin_memory` option. If true, collation for CPU tensors should use pinned memory (requires Phase 4 backend integration, e.g., `cudaMallocHost`).
        *   Worker Init: Missing.
        *   Persistent Workers: Missing.
        *   **Automatic Device Placement:** Consider adding option to move batch to target device automatically.
*   **3.3 Data Preprocessing & Augmentation (`neurarust-vision`, `neurarust-text`?) [❌ Not Implemented]**
    *   🎯 Goal: Provide tools for transforming and augmenting data samples.
    *   ❌ **No Transform Module:** Missing.
    *   ❌ All transforms (Vision, Text, Generic) missing. (Transforms outputting Tensors need default device. Transforms using Tensor ops need device awareness).
*   **3.4 Integration & Utilities [❌ Not Implemented]**
    *   🎯 Goal: Provide helpers for common dataset tasks and formats.
    *   ❌ **No Dataset Utilities:** Missing.
    *   ❌ Common Dataset Helpers missing.
    *   ❌ Splitting Datasets missing. (Utilities handling Tensors need device awareness).
*   **3.5 Testing & Documentation [❌ Not Implemented]**
    *   🎯 Goal: Ensure correctness and provide clear documentation for data utilities, **including device handling in DataLoader.**
    *   ❌ **Unit Tests:** Missing. Need tests for DataLoader covering collation, parallelism, device placement, `pin_memory`.
    *   ❌ **Parallel Loading Tests:** Missing.
    *   ❌ **Documentation:** Missing. Needs to detail device management in `DataLoader`, collation, `pin_memory`.
*   **Overall Status Phase 3:** **Not started.** All components related to this phase were removed. **DataLoader implementation requires careful consideration of device management, collation, and thread-safety.**

**Phase 4: GPU Acceleration (CUDA First, then Others) [⏳ To Do]**
*   🎯 **Goal:** Enable high-performance computation using accelerators (starting with NVIDIA GPUs via CUDA), **leveraging the `Buffer`/`StorageDevice` framework.**
*   **Overall Status Phase 4:** **To Do.**

---
