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
    *   ⏳ **Basic Backward Infrastructure:** Definition of `BackwardOp` trait removed during cleanup. Reintroduction needed for Phase 1.
*   **0.4 Initial Testing [✅ Done]**
    *   ✅ Basic Unit Tests: Added tests covering `Tensor` creation, shape validation, basic arithmetic operations (forward pass), broadcasting utility functions, and new creation functions.
*   **0.5 Overall Status & Key Issues [✅ Done]**
    *   **Status:** Project structure and foundational `Tensor` struct are implemented with explicit stride support. Basic element-wise operations (`add`, `sub`, `mul`, `div`, `neg`) use strides for data access on CPU. Standalone creation functions added. Core error handling implemented. Codebase cleaned of Phase 1-3 elements, tests pass.
    *   ✅ **Critical Issue (Lack of strides): Resolved.** `TensorData` now stores strides, and basic operations use them for indexing.
    *   ✅ **Strides Stored for Views:** Strides are stored in `TensorData`, providing the prerequisite for views. ⏳ **View Implementation (Phase 1):** Operations like `reshape`, `slice`, `transpose` need to be implemented/re-implemented as true views (sharing data) in Phase 1.
    *   ✅ **Error Handling Improvement:** Addressed. Core functions like `Tensor::new`, `sum_axes` return `Result<T, NeuraRustError>`, handling common errors like shape mismatches or invalid indices gracefully.
    *   ✅ **Thread-Safety for Parallelism:** Replaced `Rc<RefCell<TensorData<T>>>` with `Arc<RwLock<TensorData<T>>>`. Internal data buffer uses `Arc<Buffer<T>>`. This provides the necessary thread-safety foundation for future parallel computation (e.g., CPU via Rayon, GPU acceleration - Phases 4/6), resolving the limitation noted previously.

**Phase 1: Views, Autograd & Expanded CPU Ops [⏳ To Do]**
*   🎯 **Goal:** Implement view semantics, a functional dynamic autograd system, and significantly expand CPU tensor operations & API, **ensuring compatibility with the new `Arc<RwLock>`, `Buffer`, and `StorageDevice` structures.**

*   **1.1 View Semantics & Core Shape Ops [⏳ To Do]**
    *   🎯 Goal: Implement non-copying views for shape manipulation, essential for performance and memory.
    *   ⏳ **Refine `TensorData::new_view`:** Ensure it's accessible (e.g., `pub(crate)`) and correctly takes `Arc<Buffer<T>>`, `device`, `offset`, `shape`, `strides` to create `TensorData` instances representing views.
    *   ⏳ **Implement `slice` Operation:**
        *   Define `slice_op(tensor: &Tensor<T>, /* slice args */) -> Result<Tensor<T>>`.
        *   Inside, acquire read lock on input `tensor.data`.
        *   Validate slice arguments against `shape`.
        *   Calculate new `shape` and new `offset` based on original offset, slice args, and `strides`.
        *   Create new `TensorData` using `new_view` with cloned `Arc<Buffer<T>>`, original `device`, new `offset`, new `shape`, and *original* `strides`.
        *   Wrap in `Tensor { data: Arc::new(RwLock::new(new_td)) }`.
        *   Implement user-facing `Tensor::slice(...)` method.
    *   ⏳ **Implement `transpose` Operation:**
        *   Define `transpose_op(tensor: &Tensor<T>, dim1: usize, dim2: usize) -> Result<Tensor<T>>`.
        *   Acquire read lock, validate `dim1`, `dim2` against rank.
        *   Calculate new `shape` (swap dims) and new `strides` (swap strides).
        *   Create view using `new_view` (cloned buffer, original device/offset, new shape/strides).
        *   Implement `Tensor::transpose(...)`.
    *   ⏳ **Implement `permute` Operation:**
        *   Define `permute_op(tensor: &Tensor<T>, dims: &[usize]) -> Result<Tensor<T>>`.
        *   Acquire read lock, validate `dims` is a valid permutation for the rank.
        *   Calculate new `shape` and new `strides` by reordering according to `dims`.
        *   Create view using `new_view` (cloned buffer, original device/offset, new shape/strides).
        *   Implement `Tensor::permute(...)`.
    *   ⏳ **Implement `reshape` / `view` Operation:**
        *   Define `reshape_op(tensor: &Tensor<T>, new_shape: Vec<usize>) -> Result<Tensor<T>>`.
        *   Acquire read lock, validate `new_shape` product matches old product.
        *   Call `is_contiguous()` on the tensor.
        *   If contiguous: Calculate new *contiguous* `strides` for `new_shape`. Create view using `new_view` (cloned buffer, original device/offset, `new_shape`, new strides).
        *   If non-contiguous: Check if a view is *still possible* (i.e., if specific stride manipulation can achieve the reshape). If yes, calculate those strides and create view. If not possible as a view, return `Err`. (User must call `.contiguous().reshape(...)` explicitly).
        *   Implement `Tensor::reshape(...)` and potentially `Tensor::view(...)` (alias or stricter view-only version).
    *   ⏳ **Implement `contiguous()` Method:**
        *   Implement `Tensor::contiguous(&self) -> Result<Tensor<T>>`.
        *   Call `is_contiguous()`. If true, return `self.clone()`.
        *   If false:
            *   Acquire read lock.
            *   Get buffer reference (`cpu_data()?` for now). Get `device`, `shape`, `strides`, `offset`.
            *   Allocate a *new*, *contiguous* buffer (`Vec<T>` for now) on the **same `device`**.
            *   Iterate multidimensionally over `shape`.
            *   For each index set, calculate offset in the *original* buffer using `guard.get_offset()`.
            *   Read value from original buffer (CPU read for now).
            *   Write value to the *new* buffer at the current linear index.
            *   Create and return a *new* `Tensor` using `Tensor::new()` with the new buffer and shape (which calculates contiguous strides).
    *   ⏳ **Implement `is_contiguous()` Check:**
        *   Implement `TensorData::is_contiguous(&self) -> bool`.
        *   Calculate expected contiguous strides for `self.shape`.
        *   Compare `self.strides` with expected strides (handle 0/1 dim sizes).
        *   Implement `Tensor::is_contiguous(&self)` calling the `TensorData` method via read lock.
    *   ⏳ **Testing:**
        *   Unit tests for each view op (`slice`, `transpose`, `permute`, `reshape`/`view`).
        *   Test edge cases (empty tensors, 0/1 sized dimensions).
        *   Verify views share the underlying buffer (`Arc::ptr_eq` on `borrow_data_buffer()`).
        *   Verify correct `shape`, `strides`, `offset`, `device` for views.
        *   Test `is_contiguous()` correctly identifies contiguous/non-contiguous tensors.
        *   Test `contiguous()` copies only when necessary and produces a contiguous tensor on the correct device.
        *   Test that data modifications via one view are reflected when accessing via another view (requires working `get`/`set` or similar).

*   **1.2 Basic Autograd Infrastructure [⏳ To Do]**
    *   🎯 Goal: Re-establish the foundational components for automatic differentiation, **handling thread-safety and device awareness.**
    *   ⏳ **Add `TensorData` Fields:**
        *   Add `requires_grad: bool` (default `false`).
        *   Add `grad: Option<Tensor<T>>` (holds the gradient tensor, must be on same device).
        *   Add `grad_fn: Option<Arc<dyn BackwardOp<T> + Send + Sync>>` (using `Arc` for shared ownership of backward node, requires trait bounds).
    *   ⏳ **Define `BackwardOp<T>` Trait:**
        *   `pub trait BackwardOp<T: 'static + ...>: Debug + Send + Sync { ... }` (add relevant bounds for `T`).
        *   `fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>, NeuraRustError>;` (Must handle device consistency).
        *   `fn inputs(&self) -> Vec<*const RwLock<TensorData<T>>>;` (Returns stable IDs of input tensors).
    *   ⏳ **Implement `Tensor` Autograd Accessors/Mutators:**
        *   `fn requires_grad(&self) -> bool;` (read lock).
        *   `fn set_requires_grad(&self, requires_grad: bool) -> Result<(), NeuraRustError>;` (write lock, handle potential graph modifications).
        *   `fn grad(&self) -> Option<Tensor<T>>;` (read lock, clones `Tensor` if `Some`).
        *   `fn acc_grad(&self, grad_to_add: Tensor<T>) -> Result<(), NeuraRustError>;` (write lock, handles `None`, checks device, performs accumulation via device-aware `add_op`).
        *   `fn grad_fn(&self) -> Option<Arc<dyn BackwardOp<T> + Send + Sync>>;` (read lock, clones `Arc`).
        *   `fn set_grad_fn(&self, grad_fn: Option<Arc<dyn BackwardOp<T> + Send + Sync>>) -> Result<(), NeuraRustError>;` (write lock).
    *   ⏳ **Implement Graph Traversal (`autograd::graph`):**
        *   Implement topological sort function (e.g., Kahn's or DFS based).
        *   Takes starting `Tensor` pointer/ID.
        *   Uses `*const RwLock<TensorData<T>>` as node identifier.
        *   Traverses graph via `grad_fn` and `inputs()`. Needs read locks.
        *   Handles cycles (returns `Err`).
        *   Returns ordered list of node IDs for backward pass.
    *   ⏳ **Implement `Tensor::backward()` Logic:**
        *   `fn backward(&self, gradient: Option<Tensor<T>>) -> Result<(), NeuraRustError>;`
        *   Check `self.requires_grad()`.
        *   Determine initial gradient (use provided `gradient`, or default to `1.0` scalar if `self` is scalar, error otherwise). Ensure initial grad is on `self.device()`.
        *   Perform topological sort from `self`.
        *   Prepare gradient accumulation map: `HashMap<NodeId, Tensor<T>>`, initialized with initial gradient for `self`.
        *   Iterate through sorted nodes:
            *   Retrieve accumulated grad for current node from map.
            *   Acquire read lock for current node `TensorData` to get `grad_fn`.
            *   If `grad_fn` is `Some(op)`:
                *   Call `op.backward(&accumulated_grad)` -> returns input grads.
                *   Get input node IDs from `op.inputs()`.
                *   For each input node ID and its calculated grad:
                    *   Retrieve the input `Tensor` corresponding to the ID (needs mechanism to map ID back or pass `Tensor`s).
                    *   If input `tensor.requires_grad()`:
                        *   Call `input_tensor.acc_grad(calculated_grad)` (handles locking, device checks, accumulation).
            *   Optionally clear `grad` field for non-leaf nodes after processing (memory optimization).
            *   Optionally clear `grad_fn` if `retain_graph=false` (default `false`).

*   **1.3 Core Op Implementation (Forward & Backward) [⏳ To Do]**
    *   🎯 Goal: Implement forward and backward passes for essential operations, **ensuring correct handling of locks, views/strides, buffers, and devices.**
    *   ⏳ **General Pattern:**
        *   Forward (`*_op`): Takes `&Tensor` inputs. Acquires read locks. Performs device checks. Accesses data via `buffer.cpu_data()?` (for now). Creates result `Tensor` on correct device. If autograd needed, create `*Backward` struct, wrap in `Arc`, set `grad_fn` on result.
        *   Backward (`*Backward` struct): Implements `BackwardOp`. Stores necessary context (input shapes, IDs, maybe data). `backward` method calculates input gradients using `grad_output` and context, respecting devices.
    *   ⏳ **Implement `div`, `neg`, `pow` (Forward & Backward):** Follow general pattern. Handle chain rule, device awareness. `DivBackward` needs care with division by zero.
    *   ⏳ **Implement/Adapt `reduce_gradient` Utility:** Takes gradient and target shape. Performs device-aware sum reduction along broadcasted axes.
    *   ⏳ **Implement View Backwards (`SliceBackward`, `TransposeBackward`, `ReshapeBackward`, `PermuteBackward`):**
        *   Implement backward logic (often involves scattering/indexing gradients). Must be device-aware.
    *   ⏳ **Implement `matmul` (Forward & Backward - 2D):** Implement CPU forward (loop or BLAS placeholder). Implement `MatMulBackward` (matrix math, device-aware).
    *   ⏳ **Implement `sum_axes`, `mean` (Forward & Backward):** Implement CPU forward. Implement `SumAxesBackward`, `MeanBackward` (often broadcasting grad_output, device-aware).
    *   ⏳ **Implement `relu` (Forward & Backward):** Implement CPU forward. Implement `ReluBackward` (conditional gradient propagation, device-aware).

*   **1.4 Tensor API & Data Type Expansion [⏳ To Do]**
    *   🎯 Goal: Enhance `Tensor` usability, add multi-type support, **considering device management.**
    *   ⏳ **Implement Creation Methods (`arange`, `linspace`, `eye`, `rand`, `randn`):** Functions take shape/range and optional `dtype: Option<DType>`, `device: Option<StorageDevice>` (defaults CPU). Dispatch based on `T` and generate data on correct device.
    *   ⏳ **`DType` Handling:** Decide approach: Keep `Tensor<T>` generic, use traits/macros for op dispatch based on `T`. Avoid `AnyBuffer` initially.
    *   ⏳ **Type Promotion:** Implement logic within ops or helper functions to cast inputs before operation based on promotion rules (e.g., `i32+f32 -> f32`).
    *   ⏳ **Implement Type Conversion:** `Tensor::cast<NewType>(&self) -> Result<Tensor<NewType>>`. Iterate data on original device, cast, create new Tensor on same device.
    *   ⏳ **Implement `detach()`:** `Tensor::detach(&self) -> Tensor<T>`. Clones `Arc<RwLock>`, gets write lock on new data, clears `requires_grad`, `grad_fn`, `grad`. Returns new `Tensor`.
    *   ⏳ **Implement In-place Ops (`add_`, `mul_`, ...):**
        *   `Tensor::add_(&self, other: &Tensor<T>) -> Result<(), NeuraRustError>;`
        *   Requires `&self` (mutable access is via interior mutability).
        *   Acquire `write` lock on `self.data`.
        *   Perform device checks (`self` vs `other`).
        *   Check safety: Return `Err` if `self.requires_grad()` (simplest rule initially).
        *   Perform operation directly on `self.data.data` buffer (CPU/GPU specific logic).

*   **1.5 Testing (Numerical Gradients) & Documentation [⏳ To Do]**
    *   🎯 Goal: Rigorously verify gradient calculations and document Phase 1 features, **including new structures and device concepts.**
    *   ⏳ **Implement Numerical Gradient Checking Utility:**
        *   Function takes `Fn(&Tensor<T>) -> Result<Tensor<T>>`, input `Tensor<T>`, `epsilon`.
        *   Needs to handle device (e.g., move inputs to CPU for calculation, run function, move result back if needed, or run entirely on one device if possible).
        *   Calculates finite difference approximation.
        *   Compares with analytical gradient obtained via `input.grad()` after `output.backward()`.
        *   Handle precision issues.
    *   ⏳ **Add Gradient Check Tests:** Create tests using the utility for *every* implemented `BackwardOp` on CPU.
    *   ⏳ **Expand Unit Tests:**
        *   Test all view ops, API methods (`cast`, `detach`, creation ops), forward/backward ops.
        *   Include tests for device arguments and checks.
        *   Test autograd graph construction, traversal, `backward` execution.
        *   Test error conditions extensively.
    *   ⏳ **Consider Property-Based Testing (`proptest`):** Define strategies for `Tensor` generation (shapes, CPU device) and test fundamental properties.
    *   ⏳ **Documentation (`rustdoc`, Guides):**
        *   Explain `Arc<RwLock>` pattern for `TensorData` and thread-safety implications.
        *   Explain `Buffer` abstraction and `StorageDevice` (initially CPU focus).
        *   Document view semantics (sharing, strides, offset, contiguity).
        *   Document basic autograd usage (`requires_grad`, `backward`, `grad`).
        *   Document all new public APIs with examples.
        *   Create initial user guide sections on core concepts.

*   **Overall Status Phase 1:** Ready to start. Focus is on implementing views first, then re-building autograd and ops **with thread-safety and device awareness (CPU initially) from the ground up.**

**Phase 2: Neural Network Primitives & Optimization [⏳ To Do]**
*   🎯 **Goal:** Build foundational `nn` modules, loss functions, and optimization algorithms to enable basic model definition and training, **integrating device management (`CPU`/`GPU`) and leveraging the thread-safe `Tensor` structure.**
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
*   🎯 **Goal:** Develop robust and performant tools for data loading, preprocessing, and augmentation, **ensuring efficient batch creation on the target device and leveraging thread-safe structures.**
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
*   🎯 **Goal:** Enable high-performance computation using accelerators, starting with NVIDIA GPUs via CUDA, **leveraging the `Buffer`/`StorageDevice` abstraction and thread-safe `Tensor` structure.**
*   **4.1 Backend Abstraction Layer [⏳]**
    *   ⏳ Define `StorageDevice` Enum/Struct more concretely if needed (e.g., `CPU`, `Cuda(gpu_id: u32)`). (Already `CPU`/`GPU`, needs refinement for multi-GPU).
    *   ⏳ Solidify `TensorData` structure containing `device: StorageDevice` and `data: Arc<Buffer<T>>` where `Buffer<T>` can be `Cpu(Arc<Vec<T>>)` or `Gpu(...)`.
    *   ⏳ Implement `Tensor::device()` method (✅ Already Done, may need refinement for specific GPU IDs).
    *   ⏳ Implement `Tensor::to(device: StorageDevice)` method: Creates a *new* `Tensor` by copying data to a new `Buffer` allocated on the target device (CPU <-> GPU, GPU <-> GPU). Handles `Arc<RwLock>` correctly.
    *   ⏳ Design lazy initialization for CUDA contexts/devices.
*   **4.2 CUDA Integration & Infrastructure [⏳]**
    *   ⏳ Select and integrate CUDA binding crate (e.g., `cuda-rs`, `cudarc`, `accel`).
    *   ⏳ Manage CUDA Contexts (creation, destruction, current context per thread).
    *   ⏳ Manage CUDA Streams (creation, synchronization - `cudaStreamSynchronize`, `cudaEventRecord`/`cudaStreamWaitEvent`) for asynchronous operations.
*   **4.3 GPU Memory Management [⏳]**
    *   ⏳ Implement GPU memory allocation/deallocation (`cudaMalloc`, `cudaFree`) within the **`Buffer::Gpu`** variant.
    *   ⏳ Implement asynchronous data transfers (Host <-> Device, Device <-> Device) using streams (`cudaMemcpyAsync`) as part of `Tensor::to()` and potentially other operations.
    *   ⏳ Implement Pinned Memory allocation (`cudaMallocHost`/`cudaHostRegister`) to back `Buffer::Cpu` when `pin_memory=true` (used by DataLoader Phase 3).
    *   ⏳ Explore GPU memory pooling/caching allocators for `Buffer::Gpu` to reduce overhead.
*   **4.4 CUDA Kernels / Library Integration [⏳]**
    *   ⏳ **Element-wise Ops:** Implement kernels/bindings operating on pointers extracted from `Buffer::Gpu` inputs, writing to a new `Buffer::Gpu` output.
    *   ⏳ **Reductions:** Implement kernels/bindings for reductions on `Buffer::Gpu` data.
    *   ⏳ **Matrix Multiplication:** Integrate cuBLAS (`cublas<t>gemm`), taking GPU buffer pointers as input.
    *   ⏳ **Convolutions:** Integrate cuDNN (`cudnnConvolution*`), configuring it with descriptors based on tensor metadata and GPU buffer pointers.
    *   ⏳ **Pooling:** Integrate cuDNN (`cudnnPooling*`).
    *   ⏳ **Activations:** Implement kernels or use cuDNN (`cudnnActivation*`).
    *   ⏳ **Indexing/Shape Ops:** Implement GPU kernels for `gather`, `scatter`, `slice`, `cat`, `stack` operating on `Buffer::Gpu`.
    *   ⏳ **Random Number Generation:** Integrate cuRAND (`curandGenerate*`) to create `Tensor`s with `Buffer::Gpu` directly.
*   **4.5 Framework Integration [⏳]**
    *   ⏳ **Ops Dispatch:** Modify all op implementations (`neurarust-core::ops`, e.g., `add_op`) to: check `tensor.device()`; if all inputs are `StorageDevice::GPU`, call the corresponding CUDA kernel/library; if `CPU`, call CPU logic; otherwise error or copy.
    *   ⏳ **Autograd:** Ensure `BackwardOp` implementations have GPU variants. `backward()` calls must dispatch correctly based on device. Gradient accumulation must happen on the correct device (`Buffer::Gpu` or `Buffer::Cpu`).
    *   ⏳ **NN Layers:** Modify layers (`neurarust-core::nn`) to:
        *   Accept `device` on construction for `Parameter` initialization (creating `Buffer::Gpu` or `Buffer::Cpu`).
        *   Implement `.to(device)` using `Tensor::to()` for parameters/buffers.
        *   Rely on Ops Dispatch within their `forward` methods.
    *   ⏳ **DataLoader:** Integrate `pin_memory` option using GPU backend's pinned memory allocation.
*   **4.6 Mixed-Precision Training (AMP) [⏳]**
    *   ⏳ Add `f16` / `bf16` support, likely primarily within `Buffer::Gpu`.
    *   ⏳ Implement `autocast` interacting with Ops Dispatch.
    *   ⏳ Implement `GradScaler` operating potentially on GPU loss `Tensor`.
    *   ⏳ Consider FP32 master weights pattern within optimizers.
*   **4.7 Multi-GPU Support (Single Node) [⏳]**
    *   ⏳ Refine `StorageDevice::Gpu(id)` for device selection.
    *   ⏳ Implement basic `DataParallel` utility using `Tensor::to(device)` for placement and GPU communication libraries (e.g., NCCL bindings) operating on `Buffer::Gpu` pointers.
*   **4.8 Other Backends (Exploratory/Future) [⏳]**
    *   ⏳ **ROCm (AMD):** Investigate HIP bindings, potential `Buffer::Hip` variant.
    *   ⏳ **Metal (Apple Silicon):** Investigate Metal bindings, potential `Buffer::Mtl` variant.
    *   ⏳ **WebGPU:** Explore `wgpu` crate, requires WGSL kernels, potential `Buffer::Wgpu` variant.
*   **4.9 Testing & Benchmarking [⏳]**
    *   ⏳ Unit tests for GPU memory (`Buffer::Gpu`), H2D/D2H/D2D copies (`Tensor::to`).
    *   ⏳ Unit tests for individual GPU kernels/library calls (comparing results with CPU ops via `.to(CPU)`).
    *   ⏳ Integration tests for Autograd and NN layers operating on GPU `Tensor`s.
    *   ⏳ Tests for Mixed-Precision and Multi-GPU.
    *   ⏳ Benchmarks comparing CPU vs GPU performance for ops and models.
*   **4.10 Build & CI [⏳]**
    *   ⏳ Implement conditional compilation (`cfg` features) for CUDA.
    *   ⏳ Set up CI with CUDA toolkit and GPU runners.
*   **4.11 Documentation [⏳]**
    *   ⏳ Document CUDA setup.
    *   ⏳ Document `StorageDevice`, `Buffer::Gpu` concepts, `Tensor::to()`, device handling in ops/layers/training.
    *   ⏳ Document AMP and Multi-GPU usage.

**Phase 5: Advanced Features, Ecosystem & Usability [⏳ To Do]**
*   🎯 **Goal:** Implement more complex NN architectures, improve interoperability, and enhance the overall developer experience, **fully integrating device management and leveraging the core abstractions.**
*   **5.1 Advanced NN Architectures & Modules [⏳]**
    *   🎯 Goal: Build advanced, reusable NN components aware of device placement.
    *   ⏳ **Transformer Components:** Implement device-aware `MultiheadAttention`, `TransformerEncoderLayer`, etc., using device-aware ops.
    *   ⏳ **Advanced RNN Features:** Implement device-aware bidirectionality, `PackedSequence` handling (needs careful buffer/device management).
    *   ⏳ **Normalization Variants:** Implement `SyncBatchNorm` (requires multi-GPU communication on device buffers - Phase 4).
    *   ⏳ **Other Potential Modules:** Ensure device compatibility for new activations, attention mechanisms etc.
*   **5.2 ONNX Export/Import [⏳]**
    *   🎯 Goal: Allow model exchange, **handling device differences.**
    *   ⏳ **Exporter:** Access parameters/buffers via locks from their respective devices (copying to CPU for serialization likely needed). Map device-aware NeuraRust ops to ONNX.
    *   ⏳ **Importer:** Parse ONNX, map ops. Load weights, placing them onto the user-specified `device` (via `map_location`) by creating appropriate `Tensor`s/`Buffer`s.
    *   ⏳ **Testing & Coverage:** Test exported models vs ONNX Runtime, considering device. Document supported ops and device implications.
*   **5.3 Python Bindings (PyO3) (`neurarust-py`) [⏳]**
    *   🎯 Goal: Enable seamless Python integration, **exposing device management APIs.**
    *   ⏳ **Crate Setup:** Create `neurarust-py` with `PyO3`.
    *   ⏳ **Tensor Bindings:** Expose `Tensor` including `.device`, `.to(device)`. Handle NumPy conversion carefully regarding devices (copying, errors?). Expose `StorageDevice` enum.
    *   ⏳ **Autograd Bindings:** Ensure compatibility with device-aware tensors.
    *   ⏳ **NN Module Bindings:** Expose device-aware `nn.Module` (with `.to()`), layers, losses.
    *   ⏳ **Optimizer & Scheduler Bindings:** Expose device-aware versions.
    *   ⏳ **DataLoader Bindings:** Expose device options (`pin_memory`, target device).
    *   ⏳ **Packaging & Distribution:** Configure `maturin`.
    *   ⏳ **Testing:** Python-side tests covering device interactions.
    *   ⏳ **Documentation:** Provide Python API docs with device handling examples.
*   **5.4 JIT Compilation / Graph Optimization (Exploratory) [⏳]**
    *   🎯 Goal: Explore static graph optimization, **considering device-specific opportunities.**
    *   ⏳ **Tracing/Scripting:** Capture device information.
    *   ⏳ **Intermediate Representation (IR):** Must encode device placement.
    *   ⏳ **Optimization Passes:** Operator Fusion, etc., must be device-aware.
    *   ⏳ **Code Generation:** Generate code dispatching to correct device backends (CPU/GPU).
*   **5.5 Visualization & Debugging [⏳]**
    *   🎯 Goal: Improve developer experience for understanding device-aware models.
    *   ⏳ **Training Hooks:** Access data via Buffers/locks (copy to CPU if needed).
    *   ⏳ **Computation Graph Visualization:** Indicate device placement.
    *   ⏳ **Debugging Tools:** Numerical gradient checking adapted (Phase 1). Improve device mismatch errors.
*   **5.6 Documentation, Examples & Tutorials [⏳]**
    *   🎯 Goal: Provide comprehensive resources covering device management thoroughly.
    *   ⏳ **Comprehensive User Guide:** Cover `StorageDevice`, `.to()`, device handling in all components, CPU vs GPU training loops.
    *   ⏳ **API Reference Documentation:** Ensure `rustdoc` clearly explains device parameters/returns/implications.
    *   ⏳ **Gallery of Examples:** Provide examples running on CPU and GPU.
    *   ⏳ **Tutorials:** Cover device management explicitly.
    *   ⏳ **Project Website:** Host all device-aware documentation.

**Phase 6: Deployment, Specialization & Maturity [⏳ To Do]**
*   🎯 **Goal:** Target specific deployment platforms, leverage Rust's unique strengths, implement distributed training, and foster a community, **all built upon the device-aware and thread-safe core.**
*   **6.1 Deployment Targets [⏳]**
    *   🎯 Goal: Enable efficient deployment across diverse environments using appropriate backends.
    *   ⏳ **WebAssembly (WASM):** Compile core targeting CPU backend (`Buffer::Cpu`). Exclude GPU code via `cfg`. Consider single/multi-thread implications for `Arc<RwLock>`. Explore WebGPU backend later (Phase 4.8).
    *   ⏳ **Native Binary Deployment:** Leverage optimized CPU backend, or GPU backend if built with CUDA `cfg`. Facilitate static linking.
    *   ⏳ **Edge/Embedded (ARM):** Target performant CPU backend (`Buffer::Cpu`), consider `Arc<RwLock>` overhead and NEON potential.
    *   ⏳ **Server-Side Inference:** Utilize CPU/GPU backend. Leverage thread-safe `Tensor`s for sharing models across request threads.
*   **6.2 Inference Optimizations [⏳]**
    *   🎯 Goal: Reduce model size and accelerate inference speed using device-aware techniques.
    *   ⏳ **Quantization:** Implement device-aware quantization (PTQ, QAT). Requires quantized kernels/ops for CPU/GPU. May need `Buffer` variants or metadata for quantized types.
    *   ⏳ **Pruning:** Implement device-aware pruning application. Explore sparse `Buffer` representations/kernels.
    *   ⏳ **Model Distillation:** Support depends on running device-aware models and loss calculations.
*   **6.3 Distributed Training (Multi-Node) [⏳]**
    *   🎯 Goal: Enable large-scale training using device-specific communication backends.
    *   ⏳ **Communication Backend Integration:** Integrate MPI/Gloo (CPU `Buffer`) or NCCL (GPU `Buffer`).
    *   ⏳ **Distributed Primitives:** Implement collectives operating on specific `Buffer` types via appropriate backends.
    *   ⏳ **`DistributedDataParallel` (DDP):** Implement using device placement (`Tensor::to`), device-aware autograd, and communication primitives on device buffers.
*   **6.4 Leveraging Rust's Strengths [⏳]**
    *   🎯 Goal: Fully exploit Rust's features enabled by the core architecture.
    *   ⏳ **Advanced Static Optimizations (Compile-Time):** Explore device-aware macro-based optimizations.
    *   ⏳ **Enhanced Safety & Verification:** Leverage thread-safety of `Arc<RwLock>`. Verify `unsafe` blocks in `Buffer` management / FFI.
    *   ⏳ **Fearless Concurrency:** Utilize `rayon` for CPU ops thanks to thread-safe `Tensor`. Explore task-based parallelism.
*   **6.5 Tooling & Infrastructure [⏳]**
    *   🎯 Goal: Provide robust development tools reflecting the multi-device nature.
    *   ⏳ **Robust Benchmarking Suite:** Benchmark ops/models across CPU/GPU, comparing `Buffer` backends.
    *   ⏳ **Extended Continuous Integration (CI):** Test builds/runs across platforms, devices (CPU/GPU runners), and feature flags (`cfg`).
*   **6.6 Community & Ecosystem [⏳]**
    *   🎯 Goal: Foster an active community knowledgeable about the device-aware architecture.
    *   ⏳ **Governance & Contribution:** Establish clear processes.
    *   ⏳ **Community Engagement:** Communicate clearly about device support/usage.
    *   ⏳ **Ecosystem Integration:** Ensure integrations correctly handle device-aware `Tensor`s.

*(This highly detailed roadmap reflects the long-term ambition. Priorities and specific implementation details will evolve based on progress, community feedback, and emerging needs.)*

---

## 📂 Project Structure

Here is an overview of the current NeuraRust project layout:

```
.
├── Cargo.lock                # Generated by Cargo, locks dependency versions.
├── Cargo.toml                # Main workspace manifest (defines members, dependencies, metadata).
├── CODE_OF_CONDUCT.md      # Code of conduct for contributors.
├── CONTRIBUTING.md         # Guide for contributing to the project.
├── Goals.md                  # This file: Project vision, goals, and roadmap.
├── LICENSE                   # Project license (e.g., MIT, Apache 2.0).
├── neurarust-core            # Core crate containing the heart of the framework.
│   ├── Cargo.toml            # Manifest for the neurarust-core crate.
│   ├── src                   # Source code for neurarust-core.
│   │   ├── autograd          # Module for automatic differentiation.
│   │   │   ├── graph.rs      # Functions for graph traversal (e.g., topological sort).
│   │   │   └── mod.rs        # Defines the autograd engine traits (`BackwardOp`) and logic.
│   │   ├── lib.rs            # Entry point for the neurarust-core library (re-exports modules).
│   │   ├── nn                # Module for neural network components.
│   │   │   ├── layers        # Submodule for different network layers.
│   │   │   │   ├── linear.rs # Implementation of the Linear (fully connected) layer.
│   │   │   │   └── mod.rs    # Declares the layers submodule and potentially re-exports layers.
│   │   │   ├── losses        # Submodule for loss functions.
│   │   │   │   ├── mod.rs    # Declares the losses submodule and re-exports loss functions.
│   │   │   │   └── mse.rs    # Implementation of Mean Squared Error (MSE) loss.
│   │   │   ├── mod.rs        # Declares the nn module and re-exports key components (`Module`, `Parameter`, layers, losses).
│   │   │   ├── module.rs     # Defines the base `Module` trait for all nn components.
│   │   │   └── parameter.rs  # Defines the `Parameter` struct for trainable model weights.
│   │   ├── ops               # Module containing tensor operations.
│   │   │   ├── activation    # Submodule for activation functions.
│   │   │   │   ├── mod.rs    # Declares the activation submodule.
│   │   │   │   └── relu.rs   # Implementation of the ReLU activation function.
│   │   │   ├── arithmetic    # Submodule for element-wise arithmetic operations.
│   │   │   │   ├── add.rs    # Implementation of addition.
│   │   │   │   ├── div.rs    # Implementation of division.
│   │   │   │   ├── mod.rs    # Declares the arithmetic submodule.
│   │   │   │   ├── mul.rs    # Implementation of multiplication.
│   │   │   │   ├── neg.rs    # Implementation of negation.
│   │   │   │   ├── pow.rs    # Implementation of exponentiation.
│   │   │   │   └── sub.rs    # Implementation of subtraction.
│   │   │   ├── indexing.rs   # Tensor indexing and slicing operations (`slice_op`).
│   │   │   ├── linalg        # Submodule for linear algebra operations.
│   │   │   │   ├── matmul.rs # Implementation of matrix multiplication.
│   │   │   │   ├── mod.rs    # Declares the linalg submodule.
│   │   │   │   └── transpose.rs# Implementation of tensor transposition.
│   │   │   ├── loss          # Submodule for loss-related operations (currently empty/declarative).
│   │   │   │   └── mod.rs    # Declares the loss operations submodule.
│   │   │   ├── math_elem     # Submodule for element-wise mathematical functions.
│   │   │   │   ├── mod.rs    # Declares the math_elem submodule.
│   │   │   │   └── sqrt.rs   # Implementation of square root.
│   │   │   ├── mod.rs        # Declares ops submodules and re-exports common operations.
│   │   │   ├── reduction     # Submodule for reduction operations (sum, mean, max...). 
│   │   │   │   ├── mod.rs    # Declares the reduction submodule.
│   │   │   │   └── sum.rs    # Implementation of sum reduction.
│   │   │   └── stack.rs      # Operation for stacking tensors along a new dimension.
│   │   ├── tensor            # Module defining the Tensor struct and its methods.
│   │   │   ├── mod.rs        # Defines the `Tensor` struct, its core methods (creation, shape, data access), and autograd integration (`backward`, `grad`).
│   │   │   └── utils.rs      # Tensor utility functions (e.g., broadcasting helpers like `broadcast_shapes`, `reduce_gradient`).
│   │   ├── tensor_data.rs  # Defines the internal `TensorData` struct holding data, shape, grad status, etc.
│   │   ├── utils             # Module for general utility functions within the crate.
│   │   │   └── testing.rs    # Utility functions specifically for testing purposes.
│   │   └── utils.rs        # Top-level utils file in src (potentially for re-exports or crate-wide utils - currently seems minimal).
│   └── tests                 # Directory containing integration tests for neurarust-core.
│       └── training_loop.rs# Integration test simulating a basic training loop (likely incomplete).
├── neurarust-data            # Crate for data loading and handling utilities.
│   ├── Cargo.toml            # Manifest for the neurarust-data crate.
│   └── src                   # Source code for neurarust-data.
│       ├── dataloader.rs   # Implementation of the `DataLoader` for batching and iterating over datasets.
│       ├── lib.rs            # Entry point for neurarust-data library (defines `Dataset` trait, re-exports components).
│       └── vec_dataset.rs    # Simple `Dataset` implementation backed by a `Vec`.
├── neurarust-optim           # Crate dedicated to optimization algorithms.
│   ├── Cargo.toml            # Manifest for the neurarust-optim crate.
│   └── src                   # Source code for neurarust-optim.
│       ├── adam.rs         # Implementation of the Adam optimizer.
│       ├── lib.rs            # Entry point for neurarust-optim library (defines `Optimizer` trait, re-exports optimizers).
│       └── sgd.rs          # Implementation of the SGD (Stochastic Gradient Descent) optimizer.
└── README.md                 # Main project README file.
```

---

*(Note: Descriptions are based on file names, module structure, and content analysis. They can be further refined as the project evolves.)*