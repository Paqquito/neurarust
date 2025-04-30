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
    *   ✅ **Stride-Aware Indexing (Partial):** Added `TensorData::get_offset` method. Forward passes for `matmul`, `add`, and `mul` have been updated to use `get_offset` for data access, making them compatible with strides.
    *   ✅ Basic Backward Infrastructure: Defined `BackwardOp` trait and implemented `AddBackward` structure with a basic `backward` method signature and `reduce_gradient` utility (to handle gradient reduction for broadcasted ops), laying groundwork for Phase 1 Autograd.
*   **0.4 Initial Testing [✅ Done]**
    *   ✅ Basic Unit Tests: Added tests covering `Tensor` creation, shape validation, basic arithmetic operations (forward pass), broadcasting utility functions, and new creation functions.
*   **0.5 Overall Status & Key Issues [✅ Done]**
    *   **Status:** Project structure and foundational `Tensor` struct are implemented with explicit stride support. Basic element-wise operations (`add`, `mul`) and `matmul` correctly use strides for data access on CPU. Initial autograd infrastructure (`BackwardOp` trait) exists. Standalone creation functions added. **Core error handling significantly improved.**
    *   ✅ **Critical Issue (Lack of strides): Resolved.** `TensorData` now stores strides, and basic operations use them for indexing.
    *   🚧 **View Semantics Imperfection:** While strides are *stored*, operations like `reshape`, `slice`, and `transpose` currently **still perform data copies**. Implementing true "views" (new Tensors sharing data but with modified shape/strides) using the existing strides is a key task **deferred to the beginning of Phase 1**.
    *   ✅ **Error Handling Improvement:** Addressed. Core functions like `Tensor::new`, `sum_axes`, `sqrt_op`, layer/loss forward passes now return `Result<T, NeuraRustError>`, handling common errors like shape mismatches or invalid indices gracefully. Further improvements will continue in subsequent phases.
    *   📝 *Note on Parallelism:* The `Rc<RefCell>` choice for data storage needs to be revisited before significant parallel computation (CPU or GPU) can be implemented effectively (Phase 4/6).

**Phase 1: Robust Autograd & Expanded CPU Ops [🚧 In Progress]**
*   🎯 **Goal:** Implement a functional and tested dynamic autograd system, significantly expand CPU tensor operations and Tensor API.
*   **1.1 Autograd Engine (`neurarust-core::autograd`) [🚧 Partially Implemented]**
    *   🎯 Goal: Build the core engine for automatic gradient calculation.
    *   ✅ **Computation Graph:** Implemented dynamic graph structure using `BackwardOp` trait stored optionally in `TensorData` (`grad_fn`). Nodes are created during op execution if any input requires grad.
    *   ✅ **Graph Traversal:** Implemented topological sort using Depth First Search (DFS) in `autograd::graph::build_topo` to determine the correct backward execution order.
    *   ✅ **`.backward()` Method:** Implemented on `Tensor`. Initiates autograd by building the graph topo sort, then iterating through it, calling `BackwardOp::backward` for each node.
    *   ✅ **Gradient Accumulation:** Implemented in `autograd::accumulate_gradient`. Uses an external `HashMap<*const TensorData<T>, Tensor<T>>` keyed by `TensorData` pointers during the backward pass to accumulate gradients. Final accumulated gradients are then copied into `TensorData.grad` (which is an `Option<Tensor<T>>`).
    *   ✅ **Handle Non-Scalar `backward()`:** Implemented. Accepts an optional `upstream_grad` tensor. Defaults to a scalar tensor of `1.0` if `.backward()` is called on a scalar tensor and no gradient is provided.
    *   ⏳ **`Tensor.detach()`:** **Missing.** Crucial method needed to create a new Tensor sharing the same data but detached from the computation graph (i.e., `requires_grad=false`, `grad_fn=None`). Necessary for freezing parts of a model or using tensors outside autograd.
    *   ⏳ **Higher-Order Gradients:** **Missing / Not Designed.** The current architecture (storing gradients in `Option<Tensor<T>>`, single backward pass) does not support calculating gradients of gradients (e.g., `grad(grad(y, x), x)`). Would require significant redesign (e.g., persistent graphs, different gradient storage).
    *   ⏳ **Gradient Hooks:** **Missing.** No mechanism exists to register functions (hooks) that can inspect or modify gradients as they are computed during the backward pass for specific tensors or modules. Useful for debugging or advanced techniques.
    *   📝 *Note on Efficiency:* The current `accumulate_gradient` approach involves cloning tensors for accumulation within the HashMap, which might be inefficient, especially for large gradients. Exploring in-place accumulation or alternative strategies could optimize memory usage and speed.
*   **1.2 Backward Pass Implementation [🚧 Partially Implemented]**
    *   🎯 Goal: Implement the gradient computation logic for core tensor operations.
    *   **Implemented Backward Ops:**
        *   ✅ Arithmetic: `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `NegBackward`.
        *   ✅ Power/Root: `PowBackward` (scalar exponent only), `SqrtBackward` (via `SqrtOp`.
        *   ✅ Activation: `ReluBackward`.
        *   ✅ Reductions: `SumAxesBackward` (handles sum over specified axes or all axes).
        *   ✅ Linear Algebra: `MatMulBackward` (currently 2D matrices only), `TransposeBackward` (currently last 2 dims only).
        *   ✅ Indexing/Shape: `SliceBackward` (correctly handles gradient scattering to the original tensor shape), `StackBackward` (correctly handles splitting gradients back to input tensors).
    *   **Missing Backward Ops:**
        *   ⏳ Basic Element-wise Math: Backward missing for `exp`, `log`, `sin`, `cos`, etc.
        *   ⏳ Activation Functions: Backward missing for Sigmoid, Tanh, LeakyReLU, GeLU, etc.
        *   ⏳ Reductions: Backward missing for Mean, Var/Std, Max, Min, ArgMax, ArgMin, Prod, etc.
        *   ⏳ Shape Ops: Backward logic **missing or incomplete** for `reshape`, `view`, `permute`, `squeeze`, `unsqueeze`, `cat`, `flatten`, `chunk`, `split`, `repeat`. **Crucially dependent on the implementation of strides and view semantics (Phase 1.4)** to correctly calculate gradients without unnecessary data copies.
        *   ⏳ Comparison Ops: Backward logic generally not applicable (output is boolean, gradients are typically zero), but ops themselves are missing.
        *   ⏳ Indexing Ops: Backward missing for `gather`, `scatter`, `masked_fill`, `index_select`, `take`.
    *   📝 *Note on Broadcasting:* Most implemented backward operations (`Add`, `Sub`, `Mul`, `Div`, `Pow`) correctly utilize the `reduce_gradient` utility to handle broadcasting by summing gradients over the broadcasted dimensions.
*   **1.3 Expand CPU Ops (Forward Pass) [🚧 Partially Implemented]**
    *   🎯 Goal: Increase the number of available tensor operations on the CPU.
    *   **Implemented Forward Ops (beyond Phase 0):**
        *   ✅ Linear Algebra: `matmul` (2D only), `transpose` (last 2 dims only).
        *   ✅ Indexing/Shape: `slice_op`, `stack_op`.
        *   ✅ Math/Activation: `sqrt`, `relu`.
    *   **Missing Forward Ops:**
        *   ⏳ Reductions: Mean, Max, Min, Var, Std, Prod, ArgMax, ArgMin.
        *   ⏳ Comparison Ops: `eq`, `ne`, `lt`, `le`, `gt`, `ge`.
        *   ⏳ Logical Ops: `where`, `logical_and`, `logical_or`, `logical_not`.
        *   ⏳ More Linear Algebra: `dot`, `outer`. Generalize `matmul` and `transpose` beyond 2D.
        *   ⏳ More Shape Ops: `reshape` (needs view support), `view` (needs strides), `permute` (needs strides), `flatten`, `chunk`, `split`, `repeat`, `unsqueeze`, `squeeze`, `cat`.
        *   ⏳ Indexing Ops: `gather`, `scatter`, `masked_fill`, `index_select`, `take`, advanced indexing.
        *   ⏳ Trigonometric Ops: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`.
        *   ⏳ Other Math: `exp`, `log`, `log10`, `abs`, `clamp`, `round`, `floor`, `ceil`, `sign`.
*   **1.4 Tensor API & Data Types [🚧 Very Incomplete]**
    *   🎯 Goal: Enhance the `Tensor` API for better usability and support multiple data types.
    *   🚧 More Creation Methods: Only `Tensor::scalar` added since Phase 0. Key methods like `arange`, `linspace`, `eye`, `rand`, `randn` are **missing**.
    *   ⏳ Explicit Data Type Support: **Missing.** Need to introduce a `DType` enum (e.g., `Float32`, `Float64`, `Int32`, `Int64`, `Bool`) and integrate it into `TensorData` and operations. Requires generic ops or dispatch mechanisms.
    *   ⏳ Type Promotion Logic: **Missing.** Rules needed for operations involving Tensors of different `DType`s (e.g., `f32 + i32`).
    *   ⏳ Type Conversion Methods: **Missing.** Need a `.to_dtype(dtype: DType)` method for explicit type casting.
    *   ❌ **View Semantics:** **Critically Missing.** Operations like `reshape`, `transpose`, `slice` currently **perform data copies**. Proper view implementation (returning a new `Tensor` sharing the underlying data buffer but with different shape/strides/offset) is **essential** for performance and memory efficiency. This is **blocked by the lack of stored strides** (see Phase 0.2).
    *   ❌ **Contiguous Tensors:** **Missing.** Cannot check if a tensor's memory is contiguous (`.is_contiguous()`) or create a contiguous copy (`.contiguous()`). Also **blocked by the lack of stored strides**.
    *   🚧 In-place Operations (`add_`, `sub_`, `mul_`, etc.): Basic versions exist but they **do not integrate with autograd** (don't track gradients) and likely lack broadcasting support. Need significant rework to function correctly within the framework.
    *   ⏳ Basic `Device` concept: **Missing.** No way to specify or query the device (CPU vs potential future GPU) where a tensor resides. Required for Phase 4.
    *   **Critical Dependency:** The implementation of **stored strides** (fixing the Phase 0 inconsistency) is the **absolute prerequisite** for tackling View Semantics and Contiguous Tensors, which are fundamental Tensor features.
*   **1.5 Testing & Documentation [🚧 Very Incomplete]**
    *   🎯 Goal: Ensure correctness through rigorous testing and provide clear documentation.
    *   🚧 Comprehensive Gradient Tests: Basic backward tests exist for the ops implemented in 1.2. However, they **lack systematic numerical gradient checking** using finite differences to verify the analytical gradients' correctness. **Improvement Needed.**
    *   ⏳ Property-Based Testing: **Missing.** Consider using crates like `proptest` to automatically generate diverse tensor inputs and test properties of operations and autograd.
    *   🚧 Expanded Unit Tests: Need significant expansion to cover all newly added forward ops (from 1.3) and Tensor API methods (from 1.4) as they are implemented.
    *   🚧 Documentation (`rustdoc`): Docstrings are present for some parts but are often **minimal or incomplete**. Need thorough documentation for all public modules, structs, functions, and methods, including usage examples.
*   **Overall Status Phase 1:** The foundational dynamic autograd graph construction and traversal mechanism works. Backward passes are implemented for a small core set of arithmetic and basic ops. However, the implementation is **severely hampered** by:
    1.  **Missing Strides:** Blocking efficient view/contiguous tensor support.
    2.  **Incomplete Op Coverage:** Many crucial forward and backward operations are missing.
    3.  **Lack of Core Tensor Features:** Missing multiple data types, device awareness, robust in-place ops, and creation methods.
    4.  **Insufficient Testing:** Gradient tests need numerical verification.
*   **Immediate Priorities:**
    1.  **Implement Views:** Refactor `reshape`, `transpose`, `slice` (and add `view`, `permute`, etc.) to create true views (no data copy) using the new strides. Implement their corresponding backward passes correctly considering views.
    1.  **Implement Stored Strides:** Add `strides: Vec<usize>` to `TensorData` and refactor indexing/memory access logic accordingly. This unblocks views.
    2.  **Implement Views:** Refactor `reshape`, `transpose`, `slice` (and add `view`, `permute`, etc.) to create true views (no data copy) using the new strides. Implement their corresponding backward passes correctly considering views.
    3.  **Numerical Gradient Checks:** Implement a testing utility for comparing analytical gradients with numerical approximations (finite differences) and apply it to all existing and new backward ops.
    4.  **Expand Backward Coverage:** Implement backward passes for essential missing ops, prioritizing reductions (Mean) and basic shape manipulations (once views are available).

**Phase 2: Neural Network Primitives & Optimization [🚧 Barely Started]**
*   🎯 **Goal:** Build foundational `nn` modules, loss functions, and optimization algorithms to enable basic model definition and training.
*   **2.1 NN Module System (`neurarust-core::nn`) [🚧 Partially Implemented]**
    *   🎯 Goal: Define the core abstractions for building neural networks.
    *   ✅ **`Module` Trait (`nn::module.rs`):** Defined with basic method signatures: `forward` (abstract, must be implemented by layers) and `parameters` (provides a default implementation to recursively collect parameters).
    *   ✅ **`Parameter` Struct (`nn::parameter.rs`):** Defined as a wrapper around `Tensor`. Automatically sets `requires_grad=true` on the underlying tensor. Used to identify trainable weights within modules.
    *   ⏳ **Module Containers:** **Missing.** Need standard containers for composing layers:
        *   ⏳ `Sequential`: A container executing modules sequentially.
        *   ⏳ `ModuleList`: A list-like container holding modules.
        *   ⏳ `ModuleDict`: A dictionary-like container holding modules with named keys.
    *   ⏳ **Helper Methods:** **Missing.** Essential methods for inspecting and managing modules:
        *   ⏳ `named_parameters()`: Iterate over parameters with their names.
        *   ⏳ `buffers()` / `named_buffers()`: Manage non-parameter tensors (e.g., running means in BatchNorm).
        *   ⏳ `children()` / `named_children()`: Iterate over direct sub-modules.
        *   ⏳ `modules()` / `named_modules()`: Iterate over all modules recursively.
        *   ⏳ `train()` / `eval()`: Set the module and its submodules to training/evaluation mode (affects layers like Dropout, BatchNorm).
        *   ⏳ `apply(fn)`: Apply a function recursively to all submodules.
        *   ⏳ `.to(device)`: Move module parameters and buffers to a specific device (depends on Phase 4).
*   **2.2 Core Layers (`neurarust-core::nn::layers`) [🚧 Very Incomplete]**
    *   🎯 Goal: Implement fundamental neural network layers.
    *   ✅ **Linear Layer (`nn::layers::linear.rs`):**
        *   ✅ Forward Pass: Implemented, handles input matrix multiplication with weights and optional bias addition.
        *   🚧 Backward Pass: **Relies entirely on the autograd of underlying `matmul` and `add` operations.** The `Linear` module itself doesn't define a specific backward pass. Gradients for weights (`.weight`) and bias (`.bias`) are accumulated by the autograd engine via the ops used in the forward pass. **Requires `matmul` and `add` backward to be correct and implemented (Phase 1).**
        *   🚧 Parameter Initialization: Weights and biases are **currently initialized to zeros** within `Linear::new`. Needs proper initialization schemes (see 2.4).
    *   **Missing Layers:** The vast majority of standard layers are missing:
        *   ⏳ Convolution Layers: `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose*`.
        *   ⏳ Pooling Layers: `MaxPool*d`, `AvgPool*d`, `AdaptiveMaxPool*d`, `AdaptiveAvgPool*d`.
        *   ⏳ Padding Layers: `ReflectionPad*d`, `ReplicationPad*d`, `ZeroPad*d`, `ConstantPad*d`.
        *   ⏳ Normalization Layers: `BatchNorm*d`, `LayerNorm`, `GroupNorm`, `InstanceNorm*d`.
        *   ⏳ Recurrent Layers: `RNN`, `LSTM`, `GRU` (and their `Cell` variants).
        *   ⏳ Transformer Layers: `Transformer`, `MultiheadAttention`, etc. (overlaps with Phase 5.1).
        *   ⏳ Embedding Layer: `Embedding`.
        *   ⏳ Dropout Layers: `Dropout`, `Dropout*d`.
        *   ⏳ Activation Layers: Standard activation functions wrapped as `nn.Module`s (e.g., `nn.ReLU`, `nn.Sigmoid`).
*   **2.3 Loss Functions (`neurarust-core::nn::losses`) [🚧 Partially Implemented]**
    *   🎯 Goal: Implement standard functions for calculating training loss.
    *   ✅ **Mean Squared Error (`nn::losses::mse.rs`):**
        *   ✅ Forward Pass: Implemented for `Reduction::Sum` and `Reduction::Mean`.
        *   ✅ Backward Pass: Implemented for `Reduction::Sum` and `Reduction::Mean`.
        *   ⏳ `Reduction::None`: **Missing** (returning per-element loss).
    *   **Missing Loss Functions:** Crucial loss functions for classification and other tasks are missing:
        *   ⏳ Cross-Entropy / Negative Log Likelihood: `NLLLoss`, `CrossEntropyLoss` (combines LogSoftmax and NLLLoss).
        *   ⏳ Binary Cross-Entropy: `BCELoss`, `BCEWithLogitsLoss` (more numerically stable).
        *   ⏳ Other Common Losses: `L1Loss`, `SmoothL1Loss`, `KLDivLoss`, `MarginRankingLoss`, `HingeEmbeddingLoss`, `CTCLoss`, etc.
*   **2.4 Weight Initialization (`neurarust-core::nn::init`) [❌ Not Implemented]**
    *   🎯 Goal: Provide standard techniques for initializing layer weights.
    *   ❌ Module `nn::init` **does not exist**. Initialization is currently hardcoded (e.g., zeros in `Linear::new`).
    *   ⏳ Basic Initializers: Need functions like `uniform_`, `normal_`, `constant_`, `ones_`, `zeros_`.
    *   ⏳ Standard Schemes: Need implementations of `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`, `kaiming_normal_`.
    *   ⏳ Utility Functions: Helpers like `calculate_gain` for activations.
*   **2.5 Optimizers (`neurarust-optim`) [🚧 Partially Implemented]**
    *   🎯 Goal: Implement algorithms for updating model weights based on gradients.
    *   ✅ **`Optimizer` Trait (`lib.rs`):** Defined with core methods `step()` (updates parameters) and `zero_grad()` (resets parameter gradients).
    *   ✅ **SGD Implementation (`sgd.rs`):**
        *   ✅ Basic update rule (`param = param - lr * grad`) implemented.
        *   ⏳ Momentum: **Missing**.
        *   ⏳ Nesterov Momentum: **Missing**.
        *   ⏳ Weight Decay (L2 regularization): **Missing**.
        *   ⏳ Dampening: **Missing**.
    *   ✅ **Adam Implementation (`adam.rs`):**
        *   ✅ Basic Adam algorithm implemented (uses first and second moment estimates).
        *   ✅ Handles learning rate, betas, epsilon.
        *   ⏳ Weight Decay (`AdamW` variant): **Missing**.
        *   ⏳ AMSGrad variant: **Missing**.
    *   **Missing Optimizers:** Many other common optimizers are needed:
        *   ⏳ AdaGrad
        *   ⏳ RMSprop
        *   ⏳ AdaDelta
        *   ⏳ Adamax
        *   ⏳ ASGD
        *   ⏳ Rprop
        *   ⏳ LBFGS (more complex, second-order optimizer)
    *   ⏳ **Per-parameter Options:** **Missing.** Need ability to specify different learning rates, weight decay, etc., for different parameter groups within a model.
    *   ⏳ **Gradient Clipping:** **Missing.** Utilities needed for clipping gradients by norm or value before the optimizer step.
*   **2.6 Learning Rate Schedulers (`neurarust-optim::lr_scheduler`) [❌ Not Implemented]**
    *   🎯 Goal: Provide methods for adjusting the learning rate during training.
    *   ❌ Module `lr_scheduler` **does not exist** within `neurarust-optim`.
    *   ⏳ Base Class (`LRScheduler` trait/struct): **Missing**.
    *   ⏳ Common Schedulers: Need implementations like `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, etc.
*   **2.7 Integration & Training Loop [❌ Not Implemented]**
    *   🎯 Goal: Demonstrate how the components (data, model, loss, optimizer) work together.
    *   ❌ Test file `tests/training_loop.rs` exists but is **empty**. No working example of a training loop is implemented.
    *   ⏳ Clear Example: Need a basic integration test or example demonstrating a minimal training loop: data loading, forward pass, loss calculation (`loss.backward()`), gradient zeroing (`optimizer.zero_grad()`), and parameter update (`optimizer.step()`).
*   **2.8 Serialization [❌ Not Implemented]**
    *   🎯 Goal: Enable saving and loading model and optimizer states.
    *   ❌ No saving/loading capabilities (`state_dict`, `load_state_dict`) found.
    *   ⏳ **Model State:** Need mechanisms to serialize and deserialize model parameters and persistent buffers (`state_dict`). Consider using `serde` with formats like `safetensors` (preferred) or msgpack/bincode.
    *   ⏳ **Optimizer State:** Need to save/load optimizer internal state (e.g., moments in Adam, momentum buffers in SGD) to properly resume training.
*   **2.9 Testing & Documentation [🚧 Very Incomplete]**
    *   🎯 Goal: Ensure correctness of NN components and provide clear documentation.
    *   🚧 **Unit Tests:** Basic tests exist for `Linear` forward pass, `MSELoss` forward/backward, and basic `SGD`/`Adam` functionality. **Coverage is extremely low** relative to the scope of this phase. Need tests for all implemented/missing layers, losses, initializers, optimizer features, etc.
    *   ⏳ **Integration Tests:** **Missing.** Requires a working training loop (2.7) to test the interaction between components.
    *   🚧 **Documentation:** Minimal docstrings exist for the few implemented parts. **Needs significant expansion** to cover module APIs, layer arguments, loss function behavior, optimizer parameters, etc.
*   **Overall Status Phase 2:** Foundational traits and structs (`Module`, `Parameter`, `Optimizer`) and a few minimal implementations (`Linear`, `MSELoss`, `SGD`, `Adam`) exist. However, this phase is **largely unimplemented**. The vast majority of essential building blocks for constructing and training even simple neural networks (layers, losses, initializers, optimizer features, schedulers, serialization, training loop examples) are **missing**. Progress is also dependent on stabilizing Phase 1 (especially autograd correctness for ops used by layers).
*   **Priorities (Post-Phase 1 Stabilization, especially strides/views and robust backward ops):**
    1.  **Implement `nn::init`:** Provide basic weight initialization schemes and integrate them into `Linear` (and future layers).
    2.  **Implement Core Layers:** Prioritize `Conv2d` (Forward & Backward) and `MaxPool2d` (Forward & Backward).
    3.  **Implement Core Loss:** Prioritize `CrossEntropyLoss` (Forward & Backward).
    4.  **Enhance Optimizers:** Add Momentum and Weight Decay to `SGD`; implement `AdamW`.
    5.  **Implement `Sequential` Container:** Allow basic model composition.
    6.  **Create Basic Training Loop Example:** Demonstrate end-to-end usage in `tests` or examples folder.
    7.  **Implement Basic Serialization:** Add `state_dict`/`load_state_dict` for models using `safetensors`.

**Phase 3: Data Loading & Handling (`neurarust-data`) [🚧 Barely Started]**
*   🎯 **Goal:** Develop robust and performant tools for data loading, preprocessing, and augmentation.
*   **3.1 Dataset Abstractions [🚧 Partially Implemented]**
    *   🎯 Goal: Define standard interfaces for accessing datasets.
    *   ✅ **`Dataset` Trait (`lib.rs`):** Defined, representing a map-style dataset where samples are retrieved by index (`get(index)`). Similar to PyTorch's `Dataset`.
    *   ✅ **`VecDataset` (`vec_dataset.rs`):** A simple implementation of `Dataset` using an in-memory `Vec` to store samples.
    *   ⏳ **`IterableDataset` Trait/Concept:** **Missing.** Need an abstraction for datasets where data is loaded sequentially (streamed), potentially from files or databases, without requiring random access by index. Crucial for large datasets that don't fit in memory.
*   **3.2 DataLoader [🚧 Partially Implemented]**
    *   🎯 Goal: Provide an iterator for efficient batching, shuffling, and loading of datasets.
    *   ✅ **Basic `DataLoader` Struct (`dataloader.rs`):** Implemented as an iterator that yields batches of data.
    *   ✅ **Batching:** Implemented. Uses a default `collate_batch` function that stacks tensors from individual samples retrieved via `Dataset::get` using `Tensor::stack` (requires `Tensor::stack` to be implemented correctly - Phase 1).
    *   ✅ **Shuffling:** Implemented. Shuffles the dataset indices at the beginning of each epoch if `shuffle=true`.
    *   🚧 **Current Implementation:** **Single-threaded only.** Data fetching (`Dataset::get`) and collation happen sequentially in the main thread, which can become a bottleneck.
    *   **Missing Core Features:**
        *   ⏳ **Custom Collation Functions (`collate_fn`):** **Missing.** Need the ability to provide a custom function to merge a list of samples into a batch, allowing for handling of non-tensor data or complex batching strategies.
        *   ⏳ **Parallel Loading (Multithreading/Multiprocessing):** **Missing.** Crucial for performance. Need to implement multi-worker loading (e.g., using `rayon` or dedicated threads/processes) to fetch data in parallel, overlapping data loading with model computation.
        *   ⏳ **Samplers:** **Missing.** Currently uses a simple shuffled or sequential index vector. Need a flexible `Sampler` abstraction and implementations:
            *   ⏳ `Sampler` Trait: Define the interface for samplers.
            *   ⏳ `SequentialSampler`: Samples elements sequentially.
            *   ⏳ `RandomSampler`: Samples elements randomly (with/without replacement).
            *   ⏳ `BatchSampler`: Wraps another sampler to yield mini-batches of indices.
            *   ⏳ `SubsetRandomSampler`: Samples randomly from a subset of indices.
        *   ⏳ **Distributed Samplers:** **Missing.** Need samplers aware of distributed training setups (Phase 6.3) to ensure each process gets a unique subset of data (e.g., `DistributedSampler`).
        *   ⏳ **Memory Pinning (`pin_memory`):** **Missing.** Option needed to load data into pinned host memory for faster asynchronous CPU-to-GPU transfers (requires Phase 4 CUDA support).
        *   ⏳ **Worker Initialization (`worker_init_fn`):** **Missing.** Option to provide a function to configure each worker process/thread (e.g., setting random seeds).
        *   ⏳ **Persistent Workers (`persistent_workers`):** **Missing.** Option to keep worker processes alive between epochs to avoid startup overhead.
*   **3.3 Data Preprocessing & Augmentation (`neurarust-vision`, `neurarust-text`?) [❌ Not Implemented]**
    *   🎯 Goal: Provide tools for transforming and augmenting data samples.
    *   ❌ **No Transform Module:** No dedicated module or functionality for data transforms exists.
    *   ⏳ **Common Transforms Framework:** **Missing.** Need a composable transform pipeline:
        *   ⏳ Define a `Transform` trait or similar.
        *   ⏳ Implement `transforms.Compose` to chain multiple transforms together.
    *   ⏳ **Vision Transforms:** **Missing.** Need common image processing transforms (potentially in a separate `neurarust-vision` crate):
        *   ⏳ `ToTensor`: Convert PIL Image/ndarray to Tensor.
        *   ⏳ `Normalize`: Normalize tensor image with mean/std.
        *   ⏳ `Resize`, `CenterCrop`, `RandomResizedCrop`.
        *   ⏳ `RandomHorizontalFlip`, `RandomVerticalFlip`.
        *   ⏳ Color Jitter, Rotation, Affine transforms, etc.
    *   ⏳ **Text Transforms:** **Missing.** Need common text processing transforms (potentially in a separate `neurarust-text` crate):
        *   ⏳ Tokenization (using existing Rust tokenizers like `tokenizers`).
        *   ⏳ Vocabulary mapping (building/using vocabs).
        *   ⏳ Padding/Truncation.
        *   ⏳ Numericalization.
    *   ⏳ **Generic Transforms:** **Missing.** Utility like `Lambda` transform to apply arbitrary functions.
*   **3.4 Integration & Utilities [❌ Not Implemented]**
    *   🎯 Goal: Provide helpers for common dataset tasks and formats.
    *   ❌ **No Dataset Utilities:** No helpers found for reading standard dataset formats.
    *   ⏳ **Common Dataset Helpers:** **Missing.** Need utilities like:
        *   ⏳ `ImageFolder`-like dataset (reads images from nested directories).
        *   ⏳ CSV loading dataset.
        *   ⏳ Helpers for downloading/managing standard datasets (MNIST, CIFAR, etc.).
    *   ⏳ **Splitting Datasets:** **Missing.** Need a function like `random_split` to split a dataset into non-overlapping subsets (e.g., train/validation split).
*   **3.5 Testing & Documentation [🚧 Partially Implemented]**
    *   🎯 Goal: Ensure correctness and provide clear documentation for data utilities.
    *   🚧 **Unit Tests:** Basic tests exist for `VecDataset` and the single-threaded `DataLoader` (iteration, batching, shuffle). **Coverage is very low** and needs significant expansion to cover samplers, parallel loading, collation, transforms, etc., as they are implemented.
    *   ⏳ **Parallel Loading Tests:** **Missing.** Specific tests needed to verify correctness and performance of multi-worker loading.
    *   🚧 **Documentation:** Basic docstrings exist for `Dataset`, `VecDataset`, `DataLoader`. **Needs significant expansion** to cover the API design, usage patterns, available samplers, transforms, and utilities.
*   **Overall Status Phase 3:** Minimal foundational pieces for basic in-memory, single-threaded data loading exist (`Dataset` trait, `VecDataset`, basic `DataLoader`). However, the vast majority of features required for practical and performant deep learning data pipelines are **entirely missing**. This includes support for large datasets (`IterableDataset`), efficient loading (`parallel loading`, `samplers`, `pin_memory`), and essential data preparation (`transforms`, `utilities`).
*   **Priorities (Can proceed somewhat in parallel with Phase 2, but performance features depend on Phase 1 stability and threading choices):**
    1.  **Implement Sampler Trait & Basic Samplers:** Add `SequentialSampler` and `RandomSampler`, integrate into `DataLoader`.
    2.  **Implement Basic Transforms Framework:** Define `Transform` trait, `Compose`, and implement a few key vision transforms (`ToTensor`, `Normalize`, `Resize`).
    3.  **Explore Parallel Loading:** Investigate and implement multi-worker data loading using threads or processes (e.g., using `rayon` or `crossbeam-channel`).
    4.  **Implement `random_split` Utility:** Add function for dataset splitting.
    5.  **Introduce `IterableDataset` Concept:** Design and implement the trait for streaming datasets.

**Phase 4: GPU Acceleration (CUDA First, then Others) [⏳ To Do]**
*   🎯 **Goal:** Enable high-performance computation using accelerators, starting with NVIDIA GPUs via CUDA, including memory management, kernel execution, and framework integration.
*   **4.1 Backend Abstraction Layer [⏳]**
    *   ⏳ Define `Device` Enum/Struct (`CPU`, `Cuda(gpu_id: u32)`).
    *   ⏳ Integrate `Device` within `TensorData` (or a wrapper) to track tensor location.
    *   ⏳ Implement `Tensor::device()` method to query location.
    *   ⏳ Implement `Tensor::to(device: Device)` method for moving tensors between devices (CPU <-> GPU, GPU <-> GPU).
    *   ⏳ Design lazy initialization for CUDA contexts/devices.
*   **4.2 CUDA Integration & Infrastructure [⏳]**
    *   ⏳ Select and integrate CUDA binding crate (e.g., `cuda-rs`, `cudarc`, `accel`).
    *   ⏳ Manage CUDA Contexts (creation, destruction, current context per thread).
    *   ⏳ Manage CUDA Streams (creation, synchronization - `cudaStreamSynchronize`, `cudaEventRecord`/`cudaStreamWaitEvent`) for asynchronous operations.
*   **4.3 GPU Memory Management [⏳]**
    *   ⏳ Implement GPU memory allocation/deallocation (`cudaMalloc`, `cudaFree`).
    *   ⏳ Implement asynchronous data transfers (Host <-> Device, Device <-> Device) using streams (`cudaMemcpyAsync`).
    *   ⏳ Implement Pinned Memory (Host memory allocation via `cudaMallocHost`/`cudaHostRegister`) for faster asynchronous H2D/D2H transfers.
    *   ⏳ Explore GPU memory pooling/caching allocators to reduce overhead of `cudaMalloc`/`cudaFree`.
*   **4.4 CUDA Kernels / Library Integration [⏳]**
    *   ⏳ **Element-wise Ops:** Implement kernels (custom via `rust-cuda`/PTX or using libraries like `thrust` via bindings) for common arithmetic, logical, and math functions.
    *   ⏳ **Reductions:** Implement optimized reduction kernels (sum, mean, max, min, etc.) potentially using libraries or custom implementations (e.g., parallel reduction patterns).
    *   ⏳ **Matrix Multiplication:** Integrate cuBLAS (`cublas<t>gemm`) for high-performance matrix multiplication.
    *   ⏳ **Convolutions:** Integrate cuDNN (`cudnnConvolutionForward`, `cudnnConvolutionBackward*`) for high-performance convolutions (requires setting up descriptors for tensors, filters, convolution, algorithms).
    *   ⏳ **Pooling:** Integrate cuDNN (`cudnnPoolingForward`, `cudnnPoolingBackward`).
    *   ⏳ **Activations:** Implement kernels or use cuDNN (`cudnnActivationForward`, `cudnnActivationBackward`) for ReLU, Sigmoid, Tanh, etc.
    *   ⏳ **Indexing/Shape Ops:** Implement kernels for `gather`, `scatter`, complex slicing, `cat`, `stack` on GPU.
    *   ⏳ **Random Number Generation:** Integrate cuRAND (`curandGenerate*`) for creating random tensors directly on GPU.
*   **4.5 Framework Integration [⏳]**
    *   ⏳ **Ops Dispatch:** Modify CPU op implementations (`neurarust-core::ops`) to check tensor devices and dispatch to appropriate CUDA kernels/libraries if tensors are on GPU (panic or fallback if mixing CPU/GPU tensors in one op without explicit copy).
    *   ⏳ **Autograd:** Ensure `BackwardOp` implementations correctly handle GPU tensors. Gradients should be computed on the same device as the output tensor. `accumulate_gradient` needs to handle GPU tensors.
    *   ⏳ **NN Layers:** Modify layers (`neurarust-core::nn`) to:
        *   Initialize parameters on a specified device (`Linear::new(..., device)`).
        *   Implement a `.to(device)` method for modules to move all parameters/buffers.
        *   Ensure internal operations within `forward` respect tensor devices.
    *   ⏳ **DataLoader:** Integrate `pin_memory` option for faster transfer to GPU. Potentially add option to move batch directly to target device.
*   **4.6 Mixed-Precision Training (AMP) [⏳]**
    *   ⏳ Add `f16` / `bf16` support to `Tensor` (GPU only initially).
    *   ⏳ Implement `autocast` context manager/attribute to automatically select appropriate kernel versions or cast inputs/outputs for specific ops (e.g., `MatMul`, `Conv` in FP16, reductions/losses in FP32).
    *   ⏳ Implement `GradScaler` utility to manage loss scaling and prevent gradient underflow during backward pass with FP16 gradients.
    *   ⏳ Consider FP32 master weights pattern within optimizers for stability.
*   **4.7 Multi-GPU Support (Single Node) [⏳]**
    *   ⏳ Implement basic `DataParallel` utility: Replicate model on multiple GPUs, split input batch, forward on each GPU, gather outputs, scatter loss, backward on each GPU, average gradients.
    *   ⏳ Handle device affinity and inter-GPU communication (e.g., for gradient averaging using NCCL or basic `cudaMemcpy`).
*   **4.8 Other Backends (Exploratory/Future) [⏳]**
    *   ⏳ **ROCm (AMD):** Investigate HIP bindings (e.g., `hip-rs`) and library availability (rocBLAS, MIOpen). Assess porting effort.
    *   ⏳ **Metal (Apple Silicon):** Investigate Metal bindings and potential for Metal Performance Shaders (MPS).
    *   ⏳ **WebGPU:** Explore `wgpu` crate for backend. Requires significant effort to write compute shaders (WGSL) for all ops.
*   **4.9 Testing & Benchmarking [⏳]**
    *   ⏳ Unit tests for GPU memory allocation, H2D/D2H/D2D copies (sync & async).
    *   ⏳ Unit tests for individual GPU kernels/library calls (compare results against CPU versions).
    *   ⏳ Integration tests for Autograd and NN layers operating on GPU tensors.
    *   ⏳ Tests for Mixed-Precision training correctness.
    *   ⏳ Multi-GPU `DataParallel` tests.
    *   ⏳ Benchmarks (`criterion.rs` or custom): Compare CPU vs GPU op performance. Compare NeuraRust GPU vs PyTorch GPU performance for key models.
*   **4.10 Build & CI [⏳]**
    *   ⏳ Implement conditional compilation (`cfg` features) to enable/disable CUDA support during build.
    *   ⏳ Set up CI environment with CUDA toolkit and GPU runners (e.g., GitHub Actions with GPU instances) to run GPU-specific tests.
*   **4.11 Documentation [⏳]**
    *   ⏳ Document how to install CUDA toolkit and set up the environment.
    *   ⏳ Document `Device` usage, `.to()` method, and device handling concepts.
    *   ⏳ Document Mixed-Precision usage (`autocast`, `GradScaler`).
    *   ⏳ Document Multi-GPU usage.

**Phase 5: Advanced Features, Ecosystem & Usability [⏳ To Do]**
*   🎯 **Goal:** Implement more complex NN architectures, improve interoperability, and enhance the overall developer experience.
*   **5.1 Advanced NN Architectures & Modules [⏳]**
    *   ⏳ **Transformer Components:** Implement core building blocks:
        *   ⏳ `MultiheadAttention` layer.
        *   ⏳ `TransformerEncoderLayer` and `TransformerDecoderLayer`.
        *   ⏳ Standard Positional Encodings (sinusoidal, learned).
    *   ⏳ **Advanced RNN Features:**
        *   ⏳ Bidirectionality support for RNN layers.
        *   ⏳ Support for `PackedSequence` (handling variable length sequences efficiently).
        *   ⏳ Explore Peephole connections for LSTMs.
    *   ⏳ **Normalization Variants:**
        *   ⏳ Implement `SyncBatchNorm` for synchronized batch normalization across multiple GPUs (requires Phase 4.7).
    *   ⏳ **Other Potential Modules:**
        *   ⏳ Explore more activation functions (GeLU, SiLU).
        *   ⏳ Investigate other attention mechanisms (e.g., Linformer, Performer - exploratory).
*   **5.2 ONNX Export/Import [⏳]**
    *   🎯 Goal: Allow model exchange with other frameworks (PyTorch, TensorFlow).
    *   ⏳ **Exporter:**
        *   ⏳ Traverse NeuraRust computation graph (potentially traced via JIT - see 5.4).
        *   ⏳ Map NeuraRust ops (`neurarust-core::ops`) to standard ONNX opset (define target opset version, e.g., 13+).
        *   ⏳ Handle model state dictionary (`parameters`, `buffers`) serialization within the ONNX file.
        *   ⏳ Perform basic graph optimizations during export (e.g., constant folding).
    *   ⏳ **Importer (More Complex):**
        *   ⏳ Parse ONNX model file (`.onnx` format).
        *   ⏳ Map ONNX opset back to NeuraRust ops. Handle unsupported ops gracefully (error or allow custom registration).
        *   ⏳ Load weights from ONNX into NeuraRust `Parameter`s.
        *   ⏳ Handle potential layout differences (e.g., NCHW vs NHWC).
    *   ⏳ **Testing & Coverage:**
        *   ⏳ Implement tests comparing NeuraRust execution vs ONNX Runtime execution for exported models.
        *   ⏳ Document supported and unsupported ONNX ops.
*   **5.3 Python Bindings (PyO3) (`neurarust-py`) [⏳]**
    *   🎯 Goal: Enable seamless integration with the Python ecosystem for research and prototyping.
    *   ⏳ **Crate Setup:** Create `neurarust-py` crate using `PyO3`.
    *   ⏳ **Tensor Bindings:**
        *   ⏳ Expose `Tensor` class to Python.
        *   ⏳ Implement seamless NumPy conversion (`__array__` protocol, `from_numpy`), aiming for zero-copy where possible (requires careful memory management).
        *   ⏳ Expose Tensor methods (creation, manipulation, math ops).
    *   ⏳ **Autograd Bindings:** Expose `.backward()`, `.grad`, `requires_grad` functionality.
    *   ⏳ **NN Module Bindings:**
        *   ⏳ Expose `nn.Module` base class allowing subclassing from Python.
        *   ⏳ Expose core layers (`Linear`, `Conv2d`, etc.) and containers (`Sequential`).
        *   ⏳ Expose loss functions.
        *   ⏳ Expose parameter/buffer access.
    *   ⏳ **Optimizer & Scheduler Bindings:** Expose `Optimizer` classes and LR schedulers.
    *   ⏳ **DataLoader Bindings:** Expose `Dataset` and `DataLoader` (potentially allow Python datasets).
    *   ⏳ **Utilities:** Expose device handling (`.to(device)`), context managers.
    *   ⏳ **Packaging & Distribution:** Configure `maturin` for building wheels, publish to PyPI.
    *   ⏳ **Testing:** Implement comprehensive Python-side tests for all bindings.
    *   ⏳ **Documentation:** Provide Python API documentation and usage examples.
*   **5.4 JIT Compilation / Graph Optimization (Exploratory) [⏳]**
    *   🎯 Goal: Explore potential performance gains via static graph optimization and compilation.
    *   ⏳ **Tracing/Scripting:**
        *   ⏳ Develop a tracing mechanism (e.g., symbolic execution, proxy objects) to capture the computation graph during a forward pass.
        *   ⏳ Alternatively, explore a scripting approach (DSL) to define static graphs directly.
    *   ⏳ **Intermediate Representation (IR):**
        *   ⏳ Define a custom graph IR suitable for NeuraRust semantics.
        *   ⏳ Alternatively, investigate integration with existing IRs like MLIR (requires LLVM toolchain) or graph rewriting libraries (`egg`).
    *   ⏳ **Optimization Passes:**
        *   ⏳ Implement common graph optimizations: Operator Fusion (e.g., Conv+BN+ReLU), Constant Folding, Dead Code Elimination, Algebraic Simplification.
    *   ⏳ **Code Generation:**
        *   ⏳ Generate optimized Rust code for the traced graph.
        *   ⏳ Potentially target backend-specific code (e.g., generate CUDA PTX via LLVM for fused kernels - requires significant effort).
    *   ⏳ **Integration:** Provide API to invoke JIT compilation (`neurarust.jit.trace`, `neurarust.jit.script`).
*   **5.5 Visualization & Debugging [⏳]**
    *   🎯 Goal: Improve developer experience for understanding and debugging models.
    *   ⏳ **Training Hooks:**
        *   ⏳ Implement forward and backward hooks for `nn.Module` to inspect/modify activations and gradients.
        *   ⏳ Implement hooks for `Tensor` gradients.
    *   ⏳ **Computation Graph Visualization:**
        *   ⏳ Utility to export the dynamic autograd graph or a static JIT graph to standard formats (Graphviz `.dot` files, potentially TensorBoard summary format).
    *   ⏳ **Debugging Tools:**
        *   ⏳ Add a numerical gradient checking utility (finite differences) to verify backward implementations.
        *   ⏳ Improve error reporting for shape mismatches, device inconsistencies.
*   **5.6 Documentation, Examples & Tutorials [⏳]**
    *   🎯 Goal: Provide comprehensive resources for users to learn and effectively use NeuraRust.
    *   ⏳ **Comprehensive User Guide:** Structure covering Installation, Core Concepts (Tensor, Autograd), NN Modules, Optimization, Data Loading, GPU Usage, Python API, ONNX, Deployment Targets.
    *   ⏳ **API Reference Documentation:** Ensure complete, accurate, and well-formatted `rustdoc` for all public APIs, including usage examples within docstrings.
    *   ⏳ **Gallery of Examples:** Implement end-to-end examples for common tasks:
        *   ⏳ MNIST (MLP, CNN)
        *   ⏳ CIFAR-10 (CNN, ResNet-like)
        *   ⏳ IMDB Sentiment Classification (RNN, Transformer)
        *   ⏳ Potentially ImageNet Transfer Learning (using pre-trained weights via ONNX import?)
    *   ⏳ **Tutorials:** Create step-by-step guides covering: Getting Started, Building Custom Layers, Training/Evaluation Loops, Saving/Loading Models, Using the Python API, Deploying to WASM.
    *   ⏳ **Project Website:** Develop a dedicated website (e.g., using `mdbook` or a static site generator) hosting documentation, examples, tutorials, blog posts, and roadmap.

**Phase 6: Deployment, Specialization & Maturity [⏳ To Do]**
*   🎯 **Goal:** Target specific deployment platforms, leverage Rust's unique strengths, implement distributed training, and foster a community.
*   **6.1 Deployment Targets [⏳]**
    *   🎯 Goal: Enable efficient deployment of NeuraRust models across diverse environments.
    *   ⏳ **WebAssembly (WASM):**
        *   ⏳ Compile core inference engine to WASM (`wasm32-unknown-unknown`).
        *   ⏳ Optimize binary size (LTO, `wee_alloc`, code stripping).
        *   ⏳ Optimize execution speed (SIMD via `wasm_bindgen` or standard WASM SIMD).
        *   ⏳ Create JavaScript/TypeScript bindings (e.g., using `wasm-bindgen`) for easy web integration.
        *   ⏳ Provide examples of browser and Node.js usage.
    *   ⏳ **Native Binary Deployment:**
        *   ⏳ Facilitate static linking for creating self-contained executables.
        *   ⏳ Minimize external dependencies (especially system libraries).
        *   ⏳ Provide utilities or guides for packaging applications.
    *   ⏳ **Edge/Embedded (ARM):**
        *   ⏳ Ensure robust cross-compilation support for common ARM targets (e.g., `aarch64-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`).
        *   ⏳ Profile performance and memory usage on representative hardware (e.g., Raspberry Pi, mobile SoCs).
        *   ⏳ Explore ARM NEON SIMD optimizations for CPU operations.
    *   ⏳ **Server-Side Inference:**
        *   ⏳ Develop integration examples with popular Rust web frameworks (Actix, Axum, Rocket, Tonic).
        *   ⏳ Showcase asynchronous request handling and model serving patterns.
        *   ⏳ Benchmark server-side inference throughput and latency.
*   **6.2 Inference Optimizations [⏳]**
    *   🎯 Goal: Reduce model size and accelerate inference speed for deployed applications.
    *   ⏳ **Quantization:**
        *   ⏳ Implement Post-Training Static Quantization (PTQ - calibration needed).
        *   ⏳ Implement Post-Training Dynamic Quantization.
        *   ⏳ Explore Quantization-Aware Training (QAT - requires integration with training process).
        *   ⏳ Support common quantized types (`int8`, potentially `uint8`).
        *   ⏳ Provide quantized kernels (CPU, potentially GPU/accelerator specific).
    *   ⏳ **Pruning:**
        *   ⏳ Implement unstructured magnitude pruning.
        *   ⏳ Explore structured pruning techniques (filter/channel pruning).
        *   ⏳ Provide utilities for applying pruning masks and fine-tuning pruned models.
    *   ⏳ **Model Distillation:**
        *   ⏳ Provide framework support/hooks to facilitate knowledge distillation (e.g., loss functions comparing student/teacher outputs).
*   **6.3 Distributed Training (Multi-Node) [⏳]**
    *   🎯 Goal: Enable training larger models on distributed compute clusters.
    *   ⏳ **Communication Backend Integration:**
        *   ⏳ Integrate with standard backends like MPI (via `mpi-rs` or similar).
        *   ⏳ Potentially explore custom TCP/RDMA backends for higher performance.
        *   ⏳ Abstract backend details behind a common communication interface.
    *   ⏳ **Distributed Primitives:**
        *   ⏳ Implement core collective communication operations: `all_reduce`, `broadcast`, `gather`, `all_gather`, `scatter`, `barrier`.
    *   ⏳ **`DistributedDataParallel` (DDP):**
        *   ⏳ Implement DDP wrapper for `nn.Module`.
        *   ⏳ Handle gradient synchronization efficiently during backward pass (e.g., gradient bucketing).
        *   ⏳ Ensure model state synchronization (initial weights, buffers).
    *   ⏳ **Tooling & Launching:**
        *   ⏳ Provide tools or scripts for launching distributed training jobs (integrating with cluster schedulers like Slurm is a plus).
*   **6.4 Leveraging Rust's Strengths [⏳]**
    *   🎯 Goal: Fully exploit Rust's unique features for safety, performance, and concurrency.
    *   ⏳ **Advanced Static Optimizations (Compile-Time):**
        *   ⏳ Explore procedural macros (`proc_macros`) for analyzing and optimizing NN graphs *before* compilation (e.g., more aggressive fusion, static shape inference).
    *   ⏳ **Enhanced Safety & Verification:**
        *   ⏳ Use the type system more extensively to enforce constraints (e.g., dimensional correctness via const generics or dependent types if feasible, device consistency checks at compile time).
        *   ⏳ Explore formal verification methods for critical components (e.g., autograd correctness, memory safety in unsafe blocks).
    *   ⏳ **Fearless Concurrency:**
        *   ⏳ Utilize libraries like `rayon` for easy data-parallelism within CPU ops (e.g., parallelizing batch processing in element-wise ops, certain reductions).
        *   ⏳ Explore fine-grained task-based parallelism for complex operations or graph execution.
*   **6.5 Tooling & Infrastructure [⏳]**
    *   🎯 Goal: Provide robust development tools and maintain a high-quality CI/CD pipeline.
    *   ⏳ **Robust Benchmarking Suite:**
        *   ⏳ Expand benchmarks (`criterion.rs`) to cover all core ops, NN layers, and common model architectures.
        *   ⏳ Benchmark across different backends (CPU, CUDA, potentially others) and configurations (data types, batch sizes).
        *   ⏳ Regularly compare performance against baseline frameworks (PyTorch, LibTorch).
    *   ⏳ **Extended Continuous Integration (CI):**
        *   ⏳ Test builds and run tests across multiple platforms (Linux x86_64, macOS x86_64/aarch64, Windows x86_64).
        *   ⏳ Test against different Rust versions (stable, beta, nightly?).
        *   ⏳ Maintain CI jobs for different feature combinations (CPU-only, CUDA-enabled).
        *   ⏳ Implement basic distributed training tests in CI (e.g., using Docker containers or simulated environment).
*   **6.6 Community & Ecosystem [⏳]**
    *   🎯 Goal: Foster an active community and integrate NeuraRust within the broader Rust ecosystem.
    *   ⏳ **Governance & Contribution:**
        *   ⏳ Establish a clear project governance model (e.g., BDFL, Core Team, RFC process).
        *   ⏳ Refine contribution guidelines and code review processes.
    *   ⏳ **Community Engagement:**
        *   ⏳ Actively manage GitHub Discussions.
        *   ⏳ Consider setting up dedicated communication channels (Discord, Matrix, Zulip).
        *   ⏳ Encourage user feedback and contributions.
    *   ⏳ **Ecosystem Integration:**
        *   ⏳ Explore integration with other relevant Rust crates: plotting libraries (`plotters`), dataframes (`polars`, `datafusion`), scientific computing tools.
        *   ⏳ Publish helper crates or examples demonstrating integrations.

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