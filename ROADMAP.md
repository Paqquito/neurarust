# New Roadmap (Post F64 Validation - Revised)

*This roadmap outlines the planned development phases following the successful implementation and validation of core operations and autograd using F64 for gradient checking. It incorporates essential features for performance and usability aimed at PyTorch parity.*

## Phase 1 Completion: MLP Foundation, Core API & Cleanup
*   🎯 **Goal:** Finalize essential CPU tensor API features (including in-place ops), implement foundational NN primitives leading to a trainable MLP, and ensure code quality through cleanup and documentation.

*   **Sub-Phase 1.A: Code Cleanup & Documentation:**
    *   🎯 **Goal:** Eliminate compiler warnings, add comprehensive API documentation, and refactor duplicated code for better maintainability and clarity.
    *   **Detailed Steps:**
        *   **Step 1.A.1: Warning Identification & Prioritization**
            *   [✅] Run `cargo clippy --all-targets -- -D warnings` & `cargo test --workspace` to get a comprehensive list of all current warnings across the workspace.
            *   [✅] Analyze and categorize warnings (unused imports, dead code, unreachable patterns, style suggestions, potential logic issues).
            *   [✅] Prioritize warnings: Address potential logic issues and dead code first, then unused imports/variables, then style lints.
        *   **Step 1.A.2: Automated Fixes (Imports & Basic Lints)**
            *   [✅] Run `cargo fix --allow-dirty --allow-staged` to automatically fix simple issues like unused imports.
            *   [✅] Re-run `cargo clippy` & `cargo test` to verify remaining warnings.
        *   **Step 1.A.3: Manual Warning Resolution (Module by Module)**
            *   [✅] Address remaining warnings in `src/error.rs`.
            *   [✅] Address remaining warnings in `src/types.rs`.
            *   [✅] Address remaining warnings in `src/buffer.rs`.
            *   [✅] Address remaining warnings in `src/tensor_data.rs`.
            *   [✅] Address remaining warnings in `src/tensor/`. (mod.rs, create.rs, utils.rs, traits.rs, tensor_impl.rs, indexing_impl.rs, shape_impl.rs, view_methods.rs)
            *   [✅] Address remaining warnings in `src/autograd/`. (mod.rs, grad_check.rs, graph.rs)
            *   [✅] Address remaining warnings in `src/ops/`. (mod.rs, traits.rs)
            *   [✅] Address remaining warnings in `src/ops/arithmetic/`. (mod.rs, add.rs, div.rs, mul.rs, neg.rs, pow.rs, sub.rs)
            *   [✅] Address remaining warnings in `src/ops/ DType ` (mod.rs, cast.rs)
            *   [✅] Address remaining warnings in `src/ops/linalg/`. (mod.rs, matmul.rs, matmul_test.rs)
            *   [✅] Address remaining warnings in `src/ops/loss/`. (mod.rs) - *Potentially empty or minimal now*
            *   [✅] Address remaining warnings in `src/ops/nn/`. (mod.rs, activation.rs) - *ReLU only for now*
            *   [✅] Address remaining warnings in `src/ops/reduction/`. (mod.rs, max_min.rs, mean.rs, sum.rs)
            *   [✅] Address remaining warnings in `src/ops/view/`. (mod.rs, contiguous.rs, expand.rs, permute.rs, reshape.rs, slice.rs, squeeze_unsqueeze.rs, transpose.rs)
            *   [✅] Address remaining warnings in `tests/`. (grad_check_test.rs, tensor_ops_test.rs, view_ops_test.rs etc.)
            *   [✅] Re-run `cargo clippy --all-targets -- -D warnings` until no warnings remain.
        *   **Step 1.A.4: `rustdoc` Documentation - Core Structures**
            *   [✅] Module-level docs for `lib.rs`, `error.rs`, `types.rs`, `device.rs`, `buffer.rs`, `tensor_data.rs`, `tensor/mod.rs`.
            *   [✅] Structs: `NeuraRustError`, `DType`, `StorageDevice`, `Buffer`, `CpuBuffer`, `TensorData`, `Tensor`.
            *   [✅] Methods/Constructors: `Buffer::try_get_cpu_f32/f64`, `TensorData::new/new_f64/new_view/buffer/...`, `Tensor::new/new_f64`.
        *   **Step 1.A.5: `rustdoc` Documentation - Tensor Methods & Traits**
            *   [✅] Creation Functions: `tensor/create.rs` (`zeros`, `ones`, `full`, `from_vec*`, `*_like`, `arange`, `linspace`, `eye`, `rand*`)
            *   [✅] Utility Functions: `tensor/utils.rs` (`calculate_strides`, `broadcast_shapes`, `index_to_coord`)
            *   [✅] Traits: `tensor/traits.rs` (`Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`, `TensorImpl`)
            *   [✅] Tensor Method Impls:
                *   [✅] `tensor/accessors.rs` (`shape`, `strides`, `dtype`, `device`, `rank`, `numel`, `is_contiguous`, `item_*`, `*_data`, `get_*_data`)
                *   [✅] `tensor/autograd_methods.rs` (`requires_grad`, `set_requires_grad`, `grad`, `grad_fn`, `set_grad_fn`, `acc_grad`, `backward`, `detach`, `clear_grad`)
                *   [✅] `tensor/reduction_methods.rs` (`mean`, `max`)
                *   [✅] `tensor/view_methods.rs` (`slice`, `transpose`, `permute`, `reshape`, `contiguous`)
                *   [-] `tensor/data_conversion.rs` (Methods seem covered in `accessors.rs` and `view_methods.rs`)
        *   **Step 1.A.6: `rustdoc` Documentation - Autograd**
            *   [✅] Add module-level docs for `autograd/mod.rs`.
            *   [✅] Add docs for `BackwardOp` trait (`autograd/backward_op.rs`).
            *   [✅] Add docs for `graph.rs` (`NodeId`, `topological_sort`).
            *   [✅] Add docs for `grad_check.rs` (`GradCheckError`, `check_grad`, `calculate_loss`).
        *   **Step 1.A.7: `rustdoc` Documentation - Operations (`ops`)**
            *   [ ] Add module-level docs for `ops/mod.rs` and `ops/traits.rs`.
            *   [ ] Add docs for each op module (`arithmetic`, `linalg`, `nn`, `reduction`, `view`, `dtype`).
            *   [ ] Add docs for public op functions (e.g., `add_op`, `matmul_op`, `relu_op`, `sum_op`, `reshape_op`, `cast_op`, etc.).
            *   [ ] Add docs for `Backward` structs associated with each operation.
        *   **Step 1.A.8: Documentation Generation & Review**
            *   [ ] Run `cargo doc --open --no-deps` to build and view the documentation locally.
            *   [ ] Review generated docs for clarity, completeness, and correctness. Fix any issues.
        *   **Step 1.A.9: Refactoring Identification**
            *   [ ] Review code (especially in `ops` and `tensor/utils.rs`) for duplicated logic or patterns suitable for abstraction.
            *   [ ] Candidate 1: Broadcasting logic (e.g., `NdArrayBroadcastingIter` usage). Can it be centralized or simplified?
            *   [ ] Candidate 2: Gradient reduction logic (`reduce_gradient_to_shape`). Is it optimally placed and reusable?
            *   [ ] Candidate 3: CPU Kernel patterns (e.g., loops iterating over buffers). Can generic helpers be created?
            *   [ ] Candidate 4: DType dispatch logic (`match tensor.dtype()`). Can macros or traits simplify this? (Maybe later phase)
        *   **Step 1.A.10: Refactoring Implementation (Iterative)**
            *   [ ] (If applicable) Implement refactoring for Candidate 1, ensuring tests pass.
            *   [ ] (If applicable) Implement refactoring for Candidate 2, ensuring tests pass.
            *   [ ] (If applicable) Implement refactoring for Candidate 3, ensuring tests pass.
            *   [ ] Document any new utility functions/modules created during refactoring.

*   **Sub-Phase 1.B: Foundational NN Primitives & Core Tensor API:**
    *   🎯 **Goal:** Implement essential tensor methods and the basic building blocks for neural networks.
    *   **Detailed Steps:**
        *   **Step 1.B.1: Implement `Tensor::detach()`**
            *   [✅] Implement `detach()` method to create a new `Tensor` sharing the same data but detached from the autograd graph (`grad_fn = None`, `requires_grad = false`).
            *   [✅] Add tests verifying detachment and data sharing.
            *   [✅] Add `rustdoc` for `detach()`.
        *   **Step 1.B.2: Implement Scalar Extraction `Tensor::item()`**
            *   [ ] Implement `item<T: Copy>()` method to extract a single scalar value from a 0-dimensional tensor (or tensor with 1 element). Should return `Result<T, NeuraRustError>`.
            *   [ ] Add tests for correct extraction and error handling (non-scalar tensor).
            *   [ ] Add `rustdoc` for `item()`.
        *   **Step 1.B.3: Implement Basic Random Creation (`rand`, `randn`)**
            *   [ ] Implement `rand(shape)` and `randn(shape)` creation functions (likely in `src/tensor/create.rs`). Use a simple RNG initially (e.g., `rand` crate). Specify `DType` (default F32).
            *   [ ] Add tests for shape correctness and basic distribution properties (e.g., range for `rand`).
            *   [ ] Add `rustdoc` for `rand` and `randn`.
        *   **Step 1.B.4: Implement Weight Initialization Helpers (`nn::init`)**
            *   [ ] Create `src/nn/init.rs`.
            *   [ ] Implement common initializers like `kaiming_uniform_`, `kaiming_normal_`, `xavier_uniform_`, `zeros_`, `ones_`. These should operate *in-place* on a given `Tensor`.
            *   [ ] Add tests for each initializer (checking basic statistics or values).
            *   [ ] Add `rustdoc` for the `nn::init` module and functions.
        *   **Step 1.B.5: Define `Parameter` Wrapper** *(Formerly Optional 1.B.1)*
            *   [ ] Define `struct Parameter(Tensor)`.
            *   [ ] Implement `new(Tensor)` setting `requires_grad = true`.
            *   [ ] Implement `Deref`/`DerefMut`.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 1.B.6: Define Basic `Module` Trait** *(Formerly 1.B.2)*
            *   [ ] Define `trait Module` with `forward` and potentially `parameters`.
            *   [ ] Add `rustdoc`.
        *   **Step 1.B.7: Implement `nn::Linear` Layer** *(Formerly 1.B.3)*
            *   [ ] Create `src/nn/layers/linear.rs`.
            *   [ ] Define `Linear` struct (`weight: Parameter`, `bias: Option<Parameter>`).
            *   [ ] Implement `new()` using helpers from `nn::init` (Step 1.B.4).
            *   [ ] Implement `Module` trait (`forward` using existing ops).
            *   [ ] Add tests (constructor, forward, shape, autograd via `check_grad`).
            *   [ ] Add `rustdoc`.
        *   **Step 1.B.8: Implement `nn::MSELoss` Function** *(Formerly 1.B.4)*
            *   [ ] Create `src/nn/losses/mse.rs`.
            *   [ ] Define `MSELoss` struct/function with reduction options.
            *   [ ] Implement `forward` using existing ops.
            *   [ ] Add tests (forward correctness for reductions, shape, autograd via `check_grad`).
            *   [ ] Add `rustdoc`.

*   **Sub-Phase 1.C: Basic Training Loop Example:**
    *   🎯 **Goal:** Create a runnable example demonstrating a minimal end-to-end training process.
    *   **Detailed Steps:**
        *   **Step 1.C.1: Define MLP Structure** *(Unchanged)*
            *   [ ] Create `examples/basic_mlp_cpu.rs`.
            *   [ ] Define `SimpleMLP` struct, implement `Module`.
            *   [ ] Implement `forward` (`linear1 -> relu -> linear2`).
            *   [ ] (Optional) Implement `parameters()` method.
        *   **Step 1.C.2: Create Synthetic Data** *(Unchanged)*
            *   [ ] Generate `X`, `Y` tensors.
        *   **Step 1.C.3: Instantiate Model and Loss** *(Unchanged)*
            *   [ ] Instantiate `SimpleMLP`, `MSELoss`.
        *   **Step 1.C.4: Implement `zero_grad` Mechanism**
            *   [ ] Implement logic to zero gradients (e.g., method on `Parameter` or manual iteration setting `.grad = None`). Test it. Add docs.
        *   **Step 1.C.5: Implement Manual Training Loop** *(Using temporary inefficient update)*
            *   [ ] Define `learning_rate`, `num_epochs`.
            *   [ ] Loop:
                *   **Forward Pass:** `y_pred = model.forward(&X)?`.
                *   **Calculate Loss:** `loss = loss_fn.forward(&y_pred, &Y)?`.
                *   **Backward Pass:** `loss.backward()?`.
                *   **(Manual) Optimizer Step (Temporary Inefficient Version):**
                    *   Iterate through parameters `p`.
                    *   Access gradient `g`.
                    *   **Create a *new* tensor for updated weights:** `new_p_data = p.data_view()? - learning_rate * g.data_view()?`.
                    *   **Replace parameter's tensor with a new detached tensor:** `p.set_data(Tensor::new(new_p_data, p.shape()).detached())`.
                    *   *Note: This approach is simple but inefficient. Phase 1.D will introduce efficient in-place updates.*
                *   **Zero Gradients:** Use mechanism from Step 1.C.4.
                *   **(Optional) Logging:** Use `item()` from Step 1.B.2.
        *   **Step 1.C.6: Configure Example Execution** *(Unchanged)*
            *   [ ] Add `[[example]]` to `Cargo.toml`, ensure `cargo run --example basic_mlp_cpu` works.
        *   **Step 1.C.7: Add Documentation for Example** *(Unchanged)*
            *   [ ] Add comments, module docs.

*   **Sub-Phase 1.D: In-Place Operations:**
    *   🎯 **Goal:** Implement essential in-place arithmetic operations for performance and memory efficiency, critical for PyTorch parity.
    *   **Detailed Steps:**
        *   **Step 1.D.1: Implement `add_`**
            *   [ ] Implement `Tensor::add_(&mut self, other: &Tensor)`.
            *   [ ] Handle broadcasting.
            *   [ ] Modify buffer directly.
            *   [ ] **Autograd Check:** Add runtime check (e.g., `NeuraRustError::InplaceModificationError`) for leaf tensors requiring grad or nodes needed for backward.
            *   [ ] Add tests (correctness, broadcasting, autograd error).
            *   [ ] Add `rustdoc`.
        *   **Step 1.D.2: Implement `sub_`**
            *   [ ] Implement `Tensor::sub_`, similar to `add_`.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 1.D.3: Implement `mul_`**
            *   [ ] Implement `Tensor::mul_`, similar to `add_`.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 1.D.4: Implement `div_`**
            *   [ ] Implement `Tensor::div_`, similar to `add_`.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 1.D.5: Refactor Training Loop Example (Optional but Recommended)**
            *   [ ] Modify Step 1.C.5 (Optimizer Step) in `basic_mlp_cpu.rs` to use the efficient in-place operations (e.g., `p.sub_(g.mul_scalar(learning_rate))`.

*   **Phase 1 Notes:**
    *   *Other DTypes (`I64`, `I32`, `Bool`, etc.), full mixed-type operation support, and other creation functions (`arange`, `linspace`, `eye`) are deferred to later phases (e.g., Phase 2 or 4) to keep Phase 1 focused.*

## Phase 2: Optimization, Data Loading & Core DTypes
*   🎯 **Goal:** Introduce essential components for efficient model training (optimizers, data loaders) and expand core DType support to include Integers and Booleans.

*   **Sub-Phase 2.A: Optimizers (`neurarust-optim` or core):**
    *   🎯 **Goal:** Implement standard optimization algorithms.
    *   **Detailed Steps:**
        *   [ ] Define `Optimizer` trait (using in-place ops from Phase 1.D).
        *   [ ] Implement SGD optimizer.
        *   [ ] Implement Adam optimizer (requires storing momentum tensors).
        *   [ ] Add tests for optimizers.
        *   [ ] Add `rustdoc`.

*   **Sub-Phase 2.B: Data Loading (`neurarust-data` or core):**
    *   🎯 **Goal:** Implement basic data loading and batching capabilities.
    *   **Detailed Steps:**
        *   [ ] Define `Dataset` trait.
        *   [ ] Implement a simple `VecDataset`.
        *   [ ] Implement basic `DataLoader` (batching, shuffling on CPU). Handle collation of tensors (requires consistent DTypes in batch initially).
        *   [ ] Add tests for DataLoader.
        *   [ ] Add `rustdoc`.

*   **Sub-Phase 2.C: Essential DType Support (Integer, Boolean):**
    *   🎯 **Goal:** Add support for I64, I32, and Bool DTypes to core structures and operations.
    *   **Detailed Steps:**
        *   [ ] Extend `DType` enum with `I64`, `I32`, `Bool`.
        *   [ ] Extend `Buffer`/`CpuBuffer` enums with corresponding variants.
        *   [ ] Adapt creation functions (`zeros`, `ones`, `full`, `rand`, `Tensor::new`, etc.) to handle these new DTypes.
        *   [ ] Adapt core tensor methods (`item`, `cast` if exists, etc.) for new types.
        *   [ ] Adapt existing `ops` (arithmetic, linalg, reduction, view, etc.) to handle new DType combinations where it makes sense (e.g., basic arithmetic for Ints, logical ops for Bool). This involves adding `match` arms and potentially new kernels. *Focus on common, sensible operations initially.*
        *   [ ] Add tests for creating and operating on tensors with these new DTypes.
        *   [ ] Update `rustdoc`.

## Phase 3: GPU Acceleration (CUDA First)
*(Content mostly unchanged)*
*   🎯 **Goal:** Enable high-performance training and inference by adding GPU support.
*   **Sub-Phase 3.A: Backend Abstraction & CUDA Setup:**
    *   Refine `StorageDevice` / `Buffer` for GPU.
    *   Implement `Tensor::to(device)` for CPU <-> GPU copies.
    *   Integrate CUDA bindings and context management.
*   **Sub-Phase 3.B: GPU Kernels & Ops Integration:**
    *   Implement CUDA kernels or integrate libraries (cuBLAS, cuDNN) for core ops (including in-place).
    *   Adapt core operations to dispatch to GPU implementations.
*   **Sub-Phase 3.C: Device Management & Autograd:**
    *   Ensure autograd works seamlessly with GPU tensors.
    *   Adapt NN layers and optimizers for device placement.

## Phase 4: Expanding NN Capabilities & Interoperability
*(Content mostly unchanged, but can explicitly include deferred items now)*
*   🎯 **Goal:** Broaden the scope of supported architectures, DTypes, and enable interaction with the wider ML ecosystem.
*   **Sub-Phase 4.A: Advanced Layers & Architectures:**
    *   Implement Convolutional layers (Conv2d).
    *   Implement Pooling layers.
    *   Implement basic RNN layers.
    *   (Future: Attention, Transformers...)
*   **Sub-Phase 4.B: Advanced DType & Op Support:** *(Revised from former 4.B)*
    *   🎯 **Goal:** Implement full mixed-type operations and support for remaining DTypes.
    *   **Detailed Steps:**
        *   [ ] Implement robust mixed-type operations across all supported DTypes (Numeric, Bool) with clear type promotion rules (e.g., `f32 + i64 -> f32`, `f64 * bool -> f64`).
        *   [ ] Implement support for any other relevant DTypes if identified (e.g., `Complex`, smaller floats/ints).
        *   [ ] Implement remaining creation functions (`arange`, `linspace`, `eye`, etc.), ensuring DType flexibility.
        *   [ ] Add comprehensive tests for type promotion and new creation functions.
        *   [ ] Update `rustdoc`.
*   **Sub-Phase 4.C: Interoperability:**
    *   ONNX export/import capabilities.
    *   Python bindings (PyO3).

## Phase 5: Deployment & Advanced Features
*(Content mostly unchanged)*
*   🎯 **Goal:** Target diverse deployment scenarios and implement advanced optimization techniques.
*   **Sub-Phase 5.A: Deployment Targets:**
    *   WebAssembly (WASM) compilation.
    *   Native binary deployment strategies.
    *   Edge/Embedded considerations (ARM).
*   **Sub-Phase 5.B: Advanced Optimizations:**
    *   Inference Optimizations (Quantization, Pruning - Exploratory).
    *   Distributed Training (Multi-GPU/Multi-Node - Exploratory).

*(This roadmap provides a high-level overview. Specific tasks within each sub-phase will be detailed as we progress.)*