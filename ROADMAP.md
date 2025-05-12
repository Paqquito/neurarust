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
            *   [✅] Address remaining warnings in `neurarust-core/src/optim/` (optimizer modules, tests, examples)
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
            *   [✅] Add module-level docs for `ops/mod.rs` and `ops/traits.rs`.
            *   [✅] Add docs for each op module (`arithmetic`, `linalg`, `nn`, `reduction`, `view`, `dtype`).
            *   [✅] Add docs for public op functions (e.g., `add_op`, `matmul_op`, `relu_op`, `sum_op`, `reshape_op`, `cast_op`, etc.).
            *   [✅] Add docs for `Backward` structs associated with each operation.
        *   **Step 1.A.8: Documentation Generation & Review**
            *   [✅] Run `cargo doc --open --no-deps` to build and view the documentation locally.
            *   [✅] Review generated docs for clarity, completeness, and correctness. Fix any issues.
        *   **Step 1.A.9: Refactoring Identification**
            *   [✅] Review code (especially in `ops` and `tensor/utils.rs`) for duplicated logic or patterns suitable for abstraction.
            *   [✅] Candidate 1: Broadcasting logic (e.g., `NdArrayBroadcastingIter` usage). Can it be centralized or simplified? -> Addressed for arithmetic ops via helper.
            *   [✅] Candidate 2: Gradient reduction logic (`reduce_gradient_to_shape`). Is it optimally placed and reusable? -> Addressed for arithmetic ops via helper.
            *   [✅] Candidate 3: CPU Kernel patterns (e.g., loops iterating over buffers). Can generic helpers be created? -> Addressed for unary ops (`neg`, `ln`) and contiguous copy (`NdArraySimpleIter`). Also reduction ops (`sum`, `mean`, `max`) refactored using utilities.
            *   [ ] Candidate 4: DType dispatch logic (`match tensor.dtype()`). Can macros or traits simplify this? -> **Initiating Incremental Refactoring (Sub-Roadmap Below)**
                *   **Step 1.A.9.4.1: Define `NeuraNumeric` Trait**
                    *   [✅] Create a new trait `NeuraNumeric` (e.g., in `src/types/numeric.rs` or `src/ops/traits/numeric.rs`).
                    *   [✅] Define necessary bounds: `num_traits::Float`, `std::ops::{Add, Sub, Mul, Div, Neg}`, `PartialOrd`, `Debug`, `Clone`, `Copy`, `Send`, `Sync`, `'static`. (Updated bounds based on implementation)
                    *   [ ] Define associated constants if needed (e.g., `ZERO`, `ONE`, `MIN_VALUE`, `MAX_VALUE`) beyond what `num_traits::Float` provides. (Decided not needed for now)
                    *   [✅] Add `rustdoc` for the trait.
                *   **Step 1.A.9.4.2: Implement `NeuraNumeric` for F32/F64**
                    *   [✅] Implement `impl NeuraNumeric for f32`.
                    *   [✅] Implement `impl NeuraNumeric for f64`.
                    *   [✅] Add tests to verify trait bounds and constant values if applicable.
                *   **Step 1.A.9.4.3: Refactor Unary Kernel (PoC - `neg_op`)**
                    *   [✅] Create a generic kernel function `neg_kernel<T: NeuraNumeric>(data: &[T]) -> Vec<T>` in `ops/arithmetic/neg.rs`.
                    *   [✅] Modify `neg_op` function:
                        *   [✅] Keep the outer `match dtype` block.
                        *   [✅] Inside the match arms: Get the correct buffer slice (`try_get_cpu_f32`/`f64` needs access via guard -> buffer -> match).
                        *   [✅] Call the generic `neg_kernel::<f32>` or `neg_kernel::<f64>`.
                        *   [✅] Create the output `Tensor` with the correct DType (`Tensor::new`/`new_f64`).
                    *   [✅] Ensure `NegBackward` still functions correctly (verified structure and autograd tests).
                    *   [✅] Verify with `cargo test --workspace`.
                *   **Step 1.A.9.4.4: Refactor Binary Kernel Helper (PoC - e.g., `add_op`)**
                    *   [✅] **Option A (Refactor Helper):** Modify `apply_binary_op_broadcasted` (in `ops/arithmetic/mod.rs`):
                        *   [✅] Keep its signature taking `&Tensor`.
                        *   [✅] Keep the `match dtype` for buffer access and broadcaster setup.
                        *   [✅] Make the *inner loop/computation* call a new generic kernel/function `binary_kernel<T: NeuraNumeric>(a: T, b: T) -> T` (defined with the specific operation like `a + b`). (Achieved by passing closures calling the kernel)
                        *   [✅] Ensure output tensor creation uses the correct DType.
                    *   [ ] **Option B (Refactor Kernel Directly):** If helper refactoring is too complex, create a generic `add_kernel<T: NeuraNumeric>` similar to `neg_kernel` but handling two inputs (potentially with broadcasting iterators made generic or accepting slices). Modify `add_op` to call this kernel.
                    *   [✅] Choose Option A or B. Implement the chosen approach for `add_op`. (Chose modified Option A)
                    *   [✅] Ensure `AddBackward` still functions correctly. (Verified via tests)
                    *   [✅] Verify with `cargo test --workspace`.
                *   **Step 1.A.9.4.5: Evaluate PoC and Plan Wider Rollout**
                    *   [✅] Review the refactored `neg_op` and `add_op` code.
                    *   [✅] Assess: Is the `NeuraNumeric` trait sufficient? Is the pattern clean and repeatable? Does it significantly reduce kernel code duplication?
                    *   [✅] Decide:
                        *   [✅] Proceed: Apply the pattern to other ops.
                        *   [ ] Refine: Modify the trait or pattern.
                        *   [ ] Revert/Postpone: If the abstraction proves too complex or doesn't yield benefits now.
                *   **Step 1.A.9.4.6: Apply Generic Kernel Pattern Iteratively (If PoC Successful)**
                    *   [✅] Gradually refactor other arithmetic op kernels (`sub`, `mul`, `div`, `pow`, `ln`, etc.) using the established pattern.
                    *   [✅] Refactor reduction kernels (`sum_kernel`, `mean_kernel`, `max_kernel`).
                    *   [✅] Refactor comparison kernels (`equal_op`).
                    *   [ ] Refactor other relevant kernels as identified.
                    *   [✅] Ensure tests pass after each refactoring step.
        *   **Step 1.A.10: Refactoring Implementation (Iterative)**
            *   [✅] (If applicable) Implement refactoring for Candidate 1, ensuring tests pass. -> Done for arithmetic ops.
            *   [✅] (If applicable) Implement refactoring for Candidate 2, ensuring tests pass. -> Done for arithmetic ops.
            *   [✅] (If applicable) Implement refactoring for Candidate 3, ensuring tests pass. -> Done for unary ops (`neg`, `ln`), `contiguous`, and reduction ops (`sum`, `mean`, `max`) using utilities.
            *   [✅] Document any new utility functions/modules created during refactoring. -> `apply_binary_op_broadcasted`, `apply_unary_op`, `ContiguousBackward`, reduction utils (`process_reduction_axes`, `calculate_reduction_output_shape`, `calculate_grad_broadcast_shape`) documented (implicitly via usage/commits, need explicit doc review later).

*   **Sub-Phase 1.B: Foundational NN Primitives & Core Tensor API:**
    *   🎯 **Goal:** Implement essential tensor methods and the basic building blocks for neural networks.
    *   **Detailed Steps:**
        *   **Step 1.B.1: Implement `Tensor::detach()`**
            *   [✅] Implement `detach()` method to create a new `Tensor` sharing the same data but detached from the autograd graph (`grad_fn = None`, `requires_grad = false`).
            *   [✅] Add tests verifying detachment and data sharing.
            *   [✅] Add `rustdoc` for `detach()`.
        *   **Step 1.B.2: Implement Scalar Extraction `Tensor::item()`**
            *   [✅] Implement `item<T: Copy>()` method to extract a single scalar value from a 0-dimensional tensor (or tensor with 1 element). Should return `Result<T, NeuraRustError>`.
            *   [✅] Add tests for correct extraction and error handling (non-scalar tensor).
            *   [✅] Add `rustdoc` for `item()`.
        *   **Step 1.B.3: Implement Basic Random Creation (`rand`, `randn`)**
            *   [✅] Implement `rand(shape)` and `randn(shape)` creation functions (likely in `src/tensor/create.rs`). Use a simple RNG initially (e.g., `rand` crate). Specify `DType` (default F32).
            *   [✅] Add tests for shape correctness and basic distribution properties (e.g., range for `rand`).
            *   [✅] Add `rustdoc` for `rand` and `randn`.
        *   **Step 1.B.4: Implement Weight Initialization Helpers (`nn::init`)**
            *   [✅] Create `src/nn/init.rs`.
            *   [✅] Implement common initializers like `kaiming_uniform_`, `kaiming_normal_`, `xavier_uniform_`, `zeros_`, `ones_`. These should operate *in-place* on a given `Tensor`.
            *   [✅] Add tests for each initializer (checking basic statistics or values).
            *   [✅] Add `rustdoc` for the `nn::init` module and functions.
        *   **Step 1.B.5: Define `Parameter` Wrapper & Enhancements**
            *   [✅] Define `struct Parameter(Tensor)`.
            *   [✅] Add `name: Option<String>` field to `Parameter` struct.
                *   [✅] Update constructor `Parameter::new(data: Tensor, name: Option<String>)` or add `Parameter::new_with_name()`.
                *   [✅] Ensure the name is accessible (e.g., via a method `name() -> Option<&str>`).
            *   [✅] Implement `new(Tensor)` setting `requires_grad = true` (adapt for optional name).
            *   [✅] Implement `Deref`/`DerefMut` to access the underlying `Tensor`.
            *   [✅] Add tests specifically for the name functionality (creation with name, retrieval).
            *   [✅] Add/Update `rustdoc` for `Parameter`, including the name field and its usage.
        *   **Step 1.B.6: Define Basic `Module` Trait & Introspection**
            *   [✅] Define `trait Module` with a `forward(&self, input: &Tensor) -> Result<Tensor, NeuraRustError>` method.
            *   [✅] Define `parameters(&self) -> Vec<&Parameter>` method in the `Module` trait.
            *   [✅] Define `named_parameters(&self) -> Vec<(String, &Parameter)>` method in the `Module` trait.
            *   [✅] This method should collect and return references to all `Parameter`s along with their names (e.g., "linear1.weight", "bias").
            *   [✅] Implement logic to generate unique and hierarchical names for parameters within nested modules.
            *   [✅] Define `children(&self) -> Vec<&dyn Module>` method in the `Module` trait.
                *   [✅] This method should return direct child modules.
            *   [✅] Define `named_children(&self) -> Vec<(String, &dyn Module)>` method in the `Module` trait.
                *   [✅] This method should return direct child modules along with their attribute names.
            *   [✅] Define `modules(&self) -> Vec<&dyn Module>` method in the `Module` trait.
                *   [✅] This method should return an iterator over all modules in the tree (self + all descendants), depth-first.
            *   [✅] Implement these introspection methods for `nn::Linear` (parameters, named_parameters, modules; children would be empty).
            *   [✅] Implement `named_parameters` `children`, `named_children`, and modules for `SimpleMLP` in example files (`basic_mlp_cpu.rs`, `basic_mlp_cpu_inplace_optim.rs`). 
            *   [✅] Add/Update `rustdoc` for the `Module` trait and its methods, explaining their purpose and usage.
        *   **Step 1.B.7: Implement `nn::Linear` Layer**
            *   [✅] Create `src/nn/layers/linear.rs`.
            *   [✅] Define `Linear` struct (`weight: Parameter`, `bias: Option<Parameter>`).
            *   [✅] Implement `new()` using helpers from `nn::init` (Step 1.B.4) and naming parameters (e.g., "weight", "bias").
            *   [✅] Implement `Module` trait (`forward` using existing ops, and introspection methods from 1.B.6).
            *   [✅] Add tests (constructor, forward, shape, autograd via `check_grad`, introspection results).
            *   [✅] Add `rustdoc`.
        *   **Step 1.B.8: Implement `nn::MSELoss` Function**
            *   [✅] Create `src/nn/losses/mse.rs`.
            *   [✅] Define `MSELoss` struct/function with reduction options.
            *   [✅] Implement `forward` using existing ops. (Note: MSELoss is typically a function, not a module with parameters, so introspection methods might not apply or be trivial).
            *   [✅] Add tests (forward correctness for reductions, shape, autograd via `check_grad`).
            *   [✅] Add `rustdoc`.
        *   **Step 1.B.9: Additional Random Creation Functions (Deferred to align with DType support in Phase 2.C)**
            *   🎯 **Goal:** Expand tensor creation capabilities for integer and boolean types once they are supported.
            *   [ ] **`randint(low: i64, high: i64, shape: &[usize], dtype: DType, device: StorageDevice)`**
                *   [ ] Implement for integer DTypes (I32, I64) when available (Phase 2.C). Generates integers in `[low, high)`.
                *   [✅] (Optional) Initial F32/F64 version could produce floats then cast, or be skipped until Int DTypes. (Implemented F32/F64 version)
                *   [✅] Add tests for shape, range, DType correctness. (Implemented for F32/F64 version)
                *   [✅] Add `rustdoc`. (Added for F32/F64 version)
            *   [ ] **`bernoulli(p: &Tensor, dtype: DType, device: StorageDevice)` or `bernoulli_scalar(p_scalar: f64, shape: &[usize], ...)`**
                *   [ ] Implement to generate boolean tensors (from probabilities `p`) when Bool DType is available (Phase 2.C).
                *   [ ] `p` can be a scalar probability or a tensor of probabilities. Output is 0 or 1.
                *   [✅] (Optional) Initial F32/F64 version could produce 0.0/1.0. (Implemented `bernoulli_scalar` F32/F64 version)
                *   [✅] Add tests for shape, distribution according to `p`, DType correctness. (Implemented for `bernoulli_scalar` F32/F64 version)
                *   [✅] Add `rustdoc`. (Added for `bernoulli_scalar` F32/F64 version)
        *   **Step 1.B.10: Essential Shape Manipulation Methods (Non-Mutating Views)**
            *   🎯 **Goal:** Provide core methods for reshaping or inspecting tensor dimensions without data copies.
            *   **Step 1.B.10.1: Implement `Tensor::flatten(start_dim, end_dim)`**
                *   [✅] Implement `flatten(&self, start_dim: usize, end_dim: usize) -> Result<Tensor, NeuraRustError>`.
                *   [✅] Flattens a contiguous range of dims into a single dimension.
                *   [✅] Should return a view (no data copy) by adjusting shape and strides.
                *   [✅] Handle `start_dim` and `end_dim` validation (e.g., `start_dim <= end_dim`, within rank).
                *   [✅] Add tests for shape, data integrity (view), and error conditions.
                *   [✅] Add `rustdoc`.
            *   **Step 1.B.10.2: Review and Enhance `Tensor::unsqueeze(dim)` and `Tensor::squeeze(dim)`**
                *   [✅] Confirm/implement `unsqueeze(&self, dim: usize) -> Result<Tensor, NeuraRustError>` and `squeeze(&self, dim: Option<usize>) -> Result<Tensor, NeuraRustError>`.
                *   [✅] `unsqueeze` adds a new dimension of size 1 at `dim`.
                *   [✅] `squeeze` removes dimensions of size 1. If `dim` is specified, only that dimension is squeezed if it's size 1. If `dim` is None, all dimensions of size 1 are removed.
                *   [✅] Ensure these return views.
                *   [✅] Add/verify tests for various `dim` inputs, multi-dim squeezing, and error conditions.
                *   [✅] Ensure `rustdoc` is comprehensive for both in `tensor/view_methods.rs`.
            *   **Step 1.B.10.3: Implement `Tensor::expand(new_shape)`**
                *   [✅] Implement `expand(&self, new_shape: &[usize]) -> Result<Tensor, NeuraRustError>`.
                *   [✅] Expands dimensions of size 1 to match `new_shape`. Dimensions of size -1 in `new_shape` mean "do not change this dimension".
                *   [✅] Should return a view (no data copy) by adjusting strides (strides of expanded dimensions become 0).
                *   [✅] Handle `new_shape` validation (compatibility with original shape, no shrinking of non-unit dimensions).
                *   [✅] Add tests for shape, data integrity (view), stride calculation, and error conditions.
                *   [✅] Add `rustdoc`.

*   **Sub-Phase 1.C: Basic Training Loop Example:**
    *   🎯 **Goal:** Create a runnable example demonstrating a minimal end-to-end training process.
    *   **Detailed Steps:**
        *   **Step 1.C.1: Define MLP Structure**
            *   [✅] Create `examples/basic_mlp_cpu.rs`.
            *   [✅] Define `SimpleMLP` struct, implement `Module`.
            *   [✅] Implement `forward` (`linear1 -> relu -> linear2`).
            *   [✅] Implement `parameters()` and other introspection methods from 1.B.6 for `SimpleMLP`.
        *   **Step 1.C.2: Create Synthetic Data**
            *   [✅] Generate `X`, `Y` tensors.
        *   **Step 1.C.3: Instantiate Model and Loss**
            *   [✅] Instantiate `SimpleMLP`, `MSELoss`.
        *   **Step 1.C.4: Implement `zero_grad` Mechanism**
            *   [✅] Implement logic to zero gradients (e.g., iterate `model.parameters()` and call `param.clear_grad()`). Test it. Add docs.
        *   **Step 1.C.5: Implement Manual Training Loop**
            *   [✅] Define `learning_rate`, `num_epochs`.
            *   [✅] Loop:
                *   [✅] **Forward Pass:** `y_pred = model.forward(&X)?`.
                *   [✅] **Calculate Loss:** `loss = loss_fn.forward(&y_pred, &Y)?`.
                *   [✅] **Backward Pass:** `loss.backward()?`.
                *   [✅] **(Manual) Optimizer Step (Temporary Inefficient Version):**
                    *   [✅] Iterate through `model.parameters()`.
                    *   [✅] Access gradient `g` for each parameter `p`.
                    *   [✅] **Create a *new* tensor for updated weights:** `new_p_data = p.data_view()? - learning_rate * g.data_view()?`.
                    *   [✅] **Replace parameter's tensor with a new detached tensor:** `p.set_data(Tensor::new(new_p_data, p.shape()).detached())`.
                *   [✅] **Zero Gradients:** Use mechanism from Step 1.C.4.
                *   [✅] **(Optional) Logging:** Use `item()` from Step 1.B.2.
        *   **Step 1.C.6: Configure Example Execution**
            *   [✅] Add `[[example]]` to `Cargo.toml`, ensure `cargo run --example basic_mlp_cpu` works.
        *   **Step 1.C.7: Add Documentation for Example**
            *   [✅] Add comments, module docs.

*   **Sub-Phase 1.D: In-Place Operations:**
    *   🎯 **Goal:** Implement essential in-place arithmetic operations for performance and memory efficiency, critical for PyTorch parity.
    *   **Detailed Steps:**
        *   **Step 1.D.1: Implement `add_`**
            *   [✅] Implement `Tensor::add_(&mut self, other: &Tensor)`.
            *   [✅] Handle broadcasting.
            *   [✅] Modify buffer directly.
            *   [✅] **Autograd Check:** Add runtime check (e.g., `NeuraRustError::InplaceModificationError`) for leaf tensors requiring grad or nodes needed for backward.
            *   [✅] Add tests (correctness, broadcasting, autograd error).
            *   [✅] Add `rustdoc`.
        *   **Step 1.D.2: Implement `sub_`**
            *   [✅] Implement `Tensor::sub_`, similar to `add_`.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.3: Implement `mul_`**
            *   [✅] Implement `Tensor::mul_`, similar to `add_`.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.4: Implement `div_`**
            *   [✅] Implement `Tensor::div_`, similar to `mul_`, handle division by zero.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.5: Implement `pow_` (Tensor to scalar power)**
            *   [✅] Implement `Tensor::pow_(&mut self, exponent: exponent_type)` where `exponent_type` is `f32` or `f64`.
            *   [✅] Consider `NeuraNumeric` for `exponent_type` or provide separate methods for `f32`/`f64` exponents.
            *   [✅] Handle potential issues (e.g., negative base with non-integer exponent, 0^0).
            *   [✅] Add tests for correctness, edge cases, and autograd error (as it's in-place).
            *   [✅] Add `rustdoc`.
        *   **Step 1.D.6: Implement `add_scalar_`**
            *   [✅] Implement `Tensor::add_scalar_(&mut self, scalar: S)` where `S` is `f32` or `f64` (matching tensor's `DType`).
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.7: Implement `sub_scalar_`**
            *   [✅] Implement `Tensor::sub_scalar_(&mut self, scalar: S)`.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.8: Implement `mul_scalar_`**
            *   [✅] Implement `Tensor::mul_scalar_(&mut self, scalar: S)`.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.9: Implement `div_scalar_`**
            *   [✅] Implement `Tensor::div_scalar_(&mut self, scalar: S)`.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 1.D.10: Refactor Training Loop Example (Optional but Recommended)**
            *   [✅] Modify Step 1.C.5 (Optimizer Step) in `basic_mlp_cpu.rs` to use the efficient in-place operations (e.g., `p.sub_(g.mul_scalar(learning_rate))`, potentially `p.pow_f32(2.0)` or `p.add_scalar_(value)` if applicable).
                 (Création d'un nouvel exemple `basic_mlp_cpu_inplace_optim.rs`)
        *   **Step 1.D.11: Implement `clamp_` In-Place**
            *   🎯 **Goal:** Add in-place clamping operation for tensors.
            *   [✅] Implement `Tensor::clamp_(&mut self, min: Option<S>, max: Option<S>)`
                *   `S` should match the tensor's DType (e.g., `f32`, `f64`).
                *   If `min` is `Some`, all elements less than `min` are set to `min`.
                *   If `max` is `Some`, all elements greater than `max` are set to `max`.
                *   Modify the tensor's buffer directly.
                *   Perform autograd checks similar to other in-place operations (error if modifying a tensor part of a graph that requires it for backward, unless it's a leaf and `requires_grad` or if CoW is triggered).
            *   [✅] Add comprehensive unit tests:
                *   Test with `min` only, `max` only, both `min` and `max`.
                *   Test cases where no clamping occurs.
                *   Test edge cases (e.g., `min` > `max`, though this behavior might be undefined or an error).
                *   Test autograd error triggering.
            *   [✅] Add `rustdoc` for `clamp_`, detailing its behavior and arguments.
        *   **Step 1.D.12: Implement `fill_` In-Place**
            *   🎯 **Goal:** Allow in-place filling of a tensor with a scalar value.
            *   **Step 1.D.12.1: Implement `Tensor::fill_(&mut self, value: S)`**
                *   [✅] Implement `fill_(&mut self, value: S)` where `S` matches tensor's DType.
                *   [✅] Modifies the tensor buffer directly, setting all elements to `value`.
                *   [✅] Perform autograd checks (similar to other in-place ops).
                *   [✅] Add tests for correctness across DTypes and autograd error triggering.
                *   [✅] Add `rustdoc`.
        *   **Step 1.D.13: Logical In-Place Operations (Deferred to align with DType support in Phase 2.C)**
            *   🎯 **Goal:** Implement in-place logical operations for Boolean tensors once they are supported.
            *   [ ] **`Tensor::logical_and_(&mut self, other: &Tensor)`** (for Boolean Tensors)
                *   [ ] Element-wise logical AND, modifies `self` in-place.
                *   [ ] Requires `self` and `other` to be Boolean tensors and broadcastable.
                *   [ ] Autograd check (likely error if `requires_grad`, as Booleans usually don't have grads).
                *   [ ] Add tests and `rustdoc`.
            *   [ ] **`Tensor::logical_or_(&mut self, other: &Tensor)`** (for Boolean Tensors)
                *   [ ] Similar to `logical_and_`.
                *   [ ] Add tests and `rustdoc`.
            *   [ ] **`Tensor::logical_xor_(&mut self, other: &Tensor)`** (for Boolean Tensors)
                *   [ ] Similar to `logical_and_`.
                *   [ ] Add tests and `rustdoc`.

*   **Phase 1 Notes:**
    *   *Other DTypes (`I64`, `I32`, `Bool`, etc.), full mixed-type operation support, and other creation functions (`arange`, `linspace`, `eye`) are deferred to later phases (e.g., Phase 2 or 4) to keep Phase 1 focused.*
    *   *Introspection methods for `Module` (1.B.6) are foundational and will be leveraged by optimizers and serialization.*
    *   *`clamp_` (1.D.11) and `fill_` (1.D.12) are versatile utility operations often used in training loops or for specific layer implementations.*
    *   *Basic shape manipulation methods like `flatten`, `unsqueeze`, `squeeze` (1.B.10) are crucial for data preparation and defining layer computations.*

## Phase 2: Optimization, Data Loading & Core DTypes
*   🎯 **Goal:** Introduce essential components for efficient model training (optimizers, data loaders) and expand core DType support to include Integers and Booleans, paving the way for more diverse models and data types.

*   **Sub-Phase 2.A: Optimizers (`neurarust-optim` or dedicated module in core):**
    *   🎯 **Goal:** Implement standard optimization algorithms with support for parameter groups and learning rate scheduling.
    *   **Detailed Steps:**
        *   **Step 2.A.1: Define `Optimizer` Trait and Core Logic**
            *   🎯 **Goal:** Establish a common interface and lifecycle for all optimizers.
            *   [✅] Define `trait Optimizer`:
                *   [✅] `step(&mut self) -> Result<(), NeuraRustError>`: Performs a single optimization step (updates parameters).
                *   [✅] `zero_grad(&mut self)`: Clears the gradients of all parameters managed by the optimizer.
                *   [✅] `add_param_group(&mut self, param_group: ParamGroup)`: (Optional, for later) Allows adding new parameter groups.
                *   [✅] `load_state_dict(&mut self, state_dict: &OptimizerState)` and `state_dict(&self) -> OptimizerState`: For saving/loading optimizer state (e.g., momentum buffers).
            *   [✅] Define `struct ParamGroup`:
                *   [✅] Contains `params: Vec<Arc<RwLock<Parameter>>>` (or similar reference to parameters).
                *   [✅] Contains optimizer-specific hyperparameters (e.g., `lr: f32`, `weight_decay: f32`).
            *   [✅] Design `OptimizerState` enum/struct to hold state for various optimizers.
            *   [✅] Implement a mechanism for optimizers to hold and manage references to `Parameter`s (likely via `Arc<RwLock<Parameter>>` obtained from `Module::parameters()`).
            *   [✅] Add `rustdoc` for the trait and supporting structs. (Partially complete)
        *   **Step 2.A.2: Implement SGD Optimizer**
            *   🎯 **Goal:** Implement the Stochastic Gradient Descent optimizer with common features.
            *   [✅] Create `struct SgdOptimizer` implementing `Optimizer`.
            *   [✅] Constructor: `new(params: impl Iterator<Item = Arc<RwLock<Parameter>>>, lr: f32, momentum: f32, weight_decay: f32, nesterov: bool)`.
            *   [✅] Implement `step()` logic:
                *   [✅] Basic gradient descent: `p = p - lr * grad`.
                *   [✅] Momentum: `buf = momentum * buf + grad; p = p - lr * buf`.
                *   [✅] Weight decay (L2 penalty): `grad = grad + weight_decay * p` before other updates.
                *   [✅] Nesterov momentum: `grad_adjusted = grad + momentum * buf; p = p - lr * grad_adjusted` (requires careful handling of `buf` update).
            *   [✅] Implement `zero_grad()` by iterating through parameters and calling `param.clear_grad()`.
            *   [ ] Manage momentum buffers (one per parameter, stored in optimizer state). (Partial - state dict needed)
            *   [✅] Add tests: basic step, momentum, weight decay, Nesterov, state saving/loading (tests expect panic for state dict).
            *   [✅] Add `rustdoc`. (Partial)
        *   **Step 2.A.3: Implement Adam/AdamW Optimizer**
            *   🎯 **Goal:** Implement the Adam and AdamW optimizers.
            *   [✅] Create `struct AdamOptimizer` implementing `Optimizer`. (Completed and refined)
            *   [✅] Constructor: `new(params: impl Iterator<Item = Arc<RwLock<Parameter>>>, lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32, amsgrad: bool)`. (Verified, uses RwLock)
            *   [✅] Implement `step()` logic:
                *   [✅] Calculate biased first moment estimate (`m_t`).
                *   [✅] Calculate biased second raw moment estimate (`v_t`).
                *   [✅] Compute bias-corrected first moment estimate (`m_hat_t`).
                *   [✅] Compute bias-corrected second raw moment estimate (`v_hat_t`).
                *   [✅] Update parameters: `p = p - lr * m_hat_t / (sqrt(v_hat_t) + eps)`.
                *   [✅] Implement AdamW variant (decoupled weight decay: `p = p * (1 - lr * weight_decay)` applied *before* main Adam update, or directly applied to parameter outside gradient modification). (Implemented by direct application to parameter)
                *   [✅] (Optional) Implement AMSGrad variant. (Field present, logic TBD) // Logic and tests are now complete
            *   [✅] Manage first and second moment buffers (`m` and `v` per parameter) and step counter in optimizer state.
            *   [✅] Add tests: basic Adam step, bias correction, weight decay (AdamW), state saving/loading. (Core Adam logic tested; state_dict TBD)
            *   [✅] Add `rustdoc`. (Basic doc comments added)
        *   **Step 2.A.4: Implement RMSprop Optimizer**
            *   🎯 **Goal:** Implement the RMSprop optimizer.
            *   [✅] Create `struct RmsPropOptimizer` implementing `Optimizer`. (Corrected to RwLock)
            *   [✅] Constructor: `new(params: impl Iterator<Item = Arc<RwLock<Parameter>>>, lr: f32, alpha: f32, eps: f32, weight_decay: f32, momentum: f32, centered: bool)`.
            *   [✅] Implement `step()` logic:
                *   [✅] Update squared gradient average: `sq_avg = alpha * sq_avg + (1-alpha) * grad^2`.
                *   [✅] (Optional, if `centered`) Maintain average gradient: `grad_avg = alpha * grad_avg + (1-alpha) * grad`.
                *   [✅] (Optional, if `centered`) Update denominator: `denom = sqrt(sq_avg - grad_avg^2 + eps)`.
                *   [✅] (Else) Update denominator: `denom = sqrt(sq_avg + eps)`.
                *   [✅] Parameter update: `p = p - lr * grad / denom`.
                *   [✅] Implement momentum and [✅] weight decay if specified.
            *   [✅] Manage squared gradient average buffers (and optionally gradient average buffers) in optimizer state.
            *   [✅] Add tests: [✅] basic step, [✅] momentum, [✅] weight decay, [✅] centered, [✅] state saving/loading.
            *   [✅] Add `rustdoc`. (Partial)
        *   **Step 2.A.5: (Optional) Implement Adagrad Optimizer**
            *   🎯 **Goal:** Implement the Adagrad optimizer.
            *   [✅] Create `struct AdagradOptimizer` implementing `Optimizer`.
            *   [✅] Constructor: `new(params: impl Iterator<Item = Arc<RwLock<Parameter>>>, lr: f32, lr_decay: f32, weight_decay: f32, initial_accumulator_value: f32, eps: f32)`.
            *   [✅] Implement `step()` logic.
            *   [✅] Manage sum of squared gradients accumulator per parameter.
            *   [✅] Add tests and `rustdoc`.
        *   **Step 2.A.6: Learning Rate Schedulers**
            *   🎯 **Goal:** Implement common learning rate scheduling policies.
            *   [✅] Define `trait LRScheduler` (defined implicitly).
                *   [✅] `step(&mut self, epoch: Option<u64>, metrics: Option<f32>)`. (Changed epoch to u64 to match)
                *   [✅] `get_last_lr(&self) -> Vec<f32>`.
                *   [✅] `optimizer(&self) -> &O`, `optimizer_mut(&mut self) -> &mut O`.
            *   [✅] Implement `StepLR`: `new(optimizer, step_size, gamma)`.
                *   [✅] Decays LR of each parameter group by gamma every `step_size` epochs.
            *   [✅] Implement `MultiStepLR`: `new(optimizer, milestones, gamma)`.
                *   [✅] Decays LR by gamma once the number of epoch reaches one of the milestones.
            *   [✅] Implement `ReduceLROnPlateau`: `new(optimizer, mode, factor, patience, threshold, ...)`.
                *   [✅] Reduces LR when a monitored metric has stopped improving.
            *   [✅] Integrate LR schedulers with the training loop example. (`basic_mlp_cpu_inplace_optim.rs` uses StepLR)
            *   [✅] Add tests for each scheduler policy and their interaction with optimizers. (Most tests pass)
            *   [✅] Add `rustdoc`. (Partially for ReduceLROnPlateau, StepLR, MultiStepLR)
        *   **Step 2.A.7: Parameter Groups Support in Optimizers**
            *   🎯 **Goal:** Allow different hyperparameters (e.g., learning rate, weight decay) for different sets of parameters.
            *   [✅] Refine optimizer constructors to accept `Vec<ParamGroup>` or an iterator of `Parameter`s that get grouped by default. (Iter + add_param_group)
            *   [✅] Ensure `step()` and `zero_grad()` iterate through all parameter groups and apply respective hyperparameters.
            *   [✅] Ensure LR Schedulers can handle multiple parameter groups, adjusting LRs accordingly.
            *   [✅] Add tests for optimizers and schedulers with multiple parameter groups. (Tested for SGD, Adagrad)
            *   [✅] Update training loop example to demonstrate usage of parameter groups (e.g., different LR for biases).
        *   **Step 2.A.8: Gradient Clipping Utilities**
            *   🎯 **Goal:** Provide functions to clip parameter gradients to stabilize training.
            *   [✅] Implement `clip_grad_value_(parameters: impl Iterator<Item = &mut Parameter>, clip_value: f32)`.
                *   [✅] Iterates through parameters and clips their gradients in-place: `grad.clamp_(-clip_value, clip_value)`.
            *   [✅] Implement `clip_grad_norm_(parameters: impl Iterator<Item = &mut Parameter>, max_norm: f32, norm_type: f32)`.
                *   [✅] Calculates the total norm of all gradients concatenated: `total_norm = (sum(g.pow(norm_type)) for g in all_grads).pow(1.0/norm_type)`.
                *   [✅] If `total_norm > max_norm`, scales all gradients: `grad = grad * (max_norm / total_norm)`.
            *   [✅] Ensure these functions correctly handle `Option<Tensor>` for gradients.
            *   [✅] Add tests for both clipping methods with various inputs and norm types.
            *   [✅] Add `rustdoc` and usage examples (e.g., in training loop comments). // rustdoc added, examples for 2.A.9
        *   **Step 2.A.9: Create Optimizer and Scheduler Example**
            *   🎯 **Goal:** Demonstrate the usage of various optimizers, LR schedulers, and gradient clipping in a training context.
            *   [✅] Create a new example file (e.g., `examples/advanced_training_techniques.rs`).
            *   [✅] Adapt the `SimpleMLP` model or a similar small model.
            *   [✅] Demonstrate usage of SGD and Adam/AdamW.
            *   [✅] Demonstrate usage of at least one LR Scheduler (e.g., `StepLR`).
            *   [✅] Demonstrate usage of gradient clipping (`clip_grad_norm_`).
            *   [✅] Show how to configure parameter groups with different learning rates.
            *   [✅] Ensure the example runs and shows loss decreasing or a similar success metric.
            *   [✅] Add to `Cargo.toml` and document the example.

*   **Sub-Phase 2.B: Data Loading (`neurarust-data` or dedicated module in core):**
    *   🎯 **Goal:** Implement basic data loading, batching, and shuffling capabilities with flexible sampling.
    *   **Detailed Steps:**
        *   **Step 2.B.1: Define `Dataset` Trait**
            *   🎯 **Goal:** Establish a standard interface for datasets.
            *   [✅] Define `trait Dataset`:
                *   [✅] `get(&self, index: usize) -> Result<Self::Item, NeuraRustError>` (or `__getitem__`).
                *   [✅] `len(&self) -> usize` (or `__len__`).
                *   [✅] `type Item: Send + 'static` (the type of a single data sample, e.g., `(Tensor, Tensor)` or just `Tensor`).
            *   [✅] Add `rustdoc`.
        *   **Step 2.B.2: Implement `VecDataset`**
            *   🎯 **Goal:** A simple dataset wrapping a `Vec` of items.
            *   [✅] Create `struct VecDataset<T: Clone + Send + 'static>` implementing `Dataset`.
            *   [✅] Constructor: `new(data: Vec<T>)`.
            *   [✅] Implement `get` and `len`.
            *   [✅] Add tests: creation, get, len.
            *   [✅] Add `rustdoc`.
        *   **Step 2.B.3: Implement `TensorDataset`**
            *   🎯 **Goal:** A dataset for one or more tensors, where the first dimension is the batch dimension.
            *   [✅] Create `struct TensorDataset` implementing `Dataset`.
                *   [✅] `Item` type would be `Vec<Tensor>` (one slice from each input tensor).
            *   [✅] Constructor: `new(tensors: Vec<Tensor>)`.
                *   [✅] Validate that all input tensors have the same length in their first dimension.
            *   [✅] Implement `get` to return a `Vec<Tensor>` where each tensor is a slice of the original tensors at the given index (using `Tensor::slice` or a similar mechanism to get the i-th item along dim 0).
            *   [✅] Implement `len` based on the first dimension of the input tensors.
            *   [✅] Add tests: creation with single/multiple tensors, validation of lengths, `get` correctness (shape and data of slices), `len`.
            *   [✅] Add `rustdoc`.
        *   **Step 2.B.4: Define `Sampler` Trait and Basic Samplers**
            *   🎯 **Goal:** Abstract index generation for `DataLoader` to allow custom sampling strategies.
            *   [✅] Define `trait Sampler: Send + Sync`:
                *   [✅] `iter(&self, dataset_len: usize) -> Box<dyn Iterator<Item = usize> + Send + Sync>`.
                *   [✅] `len(&self, dataset_len: usize) -> usize` (number of samples the sampler will yield for a given dataset size).
            *   [✅] Implement `SequentialSampler`: `new()`.
                *   [✅] `iter` yields `0..dataset_len`.
                *   [✅] `len` returns `dataset_len`.
            *   [✅] Implement `RandomSampler`: `new(replacement: bool, num_samples: Option<usize>)`.
                *   [✅] `iter` yields random indices, with or without replacement.
                *   [✅] `num_samples` can specify a different length than `dataset_len` (especially with replacement).
                *   [✅] `len` returns `num_samples` if Some, else `dataset_len`.
            *   [✅] (Optional) Implement `SubsetRandomSampler`: `new(indices: Vec<usize>)`.
                *   [✅] Samples randomly from the provided subset of indices.
            *   [✅] Add tests for each sampler's iteration logic and length. (SequentialSampler tested)
            *   [✅] Add `rustdoc`. (For Sampler and SequentialSampler)
        *   **Step 2.B.5: Implement `DataLoader` using `Sampler`**
            *   🎯 **Goal:** Provide an iterator for loading data in batches with flexible sampling and collation.
            *   **Sub-steps:**
                *   [✅] **2.B.5.1: Définir la structure générique `DataLoader<D: Dataset, S: Sampler>`**
                    *   Champs : dataset, batch_size, sampler, drop_last, collate_fn (optionnel)
                    *   Définir le constructeur de base
                *   [✅] **2.B.5.2: Implémenter le trait `Iterator` pour `DataLoader`**
                    *   Méthode `next()` qui génère un batch
                    *   Utilisation du sampler pour obtenir les indices
                    *   Récupération des items via `dataset.get()`
                    *   Gestion de `drop_last`
                *   [✅] **2.B.5.3: Implémenter la fonction de collation par défaut**
                    *   Collate une liste de samples en batch (ex: stack pour Tensor, zip pour tuples)
                    *   Permettre une fonction personnalisée
                *   [✅] **2.B.5.4: Ajouter des tests unitaires**
                    *   Test du batching avec différents samplers
                    *   Test de `drop_last`
                    *   Test de la collation par défaut et personnalisée
                *   [✅] **2.B.5.5: Ajouter la documentation rustdoc**
                    *   Doc sur la structure, le constructeur, l'itérateur, la collation
        *   **Step 2.B.6: Create Data Loading Example**
            *   🎯 **Goal:** Demonstrate the usage of `Dataset`, `Sampler`, and `DataLoader` in a training context.
            *   [✅] Create a new example file (e.g., `examples/data_loading_example.rs`).
            *   [✅] Use `TensorDataset` with synthetic data (e.g., features and labels).
            *   [✅] Demonstrate usage of `SequentialSampler` and `RandomSampler` with the `DataLoader`.
            *   [✅] Iterate through the `DataLoader` to get batches.
            *   [✅] (Optional) Perform a mock training step (e.g., print batch shapes or feed to a dummy model) to show integration.
            *   [✅] Ensure the example runs and outputs expected batch information.
            *   [✅] Add to `Cargo.toml` and document the example.
        *   **Step 2.B.7: (Advanced - Future Consideration) `DataLoader` Enhancements**
            *   🎯 **Goal:** Lay groundwork for more performant data loading.
            *   [ ] (Placeholder) `num_workers: usize` for multi-process/multi-threaded data fetching (significant complexity, for later).
            *   [ ] (Placeholder) `pin_memory: bool` for faster CPU->GPU transfers (relevant when GPU support is mature).

*   **Sub-Phase 2.C: Essential DType Support (Integer, Boolean) & Advanced Operations:**
    *   🎯 **Goal:** Add support for I64, I32, and Bool DTypes to core structures, enable fundamental operations, and introduce versatile tensor manipulations.
    *   **Detailed Steps:**
        *   **Step 2.C.1: Extend Core `DType`, `Buffer`, `TensorData`**
            *   🎯 **Goal:** Update core enums and structs to recognize new DTypes.
            *   [✅] Extend `DType` enum with `I64`, `I32`, `Bool`.
            *   [✅] Extend `CpuBuffer` (and `Buffer` enum if it wraps `CpuBuffer` directly) with variants like `I64(Vec<i64>)`, `I32(Vec<i32>)`, `Bool(Vec<bool>)`.
            *   [✅] Adapt `TensorData` internal representation or accessors to handle these new buffer types (e.g., `try_get_cpu_i64`, `try_get_cpu_bool`).
            *   [✅] Add tests for new DType enum values and buffer variants.
        *   **Step 2.C.2: Adapt Tensor Creation Functions**
            *   🎯 **Goal:** Allow creation of tensors with new DTypes.
            *   [✅] Implement `Tensor::new_i64(data: Vec<i64>, shape: &[usize])`, `Tensor::new_i32(...)`, `Tensor::new_bool(...)`.
            *   [✅] Implement `Tensor::from_vec_i64(...)`, etc., that take `Vec<S>` and `shape` and `DType`.
            *   [✅] Adapt `zeros(shape, dtype, device)`, `ones(...)`, `full(shape, value, dtype, device)` to support new DTypes.
                *   [✅] `full` with boolean `value` for `DType::Bool`.
                *   [✅] `full` with integer `value` for integer DTypes.
            *   [✅] Implement `randint(low, high, shape, dtype, device)` (moved from 1.B.9) for I32/I64 DTypes.
            *   [✅] Implement `bernoulli(p, shape, dtype, device)` or `bernoulli_scalar(p_scalar, ...)` (moved from 1.B.9) for Bool DType.
            *   [✅] Add tests for creating tensors of each new DType using all relevant methods.
        *   **Step 2.C.3: Adapt Core Tensor Methods**
            *   🎯 **Goal:** Ensure core tensor functionalities work with new DTypes.
            *   [✅] Implement scalar extraction: `item_i64()`, `item_i32()`, `item_bool()`.
            *   [✅] Ensure `Tensor::dtype()` correctly returns the new DTypes.
            *   [✅] Ensure `Tensor::numel()`, `Tensor::shape()`, `Tensor::strides()`, etc., function correctly.
            *   [✅] Implement `Tensor::cast(new_dtype)` to support casting to/from I64, I32, Bool.
                *   Define casting rules (e.g., Float to Int truncation, Int to Float, Bool to Int/Float as 0/1).
            *   [✅] Add tests for `item_*`, `cast` across various DType combinations.
        *   **Step 2.C.4: Implement Arithmetic Operations for Integer DTypes**
            *   🎯 **Goal:** Enable basic arithmetic for integer tensors.
            *   [✅] Implement `add_op`, `sub_op`, `mul_op` for (Int, Int) -> Int.
            *   [✅] Implement `div_op` for (Int, Int) -> Int (integer division, define behavior e.g. truncation towards zero).
            *   [✅] (Optional for this phase) Consider `pow_op` for (Int, Int_scalar_exponent) -> Int.
            *   [✅] Implement corresponding `Tensor` methods (`add`, `sub`, `mul`, `div`).
            *   [✅] Implement scalar versions: `add_scalar(IntTensor, IntScalar)`, etc.
            *   [✅] Handle broadcasting.
            *   [✅] Add autograd support (gradients would typically be float, or error if trying to get int gradients for int ops without specific context).
            *   [✅] Add tests for correctness, broadcasting, and autograd (if applicable for int ops).
        *   **Step 2.C.5: Implement Comparison and Logical Operations**
            *   🎯 **Goal:** Enable comparisons producing Boolean tensors, and logical operations on Boolean tensors.
            *   [✅] Implement comparison ops (`eq_op`, `ne_op`, `lt_op`, `gt_op`, `le_op`, `ge_op`):
                *   [✅] (Int, Int) -> BoolTensor
                *   [✅] (Float, Float) -> BoolTensor (déjà existant, vérifié)
                *   [✅] (Bool, Bool) -> BoolTensor
            *   [✅] Implement `Tensor` methods for comparisons (`eq`, `ne`, `lt`, `gt`, `le`, `ge`).
            *   [✅] Implement scalar versions for comparisons.
            *   [✅] Implement logical ops (`logical_and_op`, `logical_or_op`, `logical_xor_op`, `logical_not_op`) for (Bool, Bool) -> Bool or (Bool) -> Bool.
            *   [✅] Implement `Tensor` methods for logical ops.
            *   [✅] Handle broadcasting for binary logical/comparison ops.
            *   [✅] Add tests for all combinations, broadcasting, and output DTypes.
        *   **Step 2.C.6: Implement Indexing with Integer and Boolean Tensors**
            *   🎯 **Goal:** Allow advanced indexing similar to NumPy/PyTorch.
            *   [✅] **`Tensor::index_select(dim: usize, indices: &Tensor)`**: Selects slices along `dim` using `indices` (which must be I32/I64).
                *   [✅] Implement the core operation (likely in `ops/view` or `ops/indexing`).
                *   [✅] Add autograd support (gather backward op).
                *   [✅] Add tests for correctness, out-of-bounds handling, autograd.
            *   [✅] **`Tensor::masked_select(mask: &Tensor)`**: Selects elements where `mask` (BoolTensor, broadcastable to self.shape) is true, returns a 1D tensor.
                *   [✅] Implement the core operation.
                *   [✅] Add autograd support.
                *   [✅] Add tests for correctness with various mask shapes, autograd.
            *   [✅] **`Tensor::masked_fill_(&mut self, mask: &Tensor, value: S)`**: In-place fill where `mask` is true.
                *   [✅] `S` is a scalar matching self.dtype.
                *   [✅] Implement as an in-place operation (autograd CoW checks apply).
                *   [✅] Add tests for correctness, autograd checks.
            *   [ ] (Optional for this phase, more complex) `Tensor::index_put_(&mut self, indices: &[Option<&Tensor>], values: &Tensor)` for advanced assignment.
        *   **Step 2.C.7: Implement Reduction Operations for New DTypes**
            *   🎯 **Goal:** Extend reduction operations to new DTypes.
            *   [✅] `sum_op` for IntTensors (output IntTensor) and BoolTensors (output IntTensor, counting true values).
            *   [✅] `mean_op` for IntTensors (output FloatTensor).
            *   [✅] (New Reduction) `all_op` for BoolTensors (output BoolScalarTensor).
            *   [✅] (New Reduction) `any_op` for BoolTensors (output BoolScalarTensor).
            *   [✅] Adapt `Tensor` methods (`sum`, `mean`, `all`, `any`).
            *   [✅] Handle `dim` and `keepdim` arguments.
            *   [✅] Add tests for correctness, different DTypes, `dim`/`keepdim`.
        *   **Step 2.C.8: Implement `where` Operation (Conditional Ternary)**
            *   🎯 **Goal:** Provide a conditional element selection mechanism `where(condition, x, y)`.
            *   [✅] Implement `where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, NeuraRustError>`.
                *   `condition` must be `DType::Bool`.
                *   `x` and `y` must be broadcastable with `condition` and each other, and have the same DType.
                *   Result has elements from `x` where `condition` is true, else from `y`.
            *   [✅] Add `Tensor::where_cond(condition: &Tensor, y: &Tensor)` (self is `x`).
            *   [✅] Implement autograd (gradients flow through the chosen branch).
            *   [✅] Add tests for correctness, broadcasting, DTypes, autograd.
            *   [✅] Add `rustdoc`.
        *   **Step 2.C.9: Implement `bincount` for Integer Tensors**
            *   🎯 **Goal:** Count frequency of values in an integer tensor.
            *   [✅] Implement `bincount_op(input: &Tensor, weights: Option<&Tensor>, minlength: usize) -> Result<Tensor, NeuraRustError>`.
                *   [✅] `input` must be 1D, non-negative IntTensor (I32/I64).
                *   [✅] `weights` (optional) same shape as `input`.
                *   [✅] `minlength` ensures output tensor has at least this size.
                *   [✅] Output DType: same as `weights` if provided, else I64 (for counts).
            *   [✅] Add `Tensor::bincount(...)` method.
            *   [✅] Evaluate autograd necessity (often non-differentiable use case).
            *   [✅] Add tests for counts, weights, `minlength`.
            *   [✅] Add `rustdoc`.
        *   **Step 2.C.10: Create DType and Advanced Operations Example(s)**
            *   🎯 **Goal:** Demonstrate usage of Integer/Boolean DTypes and new tensor operations.
            *   [✅] Create a new example file (e.g., `examples/dtype_operations_example.rs`) or multiple smaller ones.
            *   [✅] Show creation of Integer and Boolean tensors (`zeros`, `ones`, `full`, `randint`, `bernoulli`).
            *   [✅] Démontrer les opérations arithmétiques sur Integer et logiques sur Bool.
            *   [✅] Demonstrate Integer arithmetic and Boolean logical operations.
            *   [✅] Demonstrate `index_select`, `masked_select`, `masked_fill_`.
            *   [✅] Demonstrate `where_op` usage.
            *   [✅] Demonstrate `bincount` usage.
            *   [✅] Ensure the example(s) run and output expected results, showcasing the API.
            *   [✅] Add to `Cargo.toml` and document the example(s).
        *   **Step 2.C.11: Review and Update `NeuraNumeric` (and potentially add `NeuraIntegral`, `NeuraBoolean`)**
            *   [✅] Evaluate if current `NeuraNumeric` (designed for Floats) can be used/adapted for Integer ops, or if specific integer kernels are simpler.
            *   [✅] Consider if new marker traits like `NeuraIntegral` or `NeuraBoolean` would simplify generic code for ops specific to these types.
            *   [✅] For now, prioritize direct implementation of kernels for new DTypes and revisit generic abstractions later if clear patterns emerge.
        *   **Step 2.C.12: Update `rustdoc` for all new DType functionalities.**
            *   [✅] Document DType enum, new buffer variants, creation functions, tensor methods, and op behaviors with new DTypes.

*   **Phase 2 Notes:**
    *   *Optimizers will heavily rely on in-place operations from Phase 1.D and `Module` introspection (parameters) from Phase 1.B.*
    *   *DataLoaders will need robust `Tensor::slice` (from Phase 1 views) and potentially `Tensor::stack` (to be implemented, or manual batch construction). The new `Sampler` trait enhances flexibility.*
    *   *DType expansion is a significant undertaking affecting many parts of the codebase. Focus on CPU implementations first.*
    *   *Mixed-DType operations (e.g., FloatTensor + IntScalar) are generally deferred to Phase 4.B, but very common ones (like IntTensor.mean() -> FloatTensor) are included here.*
    *   *Gradient clipping utilities (2.A.8) are important for training stability. A dedicated example (2.A.9) will showcase optimizer features.*
    *   *The `where` operation (2.C.8) and `bincount` (2.C.9) add significant expressive power for tensor manipulations with new DTypes. Examples in 2.C.10 will demonstrate these.*
    *   *Creation of dedicated examples (2.A.9, 2.B.6, 2.C.10) is crucial for validating and showcasing new functionalities in Phase 2.*

## Phase 3: GPU Acceleration (CUDA First)
*   🎯 **Goal:** Enable high-performance training and inference by adding GPU support (initially CUDA), including backend abstraction, CUDA kernel integration, and autograd compatibility for GPU tensors, with a focus on performance and robust device management.

*   **Sub-Phase 3.A: Backend Abstraction & CUDA Core Integration (Revised & Enhanced)**

🎯 **Goal:**  
Establish a solid abstraction for the CUDA backend, enabling efficient device, memory, stream, error, and transfer management, while preparing for future extension to other backends (ROCm, Metal, etc.).
The architecture must be modular, testable, thread-safe, and allow for advanced features (profiling, multi-GPU, hooks, etc.).

### Step 3.A.1: CUDA Bindings, Context & Device Management
- [✅] **Encapsulate the CUDA binding** (`rustacuda`, `cuda-rs`, etc.) in an internal module (`cuda_api`) to ease replacement or evolution.
- [✅] **Explicit, idempotent CUDA runtime initialization** (thread-safe, multi-init safe).
- [✅] **Centralized CUDA context management:**
    - [✅] Creation, activation, explicit destruction of contexts.
    - [✅] Multi-device support (one context per device, device_id → context mapping).
    - [✅] Thread-local or scope-based context switching (inspired by `torch.cuda.device`).
    - [✅] Unit and robustness tests (context leak, double init, etc.).
- [✅] **Device management:**
    - [✅] Enumeration, selection, property query (name, memory, compute capability, etc.).
    - [✅] API to list available devices, set/get current device, etc.
    - [✅] Expose a rich `CudaDevice` struct (properties, state, etc.).
    - [✅] Robustness tests (device unavailable, OOM, etc.).
- [✅] **CUDA error handling:**
    - [✅] Exhaustive mapping of CUDA errors to `NeuraRustError` (with explicit messages).
    - [✅] Add debug/verbose mode to log CUDA errors and critical calls.
    - [✅] Unit tests for each common error type (device unavailable, OOM, etc.).
    - [✅] Clear documentation on CUDA error handling.

### Step 3.A.2: CUDA Stream & Event Management (Asynchrony & Profiling)
- [✅] **CUDA stream abstraction:**
    - [✅] `CudaStream` struct (wrapper around `CUstream`), clonable, shareable.
    - [✅] Creation, destruction, synchronization, flag management (blocking/non-blocking).
    - [✅] Default stream per thread and per device (thread-local).
    - [✅] API to get current stream, create new ones, etc.
    - [✅] Stream priority support (if supported by device).
- [✅] **CUDA event management:**
    - [✅] `CudaEvent` struct (wrapper around `CUevent`).
    - [✅] Creation, destruction, recording, synchronization, timing.
    - [✅] Integration with streams for fine synchronization.
- [✅] **Profiling and hooks:**
    - [✅] Add a hook/profiler system to measure kernel/transfer execution time, etc.
    - [✅] API to register callbacks on streams/events.
- [✅] **Unit and robustness tests:**
    - [✅] Massive creation/destruction of streams/events.
    - [✅] Cross-synchronization, deadlocks, etc.
    - [✅] Latency and throughput measurement for streams.

### Step 3.A.3: Memory Management & Caching Allocator (PyTorch-like)
- [✅] **Implement a global thread-safe CUDA memory caching allocator** (PyTorch-inspired):
    - [✅] Intercept all allocation/free calls (`cuMemAlloc_v2`, `cuMemFree_v2`).
    - [✅] Pools of free blocks by size and device.
    - [✅] Block reuse, fine cache management (memory limit, cache clearing, etc.).
    - [ ] (Optional) Free block coalescing to reduce fragmentation.
    - [ ] (Optional) Memory usage statistics, monitoring hooks.
    - [ ] (Optional) Unified Memory and pinned memory support for fast transfers.
- [ ] **`CudaBuffer` abstraction:**
    - [ ] Opaque, non-copyable struct managing device pointer lifetime.
    - [ ] Integration with the caching allocator.
    - [ ] API to get raw pointer, size, device, etc.
    - [ ] Support for async operations (copy, set, etc.) via stream.
    - [ ] Robustness tests (double free, memory leak, etc.).
- [ ] **Integration with high-level structures:**
    - [ ] Add `Buffer::Cuda(CudaBuffer)` variant in the core.
    - [ ] Support `StorageDevice::Cuda(device_id)`.
    - [ ] Integrate with `TensorData` and related methods.
    - [ ] Integration tests (Tensor → GPU, GPU → Tensor, etc.).
- [ ] **Documentation and examples:**
    - [ ] Detailed rustdoc for allocator, buffer, invariants.
    - [ ] Advanced usage examples (profiling, cache clearing, etc.).

### Step 3.A.4: Data Transfer & Device Placement (CPU <-> GPU, Multi-GPU)
- [ ] **Data transfer API:**
    - [ ] `Tensor::to(device, stream)` to transfer tensors between CPU and GPU (and between GPUs).
    - [ ] Async transfers via stream (`cuMemcpyAsync`).
    - [ ] Synchronization management (explicit wait, hooks, etc.).
    - [ ] Support for slice/view transfers.
    - [ ] Transfer error handling (OOM, device mismatch, etc.).
- [ ] **Current device and scope management:**
    - [ ] API to set default device in a scope (`with_device` or thread-local).
    - [ ] Automatic allocation on the correct device according to context.
    - [ ] Robustness tests (multi-thread, multi-device).
- [ ] **Multi-GPU support:**
    - [ ] Allocation, transfer, and execution on multiple GPUs in the same process.
    - [ ] API for tensor scatter/gather, etc.
    - [ ] Multi-GPU usage examples.
- [ ] **Tests and documentation:**
    - [ ] Transfer integrity tests (round-trip CPU→GPU→CPU).
    - [ ] Rustdoc and usage examples.

### Step 3.A.5: Backend Modularity & Extensibility
- [ ] **Backend abstraction:**
    - [ ] Define a `Backend` trait (or equivalent) to allow other backends (ROCm, Metal, etc.).
    - [ ] Generic API for device, memory, stream, transfer management.
    - [ ] Clear separation between CUDA-specific and generic code.
    - [ ] Prepare for future backend integration without breaking the API.
- [ ] **Non-regression and compatibility tests:**
    - [ ] Ensure adding a backend does not impact CUDA backend stability.
    - [ ] Extension examples (mock backend, CPU-only, etc.).
- [ ] **Architecture documentation:**
    - [ ] Diagrams and explanations on backend modularity.

### Step 3.A.6: Robustness, Profiling & Advanced Debugging
- [ ] **Advanced error handling** (see 3.D.1, but start here):
    - [ ] Exhaustive CUDA error mapping, clear propagation through the stack.
    - [ ] Debug tools (context dump, detailed logs, etc.).
- [ ] **Memory and execution time profiling:**
    - [ ] API to measure memory usage, fragmentation, kernel/transfer execution time.
    - [ ] Integration with streams/events for fine profiling.
    - [ ] Profiling usage examples.
- [ ] **Stress and fuzz testing:**
    - [ ] Intensive tests on memory management, transfers, context/stream creation/destruction.
    - [ ] Bug/crash reports.
    - [ ] Test methodology documentation.

### **Key points to reinforce (vs. initial version):**
- **Modularity:** Everything must be designed for future backend addition and evolution.
- **Robustness:** Error handling, stress tests, debug/profiling hooks.
- **Performance:** Memory cache, async transfers, multi-GPU, streams/events.
- **Documentation:** Rustdoc, architecture guides, advanced examples.
- **Testing:** Unit, integration, stress, multi-thread/multi-device.

*   **Sub-Phase 3.B: GPU Kernels & Operations Integration (Revised & Enhanced)**

🎯 **Goal:**  
Implement a modular, high-performance, and extensible system for GPU-accelerated tensor operations, supporting custom CUDA kernels, library integration (cuBLAS, cuDNN, Thrust), dynamic dispatch, and robust validation against CPU reference implementations.

### Step 3.B.1: Build System & Kernel Management
- [ ] **Flexible build system for CUDA kernels:**
    - [ ] Support for both static (build.rs + cc crate) and JIT (runtime) compilation of `.cu` files.
    - [ ] Automatic detection of CUDA architecture and PTX compatibility.
    - [ ] Hot-reload or dynamic loading of kernels for rapid development.
    - [ ] Versioning and compatibility checks for kernels.
    - [ ] Unit tests for kernel compilation, loading, and error handling.
- [ ] **Kernel registry and management:**
    - [ ] Central registry for all available GPU kernels (by op, dtype, shape, etc.).
    - [ ] API for registering, querying, and launching kernels.
    - [ ] Support for multiple kernel variants per op (e.g., optimized for different shapes or dtypes).
    - [ ] Hooks for custom user kernels (plugin system).
    - [ ] Documentation and examples for kernel registration and usage.

### Step 3.B.2: GPU Kernel Launch Utilities & Dispatch
- [ ] **Unified kernel launch API:**
    - [ ] Macro or function (e.g., `launch_kernel!`) to abstract `cuLaunchKernel` calls.
    - [ ] Automatic argument packing, grid/block calculation, and error checking.
    - [ ] Support for stream-aware launches and shared memory configuration.
    - [ ] Dynamic dispatch based on tensor device, dtype, and shape.
    - [ ] Fallback to CPU implementation if GPU kernel is unavailable or fails.
- [ ] **Dynamic op dispatch system:**
    - [ ] Central dispatch for all tensor ops (unary, binary, reduction, etc.) based on device and dtype.
    - [ ] Easy extension for new ops and kernels.
    - [ ] Support for mixed-device and mixed-dtype operations (with clear promotion rules).
    - [ ] Unit and integration tests for dispatch logic.

### Step 3.B.3: Element-wise & Broadcasted Operations
- [ ] **Element-wise unary ops (e.g., neg, relu, exp, log):**
    - [ ] Custom CUDA kernels for F32, F64 (and later Int/Bool).
    - [ ] Stream-aware, support for non-contiguous and broadcasted tensors.
    - [ ] Validation against CPU reference.
    - [ ] Profiling and performance tests.
- [ ] **Element-wise binary ops (e.g., add, sub, mul, div, pow):**
    - [ ] Custom CUDA kernels with broadcasting support.
    - [ ] Support for all relevant dtypes and mixed-dtype promotion.
    - [ ] Stream-aware, efficient memory access patterns.
    - [ ] Validation and performance tests.
- [ ] **Scalar-tensor ops:**
    - [ ] Efficient kernels for tensor + scalar, scalar * tensor, etc.
    - [ ] Support for all dtypes.
    - [ ] Tests and benchmarks.

### Step 3.B.4: Matrix & Linear Algebra Operations
- [ ] **Matrix multiplication (matmul, gemm):**
    - [ ] Integration with cuBLAS for F32/F64 (and later Int/Bool if supported).
    - [ ] Stream-aware, handle row/column-major layouts.
    - [ ] Fallback to custom kernel if cuBLAS is unavailable.
    - [ ] Validation against CPU and PyTorch.
    - [ ] Performance benchmarks.
- [ ] **Batched matmul and advanced linalg ops:**
    - [ ] Support for batched matmul, transpose, etc.
    - [ ] Integration with cuBLAS and custom kernels.
    - [ ] Tests and documentation.

### Step 3.B.5: Reduction & Scan Operations
- [ ] **Reduction ops (sum, mean, max, min, prod, etc.):**
    - [ ] Custom CUDA kernels and/or Thrust integration.
    - [ ] Support for all dtypes, axes, and keepdim.
    - [ ] Stream-aware, efficient for large and small tensors.
    - [ ] Validation and performance tests.
- [ ] **Scan/Prefix-sum ops:**
    - [ ] Thrust or custom kernel integration.
    - [ ] Tests and benchmarks.

### Step 3.B.6: View, Indexing & Memory Ops
- [ ] **View ops (reshape, permute, slice, expand, squeeze, etc.):**
    - [ ] Metadata-only for most, but `contiguous` triggers a GPU copy kernel if needed.
    - [ ] Stream-aware, efficient memory layout handling.
    - [ ] Tests for all view ops on GPU tensors.
- [ ] **Indexing ops (index_select, masked_select, gather, scatter, etc.):**
    - [ ] Custom kernels for advanced indexing.
    - [ ] Support for all dtypes and broadcasted indices/masks.
    - [ ] Validation and performance tests.
- [ ] **Memory ops (copy, fill, zero, set):**
    - [ ] Efficient device-to-device and host-device copy kernels.
    - [ ] Stream-aware, support for async and sync copies.
    - [ ] Tests and benchmarks.

### Step 3.B.7: In-place & Mutating Operations
- [ ] **In-place element-wise ops (add_, mul_, etc.):**
    - [ ] Custom kernels for in-place modification.
    - [ ] Copy-on-write logic for autograd compatibility.
    - [ ] Stream-aware, robust error handling.
    - [ ] Tests for in-place ops and CoW behavior.
- [ ] **In-place view and indexing ops:**
    - [ ] In-place scatter, masked_fill_, etc.
    - [ ] Tests and validation.

### Step 3.B.8: Library Integration (cuBLAS, cuDNN, Thrust, etc.)
- [ ] **cuBLAS integration:**
    - [ ] Context and handle management, stream association.
    - [ ] Use for matmul, gemm, and other supported ops.
    - [ ] Fallback to custom kernels if needed.
- [ ] **cuDNN integration:**
    - [ ] Context and handle management, stream association.
    - [ ] Use for activations, convolutions, batchnorm, etc.
    - [ ] Fallback to custom kernels if needed.
- [ ] **Thrust integration:**
    - [ ] Use for reductions, scans, and other supported ops.
    - [ ] Fallback to custom kernels if needed.
- [ ] **Extensibility for future libraries (CUTLASS, etc.):**
    - [ ] Modular design to allow easy integration of new libraries.

### Step 3.B.9: Validation, Profiling & Debugging
- [ ] **Cross-validation with CPU and PyTorch:**
    - [ ] Automated tests comparing GPU and CPU results for all ops.
    - [ ] Optional comparison with PyTorch for reference.
- [ ] **Profiling and benchmarking:**
    - [ ] API and tools to measure kernel execution time, memory throughput, occupancy, etc.
    - [ ] Integration with CUDA events and streams for fine-grained profiling.
    - [ ] Benchmarks for all major ops (vs. CPU and PyTorch).
- [ ] **Debugging and error handling:**
    - [ ] Robust error propagation from kernels to Rust.
    - [ ] Logging, kernel launch validation, and debug hooks.
    - [ ] Tools for inspecting device memory, kernel parameters, etc.

### Step 3.B.10: Documentation, Examples & Extensibility
- [ ] **Comprehensive rustdoc for all GPU ops and kernel APIs.**
- [ ] **Examples for custom kernel registration, advanced usage, and performance tuning.**
- [ ] **Guides for adding new GPU ops/kernels, integrating new libraries, and extending the dispatch system.**
- [ ] **Unit, integration, and stress tests for all GPU ops.**

### **Key points to reinforce:**
- **Performance:** All kernels and dispatch must be optimized for throughput, memory access, and asynchrony.
- **Modularity:** Easy to add new ops, dtypes, and libraries; clear separation CPU/GPU; plugin system for custom kernels.
- **Robustness:** Fallbacks, error handling, validation, and extensive testing.
- **Extensibility:** Designed for future ops, dtypes, and hardware.
- **Profiling & Debug:** Built-in tools for performance analysis and debugging.

*   **Sub-Phase 3.C: Autograd, Device Management & End-to-End GPU Training (Revised & Enhanced)**

🎯 **Goal:**  
Ensure the autograd system, device management, and training loop are fully GPU-aware, robust, and modular, enabling high-performance, asynchronous, and multi-device training, with an architecture inspired by PyTorch and ready for future extensibility.

### Step 3.C.1: GPU-Aware Autograd Engine
- [ ] **Autograd support for GPU tensors:**
    - [ ] All backward ops must produce gradients on the correct device (CPU or GPU).
    - [ ] Backward ops for GPU must be stream-aware and support asynchronous execution.
    - [ ] Gradients for GPU tensors must be accumulated using GPU kernels (in-place, stream-aware).
    - [ ] Support for mixed-device computation graphs (automatic device propagation, error on invalid mixes).
    - [ ] Fallback to CPU autograd for unsupported GPU ops (with warning).
    - [ ] Unit and integration tests for all autograd paths (CPU, GPU, mixed).
- [ ] **Efficient memory management in autograd:**
    - [ ] Release intermediate GPU memory as soon as possible (lifetime tracking, hooks).
    - [ ] Option to keep or free the computation graph after backward (for higher-order gradients).
    - [ ] Profiling and debugging tools for autograd memory usage.
- [ ] **Hooks and extensibility:**
    - [ ] Support for backward hooks (per-tensor, per-op) for debugging, logging, or custom logic.
    - [ ] API for user-defined custom autograd functions (forward + backward, device-aware).
    - [ ] Documentation and examples for custom autograd ops.

### Step 3.C.2: Device Context & Placement Management
- [ ] **Device context manager:**
    - [ ] API to set the default device for tensor allocation within a scope (e.g., `with_device` or context manager).
    - [ ] Thread-local device context for safe multi-threaded usage.
    - [ ] Automatic device inference for new tensors (based on inputs, context, or explicit argument).
    - [ ] Unit tests for device context switching, nested scopes, and multi-threading.
- [ ] **Module and parameter device placement:**
    - [ ] `Module::to(device, stream)` to move all parameters and buffers to the target device (stream-aware).
    - [ ] Support for moving entire models between devices (CPU↔GPU, GPU↔GPU).
    - [ ] Validation of device consistency for all parameters/buffers.
    - [ ] Tests for module/device placement, including edge cases (partial move, shared params, etc.).
- [ ] **Multi-device and distributed support:**
    - [ ] API for model/data parallelism (scatter/gather, replication, etc.).
    - [ ] Hooks for distributed training (future extensibility).
    - [ ] Documentation and examples for multi-GPU usage.

### Step 3.C.3: Optimizer & State Management for GPU
- [ ] **GPU-aware optimizers:**
    - [ ] All optimizer state (momentum, Adam buffers, etc.) must be allocated on the same device as the parameters.
    - [ ] Optimizer steps for GPU parameters must use GPU kernels (in-place, stream-aware).
    - [ ] Support for mixed-device models (error or warning on invalid mixes).
    - [ ] Efficient state transfer when moving parameters between devices.
    - [ ] Unit and integration tests for optimizer/device interaction.
- [ ] **Learning rate schedulers and hooks:**
    - [ ] Ensure all schedulers are device-agnostic and work with GPU models.
    - [ ] Support for custom optimizer hooks (profiling, logging, etc.).
    - [ ] Documentation and examples.

### Step 3.C.4: End-to-End GPU Training Loop & Examples
- [ ] **Stream-aware, asynchronous training loop:**
    - [ ] All forward, backward, and optimizer steps should be able to run asynchronously on CUDA streams.
    - [ ] Expose APIs for explicit stream synchronization (e.g., `tensor.synchronize()`, `stream.synchronize()`).
    - [ ] Support for overlapping data transfer and computation (pipelining).
    - [ ] Option to enable/disable asynchrony for debugging.
- [ ] **Comprehensive training example:**
    - [ ] Example script for end-to-end training on GPU (MLP, CNN, etc.).
    - [ ] Demonstrate device placement, stream usage, and performance profiling.
    - [ ] Show correct loss decrease and parity with CPU training.
    - [ ] Example for multi-GPU training (if hardware available).
    - [ ] Add to documentation and test suite.
- [ ] **Validation and benchmarking:**
    - [ ] Automated tests to compare CPU and GPU training results (accuracy, speed).
    - [ ] Benchmarks for training speed, memory usage, and scaling (single/multi-GPU).
    - [ ] Profiling tools for bottleneck analysis.

### Step 3.C.5: Robustness, Profiling & Debugging
- [ ] **Error handling and diagnostics:**
    - [ ] Clear error messages for device mismatches, unsupported ops, and memory issues.
    - [ ] Debug hooks for autograd, optimizer, and device placement.
    - [ ] Tools to inspect computation graph, device placement, and memory usage.
- [ ] **Profiling and logging:**
    - [ ] Built-in support for timing, memory profiling, and operation logging (per op, per step).
    - [ ] Integration with CUDA events and streams for fine-grained profiling.
    - [ ] Examples and documentation for profiling tools.
- [ ] **Stress and fuzz testing:**
    - [ ] Intensive tests for autograd, optimizer, and training loop on GPU (random graphs, large models, etc.).
    - [ ] Bug/crash reporting and test methodology documentation.

### **Key points to reinforce:**
- **Performance & Asynchrony:** All training steps must leverage CUDA streams, overlap, and efficient memory management.
- **Modularity & Extensibility:** Device management, autograd, and optimizer logic must be backend-agnostic and easy to extend.
- **Robustness:** Error handling, validation, and extensive testing for all device and autograd paths.
- **Developer Experience:** Ergonomic APIs, clear error messages, profiling, and debugging tools.
- **Documentation & Examples:** Comprehensive guides, examples, and tests for all features.

*   **Sub-Phase 3.D: GPU Extensions & Robustness (Revised & Enhanced)**

🎯 **Goal:**  
Reinforce the robustness, flexibility, and extensibility of the CUDA backend, providing advanced error handling, profiling, multi-GPU support, interop, and monitoring tools, while making the system future-proof and developer-friendly.

### Step 3.D.1: Advanced CUDA Error Handling
- [ ] **Comprehensive CUDA error mapping:**
    - [ ] Map all CUDA error codes to `NeuraRustError` with explicit, actionable messages.
    - [ ] Include context (op, device, stream, kernel, etc.) in error reports.
    - [ ] Support for error categories (OOM, launch failure, device lost, etc.).
- [ ] **Error propagation and diagnostics:**
    - [ ] Ensure all CUDA errors are propagated up the stack and never silently ignored.
    - [ ] Add debug/verbose modes for detailed error logging (with stack traces, kernel params, etc.).
    - [ ] Provide suggestions or links to documentation for common errors.
- [ ] **Unit and integration tests:**
    - [ ] Simulate and test all common CUDA error scenarios (device unavailable, OOM, launch failure, etc.).
    - [ ] Ensure error messages are clear and actionable.
- [ ] **Documentation:**
    - [ ] Document all CUDA error types, their meaning, and recommended user actions.

### Step 3.D.2: GPU Profiling, Synchronization & Monitoring
- [ ] **Explicit synchronization APIs:**
    - [ ] `Tensor::synchronize()`, `CudaStream::synchronize()`, and event-based sync.
    - [ ] Support for fine-grained synchronization (per op, per stream, per event).
    - [ ] Tests for correct synchronization and deadlock prevention.
- [ ] **Profiling tools:**
    - [ ] Built-in API for measuring kernel execution time, memory usage, and occupancy.
    - [ ] Integration with CUDA events and streams for precise timing.
    - [ ] Support for memory usage and fragmentation monitoring (per device, per op).
    - [ ] Profiling hooks for user-defined callbacks.
    - [ ] Example scripts for profiling and performance analysis.
- [ ] **Monitoring and logging:**
    - [ ] Real-time logging of GPU memory usage, allocation/free events, and kernel launches.
    - [ ] API for querying current and peak memory usage, cache stats, and fragmentation.
    - [ ] Option to enable/disable logging and monitoring at runtime.
    - [ ] Documentation and examples for monitoring tools.

### Step 3.D.3: Multi-GPU (Multi-Device) Support
- [ ] **Multi-GPU memory and execution:**
    - [ ] Support allocation and execution of tensors on multiple GPUs in the same process.
    - [ ] APIs for device-to-device transfers, peer access, and collective operations.
    - [ ] Efficient scatter/gather, broadcast, and reduce primitives.
    - [ ] Device affinity and placement policies (manual and automatic).
- [ ] **Multi-GPU synchronization and communication:**
    - [ ] Support for CUDA peer-to-peer (P2P) memory access and synchronization.
    - [ ] APIs for synchronizing across devices and streams.
    - [ ] Tests for multi-GPU correctness and performance.
- [ ] **Distributed and future-proofing:**
    - [ ] Hooks for distributed training and communication (NCCL, MPI, etc.).
    - [ ] Documentation and examples for multi-GPU and distributed usage.

### Step 3.D.4: Advanced CUDA Streams & Events Management
- [ ] **Multiple stream management:**
    - [ ] APIs for creating, destroying, and managing multiple CUDA streams per device.
    - [ ] Stream priorities, dependencies, and scheduling policies.
    - [ ] Stream pools and reuse for efficient resource management.
- [ ] **CUDA event management:**
    - [ ] APIs for creating, recording, and synchronizing CUDA events.
    - [ ] Event-based profiling and synchronization.
    - [ ] Tests for event correctness and performance.
- [ ] **Asynchronous execution and overlap:**
    - [ ] Support for overlapping computation, data transfer, and kernel launches.
    - [ ] Examples and benchmarks for async execution patterns.

### Step 3.D.5: Interoperability with Other Frameworks
- [ ] **DLPack support:**
    - [ ] Implement import/export of CUDA buffers via DLPack.
    - [ ] APIs for zero-copy tensor exchange with PyTorch, TensorFlow, etc.
    - [ ] Tests for correctness and performance of DLPack interop.
- [ ] **Examples and documentation:**
    - [ ] Example scripts for exchanging tensors with PyTorch.
    - [ ] Document API, limitations, and best practices for interop.

### Step 3.D.6: Advanced GPU Memory Management
- [ ] **Memory monitoring and fragmentation tools:**
    - [ ] APIs for querying fragmentation, cache size, and manual memory release.
    - [ ] Support for cache clearing (`empty_cache`) and memory defragmentation.
    - [ ] Real-time memory usage stats and leak detection.
- [ ] **Manual and automatic memory management:**
    - [ ] APIs for manual cache management and memory release.
    - [ ] Support for memory pools, allocation limits, and OOM prevention.
    - [ ] Documentation and examples for advanced memory management.

### Step 3.D.7: Advanced DType Support on GPU
- [ ] **Support for int, bool, float16/bfloat16 DTypes on GPU:**
    - [ ] Implement kernels and memory management for all core DTypes.
    - [ ] Ensure CPU/GPU consistency and correct type promotion.
    - [ ] Tests for all DTypes, including mixed-type ops.
- [ ] **Future-proofing for new DTypes:**
    - [ ] Modular design to allow easy addition of new DTypes.
    - [ ] Documentation and examples.

### Step 3.D.8: GPU Robustness & Fuzz Testing
- [ ] **Stress and fuzz testing for CUDA kernels:**
    - [ ] Automated tests with random inputs, large tensors, and edge cases.
    - [ ] Memory stress tests, OOM scenarios, and error injection.
    - [ ] Bug/crash reporting and test methodology documentation.
- [ ] **Continuous integration for GPU tests:**
    - [ ] Integrate GPU tests into CI pipeline (with fallback for CPU-only environments).
    - [ ] Track and report GPU-specific bugs and regressions.

### Step 3.D.9: Documentation & GPU User Guides
- [ ] **Comprehensive GPU usage guide:**
    - [ ] Best practices, known limitations, and common pitfalls.
    - [ ] Debugging and troubleshooting guide for CUDA errors.
    - [ ] Links to external resources (NVIDIA docs, Rust CUDA ecosystem, etc.).
    - [ ] Integration of GPU docs into rustdoc and README.
- [ ] **Developer and contributor documentation:**
    - [ ] Internal architecture, how to add new ops/kernels, and testing guidelines.
    - [ ] Community resources and support channels.

### **Key points to reinforce:**
- **Robustness:** Advanced error handling, stress/fuzz testing, and monitoring.
- **Extensibility:** Multi-GPU, DLPack, new DTypes, and future hardware support.
- **Profiling & Debug:** Built-in tools for performance analysis, memory monitoring, and debugging.
- **Interoperability:** Seamless exchange with other frameworks and ecosystems.
- **Developer Experience:** Clear APIs, logs, documentation, and diagnostic tools.

*   **Sub-Phase 4.A: Advanced Layers & Architectures:**
    *   🎯 **Goal:** Implement key neural network layers for computer vision and sequence modeling, along with essential normalization layers, activation functions, and flexible initialization schemes.
    *   **Detailed Steps:**
        *   **Step 4.A.1: Convolutional Layers (`nn::Conv2d`, `nn::Conv1d`, `nn::Conv3d` - Start with `Conv2d`)**
            *   🎯 **Goal:** Implement 2D convolution, the cornerstone of many vision models.
            *   [ ] **`nn::Conv2d` Implementation:**
                *   [ ] Define `Conv2d` struct (`weight: Parameter`, `bias: Option<Parameter>`, `stride`, `padding`, `dilation`, `groups`).
                *   [ ] Implement `new()` with weight/bias initialization (e.g., Kaiming for weights).
                *   [ ] Implement `Module` trait (`forward` method).
                *   [ ] Core convolution logic: Use `im2col` + `matmul` approach for CPU initially. This can be a new op `im2col_op`.
                *   [ ] GPU Path: Utilize `cudnnConvolutionForward` and `cudnnConvolutionBackwardData/Filter/Bias` from cuDNN (requires cuDNN integration from Phase 3.B.9 to be solid).
                *   [ ] Handle `stride`, `padding` (manual padding or cuDNN padding modes), `dilation`, `groups`.
            *   [ ] Add tests for `Conv2d` (CPU and GPU if available): output shape, correctness of convolution (compare with known results), autograd (`check_grad`).
            *   [ ] Add `rustdoc` for `Conv2d` and its parameters.
            *   [ ] (Optional for this phase, can be 4.A.Ext) `nn::Conv1d` and `nn::Conv3d` following a similar pattern.
        *   **Step 4.A.2: Pooling Layers (`nn::MaxPool2d`, `nn::AvgPool2d`, `nn::AdaptiveAvgPool2d`)**
            *   🎯 **Goal:** Implement common pooling operations for downsampling feature maps.
            *   [ ] **`nn::MaxPool2d` Implementation:**
                *   [ ] Define `MaxPool2d` struct (`kernel_size`, `stride`, `padding`, `dilation`, `return_indices`).
                *   [ ] Implement `Module` trait (`forward`).
                *   [ ] CPU: Manual implementation by iterating through windows.
                *   [ ] GPU: Utilize `cudnnPoolingForward` and `cudnnPoolingBackward` from cuDNN.
            *   [ ] **`nn::AvgPool2d` Implementation:**
                *   [ ] Define `AvgPool2d` struct (`kernel_size`, `stride`, `padding`, `count_include_pad`).
                *   [ ] Implement `Module` trait (`forward`).
                *   [ ] CPU/GPU similar to `MaxPool2d`.
            *   [ ] **`nn::AdaptiveAvgPool2d` / `nn::AdaptiveMaxPool2d` Implementation:**
                *   [ ] Define struct with `output_size`.
                *   [ ] Calculate `stride` and `kernel_size` dynamically based on input and output size.
                *   [ ] Reuse `MaxPool2d`/`AvgPool2d` logic or implement specific kernels/cuDNN calls if more efficient.
            *   [ ] Add tests for each pooling layer (CPU/GPU): output shape, correctness, autograd.
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.3: Normalization Layers (`nn::BatchNorm1d/2d`, `nn::LayerNorm`)**
            *   🎯 **Goal:** Implement batch and layer normalization to stabilize training and improve convergence.
            *   [ ] **`nn::BatchNorm1d` / `nn::BatchNorm2d` Implementation:**
                *   [ ] Define struct (`num_features`, `eps`, `momentum`, `affine`, `track_running_stats`).
                *   [ ] `weight: Option<Parameter>`, `bias: Option<Parameter>` if `affine` is true.
                *   [ ] `running_mean: Tensor`, `running_var: Tensor` (persistent buffers, not parameters).
                *   [ ] Implement `Module` trait (`forward`). During training, update running stats and normalize using batch stats. During eval, use running stats.
                *   [ ] CPU: Manual implementation.
                *   [ ] GPU: Utilize `cudnnBatchNormalizationForwardTraining/Inference` and `cudnnBatchNormalizationBackward`.
            *   [ ] **`nn::LayerNorm` Implementation:**
                *   [ ] Define struct (`normalized_shape`, `eps`, `elementwise_affine`).
                *   [ ] `weight: Option<Parameter>`, `bias: Option<Parameter>` if `elementwise_affine` is true.
                *   [ ] Implement `Module` trait (`forward`). Normalize over the last D dimensions.
                *   [ ] CPU/GPU: Manual implementation or custom kernels. (cuDNN has limited LayerNorm support, might need custom kernel for full features/performance on GPU).
            *   [ ] Add tests for each normalization layer (CPU/GPU): train/eval mode, affine/non-affine, correctness of normalization and running stats, autograd.
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.4: Advanced Activation Functions (`nn::GELU`, `nn::SiLU`, `nn::Softmax`, `nn::LogSoftmax`)**
            *   🎯 **Goal:** Add more sophisticated activation functions used in modern architectures.
            *   [ ] **`nn::GELU` (Gaussian Error Linear Unit):**
                *   [ ] Implement as a functional op `ops::nn::gelu_op` and `Tensor::gelu()` method.
                *   [ ] Add CPU kernel and GPU kernel (custom or check cuDNN if available).
            *   [ ] **`nn::SiLU` (Sigmoid Linear Unit, also Swish):**
                *   [ ] Implement as `ops::nn::silu_op` and `Tensor::silu()` method. `x * sigmoid(x)`.
                *   [ ] CPU/GPU kernels.
            *   [ ] **`nn::Softmax` Module/Function:**
                *   [ ] Implement `ops::nn::softmax_op(input: &Tensor, dim: i64)` and `Tensor::softmax(dim: i64)`.
                *   [ ] Implement `nn::Softmax(dim: i64)` as a `Module`.
                *   [ ] CPU/GPU kernels (GPU can use `cudnnSoftmaxForward/Backward`).
            *   [ ] **`nn::LogSoftmax` Module/Function:**
                *   [ ] Similar to `Softmax`.
            *   [ ] Add tests for each activation (CPU/GPU): correctness, autograd.
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.5: Basic Recurrent Layers (`nn::RNN`, `nn::LSTM`, `nn::GRU` - Start with `RNN`)**
            *   🎯 **Goal:** Implement fundamental recurrent layers for sequence modeling.
            *   [ ] **`nn::RNN` Implementation:**
                *   [ ] Define `RNN` struct (`input_size`, `hidden_size`, `num_layers`, `nonlinearity` (tanh/relu), `bias`, `batch_first`, `dropout`, `bidirectional`).
                *   [ ] Parameters for each layer and direction (`weight_ih_l{k}`, `weight_hh_l{k}`, `bias_ih_l{k}`, `bias_hh_l{k}`).
                *   [ ] Implement `Module` trait (`forward(input, h_0)`). Returns `(output, h_n)`.
                *   [ ] CPU: Loop over sequence, apply matrix multiplications and activation for each time step and layer.
                *   [ ] GPU: Utilize `cudnnRNNForwardTraining/Inference` and `cudnnRNNBackwardData/Weights`. Requires setting up `cudnnRNNDescriptor`, `cudnnDropoutDescriptor`, etc.
            *   [ ] (Stretch Goal for this Step, else for 4.A.Ext) **`nn::LSTM` and `nn::GRU`**: Similar structure, more complex cell computations. cuDNN provides direct support.
            *   [ ] Add tests for `RNN` (CPU/GPU): output shapes (output, hidden state), correctness of one step, multi-step, multi-layer, autograd.
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.6: Dropout Layers (`nn::Dropout`, `nn::Dropout2d`)**
            *   🎯 **Goal:** Implement dropout for regularization.
            *   [ ] **`nn::Dropout` Implementation:**
                *   [ ] Define struct (`p`, `inplace`).
                *   [ ] Implement `Module` trait (`forward`). During training, randomly zero out elements with probability `p` and scale remaining by `1/(1-p)`. During eval, it's an identity op.
                *   [ ] CPU/GPU implementations.
            *   [ ] **`nn::Dropout2d` Implementation (Spatial Dropout):**
                *   [ ] Zeros out entire channels randomly.
            *   [ ] Add tests for dropout layers (CPU/GPU): train/eval mode, correct scaling, inplace behavior, autograd (grads only flow through non-zeroed elements).
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.7: `nn::Embedding` Layer**
            *   🎯 **Goal:** Implement embedding layer for representing categorical data.
            *   [ ] Define `Embedding` struct (`num_embeddings`, `embedding_dim`, `padding_idx`, `max_norm`, `norm_type`, `scale_grad_by_freq`, `sparse`).
            *   [ ] `weight: Parameter` (the embedding matrix).
            *   [ ] Implement `Module` trait (`forward(input: &Tensor<I64/I32>)`). Input is a tensor of indices.
            *   [ ] CPU/GPU: Essentially an `index_select` operation on the weight matrix (from Phase 2.C.6).
            *   [ ] Handle `padding_idx` (sets the embedding vector for this index to zeros).
            *   [ ] (Optional for this phase) `max_norm`, `scale_grad_by_freq`, `sparse` gradients (more advanced).
            *   [ ] Add tests (CPU/GPU): output shape, correctness of lookup, `padding_idx`, autograd.
            *   [ ] Add `rustdoc`.
        *   **Step 4.A.8: Create Vision Model Example (e.g., Simple CNN for MNIST-like data)**
            *   🎯 **Goal:** Demonstrate a complete vision model using Conv2d, Pooling, Activations, Linear layers.
            *   [ ] Create `examples/simple_cnn_example.rs`.
            *   [ ] Define a CNN model (e.g., `Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear`).
            *   [ ] Use `TensorDataset` and `DataLoader` (from Phase 2) with synthetic image-like data or a very simple dataset (e.g. manually created small images/labels).
            *   [ ] Train the model for a few epochs on CPU and GPU (if available), showing loss reduction.
            *   [ ] Add to `Cargo.toml` and document the example.
        *   **Step 4.A.9: (Optional) Create Basic Sequence Model Example (e.g., Simple RNN/LSTM)**
            *   🎯 **Goal:** Demonstrate a basic recurrent model.
            *   [ ] Create `examples/simple_rnn_example.rs`.
            *   [ ] Define a model using `nn::Embedding` and `nn::RNN` (or `LSTM`).
            *   [ ] Use synthetic sequence data.
            *   [ ] Train for a few epochs, show loss reduction.
            *   [ ] Add to `Cargo.toml` and document.
        *   **Step 4.A.10: Other Normalization Layers and NN Utilities**
            *   🎯 **Goal:** Add supplementary normalization layers and common utility modules.
            *   [ ] **`nn::InstanceNorm1d/2d/3d`**: CPU/GPU (cuDNN or custom).
            *   [ ] **`nn::GroupNorm`**: CPU/GPU (custom kernels).
            *   [ ] **`nn::Flatten(start_dim, end_dim)` Module**: Wraps `flatten` op.
            *   [ ] **`nn::Unflatten(dim, unflattened_size)` Module**: Inverse of Flatten.
            *   [ ] **`nn::Identity` Module**: Placeholder/passthrough module.
            *   [ ] Add tests and `rustdoc` for each.
        *   **Step 4.A.11: Flexible Weight Initialization for `nn::Module`s**
            *   🎯 **Goal:** Allow custom initialization of module parameters post-creation.
            *   [ ] Implement `Module::apply(&mut self, F)` where `F: FnMut(&mut dyn Module)`.
                *   [ ] Traverses module tree (self and children) and applies `F`.
            *   [ ] Demonstrate usage: Iterate module tree, check type (e.g., `is Linear`), access parameters, apply `nn::init` functions.
            *   [ ] Update examples (`SimpleCNN`, `SimpleRNN`) to use `apply` for initialization.
            *   [ ] Add tests and `rustdoc`.

*   **Sub-Phase 4.B: Advanced DType & Op Support:**
    *   🎯 **Goal:** Implement robust mixed-type operations, support for remaining common DTypes, and expand tensor manipulation capabilities with advanced selection/comparison.
    *   **Detailed Steps:**
        *   **Step 4.B.1: Robust Mixed-Type Operations (Numeric)**
            *   🎯 **Goal:** Allow operations between tensors of different numeric DTypes (e.g., F32 + F64, F32 + I32) with clear promotion rules.
            *   [ ] Define type promotion rules (e.g., `(F32, I64) -> F32`, `(F64, F32) -> F64`, `(I32, F32) -> F32`). Generally promote to the more expressive float type, or largest float type involved.
            *   [ ] Refactor arithmetic ops (`add_op`, `sub_op`, `mul_op`, `div_op`, `pow_op`) to handle mixed DType inputs.
                *   This might involve casting one of the inputs to the target DType before applying the kernel.
                *   Kernels should operate on a single DType.
            *   [ ] Extend to scalar operations (e.g., `Tensor<F32>::add_scalar(i64_scalar)`).
            *   [ ] Add comprehensive tests for various mixed-type combinations and scalar ops, on CPU and GPU.
            *   [ ] Update `rustdoc` for ops to specify mixed-type behavior.
        *   **Step 4.B.2: Implement Remaining Common Tensor Creation Functions**
            *   🎯 **Goal:** Add widely used tensor creation functions like `arange`, `linspace`, `eye` with DType flexibility.
            *   [ ] **`arange(start, end, step, dtype, device)`:** Create 1D tensor with values from `start` to `end` with `step`.
            *   [ ] **`linspace(start, end, steps, dtype, device)`:** Create 1D tensor with `steps` values evenly spaced between `start` and `end`.
            *   [ ] **`eye(n, m, dtype, device)`:** Create a 2D tensor with ones on the diagonal and zeros elsewhere (identity matrix if n=m).
            *   [ ] Ensure these functions support F32, F64, I32, I64 DTypes where appropriate.
            *   [ ] Add tests for each function with various DTypes, shapes, and parameters.
            *   [ ] Add `rustdoc`.
        *   **Step 4.B.3: Advanced Tensor Manipulation Operations**
            *   🎯 **Goal:** Implement more sophisticated tensor reshaping, splitting, and joining operations.
            *   [ ] **`Tensor::unbind(dim)`**: Returns a `Vec<Tensor>` by splitting along `dim`.
            *   [ ] **`Tensor::stack(tensors: &[&Tensor], dim)`**: Stacks a sequence of tensors along a new dimension `dim`.
            *   [ ] **`Tensor::chunk(chunks: usize, dim)`**: Splits a tensor into a specific number of chunks along `dim`.
            *   [ ] **`Tensor::split(split_size_or_sections: S, dim)`**: Splits tensor by size or at sections.
            *   [ ] **`Tensor::gather(dim: usize, index: &Tensor)`**: Gathers values along `dim` according to `index`.
            *   [ ] **`Tensor::scatter_(&mut self, dim: usize, index: &Tensor, src: &Tensor)` / `scatter_add_`**: In-place write/add values from `src` at `index` along `dim`.
            *   [ ] **`Tensor::index_put_(&mut self, indices: &[Option<&Tensor>], values: &Tensor)`**: Advanced in-place assignment (generalization of `masked_fill_` using tensor indices).
            *   For each, implement CPU and GPU (custom kernels or leveraging existing ops) versions where applicable. Ensure autograd support.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 4.B.4: (Optional) Support for Complex DTypes (`Complex32`, `Complex64`)**
            *   🎯 **Goal:** Introduce basic support for complex numbers if specific use cases emerge.
            *   [ ] Add `DType::Complex32`, `DType::Complex64`.
            *   [ ] Adapt `Buffer`/`TensorData`.
            *   [ ] Implement basic arithmetic ops for complex numbers.
            *   [ ] Low priority unless a strong need is identified.
        *   **Step 4.B.5: Create Advanced Tensor Operations Example**
            *   🎯 **Goal:** Demonstrate usage of mixed-type operations and new tensor manipulation functions.
            *   [ ] Create `examples/advanced_tensor_ops_example.rs`.
            *   [ ] Show mixed-type arithmetic.
            *   [ ] Demonstrate `arange`, `linspace`, `eye`.
            *   [ ] Showcase `unbind`, `stack`, `chunk`, `split`, `gather`, `scatter_`.
            *   [ ] Add to `Cargo.toml` and document.
        *   **Step 4.B.6: Advanced Selection and Comparison Operations**
            *   🎯 **Goal:** Add richer tensor selection and sorting capabilities.
            *   [ ] **`Tensor::topk(k, dim, largest, sorted)`**: CPU/GPU.
            *   [ ] **`Tensor::sort(dim, descending)`**: CPU/GPU.
            *   [ ] **`Tensor::unique(sorted, return_inverse, return_counts, dim)`**: CPU/GPU.
            *   [ ] Add tests, autograd considerations, and `rustdoc`.

*   **Sub-Phase 4.C: Interoperability & Model Persistence:**
    *   🎯 **Goal:** Enable model persistence via native serialization and interaction with the broader ML ecosystem (Python, ONNX), demonstrated through dedicated examples.
    *   **Detailed Steps:**
        *   **Step 4.C.0: Native Serialization/Deserialization for Models and Tensors**
            *   🎯 **Goal:** Implement saving and loading of `Module` state and `Tensor` data.
            *   [ ] Use `serde` with `bincode` or `postcard`.
            *   [ ] `Tensor`: Implement `Serialize`/`Deserialize` (data, shape, strides, dtype, requires_grad). Handle CPU/GPU data (GPU might need CPU transfer for serialization).
            *   [ ] `Module::state_dict() -> Result<HashMap<String, Tensor>>` (collects parameters and persistent buffers).
            *   [ ] `Module::load_state_dict(&mut self, state_dict: HashMap<String, Tensor>, strict: bool)`.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 4.C.1: Create Native Serialization Example**
            *   🎯 **Goal:** Demonstrate saving a trained model and reloading it.
            *   [ ] Create `examples/native_serialization_example.rs`.
            *   [ ] Train a simple model (e.g., `SimpleMLP` or `SimpleCNN` from earlier examples) for a few epochs.
            *   [ ] Save its `state_dict()` to a file.
            *   [ ] Create a new instance of the model and load the `state_dict()` from the file.
            *   [ ] Perform inference with the reloaded model to verify correctness.
            *   [ ] Add to `Cargo.toml` and document the example.
        *   **Step 4.C.2: Python Bindings with PyO3 (Enhanced API Exposure)**
            *   🎯 **Goal:** Expose a comprehensive set of `NeuraRust` functionalities to Python.
            *   [ ] Set up PyO3 build process.
            *   [ ] Expose `Tensor` (creation, ops, `to(device)`, `numpy()`).
            *   [ ] Expose `Parameter`.
            *   [ ] Expose `nn::Module` trait and implemented layers from 4.A (Linear, Conv2d, RNN, BatchNorm, etc.) to Python.
            *   [ ] Expose Optimizers (SGD, Adam) and LR Schedulers.
            *   [ ] Expose `Dataset`, `DataLoader`, and `Sampler` interfaces/implementations.
            *   [ ] (Stretch) Allow custom `nn::Module` definition in Python using NeuraRust components.
            *   [ ] Refine Python error handling (NeuraRustError to Python exceptions).
            *   [ ] Add `pytest` suite for Python bindings.
            *   [ ] Add `rustdoc` and Python API documentation.
        *   **Step 4.C.3: Create Python Bindings Example**
            *   🎯 **Goal:** Demonstrate building and training a model using NeuraRust from Python.
            *   [ ] Create a Python script/notebook in `examples/python_bindings_example/`.
            *   [ ] Use the exposed NeuraRust API in Python to:
                *   [ ] Create and manipulate tensors.
                *   [ ] Define a simple model (e.g., MLP using NeuraRust layers available in Python).
                *   [ ] Create synthetic data (NumPy arrays, converted to NeuraRust tensors).
                *   [ ] Use a NeuraRust optimizer.
                *   [ ] Run a short training loop.
                *   [ ] Print loss and verify basic functionality.
            *   [ ] Document setup and execution for the Python example.
        *   **Step 4.C.4: ONNX Export (Broader Model Support)**
            *   🎯 **Goal:** Allow exporting a wider range of trained models to ONNX format.
            *   [ ] Extend ONNX export to support new layers (Conv, Pool, BatchNorm, RNN if feasible) and ops from Phase 4.
            *   [ ] Improve graph traversal and op conversion robustness.
            *   [ ] Add tests for exporting more complex models (e.g., `SimpleCNN`).
            *   [ ] Add `rustdoc`.
        *   **Step 4.C.5: Create ONNX Export Example**
            *   🎯 **Goal:** Demonstrate exporting a NeuraRust model to ONNX.
            *   [ ] Create `examples/onnx_export_example.rs`.
            *   [ ] Load or train a model (e.g., `SimpleCNN` from 4.A.8 or the reloaded model from 4.C.1).
            *   [ ] Export the model to an `.onnx` file using the `export_onnx` functionality.
            *   [ ] Include a small Python script (or instructions) to load the exported `.onnx` file using `onnxruntime` and perform a sample inference to verify the export.
            *   [ ] Add to `Cargo.toml` and document the example.
        *   **Step 4.C.6: (Stretch Goal) ONNX Import (Initial Support)**
            *   🎯 **Goal:** Allow loading simple ONNX models into `NeuraRust`.
            *   [ ] Read ONNX file and parse graph structure and initializers.
            *   [ ] Convert ONNX nodes back to `NeuraRust` ops/layers.
            *   [ ] Highly complex, start with very limited op coverage.

*   **Phase 4 Notes:**
    *   *This phase significantly expands the usability of the framework for common ML tasks. Focus on CPU implementations first for new layers/ops, then GPU if cuDNN is not a direct fit or for custom logic.*
    *   *cuDNN integration is critical for performant Conv/RNN layers on GPU.*
    *   *Mixed-type operations require careful design of promotion rules and op dispatch logic.*
    *   *Native model persistence (4.C.0) is a key feature for practical use, validated by its own example (4.C.1).*
    *   *Python bindings (PyO3) will be a major enabler for adoption, with a dedicated example (4.C.3) showcasing its usage.*
    *   *ONNX support opens up interoperability, and an export example (4.C.5) will demonstrate this capability.*
    *   *Creating comprehensive examples (CNN - 4.A.8, RNN - 4.A.9, advanced tensor ops - 4.B.5, serialization - 4.C.1, Python - 4.C.3, ONNX - 4.C.5) is key to validating these complex features.*

## Phase 5: Deployment & Advanced Features
*   🎯 **Goal:** Target diverse deployment scenarios (WASM, native binaries, edge), implement advanced training and inference optimization techniques (quantization, pruning, distributed training, JIT compilation, AMP), explore further ecosystem integrations, and enhance developer tooling, validated by targeted examples and showcases.

*   **Sub-Phase 5.A: Deployment Strategies & Targets:**
    *   🎯 **Goal:** Enable running `NeuraRust` models in various environments beyond typical server-side Python, with examples for each target.
    *   **Detailed Steps:**
        *   **Step 5.A.1: WebAssembly (WASM) Compilation & Inference Example**
            *   🎯 **Goal:** Allow compiling models and inference logic to WASM for browser/Node.js execution.
            *   [ ] Investigate and choose a WASM compilation strategy for Rust code (e.g., `wasm-pack`, `wasm-bindgen`).
            *   [ ] Identify core `NeuraRust` components needed for inference-only mode (Tensor ops, Module forward pass, no autograd, no optimizers).
            *   [ ] Refactor or gate parts of `neurarust-core` to be WASM-compatible (e.g., remove dependencies not available in WASM, handle threading differences if any).
            *   [ ] Create a minimal inference API callable from JavaScript via WASM.
            *   [ ] Create a new example `examples/wasm_inference_example/`:
                *   [ ] Include Rust code for WASM module (loading model, inference function).
                *   [ ] Include HTML + JS to load a pre-trained `NeuraRust` model (serialized via 4.C.0, or ONNX if import is available) and perform inference in the browser on sample input.
            *   [ ] Add tests for WASM-compiled inference.
            *   [ ] Add `rustdoc` and documentation for WASM deployment.
        *   **Step 5.A.2: Native Binary Deployment Strategies & Example**
            *   🎯 **Goal:** Facilitate packaging `NeuraRust` applications as standalone native executables.
            *   [ ] Document best practices for building release-optimized, self-contained Rust binaries that embed models (e.g., using `include_bytes!` for model files or loading from disk).
            *   [ ] Create a new example `examples/native_cli_inference_example.rs`:
                *   [ ] A simple command-line application that loads a trained `NeuraRust` model.
                *   [ ] Performs inference on input data (e.g., image file path provided as arg, CSV data from stdin).
            *   [ ] Discuss considerations for cross-compilation to different OS/architectures.
            *   [ ] Add to `Cargo.toml` and document the example.
        *   **Step 5.A.3: Edge/Embedded Device Considerations (Initial Exploration & Example)**
            *   🎯 **Goal:** Investigate feasibility and demonstrate basic inference on resource-constrained edge devices.
            *   [ ] Research Rust cross-compilation toolchains for ARM (e.g., `arm-linux-gnueabihf`, `aarch64-linux-gnu`).
            *   [ ] Test compiling `neurarust-core` (inference-only subset) for an ARM target.
            *   [ ] Benchmark basic tensor operations on a target ARM device.
            *   [ ] Identify potential bottlenecks (memory, compute) and areas needing specific optimization for edge.
            *   [ ] Create a new example `examples/edge_inference_example_rpi/` (or similar target):
                *   [ ] Minimal Rust code for inference with a very small model.
                *   [ ] Instructions for cross-compiling and running on the target device (e.g., Raspberry Pi).
            *   [ ] Document findings and potential pathways for edge deployment.

*   **Sub-Phase 5.B: Advanced Training & Inference Optimizations:**
    *   🎯 **Goal:** Implement techniques to improve training speed/memory and inference performance/size, with examples demonstrating their use.
    *   **Detailed Steps:**
        *   **Step 5.B.1: Gradient Accumulation**
            *   🎯 **Goal:** Allow simulating larger batch sizes by accumulating gradients over multiple mini-batches.
            *   [ ] Modify the training loop logic (or provide helpers/optimizer wrappers) to perform `loss.backward()` multiple times before `optimizer.step()`.
            *   [ ] Gradients should accumulate in `Parameter.grad` tensors.
            *   [ ] `optimizer.zero_grad()` should only be called after `optimizer.step()`.
            *   [ ] Create a new example `examples/gradient_accumulation_example.rs` or update an existing advanced training example to demonstrate this technique.
            *   [ ] Add tests and `rustdoc`.
        *   **Step 5.B.2: Gradient Checkpointing (Activation Checkpointing)**
            *   🎯 **Goal:** Reduce memory usage during training by recomputing activations in the backward pass instead of storing them all.
            *   [ ] Design an API for users to specify which parts of their model should use gradient checkpointing (e.g., a wrapper module `CheckpointModule(Module)` or a functional API `checkpoint(function, inputs)`).
            *   [ ] When checkpointing is active for a segment:
                *   During forward: Run the segment's forward pass but discard intermediate activations (only store input to the segment).
                *   During backward: When gradients arrive at the segment, re-run its forward pass (with `requires_grad=true` for its inputs) to get activations, then run its backward pass.
            *   [ ] Requires careful handling of autograd graph and detaching/re-attaching parts of it.
            *   [ ] Create a new example `examples/gradient_checkpointing_example.rs` demonstrating its use on a model that would otherwise consume significant memory.
            *   [ ] Add tests verifying memory reduction (if feasible to test directly) and correctness of gradients.
            *   [ ] Add `rustdoc`.
        *   **Step 5.B.3: Quantization (Post-Training Quantization - PTQ - Initial)**
            *   🎯 **Goal:** Reduce model size and potentially speed up inference by converting weights and activations to lower precision (e.g., INT8).
            *   [ ] Research and select a simple PTQ strategy (e.g., min-max quantization, per-tensor or per-channel).
            *   [ ] Implement functions to calibrate a trained Float32 model: run inference on a calibration dataset to determine quantization ranges (min/max values for weights and activations).
            *   [ ] Implement functions to quantize model weights to INT8 (or other target bitwidth).
            *   [ ] Develop INT8 kernels for key inference operations (e.g., `matmul_op`, `conv_op` - might need specialized libraries or careful custom implementation as direct INT8 arithmetic is different from float).
            *   [ ] Provide a way to run an inference graph with quantized weights and activations (may require specific `DType::I8` ops and casting).
            *   [ ] Create a new example `examples/quantization_ptq_example.rs` demonstrating PTQ on a simple trained model and comparing accuracy/performance with the FP32 version.
            *   [ ] Add tests and `rustdoc`. This is a large and complex area; initial support might be limited.
        *   **Step 5.B.4: (Exploratory) Pruning (Magnitude Pruning - Initial)**
            *   🎯 **Goal:** Reduce model size by removing (setting to zero) weights with small magnitudes.
            *   [ ] Implement a utility function to prune a `Module`'s parameters: iterate parameters, identify weights below a threshold, and set them to zero.
            *   [ ] Discuss strategies for fine-tuning after pruning to recover accuracy.
            *   [ ] Create a new example `examples/pruning_example.rs` demonstrating pruning on a trained model and its effect on sparsity/accuracy.
            *   [ ] This is exploratory; full support for sparse tensors and sparse kernels is a much larger effort.
        *   **Step 5.B.5: (Exploratory) Distributed Training (Data Parallel - Conceptual Outline)**
            *   🎯 **Goal:** Outline the conceptual requirements and challenges for multi-GPU/multi-node data parallel training.
            *   [ ] Research common distributed training frameworks (e.g., PyTorch `DistributedDataParallel`, Horovod).
            *   [ ] Identify key components needed:
                *   Process group management (NCCL for NVIDIA GPUs, Gloo for CPU/cross-platform).
                *   Gradient synchronization (e.g., all-reduce operation on gradients).
                *   Model replication across devices/nodes.
                *   Distributed samplers for `DataLoader`.
            *   [ ] This step is primarily for research and design documentation in Phase 5. Actual implementation is a very large undertaking and might be a separate major version/extension.
            *   [ ] Document conceptual design and challenges.
        *   **Step 5.B.6: (Exploratory) Inference Graph Compilation (JIT-like)**
            *   🎯 **Goal:** Investigate techniques for compiling parts of the computation graph for optimized inference.
            *   [ ] Research existing Rust JIT compilation libraries or graph optimization frameworks (e.g., LLVM-based, or simpler graph rewriting).
            *   [ ] Define a mechanism to "trace" or capture the computation graph from a `Module`'s `forward` pass (si non déjà fait pour ONNX, sinon l'étendre).
            *   [ ] Implement basic graph optimization passes:
                *   [ ] Operator fusion (e.g., conv + bias + relu).
                *   [ ] Constant folding.
            *   [ ] (Très avancé) Génération de code optimisé pour CPU (e.g., via LLVM) ou pour GPU (PTX, si non couvert par kernels existants).
            *   [ ] Create an example comparing inference speed of a model with and without graph compilation.
            *   [ ] This is highly exploratory and complex.
        *   **Step 5.B.7: Full Mixed Precision Training (AMP) Support for GPU**
            *   🎯 **Goal:** Provide robust and easy-to-use automated mixed precision training capabilities.
            *   [ ] Implement `GradScaler` object for dynamic loss scaling to prevent underflow of FP16 gradients.
            *   [ ] Implement an `autocast` context manager or similar API:
                *   [ ] Automatically casts inputs of selected ops (e.g., matmul, conv) to FP16/BF16.
                *   [ ] Ensures other ops (e.g., reductions, losses) run in FP32 for stability.
            *   [ ] Ensure optimizers can handle scaled gradients and unscale them before parameter updates.
            *   [ ] Update GPU kernels (or ensure cuBLAS/cuDNN calls) to correctly support FP16/BF16 where beneficial (requires F16/BF16 DType support from Phase 3.C.6 to be mature).
            *   [ ] Create an example demonstrating AMP training on a model like `SimpleCNN` or `ResNet-like` (from 5.C.4) and showing speedup/memory reduction.
            *   [ ] Add tests and comprehensive `rustdoc`.

*   **Sub-Phase 5.C: Ecosystem & Usability Enhancements:**
    *   🎯 **Goal:** Further improve developer experience, community engagement, and integration possibilities, showcased through comprehensive examples and basic tooling.
    *   **Detailed Steps:**
        *   **Step 5.C.1: Enhanced Model Hub / Pre-trained Model Access (Conceptual)**
            *   🎯 **Goal:** Design a system for easily sharing and using pre-trained `NeuraRust` models.
            *   [ ] Define a manifest format for model metadata (architecture, weights file, pre-processing info, license).
            *   [ ] Conceptualize a CLI tool or API for downloading/listing models from a central (or distributed) repository.
            *   [ ] Consider integration with native serialization (4.C.0) for model weight storage.
            *   [ ] This is primarily design; implementation might be a community effort or later phase.
        *   **Step 5.C.2: Advanced Python Bindings (More PyTorch Parity)**
            *   🎯 **Goal:** Achieve closer API parity with PyTorch for commonly used Python functionalities.
            *   [ ] Based on user feedback and common PyTorch patterns, identify more `Tensor` methods or `nn` utilities to expose via PyO3.
            *   [ ] Improve conversion between NeuraRust Tensors and NumPy arrays (zero-copy if possible for CPU tensors).
            *   [ ] Enhance support for custom `nn.Module` defined in Python that can be part of a larger NeuraRust graph (if feasible with PyO3 and autograd).
        *   **Step 5.C.3: Community Building & Contribution Guidelines**
            *   🎯 **Goal:** Foster a community and make it easier for others to contribute.
            *   [ ] Create comprehensive `CONTRIBUTING.md` guidelines (code style, testing, PR process).
            *   [ ] Improve developer documentation (internal architecture, how to add new ops/layers).
            *   [ ] Set up forums/channels for discussion (e.g., GitHub Discussions, Discord).
            *   [ ] Identify and label good first issues for new contributors.
        *   **Step 5.C.4: Create Comprehensive End-to-End Project Examples**
            *   🎯 **Goal:** Showcase `NeuraRust` capabilities with more complex, real-world (or near real-world) examples.
            *   [ ] Create a new example `examples/image_classification_cifar10.rs` (or similar standard dataset):
                *   [ ] Implement a ResNet-like or VGG-like architecture.
                *   [ ] Demonstrate data loading for image datasets (may require basic image loading/processing ops or a small helper crate - see 5.C.6).
                *   [ ] Full training loop with GPU support, optimizers, LR schedulers, AMP (if available).
                *   [ ] Evaluation metrics (accuracy).
                *   [ ] (Optional) Provide pre-trained weights using native serialization.
            *   [ ] Create a new example `examples/text_sentiment_analysis_imdb.rs` (or similar standard dataset):
                *   [ ] Use `nn::Embedding`, RNN/LSTM layers.
                *   [ ] Demonstrate handling of sequence data, padding, batching for text.
                *   [ ] Full training loop and evaluation.
            *   [ ] These examples would integrate many features from Phases 1-5 and serve as key showcases.
            *   [ ] Add to `Cargo.toml` and document thoroughly.
        *   **Step 5.C.5: (Exploratory) Basic Visualization Utilities**
            *   🎯 **Goal:** Provide simple tools for inspecting tensors and model graphs.
            *   [ ] **Tensor Summaries**: Function `tensor_summary(tensor: &Tensor)` to print shape, dtype, device, min, max, mean, std.
            *   [ ] **Graph Visualization**: Utility to output autograd graph or `Module` structure to DOT format for Graphviz.
            *   [ ] Create an example showcasing these utilities with a simple model.
            *   [ ] Add `rustdoc`.
        *   **Step 5.C.6: (Exploratory) Integration with Rust Data Augmentation/Image Processing Libraries**
            *   🎯 **Goal:** Facilitate creating `Dataset`s that use existing Rust crates for data preprocessing, demonstrated by a specific example.
            *   [ ] Research Rust image processing (e.g., `image` crate) and generic data transform crates.
            *   [ ] Develop `Dataset` wrappers or helper functions showing how to integrate these crates for on-the-fly image augmentation (random crop, flip, rotate, color jitter) or other data transforms.
            *   [ ] Create a new example `examples/data_augmentation_example.rs` that uses such wrappers/helpers with a simple dataset to demonstrate the augmentation pipeline.
            *   [ ] Update comprehensive examples (e.g., CIFAR-10 in 5.C.4) to use these augmentation capabilities if applicable and stable.
            *   [ ] Focus on interoperability and demonstrating patterns.
        *   **Step 5.C.7: (Exploratory) User-Defined Differentiable Operations API**
            *   🎯 **Goal:** Investigate mechanisms for users to define custom operations with associated backward passes without modifying NeuraRust core.
            *   [ ] Define a Rust trait (e.g., `CustomAutogradFunction`) with `forward` and `backward` methods that users can implement.
            *   [ ] Explore how such functions could be registered and used within the autograd graph.
            *   [ ] (Très avancé) Investigate exposing this capability to Python via PyO3.
            *   [ ] Document design ideas and challenges.
        *   **Step 5.C.8: (Exploratory) Integrated Profiling Tools**
            *   🎯 **Goal:** Provide basic tools for profiling NeuraRust model execution.
            *   [ ] Design hooks or callbacks within op execution and autograd to record timing and memory usage.
            *   [ ] Develop a utility to aggregate and display profiling data (e.g., time per op, device, memory peak on CPU/GPU).
            *   [ ] Create an example demonstrating how to use the profiler with a training or inference workload.
            *   [ ] Focus on CPU and GPU time for ops initially; detailed memory profiling and host-device transfer profiling are more complex.
            *   [ ] Document design and usage.

*   **Phase 5 Notes:**
    *   *Deployment targets (WASM, native, edge) open up new use cases beyond Python-centric server environments, each validated by an example.*
    *   *Advanced optimizations (gradient accumulation, checkpointing, quantization, pruning, JIT, AMP) are key for SOTA model training and efficient inference, with examples for each demonstrating their impact.*
    *   *Distributed training is a massive undertaking, Phase 5 focuses on conceptualization and prerequisites.*
    *   *Ecosystem and usability (model hub, better Python bindings, community, basic tooling) are crucial for long-term adoption and growth.*
    *   *Comprehensive end-to-end examples (5.C.4) are critical for demonstrating the framework's maturity and capabilities on realistic tasks, potentially leveraging data augmentation (5.C.6) demonstrated in its own example.*

## Phase 6: Frontier Innovations & Rust-Native Advantages
*   🎯 **Goal:** Leverage Rust's unique strengths and explore frontier ML research to establish `NeuraRust` as a leader in specific areas, offering distinct advantages over existing frameworks like PyTorch in terms of safety, performance predictability, and novel capabilities. This phase assumes a mature and feature-rich NeuraRust from Phases 1-5.

*   **Sub-Phase 6.A: Next-Generation Performance & Safety (Rust-Leveraged)**
    *   🎯 **Goal:** Push the boundaries of performance and safety using advanced Rust features, memory management strategies, and sophisticated debugging tools.
    *   **Detailed Steps:**
        *   **Step 6.A.1: Shaped Tensors for Compile-Time Shape Safety (Research & Prototyping)**
            *   🎯 **Goal:** Investigate and prototype an API for tensors whose shapes are (partially or fully) tracked by the Rust type system.
            *   [ ] Research advanced const generics, GATs, and trait systems for encoding tensor dimensions and axes.
            *   [ ] Develop a proof-of-concept API for a subset of operations with compile-time shape checking.
            *   [ ] Evaluate performance implications and ergonomic trade-offs.
            *   [ ] Document findings and a potential path to broader integration (likely a major breaking change or an optional API).
            *   *Potential Added Value vs PyTorch: Significant reduction in runtime shape errors, increased code robustness, safer refactoring.*
        *   **Step 6.A.2: Advanced Arena/Region-Based Memory Management for Tensors**
            *   🎯 **Goal:** Implement highly optimized memory allocation strategies for critical performance paths.
            *   [ ] Design an API for creating "computation scopes" or "memory arenas" where intermediate tensors are allocated.
            *   [ ] Implement arena allocators that can be reset efficiently after a scope (e.g., after a forward/backward pass for activations not needed long-term).
            *   [ ] Integrate this with the existing caching allocator (Phase 3.A) for a tiered memory strategy.
            *   [ ] Create examples and benchmarks demonstrating performance gains in latency-sensitive or memory-constrained scenarios.
            *   *Potential Added Value vs PyTorch: More predictable latency, reduced memory fragmentation, potentially lower overall memory footprint in specific use cases.*
        *   **Step 6.A.3: Fearless Concurrency for CPU Kernels using Advanced `rayon` Patterns**
            *   🎯 **Goal:** Maximize CPU utilization through highly optimized parallel execution of compute kernels.
            *   [ ] Profile existing CPU kernels and identify areas for further parallelization using `rayon`'s advanced features (custom `ParallelIterator`, `join_context`, etc.).
            *   [ ] Explore auto-parallelization strategies for user-defined operations or model graphs on CPU.
            *   [ ] Ensure thread-pool management is optimal.
            *   *Potential Added Value vs PyTorch: Potentially more efficient and easier-to-maintain CPU parallelism due to Rust's safety guarantees.*
        *   **Step 6.A.4: Integrated Profiling Tools**
            *   🎯 **Goal:** Provide basic tools for profiling NeuraRust model execution.
            *   [ ] Design hooks or callbacks within op execution and autograd to record timing and memory usage.
            *   [ ] Develop a utility to aggregate and display profiling data (e.g., time per op, device, memory peak on CPU/GPU).
            *   [ ] Create an example demonstrating how to use the profiler with a training or inference workload.
            *   [ ] Focus on CPU and GPU time for ops initially; detailed memory profiling and host-device transfer profiling are more complex.
            *   [ ] Document design and usage.
        *   **Step 6.A.5: Advanced Autograd and GPU Debugging Tools**
            *   🎯 **Goal:** Develop sophisticated tools for diagnosing issues in autograd and GPU execution.
            *   [ ] **Gradient Flow Visualization**: Extend graph visualization (from 5.C.5) to show gradient magnitudes and flow, helping identify vanishing/exploding gradients.
            *   [ ] **NaN/Inf Tracker**: Implement hooks in ops/autograd to detect and report the origin of NaNs or Infs in tensors.
            *   [ ] **Detailed GPU Memory Inspector**: Tools to query and visualize GPU memory allocation patterns, fragmentation (if using custom allocator), and occupancy beyond basic profiler stats.
            *   [ ] Create examples demonstrating how to use these debugging tools to diagnose common training problems.
            *   [ ] Add `rustdoc`.

*   **Sub-Phase 6.B: Innovative Modeling & Autograd Capabilities**
    *   🎯 **Goal:** Introduce novel features for model definition, differentiation, specialized data representations, and advanced quantization techniques that go beyond current mainstream framework offerings.
    *   **Detailed Steps:**
        *   **Step 6.B.1: Static/Compilable Autograd Backend (Optional & Exploratory)**
            *   🎯 **Goal:** Provide an optional backend for autograd that compiles the backward pass for specific model architectures.
            *   [ ] Extend JIT/Graph compilation work (from 5.B.6) to automatically derive and compile the backward pass from a statically defined forward graph.
            *   [ ] This could involve symbolic differentiation or advanced program transformation techniques.
            *   [ ] Benchmark against the dynamic autograd system for performance on supported models.
            *   *Potential Added Value vs PyTorch: Potentially faster backward passes for static graph models, reduced overhead compared to dynamic tape-based autograd.*
        *   **Step 6.B.2: Enhanced Support for Higher-Order Differentiation**
            *   🎯 **Goal:** Make computation of second-order (or higher) derivatives more ergonomic and efficient.
            *   [ ] Review and optimize the autograd engine for efficient computation of Hessians, Jacobians, etc.
            *   [ ] Provide clear APIs for `grad(grad(y, x1), x2)` or similar patterns.
            *   [ ] Develop examples for applications like meta-learning or physics-informed neural networks that require higher-order derivatives.
            *   *Potential Added Value vs PyTorch: While PyTorch supports it, Rust's performance characteristics might offer an edge if the autograd engine is specifically optimized for this.*
        *   **Step 6.B.3: Native Support for Sparse Tensors and Operations**
            *   🎯 **Goal:** Fully integrate sparse tensors as a first-class citizen for memory and compute efficiency in relevant domains.
            *   [ ] Define sparse tensor data structures (e.g., COO, CSR, CSC formats).
            *   [ ] Implement core sparse operations: sparse-dense matrix multiplication, sparse-sparse operations, element-wise ops involving sparse tensors.
            *   [ ] Develop autograd support for sparse operations.
            *   [ ] Provide CPU and GPU implementations (e.g., using cuSPARSE for CUDA).
            *   [ ] Create examples for GNNs or NLP tasks leveraging sparse tensors.
            *   *Significance: Crucial for graph neural networks, some NLP tasks, and models with massive embedding tables or inherently sparse structures.*
        *   **Step 6.B.4: User-Defined Differentiable Operations API**
            *   🎯 **Goal:** Investigate mechanisms for users to define custom operations with associated backward passes without modifying NeuraRust core.
            *   [ ] Define a Rust trait (e.g., `CustomAutogradFunction`) with `forward` and `backward` methods that users can implement.
            *   [ ] Explore how such functions could be registered and used within the autograd graph.
            *   [ ] (Très avancé) Investigate exposing this capability to Python via PyO3.
            *   [ ] Document design ideas and challenges.
        *   **Step 6.B.5: Comprehensive Quantization Aware Training (QAT) Support**
            *   🎯 **Goal:** Implement full support for QAT to produce highly optimized quantized models.
            *   [ ] Define "Fake Quantization" ops (e.g., `FakeQuantizeAffine`, `FakeQuantizePerChannel`) that simulate quantization effects during training (forward and backward passes).
            *   [ ] Develop a workflow/API to insert these fake quantization ops into a model (e.g., `prepare_qat(model)`).
            *   [ ] Implement a function `convert_qat(model)` to transform a QAT-trained model (with fake quant ops) into a truly quantized model using INT8 (or other bitwidth) ops for inference.
            *   [ ] Ensure INT8 kernels (from Phase 5.B.3) are robust and cover ops typically used in QAT-prepared models (Conv, Linear, etc.).
            *   [ ] Create a new example `examples/quantization_qat_example.rs` showing the QAT process on a model and comparing its accuracy/performance to FP32 and PTQ versions.
            *   [ ] Add tests for fake quant ops and the QAT conversion process.
            *   [ ] Add `rustdoc`.

*   **Sub-Phase 6.C: Next-Generation Interoperability, Deployment & Ecosystem**
    *   🎯 **Goal:** Position `NeuraRust` for emerging deployment targets, deeper integration with a multi-language/multi-backend world, foster a rich ecosystem, and explore frontier hardware paradigms.
    *   **Detailed Steps:**
        *   **Step 6.C.1: MLIR-Based Compilation Backend (Exploratory to Full Integration)**
            *   🎯 **Goal:** Utilize MLIR (Multi-Level Intermediate Representation) for targeting a wide array of hardware backends and enabling advanced graph optimizations.
            *   [ ] Deepen integration with MLIR (maturing JIT from 5.B.6):
                *   [ ] Convert NeuraRust computation graphs to MLIR dialects (e.g., TOSA, Linalg).
                *   [ ] Leverage MLIR's optimization passes (operator fusion, constant folding, layout optimization, dead code elimination).
                *   [ ] Use MLIR to compile to LLVM for CPUs (with SIMD), SPIR-V for Vulkan (potential cross-vendor GPU), and other vendor-specific backends.
            *   [ ] This could eventually unify or replace parts of the backend-specific (Phase 3) efforts with a more general solution.
            *   *Potential Added Value vs PyTorch: Superior hardware portability, access to a vibrant compiler research community, potentially more aggressive cross-operator optimizations. Significance: Can significantly boost inference and potentially training performance.*
        *   **Step 6.C.2: Advanced Support for Privacy-Preserving ML (Federated Learning, Homomorphic Encryption Hooks)**
            *   🎯 **Goal:** Provide foundational tools and abstractions to facilitate research and development in privacy-preserving ML.
            *   [ ] Implement primitives for secure aggregation (relevant to Federated Learning).
            *   [ ] Explore and provide hooks for integrating with Rust-based homomorphic encryption libraries.
            *   [ ] Develop examples showcasing how NeuraRust could be used in these contexts (e.g., basic federated averaging, inference on encrypted data with a compatible HE scheme).
            *   *Potential Added Value vs PyTorch: Rust's safety and performance make it an attractive language for security-sensitive ML. NeuraRust could become a go-to framework for this niche.*
        *   **Step 6.C.3: Formalized Multi-Backend Accelerator API**
            *   🎯 **Goal:** Mature the backend abstraction (from Phase 3 notes) into a stable API allowing third-party accelerator backends (e.g. ROCm, Metal).
            *   [ ] Define a clear trait-based system for `Device`, `Buffer`, `KernelLauncher`, `OpDispatcher` for different accelerators.
            *   [ ] Provide reference implementations for CPU and CUDA.
            *   [ ] Document how to add a new hardware backend.
            *   *Significance: Increases portability and accessibility of NeuraRust across different hardware platforms. Potential Added Value: A more community-driven and potentially more straightforward way to add new hardware support.*
        *   **Step 6.C.4: Higher-Level Training Loop Abstractions**
            *   🎯 **Goal:** Provide optional utility frameworks to simplify common training, validation, and testing workflows.
            *   [ ] Define traits for `Callback`s (for logging, early stopping, model checkpointing, LR scheduling integration).
            *   [ ] Design a `Trainer` object that encapsulates the training loop, optimizer, dataloaders, and device management.
            *   [ ] Handle distributed training setup (if Phase 5.B.5 progresses to implementation) within the Trainer.
            *   [ ] Ensure flexibility for users to customize components or use the core API directly.
            *   *Significance: Reduces boilerplate, promotes best practices, and makes complex training setups more accessible. Similar to PyTorch Lightning or Hugging Face Accelerate.*
        *   **Step 6.C.5: Enhanced ONNX Interoperability (Broader OpSet Support, Import Robustness)**
            *   🎯 **Goal:** Achieve more comprehensive ONNX export and import capabilities.
            *   [ ] Support a wider range of ONNX opsets for export.
            *   [ ] Improve robustness of ONNX import, handling more complex graph structures and ops.
            *   *Significance: Critical for interoperability with the wider ML ecosystem and leveraging pre-trained models.*
        *   **Step 6.C.6: Richer Ecosystem of Pre-trained Models & Domain-Specific Libraries**
            *   🎯 **Goal:** Foster the development of a NeuraRust model zoo and libraries for specific domains (vision, text, audio).
            *   [ ] Mature the Model Hub concept (from 5.C.1) into a functional platform.
            *   [ ] Encourage community contributions of pre-trained models in NeuraRust format.
            *   [ ] Develop or support the creation of libraries similar to `torchvision`, `torchaudio`, `torchtext` for NeuraRust.
            *   *Significance: Lowers the barrier to entry for new users and enables rapid application development.*
        *   **Step 6.C.7: (Exploratory) Integration with Emerging Hardware Paradigms**
            *   🎯 **Goal:** Investigate and prototype initial integrations with non-von Neumann or highly specialized AI accelerators.
            *   [ ] Research Rust APIs or FFI capabilities for interacting with neuromorphic computing simulators/emulators (e.g., Loihi, SpiNNaker if accessible).
            *   [ ] Explore interfaces for probabilistic computing hardware/frameworks if Rust bindings become available.
            *   [ ] Develop a proof-of-concept example: a very simple NeuraRust model or computation offloaded to such a simulator/emulator, demonstrating data exchange and control flow.
            *   [ ] Document challenges, potential, and limitations for these integrations.
            *   *Potential Added Value vs PyTorch: Early mover advantage in Rust for next-gen AI hardware, leveraging Rust's suitability for systems-level programming.*

*   **Phase 6 Notes:**
    *   *This phase is highly ambitious and focuses on areas where Rust's unique properties can offer significant advantages or where NeuraRust can pioneer novel ML framework features.*
    *   *Many steps are research-oriented initially and may require significant breakthroughs or community contributions to fully realize.*
    *   *Success in this phase would position NeuraRust not just as an alternative, but as an innovator in certain niches of the ML landscape.*
    *   *Performance, safety, and next-generation deployment are key themes.*
    *   *Close collaboration with the Rust academic and research community would be beneficial for topics like Shaped Tensors and MLIR integration.*
    *   *Advanced debugging tools (6.A.5), full QAT support (6.B.5), and exploration of emerging hardware (6.C.7) solidify NeuraRust's position at the cutting edge.*