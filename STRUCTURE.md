.
├── Cargo.lock
├── Cargo.toml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Goals copy.md
├── Goals.md
├── LICENSE
├── neurarust-core
│   ├── Cargo.toml
│   ├── examples
│   │   ├── advanced_training_techniques.rs
│   │   ├── basic_mlp_cpu_inplace_optim.rs
│   │   ├── basic_mlp_cpu_manual_inplace.rs
│   │   ├── basic_mlp_cpu_param_groups.rs
│   │   └── basic_mlp_cpu.rs
│   ├── src
│   │   ├── autograd
│   │   │   ├── backward_op.rs
│   │   │   ├── grad_check.rs
│   │   │   ├── graph.rs
│   │   │   └── mod.rs
│   │   ├── buffer.rs
│   │   ├── device.rs
│   │   ├── error.rs
│   │   ├── lib.rs
│   │   ├── model
│   │   │   ├── mod.rs
│   │   │   └── sequential.rs
│   │   ├── nn
│   │   │   ├── init.rs
│   │   │   ├── init_test.rs
│   │   │   ├── layers
│   │   │   │   ├── linear.rs
│   │   │   │   ├── linear_test.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── relu.rs
│   │   │   ├── losses
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mse.rs
│   │   │   │   └── mse_test.rs
│   │   │   ├── mod.rs
│   │   │   ├── module.rs
│   │   │   ├── parameter.rs
│   │   │   └── parameter_test.rs
│   │   ├── ops
│   │   │   ├── activation
│   │   │   │   ├── mod.rs
│   │   │   │   ├── relu.rs
│   │   │   │   └── relu_test.rs
│   │   │   ├── arithmetic
│   │   │   │   ├── add.rs
│   │   │   │   ├── add_test.rs
│   │   │   │   ├── div.rs
│   │   │   │   ├── div_test.rs
│   │   │   │   ├── max_elemwise.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mul.rs
│   │   │   │   ├── mul_test.rs
│   │   │   │   ├── neg.rs
│   │   │   │   ├── neg_test.rs
│   │   │   │   ├── pow.rs
│   │   │   │   ├── pow_test.rs
│   │   │   │   ├── sub.rs
│   │   │   │   └── sub_test.rs
│   │   │   ├── comparison
│   │   │   │   ├── equal.rs
│   │   │   │   ├── equal_test.rs
│   │   │   │   ├── ge.rs
│   │   │   │   └── mod.rs
│   │   │   ├── dtype
│   │   │   │   ├── cast.rs
│   │   │   │   └── mod.rs
│   │   │   ├── linalg
│   │   │   │   ├── matmul.rs
│   │   │   │   ├── matmul_test.rs
│   │   │   │   └── mod.rs
│   │   │   ├── loss
│   │   │   │   └── mod.rs
│   │   │   ├── math_elem
│   │   │   │   ├── ln.rs
│   │   │   │   ├── ln_test.rs
│   │   │   │   └── mod.rs
│   │   │   ├── mod.rs
│   │   │   ├── reduction
│   │   │   │   ├── max.rs
│   │   │   │   ├── max_test.rs
│   │   │   │   ├── mean.rs
│   │   │   │   ├── mean_test.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── sum.rs
│   │   │   │   ├── sum_test.rs
│   │   │   │   └── utils.rs
│   │   │   ├── traits
│   │   │   │   └── numeric.rs
│   │   │   ├── traits.rs
│   │   │   └── view
│   │   │       ├── contiguous.rs
│   │   │       ├── expand.rs
│   │   │       ├── expand_test.rs
│   │   │       ├── mod.rs
│   │   │       ├── permute.rs
│   │   │       ├── permute_test.rs
│   │   │       ├── reshape.rs
│   │   │       ├── reshape_test.rs
│   │   │       ├── slice.rs
│   │   │       ├── slice_test.rs
│   │   │       ├── squeeze_unsqueeze.rs
│   │   │       ├── transpose.rs
│   │   │       ├── transpose_test.rs
│   │   │       └── utils.rs
│   │   ├── optim
│   │   │   ├── adagrad.rs
│   │   │   ├── adagrad_test.rs
│   │   │   ├── adam.rs
│   │   │   ├── adam_test.rs
│   │   │   ├── grad_clipping.rs
│   │   │   ├── grad_clipping_test.rs
│   │   │   ├── lr_scheduler.rs
│   │   │   ├── lr_scheduler_tests.rs
│   │   │   ├── mod.rs
│   │   │   ├── optimizer_state.rs
│   │   │   ├── optimizer_trait.rs
│   │   │   ├── param_group.rs
│   │   │   ├── rmsprop.rs
│   │   │   ├── rmsprop_test.rs
│   │   │   ├── sgd.rs
│   │   │   └── sgd_test.rs
│   │   ├── tensor
│   │   │   ├── accessors.rs
│   │   │   ├── autograd_methods.rs
│   │   │   ├── autograd.rs
│   │   │   ├── broadcast_utils.rs
│   │   │   ├── create.rs
│   │   │   ├── create_test.rs
│   │   │   ├── debug.rs
│   │   │   ├── inplace_arithmetic_methods.rs
│   │   │   ├── inplace_ops
│   │   │   │   ├── add.rs
│   │   │   │   ├── add_scalar.rs
│   │   │   │   ├── clamp.rs
│   │   │   │   ├── div.rs
│   │   │   │   ├── div_scalar.rs
│   │   │   │   ├── fill.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mul.rs
│   │   │   │   ├── mul_scalar.rs
│   │   │   │   ├── pow.rs
│   │   │   │   ├── sub.rs
│   │   │   │   └── sub_scalar.rs
│   │   │   ├── inplace_ops_tests
│   │   │   │   ├── add_scalar_test.rs
│   │   │   │   ├── add_test.rs
│   │   │   │   ├── clamp_test.rs
│   │   │   │   ├── div_scalar_test.rs
│   │   │   │   ├── div_test.rs
│   │   │   │   ├── fill_test.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mul_scalar_test.rs
│   │   │   │   ├── mul_test.rs
│   │   │   │   ├── pow_test.rs
│   │   │   │   ├── sub_scalar_test.rs
│   │   │   │   └── sub_test.rs
│   │   │   ├── iter_utils.rs
│   │   │   ├── mod.rs
│   │   │   ├── reduction_methods.rs
│   │   │   ├── traits.rs
│   │   │   ├── utils.rs
│   │   │   ├── utils_test.rs
│   │   │   ├── view_methods.rs
│   │   │   └── view_methods_test.rs
│   │   ├── tensor_data.rs
│   │   ├── test_utils.rs
│   │   ├── types.rs
│   │   ├── utils
│   │   │   └── testing.rs
│   │   └── utils.rs
│   └── tests
│       ├── common.rs
│       ├── tensor_basic_ops.rs
│       ├── tensor_creation.rs
│       ├── tensor_utils.rs
│       └── tensor_view_ops.rs
├── README.md
├── ROADMAP.md
├── STRUCTURE.md
└── text.txt

26 directories, 161 files