.
├── Cargo.lock
├── Cargo.toml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── examples
│   ├── basic_mlp_cpu_inplace_optim.rs
│   └── basic_mlp_cpu.rs
├── Goals copy.md
├── Goals.md
├── LICENSE
├── neurarust-core
│   ├── Cargo.toml
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
│   │   ├── nn
│   │   │   ├── init.rs
│   │   │   ├── init_test.rs
│   │   │   ├── layers
│   │   │   │   ├── linear.rs
│   │   │   │   ├── linear_test.rs
│   │   │   │   └── mod.rs
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
│   │   │       ├── transpose.rs
│   │   │       ├── transpose_test.rs
│   │   │       └── utils.rs
│   │   ├── optim
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
│   │   │   │   ├── div.rs
│   │   │   │   ├── div_scalar.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mul.rs
│   │   │   │   ├── mul_scalar.rs
│   │   │   │   ├── pow.rs
│   │   │   │   ├── sub.rs
│   │   │   │   └── sub_scalar.rs
│   │   │   ├── inplace_ops_tests
│   │   │   │   ├── add_scalar_test.rs
│   │   │   │   ├── add_test.rs
│   │   │   │   ├── div_scalar_test.rs
│   │   │   │   ├── div_test.rs
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
│   │   │   └── view_methods.rs
│   │   ├── tensor_data.rs
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

24 directories, 128 files