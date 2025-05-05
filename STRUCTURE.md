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
│   │   │   ├── layers
│   │   │   │   ├── linear.rs
│   │   │   │   └── mod.rs
│   │   │   ├── losses
│   │   │   │   ├── mod.rs
│   │   │   │   └── mse.rs
│   │   │   ├── mod.rs
│   │   │   ├── module.rs
│   │   │   └── parameter.rs
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
│   │   │   │   └── sum_test.rs
│   │   │   └── view
│   │   │       ├── expand.rs
│   │   │       ├── mod.rs
│   │   │       ├── permute.rs
│   │   │       ├── reshape.rs
│   │   │       ├── reshape_test.rs
│   │   │       ├── slice.rs
│   │   │       ├── transpose.rs
│   │   │       ├── transpose_test.rs
│   │   │       └── utils.rs
│   │   ├── optim
│   │   ├── tensor
│   │   │   ├── autograd_methods.rs
│   │   │   ├── broadcast_utils.rs
│   │   │   ├── create.rs
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs
│   │   │   ├── utils.rs
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
└── STRUCTURE.md

20 directories, 85 files