#[macro_export]
macro_rules! assert_relative_eq {
    ($left:expr, $right:expr, epsilon = $epsilon:expr) => {
        {
            let left_val = $left;
            let right_val = $right;
            let eps = $epsilon;
            if (left_val - right_val).abs() > eps {
                panic!(
                    "assertion failed: `(left == right)` (within epsilon {LEFT_SUB_RIGHT_ABS} > {EPS})\n  left: `{:?}`\n right: `{:?}`",
                    left_val,
                    right_val,
                    LEFT_SUB_RIGHT_ABS = (left_val - right_val).abs(),
                    EPS = eps
                );
            }
        }
    };
    ($left:expr, $right:expr) => {
        assert_relative_eq!($left, $right, epsilon = 1e-9_f64);
    };
} 