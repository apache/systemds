# OOC Covariance and TSMM Benchmark Plan

## Scope

Benchmarks compare local CP execution against local OOC execution only.

## Operators

- cov(A, B)
- cov(A, B, W)
- t(X) %*% X
- X %*% t(X)

## Benchmark Variables

- Dense and sparse inputs
- Weighted and unweighted covariance
- TSMM LEFT: t(X) %*% X
- TSMM RIGHT: X %*% t(X)
- Single-tile and multi-tile TSMM outputs
- Block sizes: 500, 1000, and 2000 where feasible
- Warm-up runs before measured runs

## Initial Matrix Plan

| Operator | Case | Matrix size | Sparsity | Block size | Comparison |
|---|---:|---:|---:|---:|---|
| cov(A,B) | dense | 10000 x 1 | 0.9 | 1000 | CP vs OOC |
| cov(A,B) | sparse | 10000 x 1 | 0.1 | 1000 | CP vs OOC |
| cov(A,B,W) | dense | 10000 x 1 | 0.9 | 1000 | CP vs OOC |
| cov(A,B,W) | sparse | 10000 x 1 | 0.1 | 1000 | CP vs OOC |
| t(X)%*%X | LEFT single-tile | 10000 x 100 | 0.9 | 1000 | CP vs OOC |
| t(X)%*%X | LEFT multi-tile | 10000 x 3000 | 0.2 | 1000 | CP vs OOC |
| X%*%t(X) | RIGHT single-tile | 100 x 10000 | 0.9 | 1000 | CP vs OOC |
| X%*%t(X) | RIGHT multi-tile | 3000 x 100 | 0.2 | 1000 | CP vs OOC |

## Measurement Plan

- 1 or 2 warm-up runs
- 3 measured runs
- Report average runtime
- Report matrix dimensions, sparsity, block size, execution mode, and operator
- Verify correctness against CP output before interpreting runtime
