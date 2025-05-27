# PASS 1 – Top-Level Blueprint
| Milestone                  | Outcome                                                               | Why first?                                          |
| -------------------------- | --------------------------------------------------------------------- | --------------------------------------------------- |
| **M0 – Array Core**        | 4-D const-generic ndarray, views, RNG, thread-pool, CPU SIMD          | Everything else builds on it                        |
| **M1 – Linear Algebra**    | GEMM, LU/QR/Chol, serialisation                                       | Needed for analytical ML solvers                    |
| **M2 – Classical ML v1**   | Linear/Ridge/Lasso + optimisers, DataLoader, Trainer                  | Validates autograd + optimiser loop on small models |
| **M3 – Tree Models**       | Decision Tree ➜ RF ➜ XGBoost                                          | Non-gradient path, stress-test Dataset+Loader       |
| **M4 – DL Core**           | Dynamic autograd, `Module`, `network!` macro, MLP/CNN                 | Foundation for all DL variants                      |
| **M5 – Seq & Attention**   | RNNs, Transformer, tokenisers, embeddings, LoRA                       | Unlocks modern NLP + LoRA adapters                  |
| **M6 – Generative**        | GAN, VAE, Diffusion                                                   | Uses Trainer, schedulers, mixed precision           |
| **M7 – GPU**               | CUDA backend, GPU array + GEMM kernels                                | Accelerates all previous compute                    |
| **M8 – RAG**               | Tokenizer + Embedder + VectorIndex + Retriever + Generator + pipeline | Combines NLP & Vector search                        |
| **M9 – Release Hardening** | Docs, benchmarks, CI badges                                           | Ship-ready                                          |

# PASS 2 – Iterative Chunk-Level Roadmap
| Milestone                  | Chunk                         | Goal / Deliverable                                      |
| -------------------------- | ----------------------------- | ------------------------------------------------------- |
| **M0 – Array Core**        | **C1** Box Buffer & Alignment | `AlignedBox` (32-byte) + unit/bench tests               |
|                            | **C2** Array1 & Indexing      | safe/unsafe indexing, length, perf tests                |
|                            | **C3** `tensor![]` Macro      | compile-time nested literals to rank ≤ 4                |
|                            | **C4** Views (Shape + Stride) | zero-copy transpose, Python-style slice                 |
|                            | **C5** Broadcast Engine       | compile-time resolver + `add_into`                      |
|                            | **C6** Bitset Masking         | compact mask, `mask_select`, `mask_view`, `mask_assign` |
|                            | **C7** ThreadPool             | global pool + `scope()`; parallel reductions            |
|                            | **C8** PRNG & Initialisers    | PCG, uniform/normal, Xavier/He helpers                  |
| **M1 – Linear Algebra**    | **C1** Blocked GEMM           | SIMD + threaded `matmul_into`                           |
|                            | **C2** Partial-Pivot LU       | in-place factor + pivot indices                         |
|                            | **C3** Householder QR         | no-pivot in-place QR                                    |
|                            | **C4** Cholesky (Lower)       | `chol` in-place lower-tri factor                        |
|                            | **C5** Row→Col Copy           | allocate new contiguous col-major array                 |
|                            | **C6** Binary Serialiser      | `.nmr` read/write arrays                                |
| **M2 – Classical ML v1**   | **C1** Analytical Solver      | normal-equation (Cholesky) linear/ridge                 |
|                            | **C2** SGD Solver             | LR enum, early-stop, partial\_fit                       |
|                            | **C3** Coordinate Descent     | Lasso-optimised CD                                      |
|                            | **C4** Optimiser Suite        | SGD, Adam/AdamW, RMSProp, AdaGrad, LBFGS                |
|                            | **C5** Dataset + DataLoader   | lazy, shuffle, prefetch (num\_workers)                  |
|                            | **C6** Trainer v1             | metrics (console, CSV opt-in), checkpointing            |
| **M3 – Tree Models**       | **C1** Decision Tree          | CART classifier/regressor                               |
|                            | **C2** Random Forest          | bagging + feature sub-sample                            |
|                            | **C3** XGBoost                | gradient-boosted trees                                  |
|                            | **C4** Docs & Bench           | docs, unit tests, speed benchmarks                      |
| **M4 – DL Core**           | **C1** Dynamic Autograd       | tape, `.backward()`                                     |
|                            | **C2** `Module` Trait         | owns params, `forward()`                                |
|                            | **C3** Base Layers            | Linear, Conv2d, activation ops                          |
|                            | **C4** `network!{}` Macro     | anonymous model builder                                 |
|                            | **C5** Regularisers           | Dropout, BatchNorm (Mode flag)                          |
|                            | **C6** Param Groups           | optimiser grouping A/B/C                                |
| **M5 – Seq & Attention**   | **C1** Tokeniser (BPE/WP)     | vocab build + encode/decode                             |
|                            | **C2** Embedding Layers       | learned + positional                                    |
|                            | **C3** RNN (GRU/LSTM)         | time-step unroll, batching                              |
|                            | **C4** Transformer            | encoder/decoder, multi-head attn                        |
|                            | **C5** LoRA                   | auto + manual wrappers                                  |
| **M6 – Generative**        | **C1** VAE                    | reparam trick, KL loss                                  |
|                            | **C2** GAN                    | DCGAN baseline                                          |
|                            | **C3** Diffusion (DDPM)       | scheduler utilities                                     |
|                            | **C4** Mixed Precision        | fp16/bf16 training toggle                               |
| **M7 – GPU Acceleration**  | **C1** `GpuContext` Scope     | host/device scope guard                                 |
|                            | **C2** Device Buffers         | persistent caching                                      |
|                            | **C3** CUDA GEMM              | cuBLAS-like kernel                                      |
|                            | **C4** Broadcast/Reduce       | element-wise & reductions kernels                       |
|                            | **C5** GPU Autodiff           | kernel registry in tape                                 |
| **M8 – RAG**               | **C1** VectorIndex (Exact)    | brute-force k-NN                                        |
|                            | **C2** VectorIndex (HNSW)     | approximate k-NN                                        |
|                            | **C3** Retriever              | top-k, score filter                                     |
|                            | **C4** Generator              | seq-to-seq transformer w/ beam                          |
|                            | **C5** RAG Pipeline           | end-to-end `generate(query)`                            |
|                            | **C6** Component API          | embedder, retriever, etc.                               |
| **M9 – Release Hardening** | **C1** Docs                   | mdBook + docs.rs                                        |
|                            | **C2** Benchmarks             | criterion, comparison table                             |
|                            | **C3** CI Matrix              | Linux/macOS/Win, stable/nightly                         |
|                            | **C4** Crates.io Publish      | 1.0-alpha release                                       |

# PASS 3 – Micro-Step Backlog (Milestone M0 Only)
Pattern: S0.x where x = sequential micro-step number.
| Step      | Description                                                                                                  | Acceptance Test / Safety   |
| --------- | ------------------------------------------------------------------------------------------------------------ | -------------------------- |
| **S0.1**  | Create Cargo workspace `numeru` with member crate `numeru_core`; add nightly toolchain & CI (`cargo check`). | CI green                   |
| **S0.2**  | Implement `AlignedBox<T, N>` newtype with 32-byte alignment via `#[repr(align(32))]`.                        | Unit test pointer %32 == 0 |
| **S0.3**  | Add unit test `aligned_box.rs` validating alignment & length.                                                | `cargo test` passes        |
| **S0.4**  | Add Criterion bench `alloc.rs` comparing `AlignedBox` vs `Box`.                                              | Bench compiles             |
| **S0.5**  | Implement `Array1<T, N>` storing `AlignedBox`.                                                               | `len()==N`                 |
| **S0.6**  | Safe `Index` & `IndexMut` with bounds check panic.                                                           | OOB test panics            |
| **S0.7**  | Add `unsafe fn get_unchecked()`; run under Miri.                                                             | Miri clean                 |
| **S0.8**  | Configure release build to elide bounds via `cfg!(debug_assertions)`.                                        | Perf unit test faster      |
| **S0.9**  | Implement `tensor![]` macro for 1-D literals.                                                                | Compile macro test         |
| **S0.10** | Extend macro to infer rank ≤ 4 (nested arrays).                                                              | Shape tests                |
| **S0.11** | Compile-fail tests for inconsistent shapes.                                                                  | `trybuild` tests           |
| **S0.12** | Add `ArrayView<'a, T, R>` & `ArrayViewMut`.                                                                  | Lifetime tests             |
| **S0.13** | Implement zero-copy `transpose()`.                                                                           | Stride check               |
| **S0.14** | Python-style `slice![]` macro incl. neg indices & steps.                                                     | Slice tests                |
| **S0.15** | Validate stride math via property tests (`proptest`).                                                        | All pass                   |
| **S0.16** | Broadcast-shape const-generic resolver.                                                                      | Compile-time assert        |
| **S0.17** | Element-wise `add_into(&mut dst, &a, &b)`.                                                                   | Numeric test               |
| **S0.18** | Panic on broadcast mismatch.                                                                                 | Unit test                  |
| **S0.19** | Implement `BitMask<const N: usize>` (compact bitset).                                                        | Bit ops tests              |
| **S0.20** | `mask_select` alloc copy 1-D array.                                                                          | Size check                 |
| **S0.21** | `mask_view` iterator view (zero alloc).                                                                      | Iterator tests             |
| **S0.22** | `mask_assign(&mask, src)` length-checked.                                                                    | Panic test                 |
| **S0.23** | Masking unit + property tests.                                                                               | Pass                       |
| **S0.24** | Implement global `ThreadPool` + `scope`.                                                                     | Parallel sum test          |
| **S0.25** | Parallel reductions for `sum`, `mean`, etc.                                                                  | Compare serial             |
| **S0.26** | Stress test ThreadPool (2–8 threads).                                                                        | No data race               |
| **S0.27** | PCG-64 per-thread PRNG impl.                                                                                 | Statistical test           |
| **S0.28** | `set_global_seed` + auto-seed fallback.                                                                      | Repro test                 |
| **S0.29** | `rand_uniform_into`, `rand_normal_into`.                                                                     | Mean/var assert            |
| **S0.30** | Xavier/He initialiser helpers (shape-inferred).                                                              | Var check                  |
| **S0.31** | RNG reproducibility & independence tests.                                                                    | Pass                       |
| **S0.32** | Optional benchmark RNG throughput vs `rand` crate.                                                           | Bench compiles             |
| **S1.1** | C1 GEMM | Add `Array2<T, R, C>` type alias (`Array<T, RowMajor, R, C>`) and ctor `zeros()` | `assert_eq!(a.shape(), (R,C))` |
| **S1.2** | C1 GEMM | Implement **naïve** `matmul_into_naive(&mut dst, &a, &b)` (no blocking)          | Unit test random 4×4 multiply  |
| **S1.3** | C1 GEMM | Shape-compatibility guard → panic on mismatch                                    | Mismatch test panics           |
| **S1.4** | C1 GEMM | Add **tiling parameters** consts (`BLOCK_M/N/K = 64`)                            | Compile passes                 |
| **S1.5** | C1 GEMM | Implement blocked kernel (still scalar inner loop)                               | Numerical equality vs naïve    |
| **S1.6** | C1 GEMM | Integrate `std::simd` 256-bit f64 vector into inner loop                         | Perf unit test ≥ 2× naïve      |
| **S1.7** | C1 GEMM | Wrap blocked kernel with `ThreadPool::install`; parallel over tiles              | Stress test 1–8 threads        |
| **S1.8** | C1 GEMM | Criterion bench: naïve vs blocked vs blocked+SIMD+threads                        | Bench compiles                 |
| **S1.9** | C1 GEMM | Public re-export `matmul_into` (blocked fast path) & docs                        | `cargo doc` shows fn           |
| **S1.10** | C2 LU | Add helper `swap_rows(&mut Array2, i, j)` | Unit swap test |
| **S1.11** | C2 LU | Implement partial-pivot in-place `lu_decompose(a: &mut Array2, piv: &mut [usize])` | Pivot indices len == rows |
| **S1.12** | C2 LU | Detect zero pivot → panic "Singular" | Singular matrix test panics |
| **S1.13** | C2 LU | Provide `lu_solve(&lu, &piv, b: &Array1, out: &mut Array1)` | Reconstruct Ax=b test |
| **S1.14** | C2 LU | Verify residual ∥Ax-b∥/∥b∥ ≤ 1e-10 for random 64×64 | Unit test |
| **S1.15** | C2 LU | Criterion bench: LU solve vs naïve Gaussian elimination | Bench compiles |
| **S1.16** | C2 LU | Add docs + example in README fragment | cargo doc link OK |
| **S1.17** | C3 QR | Implement `householder_reflect(v: &mut [f64])` -> tau utility | Unit: reflector zeros lower part |
| **S1.18** | C3 QR | In-place `qr_decompose(a: &mut Array2, tau: &mut Array1)` (no pivot) | Reassemble Q*R error <1e-10 |
| **S1.19** | C3 QR | Function `form_q(a: &Array2, tau, out:&mut Array2) explicit Q` | Orthogonality QᵀQ≈I test |
| **S1.20** | C3 QR | Provide `qr_solve(&qr,&tau,b,out)` using back-sub | Solve residual test |
| **S1.21** | C3 QR | Bench QR vs LU for least-squares 256×64 | Bench compiles |
| **S1.22** | C3 QR | Add doc examples | Docs pass |
| **S1.23** | C4 Chol | Implement `chol_lower(a:&mut Array2)` with PD check | A_recon = L Lᵀ residual |
| **S1.24** | C4 Chol | Panic "Not PD" if diag ≤ 0 | Non-PD matrix test panics |
| **S1.25** | C4 Chol | `chol_solve(&chol, b, out)` | Solve test residual |
| **S1.26** | C4 Chol | Bench Cholesky vs LU for SPD 512×512 | Bench compiles |
| **S1.27** | C4 Chol | Docs + example | Docs OK |
| **S1.28** | C5 Row→Col | Implement `to_col_major(&self) -> ColArray2` (alloc copy) | Verify col strides contiguous |
| **S1.29** | C5 Row→Col | FFI safety: round-trip row→col→row equals orig | Unit test |
| **S1.30** | C5 Row→Col | Bench copy speed vs transpose view | Bench compiles |
| **S1.31** | C6 Ser | Define `.nmr` header: magic NMR1, u8 rank, u8 endian flag | Spec file in docs |
| **S1.32** | C6 Ser | Implement `write_nmr(&array, writer)` (row-major only) | Write then read equals orig |
| **S1.33** | C6 Ser | Implement `read_nmr(reader) -> ArrayDyn` | Unit round-trip various sizes |
| **S1.34** | C6 Ser | Support col-major arrays (detect flag + copy) | Unit test |
| **S1.35** | C6 Ser | Fuzz test with random sizes < 1 MiB (cargo-fuzz) | Fuzz run passes |
| **S1.36** | C6 Ser | Docs & README snippet with example save/load | Docs OK |
| Step     | Chunk       | Description (≤ 2 h)                                                                     | Acceptance / Test                                      |
| -------- | ----------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **S2.1** | C1 Analytic | Add blank `ml` module + `Estimator` trait `{ fit, predict }` (generic over `Array2/1`). | Crate compiles.                                        |
| **S2.2** | C1 Analytic | Define `LinearRegression` struct `{ weights: Array1 }`.                                 | `new()` returns struct.                                |
| **S2.3** | C1 Analytic | Implement `fit()` using normal-equation via existing `chol_lower` solver.               | Fit on tiny 3×2 data, MSE ≈ 0.                         |
| **S2.4** | C1 Analytic | Implement `predict()` (matrix–vector mult).                                             | Unit: ŷ dims and values correct.                      |
| **S2.5** | C1 Analytic | `RidgeRegression { lambda }` adds `λI` regulariser.                                     | Residual smaller than Linear for ill-conditioned demo. |
| **S2.6** | C1 Analytic | `LassoRegression` placeholder struct (will call CD later).                              | Compiles with TODO.                                    |
| **S2.7** | C1 Analytic | Add helper `design_matrix_bias(X)` for bias column.                                     | Test bias col of ones.                                 |
| **S2.8** | C1 Analytic | Doc examples in rust-doc.                                                               | `cargo doc` shows examples.                            |
| **S2.9** | C1 Analytic | Criterion bench: analytic Linear vs SGD (placeholder).                                  | Bench compiles.                                        |
| **S2.10** | C2 SGD | Create `SgdParams { lr: LRPolicy, epochs, tol } enum & struct`. | Compile. |
| **S2.11** | C2 SGD | Implement `LRPolicy enum: Constant, InvSqrt, Cosine`. | Unit: value at t=0,1. |
| **S2.12** | C2 SGD | Integrate `fit_sgd()` for LinearRegression (batch). | Converges on 10×3 dataset (MSE<1e-4). |
| **S2.13** | C2 SGD | Add `partial_fit()` (single mini-batch update). | Weight diff ≠ 0 after call. |
| **S2.14** | C2 SGD | Early-stopping on tol against previous epoch loss. | Stops < max_epochs in test. |
| **S2.15** | C2 SGD | Support shuffle bool (Fisher–Yates). | Epoch loss differs when shuffle toggle. |
| **S2.16** | C2 SGD | Integrate `ThreadPool` for mini-batch gradient calc. | Speed test faster multi-thread. |
| **S2.17** | C2 SGD | Expose solver field in model defaulting to Auto. | Compile; default = analytic if n_features<1k else SGD. |
| **S2.18** | C2 SGD | Unit tests analytic vs SGD predictions (close). | R² diff < 1e-3. |
| **S2.19** | C2 SGD | Docs & README snippet. | Docs OK. |
| **S2.20** | C3 CD | Implement soft-threshold util `soft_thresh(x, λ)` (no alloc). | Unit tests. |
| **S2.21** | C3 CD | Implement coordinate-descent `fit_cd()` for LassoRegression. | L1 error small on sparse toy set. |
| **S2.22** | C3 CD | Add max_iter, tol params; stop when max |Δw| < tol. | Converges ≤ 1000 iters in test. |
| **S2.23** | C3 CD | Warm-start from existing weights if present. | Two successive fits reuse weights. |
| **S2.24** | C3 CD | `partial_fit()` CDC update for mini-batch. | Weights update diff ≠ 0. |
| **S2.25** | C3 CD | Compare analytic ridge vs CD lasso on dense data. | Bench compiles. |
| **S2.26** | C3 CD | Panic on colinear + λ=0 (detect divergence). | Unit panic. |
| **S2.27** | C3 CD | Docs example. | Docs OK. |
| **S2.28** | C4 Opt | Create optimizer module + `Optimizer trait { step, zero_grad }`. | Compiles. |
| **S2.29** | C4 Opt | Implement basic `Sgd` (with momentum & weight-decay). | Unit: w update vs numpy ref. |
| **S2.30** | C4 Opt | Implement `Adam` + `AdamW`. | Compare to PyTorch for one step (<1e-8). |
| **S2.31** | C4 Opt | Implement `RmsProp` and `AdaGrad`. | Unit stats update checks. |
| **S2.32** | C4 Opt | Implement `Lbfgs` (two-loop recursion, batch). | Converges on Rosenbrock (f<1e-8). |
| **S2.33** | C4 Opt | Add enum `LrScheduler` with Step, Exp, Cosine, OneCycle. | Compile. |
| **S2.34** | C4 Opt | Wire schedulers into Optimizer. | Unit: lr changes across epochs. |
| **S2.35** | C4 Opt | Add gradient-clipping util (norm & value). | Clip test passes. |
| **S2.36** | C4 Opt | Parameter-group support (A/B/C modes). | Unit: diff lrs in groups. |
| **S2.37** | C4 Opt | Docs & README table summarising opts. | Docs OK. |
| **S2.38** | C4 Opt | Criterion bench Adam vs SGD on 1e5 params. | Bench compiles. |
| **S2.39** | C4 Opt | Miri run on Optimiser trait objects. | Miri clean. |
| **S2.40** | C5 Data | Define `Dataset trait { len, get(idx)->Sample }`. | Example dummy dataset compiles. |
| **S2.41** | C5 Data | Implement `ArrayDataset::from_arrays(X, y)`. | len == rows test. |
| **S2.42** | C5 Data | Implement `DataLoader` iterator (single-thread). | Iter yields batches sized N. |
| **S2.43** | C5 Data | Add shuffle bool (epoch-wise). | Order differs when true. |
| **S2.44** | C5 Data | Add batch_size & automatic last-batch drop/keep. | Tests. |
| **S2.45** | C5 Data | Prefetch worker pool (num_workers) with channel. | Throughput benchmark > 1× single. |
| **S2.46** | C5 Data | Implement map(transform) closure to dataset. | Transform doubles X test. |
| **S2.47** | C5 Data | Timeout & graceful shutdown on drop. | Leak-test passes. |
| **S2.48** | C5 Data | Docs + example training loop snippet. | Docs OK. |
| **S2.49** | C5 Data | Add unit covering multi-worker determinism when seed fixed. | Same batch order test. |
| **S2.50** | C5 Data | Bench single vs 4-worker load for 1 GiB random data. | Bench compiles. |
| **S2.51** | C5 Data | Allow custom collate `fn (fn(Vec<Sample>)->(X,y))`. | Compile. |
| **S2.52** | C5 Data | Miri run DataLoader drop safety. | Clean. |
| **S2.53** | C6 Trainer | Create Trainer struct with builder pattern. | Trainer::new(&mut model,opt) compiles. |
| **S2.54** | C6 Trainer | Implement loop: for each epoch iterate DataLoader, forward → loss → backward → opt.step. | Runs 1 epoch on toy data. |
| **S2.55** | C6 Trainer | Integrate LR scheduler `.step()` each epoch/batch flag. | LR logged. |
| **S2.56** | C6 Trainer | Progress-bar console using indicatif (std-only fallback). | Shows bars in tests (captured). |
| **S2.57** | C6 Trainer | Metric logger trait + console logger default. | val_loss printed. |
| **S2.58** | C6 Trainer | CSV logger optional; file path setter. | CSV rows == epochs. |
| **S2.59** | C6 Trainer | Validation loop & metric computation. | val_loss reasonable. |
| **S2.60** | C6 Trainer | Checkpoint manager: metric-based best model save. | File created when val improves. |
| **S2.61** | C6 Trainer | Epoch/step interval & metric-based checkpoint config. | Config test. |
| **S2.62** | C6 Trainer | Callback hooks (on_epoch_start, on_batch_end, etc.). | Custom hook mutates counter. |
| **S2.63** | C6 Trainer | Gradient-clipping integration before opt.step. | Norm ≤ clip. |
| **S2.64** | C6 Trainer | Early-stop callback (if no improve N epochs). | Stops early in test. |
| **S2.65** | C6 Trainer | Docs: full training example `LinearRegression` on DataLoader. | Builds & runs (doc test). |