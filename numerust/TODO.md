# NUMERU – Development Checklist
> Mark each `[ ]` as `[x]` when done.  
> Micro-steps (Sx.y) for **M0** are written out in full to guide the pattern;  
> later milestones list chunk-level tasks you will further explode into micro-steps following the same template.

---

## M0  Array Core (Foundation)

### C1  Box Buffer & Alignment
- [ ] **S0.1** Create Cargo workspace `numeru` with crate `numeru_core`, nightly toolchain & CI skeleton  
- [ ] **S0.2** Implement `AlignedBox<T, N>` (32-byte aligned)  
- [ ] **S0.3** Add unit test ensuring alignment & length  
- [ ] **S0.4** Add Criterion benchmark comparing `AlignedBox` vs plain `Box`

### C2  Array1 & Indexing
- [ ] **S0.5** Implement `Array1<T, N>` with `zeros()` ctor  
- [ ] **S0.6** Safe `Index` / `IndexMut` (bounds-checked)  
- [ ] **S0.7** `unsafe fn get_unchecked()` + Miri test  
- [ ] **S0.8** Release build uses unchecked path; add perf unit test

### C3  `tensor![]` Macro
- [ ] **S0.9** Implement compile-time nested-literal macro for 1-D  
- [ ] **S0.10** Extend macro for rank≤4 inference  
- [ ] **S0.11** Unit tests: various shapes, compile-fail mis-shape

### C4  Views, Shape + Stride
- [ ] **S0.12** Add `ArrayView` / `ArrayViewMut` structs  
- [ ] **S0.13** Implement zero-copy transpose  
- [ ] **S0.14** Python-style slicing (`slice![.., -1, 0..5;2]`)  
- [ ] **S0.15** Tests: stride correctness, negative indices

### C5  Broadcast Engine
- [ ] **S0.16** Implement compile-time broadcast shape resolver  
- [ ] **S0.17** Element-wise `add_into` using resolver  
- [ ] **S0.18** Tests: scalar + tensor, mismatched panic

### C6  Bitset Masking
- [ ] **S0.19** Implement compact bitset type (`BitMask<N>`)  
- [ ] **S0.20** `mask_select` → compact copy  
- [ ] **S0.21** `mask_view` → iterator view  
- [ ] **S0.22** `mask_assign` (array length must match)  
- [ ] **S0.23** Tests: selection, view iteration, assign error

### C7  ThreadPool Infrastructure
- [ ] **S0.24** Implement global `ThreadPool::install` scope guard  
- [ ] **S0.25** Parallel map for Array1 reductions  
- [ ] **S0.26** Unit + stress tests across 2–8 threads

### C8  PRNG & Initialisers
- [ ] **S0.27** Implement PCG-64 PRNG (per-thread TLS)  
- [ ] **S0.28** `set_global_seed(u64)` + auto-seed fallback  
- [ ] **S0.29** `rand_uniform_into`, `rand_normal_into`  
- [ ] **S0.30** Xavier/He helpers (shape-inferred)  
- [ ] **S0.31** Tests: statistical mean/var, seed reproducibility  
- [ ] **S0.32** Bench RNG throughput vs rand crate (optional)

---

## M1  Linear Algebra

### C1  Blocked GEMM (threaded)
- [ ] Implement CPU-SIMD + ThreadPool GEMM (`matmul_into`)
- [ ] Tests vs naïve multiply, compare perf

### C2  Partial-Pivot LU (in-place)
- [ ] In-place factor overwriting, return pivot indices
- [ ] Solve helper `lu_solve` + tests (Ax=b round-trip)

### C3  Householder QR (no pivot)
- [ ] In-place factors `Q` (implicit) & `R`
- [ ] Recompose test `Q * R` ≈ original

### C4  Cholesky Lower
- [ ] Lower-triangular in-place `chol`
- [ ] Positive-definite check, unit tests

### C5  Row→Col Major Copy
- [ ] `to_col_major()` allocates new contiguous buffer
- [ ] FFI safety tests

### C6  Binary Serialiser (`.nmr`)
- [ ] Write & read arrays (shape, stride, data)
- [ ] Round-trip & endianness test

*(continue chunk checklists similarly for M2 – M8)*  

---

## M2  Classical ML v1 – Linear Models & Trainer

- [ ] C1 Analytical solver via normal-eq (Cholesky)
- [ ] C2 SGD solver (lr enum, early stop)
- [ ] C3 Coordinate Descent (Lasso focus)
- [ ] C4 Optimiser suite (SGD, Adam, etc.)
- [ ] C5 Dataset + DataLoader (+prefetch)
- [ ] C6 Trainer v1 (metrics console, CSV opt-in)

---

## M3  Tree Models  
- [ ] Decision Tree classifier/regressor  
- [ ] Random Forest (bagging, feature sub-sample)  
- [ ] XGBoost gradient-boosted trees  

---

## M4  DL Core  
- [ ] Dynamic autograd tape  
- [ ] `Module` trait + `network!{}` macro (anonymous)  
- [ ] MLP, CNN layers, Dropout, BatchNorm  
- [ ] Optimiser param-groups (A / B / C)  

---

## M5  Seq & Attention  
- [ ] Tokenizer API (BPE, WordPiece)  
- [ ] Embedding layers (positional + learned)  
- [ ] RNN (GRU/LSTM)  
- [ ] Transformer encoder/decoder  
- [ ] LoRA helpers (auto + manual)  

---

## M6  Generative  
- [ ] VAE (reparam trick)  
- [ ] GAN (DCGAN baseline)  
- [ ] Diffusion (DDPM) + scheduler utilities  
- [ ] Mixed-precision (fp16/bf16) toggle  

---

## M7  GPU Acceleration  
- [ ] `GpuContext` scope guard  
- [ ] Host↔device persistence caching  
- [ ] CUDA GEMM kernel  
- [ ] Broadcast & reduction kernels  
- [ ] GPU autodiff ops  

---

## M8  Retrieval-Augmented Generation  
- [ ] VectorIndex (exact + HNSW)  
- [ ] Retriever API  
- [ ] Generator (seq-to-seq transformer)  
- [ ] `RagPipeline` turnkey  
- [ ] Component modular exports  

---

## M9  Release Hardening  
- [ ] Public docs (`mdbook` / docs.rs)  
- [ ] Benchmarks & comparison table  
- [ ] CI matrix (Linux/macOS/Win, stable/nightly)  
- [ ] License & Code of Conduct  
- [ ] 1.0-alpha crates.io publish  

---

### Conventions
- Write **tests first** (TDD) for each micro-step.  
- CI must stay **green at every commit**.  
- Keep each PR ≤ **200 LOC** net change.  

---