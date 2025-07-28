# NUMERU

**Numerical Methods & AI in Pure Rust (std-only)**  
_Work-In-Progress — API subject to rapid change_

---

## ✨ Vision

NUMERU aims to deliver a **fast, memory-efficient, fully type-safe** foundation for numerical computing and machine-learning in Rust — **without depending on external C/Fortran libraries or even `std` extensions.**  
When finished, the crate family will let you:

* Manipulate **const-generic N-D tensors** with zero-copy views, broadcasting, masking, and Python-style slicing.
* Execute high-performance **linear-algebra kernels** (GEMM, LU, QR, Cholesky…) built on portable-SIMD, multi-threading, and optional CUDA offload.
* Train classical ML models (linear/regularised regression, trees, SVM, k-NN, Naïve Bayes) and state-of-the-art DL architectures (Transformers, VAEs, GANs, Diffusion) **entirely in Rust**.
* Run fully-featured **autograd**, optimiser, scheduler, and trainer stacks — with LoRA adapters, mixed precision, and Retrieval-Augmented Generation pipelines.
* Save, load, and exchange artefacts via a compact binary **`.nmr` format** or **ONNX** export.

Everything is developed in **small, test-driven increments**; every commit keeps CI green.

---

## 🔩 Core Principles

| Principle | What it means here |
|-----------|-------------------|
| **Performance First** | 32-byte-aligned buffers, cache-blocked kernels, portable-SIMD, built-in thread pool, future CUDA kernels. |
| **Safety by Default** | Bounds-checked interfaces in debug builds, `unsafe` contained in leaf modules, Miri-clean codebase. |
| **Compile-Time Guarantees** | Const-generic shapes, deterministic broadcasting, compile-time reduction axes, nightly features (`generic_const_exprs`, `std::simd`). |
| **In-Place Everything** | Every heavy operation accepts a destination buffer to avoid unwanted allocations. |
| **Extensibility** | Trait-based APIs, parameter groups, callback hooks, rule-based configuration. |
| **Pure‐Rust Std-Only** | No `ffi`, no `rand`, no Rayon; we build what we need and treat the standard library as the boundary. |

---

## 🗺️ High-Level Roadmap

| Milestone | Status | Highlights |
|-----------|--------|------------|
| **M0 Array Core** | ⏳ | const-generic tensors, views, broadcast, RNG, thread pool |
| **M1 Classical ML** | ⭘ | Linear/Ridge/Lasso (analytic + SGD + CD), optimiser suite, data loader, trainer |
| **M2 Linear Algebra** | ⭘ | GEMM, LU, QR, Chol, `.nmr` serialiser |
| **M3 Tree Models** | ⭘ | Decision Tree, Random Forest, XGBoost |
| **M4 DL Core** | ⭘ | Autograd, `Module` trait, MLP/CNN, Dropout, BatchNorm |
| **M5 Seq & Attention** | ⭘ | Tokenisers, embeddings, RNNs, Transformer, LoRA |
| **M6 Generative** | ⭘ | VAE, GAN, Diffusion, mixed precision |
| **M7 GPU Acceleration** | ⭘ | CUDA kernels, device scopes, GPU autograd |
| **M8 RAG Pipeline** | ⭘ | Vector index (exact + HNSW), retriever, generator, turnkey pipeline |
| **M9 Release Hardening** | ⭘ | Docs, benchmarks, CI matrix, crates.io publish |

---

## 🚧 Current Status

Done: 
* The repository skeleton and CI are in place.  
* Basic numerical (f64) Tensor struct and implementations.

WIP:
* Linear Regression

To Do:
* 
---

## 📜 License

NUMERU will be released under the **MIT + Apache-2.0** dual license.  
Until the first public tag, the project is private to collaborators.