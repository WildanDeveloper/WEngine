# WEngine: Elis AI Engine

Lightweight, blazing-fast CPU-first AI Engine built in **Rust** + **C++ (AVX-512)** by **WildanDev**.

Designed for high-throughput neural inference and LLMs on commodity CPU instances without requiring expensive GPUs.

---

## Highlights

- **Blazing Fast CPU Inference**: Native C++ AVX-512 + FMA microkernels (`_mm512_fmadd_ps`) delivering **31+ GFLOPS** sustained on standard x86_64 server CPUs.
- **Multithreaded Scaling**: Automatically utilizes all available CPU cores via parallel slice GEMM.
- **Ultra-Lightweight Footprint**: Full **Elis-15M** Transformer uses only **~62 MB RAM**, making it deployable on minimal VPS/cloud nodes.
- **Throughput**: ~**1,000 tokens/sec** on 4 vCPUs (AMD EPYC).
- **Modern Transformer Architecture**: Multi-Head Attention, RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU activation.
- **Zero Allocation per Forward**: Pre-allocated scratch buffers eliminate runtime heap overhead.
- **Extensible API**: Includes base `Elis` MLP engine (with full analytical backpropagation, numerical gradient checks, SGD) and `ElisTransformer` language model.

---

## Architecture Overview

```
WEngine/
├── src/
│   ├── lib.rs           # Core module exports
│   ├── elis.rs          # Elis Neural Net (MLP, autograd, SGD)
│   ├── transformer.rs   # ElisTransformer (RoPE, RMSNorm, SwiGLU, MHA)
│   ├── tensor.rs        # Tensor 2D primitives
│   ├── rng.rs           # Fast PCG32 deterministic PRNG
│   └── accel.rs         # Zero-overhead Rust FFI to native C++ AVX-512 kernels
├── native/
│   ├── wildandev_accel.cpp  # AVX-512 / AVX2 GEMM, RMSNorm, SwiGLU microkernels
│   └── Makefile             # C++ shared library compilation
├── tests/
│   └── elis_core_test.rs # Exact analytic gradient check & XOR convergence tests
└── examples/
    ├── bench_scale.rs   # Elis-15M 4-core AVX-512 benchmark
    ├── bench_tuned.rs   # Zero-alloc transformer benchmark
    ├── bench_elis.rs    # Elis XOR training benchmark
    └── chat_elis.rs     # Autoregressive text generation sample
```

---

## Quick Start

### Prerequisites
- Linux x86_64 with AVX2 or AVX-512 support
- Rust toolchain (`rustc`, `cargo`)
- `g++` (supporting C++17)

### Build Native Accelerators & Run Tests
```bash
# 1. Compile C++ AVX-512 shared library
cd native && make && cd ..

# 2. Run unit tests (Analytic gradient check + XOR convergence)
cargo test

# 3. Run Elis-15M Benchmark on your CPUs
cargo run --release --example bench_scale

# 4. Run interactive text generation
cargo run --release --example chat_elis
```

---

## Benchmarks (AMD EPYC 4 vCPUs)

| Model | Parameters | Context Length | RAM Usage | Latency | Throughput |
|---|---|---|---|---|---|
| **Elis-Mini** | 398 K | 32 | 1.68 MB | 1.1 ms | 28,343 tok/s |
| **Elis-15M** | 16.03 M | 64 | 62.15 MB | 65.7 ms | 974 tok/s |

---

## License

Created by **WildanDev**. All rights reserved.
