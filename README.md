# CUDA Dictionary Collisions

This repository now contains a clean-room CUDA implementation of the `collisions` executable described in `CLEANROOM_SPEC.md`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The executable is written to `build/collisions`.

## Run

Type-mode example:

```bash
./build/collisions --no-dict --min 1 --max 1 --threads 1 --force 61
```

Field-mode example with logging:

```bash
./build/collisions --field --no-dict --prefix t --suffix Structure --min 2 --max 2 --threads 1 --force --log 6929265
```

## Notes

- Runtime data files are loaded by relative path from the current working directory.
- The search engine precomputes token hashes and lengths, stores targets in a device-side open-addressed hash table, and batches suffix enumeration on the GPU.
- If no CUDA device is available at runtime, the executable falls back to a CPU search path while keeping the same CLI behavior.