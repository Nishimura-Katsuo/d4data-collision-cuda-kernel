# CUDA Dictionary Collisions

This repository now contains a clean-room CUDA implementation of the `collisions` executable described in `CLEANROOM_SPEC.md`.

## Build

Requirements:

- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit

Linux:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable is written to `build/collisions`.

Windows (Visual Studio generator):

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel
```

The executable is written to `build\Release\collisions.exe`.

## Run

Linux type-mode example:

```bash
./build/collisions --no-dict --min 1 --max 1 --threads 1 --force 61
```

Linux field-mode example with logging:

```bash
./build/collisions --field --no-dict --prefix t --suffix Structure --min 2 --max 2 --threads 1 --force --log 6929265
```

Windows PowerShell type-mode example:

```powershell
.\build\Release\collisions.exe --no-dict --min 1 --max 1 --threads 1 --force 61
```

Windows PowerShell field-mode example with logging:

```powershell
.\build\Release\collisions.exe --field --no-dict --prefix t --suffix Structure --min 2 --max 2 --threads 1 --force --log 6929265
```

## Notes

- Runtime data files are loaded by relative path from the current working directory.
- The search engine precomputes token hashes and lengths, stores targets in a device-side open-addressed hash table, and batches suffix enumeration on the GPU.
- If no CUDA device is available at runtime, the executable falls back to a CPU search path while keeping the same CLI behavior.