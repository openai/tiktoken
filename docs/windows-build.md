# Building tiktoken from source on Windows (MSVC)

This document explains how to build and test `tiktoken` from source on Windows using
the Microsoft Visual C++ (MSVC) toolchain.

`tiktoken` includes a Rust extension module (`_tiktoken`). On Windows, this extension
must be compiled locally. If the build environment is incomplete, you may encounter
errors such as:

- `error: linker 'link.exe' not found`
- `ImportError: cannot import name '_tiktoken' from partially initialized module`
- `Failed building editable for tiktoken`

This guide walks through the required setup step by step.

---

## Requirements

### Supported platforms

- Windows 10 or Windows 11 (64-bit)
- Python 3.10+ (Python 3.12 recommended)
- Rust (stable, MSVC toolchain)
- Microsoft Visual Studio Build Tools (C++)

---

## Step 1: Install Python

Install Python from https://www.python.org and ensure the Python launcher works:

```
py -3.12 --version
```

Use `py -3.12` in all commands below to avoid ambiguity when multiple Python versions
are installed.

---

## Step 2: Install Rust (MSVC toolchain)

Install Rust using rustup:

```
winget install --id Rustlang.Rustup -e
```

Open a new terminal and verify:

```
rustc --version
cargo --version
```

Ensure the MSVC toolchain is selected (recommended on Windows):

```
rustup default stable-x86_64-pc-windows-msvc
rustup show
```

---

## Step 3: Install Visual Studio Build Tools (C++)

`tiktoken` requires the MSVC linker (`link.exe`). VS Code alone is not sufficient.

Install **Build Tools for Visual Studio 2022**:

```
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

During installation, select the workload:

- **Desktop development with C++**

Make sure the following components are included:
- MSVC v143 (or newer) C++ build tools
- Windows 10/11 SDK

After installation, restart your terminal (or open the
**x64 Native Tools Command Prompt for VS 2022**) and verify:

```
where link
```

If `link.exe` is found, the toolchain is correctly installed.

---

## Step 4: Clone and prepare the repository

```
git clone https://github.com/openai/tiktoken.git
cd tiktoken
```

Upgrade build tooling:

```
py -3.12 -m pip install -U pip setuptools wheel
```

---

## Step 5: Install tiktoken in editable mode

From the repository root:

```
py -3.12 -m pip install -e .
```

This step compiles the Rust extension and installs `tiktoken` in editable mode.

Verify the installation:

```
py -3.12 -c "import tiktoken; print(tiktoken.__version__)"
```

---

## Step 6: Install test dependencies

The test suite depends on `pytest` and `hypothesis`:

```
py -3.12 -m pip install -U pytest hypothesis
```

---

## Step 7: Run tests

```
py -3.12 -m pytest -q
```

All tests should pass.

---

## Troubleshooting

### `error: linker 'link.exe' not found`

Cause:
- MSVC build tools are missing or not available in the environment.

Fix:
1. Install **Visual Studio Build Tools** with **Desktop development with C++**
2. Restart the terminal
3. Confirm availability:

```
where link
```

---

### `ImportError: cannot import name '_tiktoken'`

Cause:
- The Rust extension failed to build.

Fix:
1. Ensure the MSVC Rust toolchain is active:

```
rustup default stable-x86_64-pc-windows-msvc
```

2. Reinstall in editable mode:

```
py -3.12 -m pip install -e . --force-reinstall
```

---

### `ModuleNotFoundError: No module named 'hypothesis'`

Fix:

```
py -3.12 -m pip install hypothesis
```

---

### PowerShell error with `-e`

If PowerShell reports an error about parameter `-e`, ensure you are not pasting
the prompt prefix (`PS C:\...>`). Use exactly:

```
py -3.12 -m pip install -e .
```

---

## Notes

- The MSVC toolchain is the recommended and supported build path on Windows.
- Using the Python launcher (`py -3.12`) avoids confusion with multiple Python installs.
- If issues persist, include the output of:
  - `where link`
  - `rustup show`
  - `py -3.12 -m pip install -e .`
