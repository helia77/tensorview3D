# 3D Tensorview and Tensorvoting with CUDA

This repository provides a tensor visualization tool to visualize 3D tensor fields with built-in functions and performs efficient 3D tensor voting using parallel computations with CUDA.

## Table of Contents
1. [Overview](#overview)
2. [Demo] (#demo)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)

---

## Overview
The repository focuses on:
- Tools for generating, processing, and visualizing tensor fields and volumetric data.
- Parallelized **tensor voting** computations using CUDA for GPU acceleration.
- Efficient parallelized **eigendecomposition** of tensors for 3D data.
- Synthetic dataset generation for validation and testing.

This project is designed for research in large-scale **tensor voting frameworks**, particularly for 3D data analysis and segmentation. 
For the complete repository (including the 2D version, image-to-tensor library, etc.) visit the [STIM LAB Repository](https://github.com/STIM-Lab/tensor).

## Demo
This demo shows the working experience with the TensorVeiw3D program. A tensor field of the Serial block-face scanning electron microscopy (SBF-SEM) data
has been loaded to the program, and different tools are applied.
![](https://github.com/helia77/tensorview3D/demo.gif)
---

## Dependencies
- **Python 3.8+**
- Required Python packages:
   - `numpy`
   - `scikit-image`
   - `scipy`
- **C++ Libraries**:
   - OpenGL
   - FreeGLUT
   - Boost Program Options
   - Eigen3
   - GLEW
   - ImGui
   - GLFW3
- **CUDA Toolkit** (for GPU-accelerated computations)

## Installation
### Prerequisites
- **Windows OS** (project does not currently support macOS)
- **CUDA Toolkit**: Install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- **CMake**: Version 3.29 or higher.
- **vcpkg** (for managing C++ dependencies).

### Setting Up vcpkg
Install `vcpkg` to manage the required C++ packages:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```

### Install Dependencies with vcpkg
Run the following commands to install required libraries:
```bash
vcpkg install glfw3 glm imgui[core,glfw-binding,opengl3-binding] glew boost-program-options eigen3
```
These commands install:
- `glfw3` for window handling
- `glm` for math operations
- `imgui` for the GUI
- `GLEW` for OpenGL extensions
- `Boost` (Program Options) for argument parsing
- `Eigen3` for linear algebra

> **Note**: Replace `vcpkg` paths in CMake if necessary (e.g., `CMAKE_TOOLCHAIN_FILE`).

### Build with CMake
1. Clone the repository:
   ```bash
   git clone https://github.com/helia77/tensorview3D.git
   cd tensorview3D
   ```
2. Create a build directory and configure CMake:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_TOOLCHAIN_FILE="path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
   ```
3. Build the project:
   ```bash
   cmake --build . --config Release
   ```

### Install Python Dependencies
To install Python libraries, run on your conda terminal:
```bash
conda install numpy scikit-image scipy
```

---

## Usage
### C++ Implementation
- Build the C++ executable using CMake (steps above) and run:
   ```bash
   ./tensorview3 --input TENSOR_FILENAME
   ```


## Contributing
Contributions are welcome! If you'd like to improve this repository, follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes and push the branch.
4. Open a Pull Request.
