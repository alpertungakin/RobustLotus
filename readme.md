# VoxelShellProcessor: Point Cloud to Watertight Mesh

A high-performance Python pipeline that converts **Point Cloud Data (.ply)** into clean, **watertight 3D meshes (.obj)**. 

It is specifically designed for building data, utilizing a **Voxel-based Shell Extraction** technique accelerated by **CUDA** to handle large datasets efficiently.



## 🚀 Key Features

* **Voxelization & Filtering:** Converts raw points into a voxel grid, filtering "empty" space using a highly optimized **CUDA Kernel**.
* **Shell Extraction:** Identifies the outer "shell" of the object using multi-directional raycasting, removing internal voxels to create a watertight mesh.
* **Smoothing:** Applies Taubin smoothing to reduce the "Lego-block" aliasing of voxels while preserving the building's volume.
* **Hybrid Engine:** Combines the speed of **Open3D** (C++), the parallel power of **Numba (CUDA)**, and the robust topology tools of **Trimesh**.

---

## 🛠️ Prerequisites

You need an NVIDIA GPU with CUDA drivers installed to run the acceleration kernels.

### Python Dependencies
The following libraries are required:

```bash
pip install open3d numpy numba trimesh scipy
conda install cudatoolkit