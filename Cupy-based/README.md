<<<<<<< HEAD
=======
# Parametric Gauss Reconstruction (PGR)

## Update (2024/09/12): 

- Check out our newest update to PGR here: [WNNC](https://github.com/jsnln/WNNC):
  - Much lower complexity and higher efficiency: O(N^3) => O(NlogN) and can handle millions of points
  - Much better normal accuracy
  - PyTorch interfaces that are easy to use
  - Unofficial implementations of [GaussRecon](https://dl.acm.org/doi/10.1145/3233984) as a by-product, faster and more noise-resilient to the mainstream [PoissonRecon](https://github.com/mkazhdan/PoissonRecon).


This repository contains the implementation of the paper:
>>>>>>> f2f57119d3a09c5117f695df279cc4c619c6c02d

This is the code for the following article:

[**Anisotropic Gauss Reconstruction for Unoriented Point Clouds**]

The code is modified based on the code provided in the following article

[**Surface Reconstruction from Point Clouds without Normals by Parametrizing the Gauss Formula (ACMTOG 2022)**]

Thank you to author Lin Siyou for his open-source work on the code.
The original code can be obtained through the following link:
https://github.com/jsnln/ParametricGaussRecon             Or through: https://jsnln.github.io/


## Instructions on Running the Program

### Environment and Dependencies

This program has been tested on Ubuntu 20.04 with Python 3.8 and CUDA 11.1.  This program requires these Python packages: `numpy`, `cupy`, `scipy` and `tqdm`.

### Compiling Source Files

This program uses third-party libraries [CLI11](https://github.com/CLIUtils/CLI11) and [cnpy](https://github.com/rogersce/cnpy). We provide copies of them in this repository for convenicence. If you have [cmake](https://cmake.org/), you can build with:

```bash
mkdir build
cd build
cmake ..
make -j8
```

Or you can also build with `g++` directly:

```bash
cd src
g++ PGRExportQuery.cpp Cube.cpp Geometry.cpp MarchingCubes.cpp Mesh.cpp Octnode.cpp Octree.cpp ply.cpp plyfile.cpp cnpy/cnpy.cpp -ICLI11 -o ../apps/PGRExportQuery -lz -O2
g++ PGRLoadQuery.cpp Cube.cpp Geometry.cpp MarchingCubes.cpp Mesh.cpp Octnode.cpp Octree.cpp ply.cpp plyfile.cpp cnpy/cnpy.cpp -ICLI11 -o ../apps/PGRLoadQuery -lz -O2
```

If successful, this will generate two executables `PGRExportQuery` and `PGRLoadQuery` in `ParametricGaussRecon/apps`. The former builds an octree and exports grid corner points for query; the latter loads query values solved by PGR and performs iso-surfacing.

### Run the All-in-one Script

We provide a script `run_AGR.py` to run the complete pipeline. Usage:

```
python run_AGR.py point_cloud.xyz [-wk WIDTH_K] [-wmax WIDTH_MAX] [-wmin WIDTH_MIN]
                  [-a ALPHA] [-m MAX_ITERS] [-d MAX_DEPTH] [-md MIN_DEPTH]
                  [--cpu] [--save_r] [-c direction Module length]
```

Note that `point_cloud.xyz` should __NOT__ contain normals. The meaning of the options can be found by typing


````

### All code must run on a GPU, only the GPU version is provided here.(It can be run directly on a Linux server (ensuring that PGRExportQuery and PGRLoadQuery have set read-write permissions))

```bash
python run_AGR.py data/xyz/bunny_10000.xyz --alpha 2 -wk 16	

This will create a folder `results/bunny_10000.xyz` together with three subfolders: `recon`, `samples` and `solve` :

- `recon` contains the reconstructed meshes in PLY format.
- `samples` contains the input point cloud in XYZ format, the normalized input point cloud and the octree grid corners as query set in NPY format.
- `solve` contains the solved Linearized Surface Elements in XYZ and NPY formats, the queried values, query set widths in NPY format, and the iso-value in TXT format.
```

```bash
python run_AGR.py data/xyz/Utah_teapot_5000.xyz --alpha 2 -wk 16	
python run_AGR.py data/xyz/Utah_teapot_3000.xyz --alpha 2 -wk 16	
python run_AGR.py data/xyz/Utah_teapot_1000.xyz --alpha 2 -wk 16	


For the convenience of operation, we provide a version that can be directly reconstructed from. py (randomly selected points from the mesh), with a default selection of 5K points in this version.

python runply.py data/ply/bunny.ply --alpha 2 -wk 16
python runply.py data/ply/light.ply --alpha 2 -wk 16

More files(.ply) can be obtained by run:

python download_datasets_abc.py
python download_datasets_thingi10k.py
python download_datasets_real_world.py
python download_datasets_famous.py
in /data




