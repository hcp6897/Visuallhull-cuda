# Visuallhull-cuda

## Setup

```
conda create -n cuda
conda activate cuda

//use 'nvidia-smi' to know which cuda version you have
conda install cudatoolkit=10.1
conda install numba

conda install -c conda-forge opencv
pip install scikit-image
pip install open3d
```

- Errors

    - ImportError: libGL.so.1: cannot open shared object file: No such file or directory。
    
        ```
        apt update
        apt install libgl1-mesa-glx
        ```

