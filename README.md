# UAV Localization Using Particle Filter

Preview of the results:
https://www.youtube.com/watch?v=tcz_gFbivqA

The code was used with dataset available from zenodo:
https://zenodo.org/record/1211730

# Dependencies

* OpenCV 3.3+ with CUDA support
* libboost
* GeographicLib
* Eigen3
* libgdal
* libcurl

# Running the project

1. Install the dependencies
2. Compile the project:
```
mkdir build
cmake ..
make
``` 
3. Download sample dataset, run the download script from project root directory:
```
bash download_data.sh
```
4. Run the localization, from project root:
```
./build/dataset-match --map-image dataset/urban/m_3809028_ne_15_1_20140720/m_3809028_ne_15_1_20140720.tif --dataset dataset/UL-200 --preview
```

# TODO List

~~1. Remove dependency on GPU~~
~~2. Speed up runtime by using FastMATCH approach to calculate similarity: random pixel sampling with photometric invariance~~
