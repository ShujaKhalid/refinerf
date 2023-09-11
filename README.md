# RefiNeRF: Modelling dynamic neural radiance fields with inconsistent or missing camera parameters

Paper: https://arxiv.org/abs/2303.08695
Venue: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV - 2024)
Status: Submitted (In review)

### NOTE: This is an early release of the project and you might encounter some bugs - please open an issue if you do. I still have quite a few TODOs in the code. Pull requests are welcome :)

### NOTE: This code borrows heavily from https://github.com/ShujaKhalid/wildNeRF - In case of issues, please refer to it if I'm not able to get back to you in time.

### TODO: 
1. Refactor code and attend to in-code TODOs and FIXMEs (about 50)
2. Add tests to ensure compliance when using the repo on custom datasets   
3. ~~Add ablation study files and scripts (along with associated outputs)~~
4. Add gif outputs to README for training and inference
5. ~~Add COLMAP instructions~~

## Installation 

To install the required pre-requisites to run this code,   
first, create and activate a conda environment:

```
conda create -n "venv-ngp" python=3.9
conda activate venv-ngp
```

run the requirements.txt file

```
pip install requirements.txt
```

To run the code, COLMAP files are required for image registration. We provide a few options to generate the COLMAP data.

```
./runner.sh --extract --nvidia
```
for registering the NVIDIA dynamic scenes dataset

```
runner.sh --extract --custom
```
for registering images from a CUSTOM dataset

## Training/Inference

The training is relatively straight forward and requires that the COLMAP files and associated images be placed according to the following folder structure:

```
├── assets
├── dnerf
├── ffmlp
│   └── src
├── gridencoder
│   └── src
├── nerf
├── raymarching
│   └── src
├── results
│   ├── gt
│   │   ├── Balloon1
│   │   ├── Balloon2
│   │   ├── Jumping
│   │   ├── Playground
│   │   ├── Skating
│   │   ├── Truck
│   │   └── Umbrella
│   └── Ours
│       ├── Balloon1
│       ├── Balloon2
│       ├── custom
│       ├── Jumping
│       ├── Playground
│       ├── Skating
│       └── Umbrella
├── scripts
├── sdf
├── shencoder
│   └── src
├── tensoRF
├── testing
└── utils
    ├── midas
    └── RAFT
        └── utils
```


1. Inference on a folder of videos:

```
python runner_sa160.py
```

We created this script to register and reconstruct short video clips in the SurgicalActions160 dataset.



### Acknowledgements
Credits to Thomas Müller for the amazing tiny-cuda-nn and instant-ngp:

```
@misc{tiny-cuda-nn,
    Author = {Thomas M\"uller},
    Year = {2021},
    Note = {https://github.com/nvlabs/tiny-cuda-nn},
    Title = {Tiny {CUDA} Neural Network Framework}
}

@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```
The framework of NeRF is adapted from nerf_pl:

```
@misc{queianchen_nerf,
    author = {Quei-An, Chen},
    title = {Nerf_pl: a pytorch-lightning implementation of NeRF},
    url = {https://github.com/kwea123/nerf_pl/},
    year = {2020},
}
The official TensoRF implementation:

@article{TensoRF,
  title={TensoRF: Tensorial Radiance Fields},
  author={Chen, Anpei and Xu, Zexiang and Geiger, Andreas and Yu, Jingyi and Su, Hao},
  journal={arXiv preprint arXiv:2203.09517},
  year={2022}
}
```
The NeRF GUI is developed with DearPyGui.


### Citations
If you've found this library useful, please cite us!

```
@article{khalid2022wildnerf,
  title={wildNeRF: Complete view synthesis of in-the-wild dynamic scenes captured using sparse monocular data},
  author={Khalid, Shuja and Rudzicz, Frank},
  journal={arXiv preprint arXiv:2209.10399},
  year={2022}
}
```


```
@article{khalid2023refinerf,
  title={RefiNeRF: Modelling dynamic neural radiance fields with inconsistent or missing camera parameters},
  author={Khalid, Shuja and Rudzicz, Frank},
  journal={arXiv preprint arXiv:2303.08695},
  year={2023}
}
```