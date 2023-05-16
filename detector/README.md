## Installation

The object and attribute detection model/repo we chose is quite difficult to build, however to ease this process, we provide a docker image and dev container for those wanting to use the same model. The docker is created for the sole purpose of running the below files in an attempt to be as lightweight as possible.

Note: The docker container assumes compatibility between your GPU and the CUDA/PyTorch versions; in other words -- newer GPUs are not supported. 

### Using Dev Containers

- Modify the included volume to suit your own file organization and/or data location.
- Then simply open in a dev container.

### Using Docker CLI

`docker run -it --gpus=all -v .\multimodal-robustness-xmai\:/home/appuser/multimodal-robustness --rm claws:mm-robust-detector`

### Capturing Objects and Attributes

Initially, move the provided notebook `augment_caption.ipynb` into `bottom-up-attention.pytorch/utils/`, since this is the assumed working directory. We provide sample csv creations for our two datasets, however, modifying these are quite easy to do with your own data.


### Future Plans

- [] Attempt to build docker for newer GPUs and versions.