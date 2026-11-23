# Build base image
FROM nvcr.io/nvidia/pytorch:25.03-py3 AS base

# tmux is for debugging, osmesa is for rendering. Put all apt-get installs in this line
RUN apt update && apt-get install -y tmux libosmesa6-dev

RUN pip install --upgrade pip

# Pin setuptools to match pyproject.toml (required by torch==2.1.0 cpp_extension for pkg_resources)
RUN pip install "setuptools>=45,<78"

# Install dependencies
RUN pip install h5py hydra-core matplotlib scikit-learn scipy tensorboard trimesh
RUN pip install numpy==1.26.4 tqdm pyyaml pytest

# Install scene_synthesizer
RUN pip install scene-synthesizer[recommend]

RUN pip install imageio python-fcl

# Install pyrender
RUN pip install pyrender && pip install PyOpenGL==3.1.5

# Diffusers supplies the DDPM scheduler; Transformers is not used by GraspGenX.
# NGC pins older versions in PIP_CONSTRAINT. Release only the model/image
# packages being upgraded, retaining its framework and CUDA constraints.
RUN if [ -n "${PIP_CONSTRAINT:-}" ]; then \
        sed -i -E '/^(diffusers|huggingface[-_]hub|safetensors|pillow|filelock|onnx)([<>=!~[:space:]]|$)/Id' "$PIP_CONSTRAINT"; \
    fi
RUN pip install diffusers==0.38.0 timm==1.0.15 tensordict

# URDF parsing (both yourdfpy and urdfpy are used by graspgenx)
RUN pip install urdfpy yourdfpy==0.0.56

# Visualizer for demo scripts (demo_object_pc / demo_scene_pc / demo_object_mesh)
RUN pip install viser

# PointTransformerV3 / sparse-conv dependencies and training utilities
RUN pip install addict yapf==0.40.1 tensorboardx sharedarray torch-geometric

# For dataset management and manifold
RUN pip install webdataset
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN pip install objaverse==0.1.7
RUN mkdir -p /install/
RUN cd /install/ && git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
RUN mkdir -p /install/Manifold/build
RUN cd /install/Manifold/build
RUN cd /install/Manifold/build && cmake .. -DCMAKE_BUILD_TYPE=Release
RUN cd /install/Manifold/build && make
ENV PATH "${PATH}:/install/Manifold/build/"

# Remove unused Transformers inherited from the base image, and upgrade any
# vulnerable image/model tooling left by the base or earlier installation steps.
# ONNX 1.22 also needs newer protobuf, ml-dtypes and typing-extensions.
RUN if [ -n "${PIP_CONSTRAINT:-}" ]; then \
        sed -i -E '/^(protobuf|ml[-_]dtypes|typing[-_]extensions)([<>=!~[:space:]]|$)/Id' "$PIP_CONSTRAINT"; \
    fi && pip uninstall -y transformers && pip install --upgrade \
    diffusers==0.38.0 safetensors==0.8.0 onnx==1.22.0 Pillow==13.0.0 filelock==4.3.1

WORKDIR /code/
