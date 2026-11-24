#!/usr/bin/env bash
set -euo pipefail

workspace=${1:?Pass the Kaggle working directory}
export UV_CACHE_DIR="$workspace/.uv-cache"
export DGLBACKEND=pytorch
export MAMBA_ROOT_PREFIX="$workspace/.mamba"
venv="$workspace/.dgmae-venv"
micromamba="$workspace/bin/micromamba"

if [[ ! -x "$micromamba" ]]; then
    archive="$workspace/micromamba.tar.bz2"
    curl --fail --location --silent --show-error --retry 3 \
        https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.3.3-0.tar.bz2 -o "$archive"
    printf '%s  %s\n' e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2 "$archive" | sha256sum -c -
    tar -xjf "$archive" -C "$workspace" bin/micromamba
fi
if [[ ! -x "$venv/bin/python" ]]; then
    "$micromamba" create --yes --prefix "$venv" --override-channels \
        -c https://conda.anaconda.org/dglteam/label/cu121 -c conda-forge \
        'python=3.11' 'dgl=2.1.0.cu121=py311_0' 'numpy=1.26.4' 'scipy=1.12.0'
fi
uv pip install --python "$venv/bin/python" \
    --index https://download.pytorch.org/whl/cu121 'torch==2.2.1+cu121'
uv pip install --python "$venv/bin/python" \
    'torchdata==0.7.1' \
    'torch-geometric==2.5.3' 'ogb==1.3.6' 'PyYAML==6.0.2' 'scikit-learn==1.5.2' \
    'packaging==24.2' 'pydantic==2.10.6'
uv pip install --python "$venv/bin/python" --no-index --no-deps \
    --find-links 'https://data.pyg.org/whl/torch-2.2.0+cu121.html' \
    'torch-scatter==2.1.2+pt22cu121' 'torch-sparse==0.6.18+pt22cu121'

"$venv/bin/python" -c 'import dgl, dgl.sparse, torch, torch_sparse; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print("Torch:", torch.__version__, "DGL:", dgl.__version__); dgl.graph(([0], [0])).to("cuda:0")'
