#!/usr/bin/env bash
set -u

docker system df
for image in shiyunqi/open3dbench:place orfs-dreamplace:latest; do
  printf 'image=%s\n' "$image"
  docker run --rm --gpus all "$image" bash -lc '
    set -e
    python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    python3 -c "import torch_geometric; print(torch_geometric.__version__)"
    command -v nvcc
    command -v openroad
    openroad -version | head -n 1
  ' || true
done
