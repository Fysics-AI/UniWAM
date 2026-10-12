import os
import socket

# Each distributed worker needs its own DeepSpeed Triton autotune files. The
# shared cache uses a fixed `.tmp` name and races when several ranks exit.
_triton_cache_root = os.environ.get("TRITON_CACHE_DIR")
if _triton_cache_root:
    _local_rank = os.environ.get("LOCAL_RANK", str(os.getpid()))
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        _triton_cache_root,
        socket.gethostname(),
        f"local_rank_{_local_rank}",
    )

import hydra
from omegaconf import DictConfig

from uniwam.runtime import run_training
from uniwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
