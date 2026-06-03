import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


MIMO_VL_7B_SFT_2508 = "XiaomiMiMo/MiMo-VL-7B-SFT-2508"
MIMO_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)
MIMO_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.safetensors",
)


def _snapshot_complete(snapshot_path: str) -> bool:
    root = Path(snapshot_path)
    return all((root / filename).exists() for filename in MIMO_REQUIRED_FILES)


def _local_rank() -> int:
    for env_name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _resolve_mimo_snapshot(pretrained: str) -> str:
    if pretrained != MIMO_VL_7B_SFT_2508 or Path(pretrained).exists():
        return pretrained

    revision = os.environ.get("MIMO_VL_REVISION")
    snapshot_kwargs = {
        "repo_id": pretrained,
        "revision": revision,
        "allow_patterns": MIMO_ALLOW_PATTERNS,
        "max_workers": 1,
    }

    try:
        snapshot_path = snapshot_download(local_files_only=True, **snapshot_kwargs)
        if _snapshot_complete(snapshot_path):
            return snapshot_path
    except Exception:
        snapshot_path = None

    if _local_rank() == 0:
        eval_logger.info("Resolving complete MiMo-VL checkpoint snapshot before model load.")
        snapshot_path = snapshot_download(**snapshot_kwargs)
        if not _snapshot_complete(snapshot_path):
            missing = [f for f in MIMO_REQUIRED_FILES if not (Path(snapshot_path) / f).exists()]
            raise FileNotFoundError(f"Incomplete MiMo-VL snapshot at {snapshot_path}; missing: {missing}")
        return snapshot_path

    deadline = time.time() + int(os.environ.get("MIMO_VL_SNAPSHOT_WAIT_SECONDS", "3600"))
    while time.time() < deadline:
        try:
            snapshot_path = snapshot_download(local_files_only=True, **snapshot_kwargs)
            if _snapshot_complete(snapshot_path):
                return snapshot_path
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError(
        "Timed out waiting for rank 0 to prepare the MiMo-VL checkpoint snapshot. "
        "Run rank 0 once with network access or prefetch the model with huggingface-cli."
    )


@register_model("mimo_vl")
class MiMo_VL(Qwen2_5_VL):
    """MiMo-VL-7B model wrapper.

    MiMo-VL is fully compatible with the Qwen2_5_VL architecture
    (Qwen2_5_VLForConditionalGeneration) so we reuse that implementation.
    Default pretrained model is set to MiMo-VL-7B-SFT-2508.
    """

    def __init__(
        self,
        pretrained: str = MIMO_VL_7B_SFT_2508,
        **kwargs,
    ):
        pretrained = _resolve_mimo_snapshot(pretrained)
        super().__init__(pretrained=pretrained, **kwargs)
