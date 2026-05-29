from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _resolve_local_snapshot(pretrained: str) -> str:
    """Use an already-cached HF snapshot when available to avoid rank races."""
    if not pretrained.startswith("allenai/Molmo2-"):
        return pretrained

    import os

    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        cache_root = str(hf_home / "hub")

    model_dir = Path(cache_root) / f"models--{pretrained.replace('/', '--')}"
    ref_path = model_dir / "refs" / "main"
    try:
        revision = ref_path.read_text().strip()
    except OSError:
        return pretrained

    snapshot = model_dir / "snapshots" / revision
    has_weights = (snapshot / "model.safetensors").exists() or (snapshot / "model.safetensors.index.json").exists()
    if (snapshot / "config.json").exists() and has_weights:
        eval_logger.info(f"Using local cached snapshot for {pretrained}: {snapshot}")
        return str(snapshot)
    return pretrained


@register_model("molmo2")
class Molmo2(lmms):
    """Molmo2 video QA adapter using the official Transformers interface."""

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        dtype: str = "auto",
        use_cache: bool = True,
        num_frames: Optional[int] = None,
        max_fps: Optional[float] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        pretrained = _resolve_local_snapshot(pretrained)

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self._patch_decord_fallback()
        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            trust_remote_code=True,
            dtype=dtype,
            device_map=self.device_map,
        ).eval()

        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            eval_logger.warning("Molmo2 adapter currently processes requests one at a time; forcing batch_size=1")
            self.batch_size_per_gpu = 1
        self.use_cache = use_cache
        self.num_frames = num_frames
        self.max_fps = max_fps
        self.system_prompt = system_prompt
        self._gen_overrides = {}
        if max_new_tokens is not None:
            self._gen_overrides["max_new_tokens"] = int(max_new_tokens)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with independent data parallel inference")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _patch_decord_fallback(self) -> None:
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is None or not hasattr(video_processor, "fetch_videos"):
            return

        original_fetch = video_processor.fetch_videos
        load_video = original_fetch.__func__.__globals__.get("load_video")
        if load_video is None:
            return

        def fetch_videos_with_pyav_fallback(vp, video_url_or_urls, sample_timestamps_fn=None):
            try:
                return original_fetch(video_url_or_urls, sample_timestamps_fn=sample_timestamps_fn)
            except Exception as exc:
                message = str(exc)
                is_decord_error = exc.__class__.__name__ == "DECORDError" or "Error sending packet" in message
                if not is_decord_error:
                    raise
                eval_logger.warning(f"Decord failed for {video_url_or_urls}; retrying Molmo2 video decode with PyAV")
                if isinstance(video_url_or_urls, list):
                    return list(
                        zip(
                            *[
                                fetch_videos_with_pyav_fallback(vp, item, sample_timestamps_fn=sample_timestamps_fn)
                                for item in video_url_or_urls
                            ]
                        )
                    )
                return load_video(video_url_or_urls, backend="pyav", sample_timestamps_fn=sample_timestamps_fn)

        video_processor.fetch_videos = MethodType(fetch_videos_with_pyav_fallback, video_processor)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Molmo2")

    def _visual_to_content(self, visual):
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return {"type": "video", "video": visual}
        return {"type": "image", "image": visual}

    def _processor_kwargs(self) -> dict:
        kwargs = {}
        if self.num_frames is not None:
            kwargs["num_frames"] = int(self.num_frames)
        if self.max_fps is not None:
            kwargs["max_fps"] = self.max_fps
        return kwargs

    def _build_messages(self, context: str, visuals: list) -> list[dict]:
        content = [self._visual_to_content(visual) for visual in visuals]
        content.append({"type": "text", "text": context})
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=1, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            context = contexts[0].replace("<image>", "")
            gen_kwargs = dict(all_gen_kwargs[0])
            until = gen_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected until to be str or list, got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            visuals = doc_to_visual[0](self.task_dict[task[0]][split[0]][doc_id[0]])
            messages = self._build_messages(context, visuals or [])
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **self._processor_kwargs(),
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            current_gen_kwargs = {
                "max_new_tokens": 128,
                "do_sample": False,
                "num_beams": 1,
                **gen_kwargs,
                **self._gen_overrides,
            }
            current_gen_kwargs.pop("temperature", None)
            current_gen_kwargs.pop("top_p", None)
            current_gen_kwargs.pop("top_k", None)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=self.use_cache,
                    **current_gen_kwargs,
                )

            generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
            ans = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            for term in until:
                if term:
                    ans = ans.split(term)[0]
            res.append(ans)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
