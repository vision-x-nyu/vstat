"""LLaVA-OneVision-2 (OneVision encoder + Qwen3-8B) model wrapper for lmms-eval.

Model: https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
Loaded via transformers AutoProcessor / AutoModelForImageTextToText with
trust_remote_code=True. Supports single image, multi-image and video inputs.

Two video backends are exposed through the processor:
  - "frames": uniform frame sampling, controlled by `num_frames`.
  - "codec":  codec-aware canvas packing (needs codec-video-prep + ffmpeg).

Usage (model_args, comma-separated):
  pretrained=lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct,num_frames=32,max_pixels=200704,video_backend=frames

Input spec:
  doc_to_visual yields either a video file path (str ending in .mp4/.avi/.mov)
  or PIL.Image objects.

Output spec:
  generate_until returns one decoded string per request.
"""

from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@register_model("llava_onevision2")
class Llava_OneVision2(lmms):
    """LLaVA-OneVision-2-8B-Instruct wrapper (data-parallel via accelerate)."""

    def __init__(
        self,
        pretrained: str = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        num_frames: int = 32,
        max_pixels: int = 200704,
        video_backend: str = "frames",
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert video_backend in ("frames", "codec"), f"Unknown video_backend: {video_backend}"
        assert int(batch_size) == 1, "Only batch_size=1 is supported for llava_onevision2"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {"dtype": torch.bfloat16, "device_map": self.device_map, "trust_remote_code": True}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self._tokenizer = self.processor.tokenizer
        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            self.processor.video_processor.max_pixels = max_pixels

        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.video_backend = video_backend
        self.system_prompt = system_prompt
        self.max_new_tokens_override = max_new_tokens

        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in (DistributedType.FSDP, DistributedType.MULTI_GPU), "Only DDP and FSDP are supported."
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data-parallel inference")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

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
        return self._tokenizer.eos_token_id

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
        raise NotImplementedError("Loglikelihood is not implemented for LLaVA-OneVision-2")

    def _build_message(self, context: str, visuals: list) -> Tuple[list, list, list]:
        content = []
        images, videos = [], []
        for visual in visuals or []:
            if isinstance(visual, str) and visual.lower().endswith(VIDEO_EXTENSIONS):
                content.append({"type": "video"})
                videos.append(visual)
            elif isinstance(visual, Image.Image):
                content.append({"type": "image"})
                images.append(visual.convert("RGB"))
        content.append({"type": "text", "text": context})

        message = []
        if self.system_prompt:
            message.append({"role": "system", "content": self.system_prompt})
        message.append({"role": "user", "content": content})
        return message, images, videos

    def _build_inputs(self, text: str, images: list, videos: list):
        proc_kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
            if self.video_backend == "codec":
                proc_kwargs["video_backend"] = "codec"
                proc_kwargs["max_pixels"] = self.max_pixels
            else:
                proc_kwargs["num_frames"] = self.num_frames
        return self.processor(**proc_kwargs)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            context = context.replace("<image>", "")
            message, images, videos = self._build_message(context, visuals)
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            inputs = self._build_inputs(text, images, videos).to(self.device)

            until = gen_kwargs.get("until", [self._tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [u for u in until if u and u != "\n\n"]

            max_new_tokens = self.max_new_tokens_override or gen_kwargs.get("max_new_tokens", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = bool(gen_kwargs.get("do_sample", temperature and temperature > 0))

            # Mirror the reference demo_inference.py generation call exactly:
            # greedy -> do_sample=False, num_beams=1; otherwise sampling kwargs.
            pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": self.use_cache,
                "eos_token_id": self._tokenizer.eos_token_id,
                "pad_token_id": pad_id,
            }
            if do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = gen_kwargs.get("top_p")
            else:
                generate_kwargs["do_sample"] = False
                generate_kwargs["num_beams"] = 1

            with torch.inference_mode():
                cont = self.model.generate(**inputs, **generate_kwargs)

            prompt_len = inputs["input_ids"].shape[-1]
            answer = self._tokenizer.batch_decode(cont[:, prompt_len:], skip_special_tokens=True)[0].strip()
            for term in until:
                answer = answer.split(term)[0]

            res.append(answer)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
