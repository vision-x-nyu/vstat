import re
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

process_vision_info, _has_keye_vl = optional_import("keye_vl_utils", "process_vision_info")
if not _has_keye_vl:
    eval_logger.warning("Failed to import keye_vl_utils; install via `pip install keye-vl-utils==1.5.2`")


@register_model("keye_vl")
class Keye_VL(lmms):
    """Kwai Keye-VL 1.5 model wrapper.

    https://huggingface.co/Kwai-Keye/Keye-VL-1_5-8B
    Uses AutoModel with trust_remote_code=True. The model has three reasoning
    modes triggered by suffixes on the user text: "/think", "/no_think", or
    no suffix (auto-think). Set `think_mode` to one of {"think", "no_think",
    "auto"} to control behavior; default is "think".
    """

    def __init__(
        self,
        pretrained: str = "Kwai-Keye/Keye-VL-1_5-8B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        think_mode: str = "think",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        valid_think_modes = {"think", "no_think", "auto"}
        if think_mode not in valid_think_modes:
            raise ValueError(f"think_mode must be one of {valid_think_modes}, got {think_mode}")
        self.think_mode = think_mode

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModel.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_num_frames = max_num_frames
        self.fps = fps

        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.system_prompt = system_prompt

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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
        raise NotImplementedError("Loglikelihood is not implemented for Keye-VL")

    def _apply_think_mode(self, text: str) -> str:
        if self.think_mode == "think":
            return text.rstrip() + " /think"
        if self.think_mode == "no_think":
            return text.rstrip() + " /no_think"
        return text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                context = self._apply_think_mode(context)
                contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            video_entry = {
                                "type": "video",
                                "video": visual,
                                "max_frames": self.max_num_frames,
                            }
                            if self.fps is not None:
                                video_entry["fps"] = self.fps
                            processed_visuals.append(video_entry)
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append({"type": "image", "image": visual})

                message = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    },
                ]
                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(batched_messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs,
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
