import gc
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3_5")
class Qwen3_5(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "device_map": self.device_map,
            "dtype": torch.bfloat16,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._config = self.model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 262144)

        self._gen_overrides = {}
        if max_new_tokens is not None:
            self._gen_overrides["max_new_tokens"] = int(max_new_tokens)
        if do_sample is not None:
            self._gen_overrides["do_sample"] = bool(do_sample)
        if temperature is not None:
            self._gen_overrides["temperature"] = float(temperature)
        if top_p is not None:
            self._gen_overrides["top_p"] = float(top_p)
        if top_k is not None:
            self._gen_overrides["top_k"] = int(top_k)

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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_5")

    def _visual_content(self, visual):
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            content = {"type": "video", "url": visual, "num_frames": self.max_num_frames}
            if self.fps is not None:
                content["fps"] = self.fps
            return content
        if isinstance(visual, Image.Image):
            return {"type": "image", "image": visual}
        if isinstance(visual, str):
            return {"type": "image", "url": visual}
        return None

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
            # Qwen3.5 thinking responses use "</think>\n\n<answer>". Keep the
            # final answer available for the VSTAT parser if this stop leaks in.
            until = [item for item in until if item != "\n\n"]

            batched_messages = []
            contexts = list(contexts)
            for i, context in enumerate(contexts):
                context = context.replace("<image>", "")
                contexts[i] = context
                content = []
                for visual in visual_list[i] or []:
                    item = self._visual_content(visual)
                    if item is not None:
                        content.append(item)
                content.append({"type": "text", "text": context})
                batched_messages.append([{"role": "user", "content": content}])

            inputs = self.processor.apply_chat_template(
                batched_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=True,
                processor_kwargs={"padding": True},
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            current_gen_kwargs = {
                "max_new_tokens": 2048,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            }
            current_gen_kwargs.update(self._gen_overrides)

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                top_k=current_gen_kwargs["top_k"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            del inputs, cont, generated_ids_trimmed, answers
            del batched_messages, visual_list, current_gen_kwargs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Qwen3_5")
