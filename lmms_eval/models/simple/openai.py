import base64
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, List, Optional, Tuple, Union
from urllib.parse import unquote

import numpy as np
from accelerate import Accelerator, DistributedType
from dotenv import load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.models.model_utils.media_encoder import (
    encode_image_to_base64,
    encode_image_to_base64_with_size_limit,
)
from lmms_eval.models.model_utils.usage_metrics import is_budget_exceeded, log_usage

try:
    from openai import DefaultHttpxClient
except ImportError:
    DefaultHttpxClient = None

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


def _normalize_openai_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
                continue
            if isinstance(part.get("content"), str):
                text_parts.append(part["content"])
        return "".join(text_parts)
    return str(content)


def _uses_reasoning_chat_params(model_version: str) -> bool:
    model_id = model_version.lower()
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_responses_output_text(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str):
        return output_text

    text_parts = []
    for item in _get_value(response, "output", []) or []:
        if _get_value(item, "type") != "message":
            continue
        for part in _get_value(item, "content", []) or []:
            if _get_value(part, "type") in {"output_text", "text"}:
                text = _get_value(part, "text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts)


def _extract_responses_reasoning_summary(response: Any) -> Optional[str]:
    summary_parts = []
    for item in _get_value(response, "output", []) or []:
        if _get_value(item, "type") != "reasoning":
            continue
        for part in _get_value(item, "summary", []) or []:
            text = _get_value(part, "text")
            if isinstance(text, str) and text.strip():
                summary_parts.append(text.strip())
    return "\n".join(summary_parts) if summary_parts else None


@register_model("openai")
class OpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 10,
        retry_backoff_s: float = 1.0,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        azure_openai: bool = False,
        max_frames_num: int = 10,
        video_fps: Optional[float] = None,
        video_as_url: bool = False,
        httpx_trust_env: bool = True,
        batch_size: int = 64,
        num_concurrent: int = 32,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        use_responses_api: bool = False,
        **kwargs,
    ) -> None:
        """
        :param httpx_trust_env: bool
            httpx.Client used by openai-python has trust_env set to True by default. A
            False value of this param constructs a httpx.Client with trust_env set to
            False.  Such a httpx.Client ignores environment variables (HTTP_PROXY,
            HTTPS_PROXY, ALL_PROXY) and macOS proxy server settings.
        """
        super().__init__()
        # Accept both `model` and `model_version` for convenience, since
        # `--model_args model=xxx` is a common user expectation.
        if model is not None:
            model_version = model
        if kwargs:
            eval_logger.warning(f"Unknown model_args ignored: {list(kwargs.keys())}. " f"Check the supported parameters for the 'openai' backend.")
        self.model_version = model_version
        self.timeout = timeout
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        self.max_frames_num = max_frames_num
        self.video_fps = float(video_fps) if video_fps is not None else None
        self.video_as_url = parse_bool(video_as_url)
        self.num_concurrent = max(1, int(num_concurrent))
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = None if reasoning_summary in (None, "", "none", "None") else str(reasoning_summary)
        self.use_responses_api = parse_bool(use_responses_api)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        # In China mainland, people usually use a VPN client to access international web
        # sites such as Google. Such a client usually configures macOS proxy server
        # settings. openai-python uses a httpx.Client with trust_env set to True. Such a
        # httpx.Client uses macOS proxy server settings. Adding httpx_trust_env option
        # allows httpx to ignore proxy server settings set by VPN clients.
        if not httpx_trust_env and DefaultHttpxClient is None:
            eval_logger.warning("DefaultHttpxClient is unavailable in current openai package; " "falling back to default HTTP client with trust_env=True.")
            http_client = None
        else:
            if not httpx_trust_env and DefaultHttpxClient is not None:
                http_client = DefaultHttpxClient(trust_env=httpx_trust_env)
            else:
                http_client = None

        # Use provided parameters or fall back to environment variables
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_API_BASE")

        # Fix URL encoding issue - decode if it's URL encoded
        if base_url and "%" in base_url:
            base_url = unquote(base_url)

        # Remove trailing slash if present
        if base_url and base_url.endswith("/"):
            base_url = base_url.rstrip("/")

        self.client = (
            OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
            if not azure_openai
            else AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                http_client=http_client,
            )
        )

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(self, string: str):
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens):
        return ""

    @property
    def eot_token_id(self):
        return 0

    @property
    def rank(self):
        return self._rank

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            with Image.open(image) as loaded_image:
                return encode_image_to_base64_with_size_limit(
                    loaded_image.convert("RGB"),
                    max_size_bytes=self.max_size_in_mb * 1024 * 1024,
                    image_format="PNG",
                    convert_rgb=False,
                    quality=None,
                    copy_if_pil=False,
                    resize_factor=0.75,
                    min_side=100,
                    resample=Image.Resampling.LANCZOS,
                )
        return encode_image_to_base64_with_size_limit(
            image,
            max_size_bytes=self.max_size_in_mb * 1024 * 1024,
            image_format="PNG",
            convert_rgb=False,
            quality=None,
            copy_if_pil=False,
            resize_factor=0.75,
            min_side=100,
            resample=Image.Resampling.LANCZOS,
        )

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        if total_frame_num <= 0:
            return []

        frame_idx = []

        if self.video_fps is not None and self.video_fps > 0:
            source_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
            if source_fps > 0:
                step = max(1, int(round(source_fps / self.video_fps)))
                frame_idx = list(range(0, total_frame_num, step))
                if frame_idx and frame_idx[-1] != total_frame_num - 1:
                    frame_idx.append(total_frame_num - 1)

        if not frame_idx:
            sample_count = min(max(1, int(for_get_frames_num)), total_frame_num)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_count, dtype=int)
            if total_frame_num - 1 not in uniform_sampled_frames:
                uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)
            frame_idx = uniform_sampled_frames.tolist()
        elif for_get_frames_num and len(frame_idx) > int(for_get_frames_num):
            keep = np.linspace(0, len(frame_idx) - 1, int(for_get_frames_num), dtype=int)
            frame_idx = [frame_idx[i] for i in keep]
            if frame_idx[-1] != total_frame_num - 1:
                frame_idx.append(total_frame_num - 1)

        frame_idx = sorted(set(frame_idx))
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            base64_frames.append(
                encode_image_to_base64(
                    img,
                    image_format="PNG",
                    convert_rgb=False,
                    quality=None,
                )
            )

        return base64_frames

    _VIDEO_MIME_TYPES = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/mov",
        ".mpeg": "video/mpeg",
        ".avi": "video/mp4",
        ".flv": "video/mp4",
        ".wmv": "video/mp4",
        ".mkv": "video/webm",
    }

    @staticmethod
    def encode_video_to_data_url(video_path: str) -> str:
        ext = os.path.splitext(video_path)[1].lower()
        mime = OpenAICompatible._VIDEO_MIME_TYPES.get(ext, "video/mp4")
        with open(video_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def encode_audio_file(self, audio_path: str):
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        audio_format = ext if ext in {"wav", "mp3", "flac", "aac", "ogg", "m4a"} else "wav"
        with open(audio_path, "rb") as handle:
            audio_b64 = base64.b64encode(handle.read()).decode("utf-8")
        return audio_b64, audio_format

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[GenerationResult]:
        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        ordered_requests = []
        for single_request in re_ords.get_batched(n=1, batch_fn=None):
            ordered_requests.extend(single_request)

        if not ordered_requests:
            return []

        pbar = tqdm(
            total=len(ordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        reordered_responses: List[Union[GenerationResult, None]] = [None] * len(ordered_requests)
        current_concurrency = min(
            self.num_concurrent,
            self.adaptive_config.max_concurrency,
        )
        dispatch_order = list(range(len(ordered_requests)))
        if self.prefix_aware_queue:
            dispatch_order.sort(
                key=lambda idx: (
                    make_prefix_hash(str(ordered_requests[idx][0]), self.prefix_hash_chars),
                    idx,
                ),
            )
        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        request_latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(
            1,
            self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency,
        )

        def _to_responses_payload(payload: dict) -> dict:
            content = []
            for part in payload["messages"][0]["content"]:
                part_type = part.get("type")
                if part_type == "text":
                    content.append({"type": "input_text", "text": part["text"]})
                elif part_type == "image_url":
                    content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
                elif part_type == "input_audio":
                    content.append(part)
                else:
                    raise ValueError(f"Responses API conversion does not support content part type: {part_type}")

            responses_payload = {
                "model": payload["model"],
                "input": [{"role": "user", "content": content}],
            }

            max_output_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
            if max_output_tokens is not None:
                responses_payload["max_output_tokens"] = max_output_tokens

            if "temperature" in payload:
                responses_payload["temperature"] = payload["temperature"]

            reasoning = {}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.reasoning_summary:
                reasoning["summary"] = self.reasoning_summary
            if reasoning:
                responses_payload["reasoning"] = reasoning

            return responses_payload

        def process_single_request(local_index: int, payload: dict):
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"

            for attempt in range(self.max_retries):
                try:
                    if self.use_responses_api:
                        response = self.client.responses.create(**_to_responses_payload(payload))
                        response_text = _extract_responses_output_text(response)
                        reasoning = _extract_responses_reasoning_summary(response)
                    else:
                        response = self.client.chat.completions.create(**payload)
                        response_text = _normalize_openai_message_content(response.choices[0].message.content)
                        reasoning = None
                    token_counts = None
                    if hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "prompt_tokens", None)
                        if input_tokens is None:
                            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                        output_tokens = getattr(response.usage, "completion_tokens", None)
                        if output_tokens is None:
                            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
                        output_details = getattr(response.usage, "completion_tokens_details", None)
                        if output_details is None:
                            output_details = getattr(response.usage, "output_tokens_details", None)
                        reasoning_tokens = (getattr(output_details, "reasoning_tokens", 0) or 0) if output_details else 0
                        log_usage(
                            model_name=self.model_version,
                            task_name=None,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                            source="model",
                        )
                        token_counts = TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=reasoning_tokens,
                        )
                    latency = time.time() - started_at
                    return response_text, local_index, True, rate_limited, latency, token_counts, reasoning
                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg
                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(self.retry_backoff_s)

            latency = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
            return failure_content, local_index, False, rate_limited, latency, None, None

        def maybe_update_concurrency(force: bool = False) -> None:
            nonlocal current_concurrency
            nonlocal failed_requests
            nonlocal rate_limited_requests
            nonlocal request_latencies
            nonlocal completed_since_adapt

            if not self.adaptive_concurrency:
                return

            sample_threshold = max(4, current_concurrency)
            if not force and completed_since_adapt < sample_threshold:
                return
            if completed_since_adapt <= 0:
                return

            decision = decide_next_concurrency(
                current_concurrency=current_concurrency,
                total_requests=completed_since_adapt,
                failed_requests=failed_requests,
                rate_limited_requests=rate_limited_requests,
                latencies=request_latencies,
                config=self.adaptive_config,
            )
            if decision.next_concurrency != decision.current_concurrency:
                eval_logger.info(
                    "Adaptive concurrency update: "
                    f"{decision.current_concurrency} -> "
                    f"{decision.next_concurrency} "
                    f"(fail_rate={decision.failure_rate:.3f}, "
                    f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                    f"p95_latency={decision.p95_latency_s:.3f}s)"
                )
            current_concurrency = decision.next_concurrency
            failed_requests = 0
            rate_limited_requests = 0
            request_latencies = []
            completed_since_adapt = 0

        def build_payload_for_index(global_index: int):
            (
                context,
                gen_kwargs,
                doc_to_visual_fn,
                doc_id_single,
                task_name,
                split_name,
            ) = ordered_requests[global_index]
            visuals = [doc_to_visual_fn(self.task_dict[task_name][split_name][doc_id_single])]
            if None in visuals:
                imgs = []
            else:
                visuals = self.flatten(visuals)
                imgs = []
                for visual in visuals:
                    if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual or ".webm" in visual or ".mkv" in visual):
                        if self.video_as_url:
                            imgs.append({"video_data_url": self.encode_video_to_data_url(visual)})
                        else:
                            frames = self.encode_video(visual, self.max_frames_num)
                            imgs.extend(frames)
                    elif isinstance(visual, str) and (".wav" in visual or ".mp3" in visual or ".flac" in visual or ".aac" in visual or ".ogg" in visual or ".m4a" in visual):
                        audio_b64, audio_format = self.encode_audio_file(visual)
                        imgs.append({"audio_b64": audio_b64, "audio_format": audio_format})
                    elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        imgs.append(self.encode_image(visual))
                    elif isinstance(visual, Image.Image):
                        imgs.append(self.encode_image(visual))

            request_gen_kwargs = dict(gen_kwargs)
            max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
            temperature = request_gen_kwargs.get("temperature", 0)

            payload = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": []}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            payload["messages"][0]["content"].append({"type": "text", "text": context})
            for img in imgs:
                if isinstance(img, dict) and "video_data_url" in img:
                    payload["messages"][0]["content"].append(
                        {
                            "type": "video_url",
                            "video_url": {"url": img["video_data_url"]},
                        }
                    )
                elif isinstance(img, dict) and "audio_b64" in img:
                    payload["messages"][0]["content"].append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": img["audio_b64"], "format": img["audio_format"]},
                        }
                    )
                else:
                    payload["messages"][0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )

            if _uses_reasoning_chat_params(self.model_version):
                payload.pop("temperature")
                payload.pop("max_tokens")
                payload["max_completion_tokens"] = max_new_tokens
                if self.reasoning_effort:
                    payload["reasoning_effort"] = self.reasoning_effort
                if self.model_version.lower().startswith(("o1", "o3")):
                    payload["response_format"] = {"type": "text"}

            return payload

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    if is_budget_exceeded():
                        reordered_responses[dispatch_order[cursor]] = GenerationResult(text="", token_counts=None)
                        pbar.update(1)
                        cursor += 1
                        continue
                    request_index = dispatch_order[cursor]
                    payload = build_payload_for_index(request_index)
                    future = executor.submit(process_single_request, request_index, payload)
                    in_flight[future] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    (
                        response_text,
                        local_index,
                        success,
                        rate_limited,
                        latency,
                        token_counts,
                        reasoning,
                    ) = future.result()
                    in_flight.pop(future, None)
                    reordered_responses[local_index] = GenerationResult(text=response_text, token_counts=token_counts, reasoning=reasoning)
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    request_latencies.append(latency)
                    completed_since_adapt += 1
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

        maybe_update_concurrency(force=True)

        pbar.close()
        completed_responses = [response if response is not None else GenerationResult(text="", token_counts=None) for response in reordered_responses]
        return re_ords.get_original(completed_responses)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")
