#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_PROMPT = (
    "Describe only what is literally visible in this image in one short sentence. "
    "Focus on salient people, objects, text, or UI when they are clearly present. "
    "If the image looks like a screen, webpage, dashboard, or document, say that directly. "
    "Do not invent people or actions that are not clearly visible, and do not mention that this is an AI task."
)
_TORCHVISION_COMPAT_LIB = None


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def non_empty_env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default).strip()
    return value


def base_to_host_port(base: str) -> Tuple[str, int]:
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3032
    return host, port


def ensure_torchvision_import_compat(torch_module: Any) -> None:
    global _TORCHVISION_COMPAT_LIB

    if os.environ.get("CHEK_DISABLE_TORCHVISION_NMS_STUB", "0") == "1":
        return
    try:
        import torchvision  # noqa: PLC0415,F401

        return
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)
    try:
        torch_module._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError:
        pass
    try:
        compat_lib = torch_module.library.Library("torchvision", "DEF")
        compat_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_COMPAT_LIB = compat_lib
    except Exception:
        return


def resolve_torch_device(torch_module: Any, requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        return "cpu"
    return requested


def torch_dtype_for_device(torch_module: Any, device: str) -> Any:
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32


def move_batch_to_device(torch_module: Any, batch: Any, device: str, dtype: Any | None = None) -> Any:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch_module.Tensor):
            if dtype is not None and torch_module.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return batch.__class__(moved)


def normalize_caption(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip())
    if not collapsed:
        return "Scene observed from the phone main view."
    return collapsed.rstrip(".,;:") + "."


def is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).strip().lower()
    return any(
        token in message
        for token in (
            "cuda runtime error",
            "cublas error",
            "cudacachingallocator",
            "nvml",
            "device-side assert",
            "driver/library version mismatch",
        )
    )


def infer_action_from_caption(caption: str) -> str:
    lowered = caption.lower()
    if any(token in lowered for token in ("reach", "reaching", "grab", "picking", "pick up")):
        return "reaching_object"
    if any(token in lowered for token in ("hold", "holding", "carry", "carrying")):
        return "holding_object"
    if any(token in lowered for token in ("walk", "walking", "move", "moving")):
        return "moving_phone"
    if any(token in lowered for token in ("sit", "sitting", "desk", "table")):
        return "steady_capture"
    return "steady_capture"


def caption_keywords(caption: str) -> Tuple[list[str], list[str]]:
    lowered = caption.lower()
    tags: list[str] = []
    objects: list[str] = []
    tag_keywords = {
        "person": "person_visible",
        "people": "person_visible",
        "hand": "hand_visible",
        "hands": "hand_visible",
        "screen": "screen_ui",
        "webpage": "screen_ui",
        "website": "screen_ui",
        "dashboard": "screen_ui",
        "button": "screen_ui",
        "page": "screen_ui",
        "text": "text_heavy_scene",
        "document": "text_heavy_scene",
        "desk": "desk_scene",
        "table": "desk_scene",
        "office": "office_scene",
        "room": "indoor_scene",
        "street": "street_scene",
        "city": "street_scene",
    }
    object_keywords = [
        "phone",
        "hand",
        "mug",
        "cup",
        "bottle",
        "laptop",
        "keyboard",
        "desk",
        "table",
        "chair",
        "door",
        "arm",
        "sanitizer",
        "bag",
        "screen",
        "page",
        "dashboard",
        "button",
        "text",
    ]
    for token, tag in tag_keywords.items():
        if token in lowered and tag not in tags:
            tags.append(tag)
    for token in object_keywords:
        if token in lowered and token not in objects:
            objects.append(token)
    return tags, objects


@dataclass
class LoadedRuntime:
    model_alias: str
    model_path: str
    device: str
    dtype_name: str
    processor: Any
    model: Any
    processor_call_kwargs: Dict[str, Any]
    image_seq_len: int
    longest_side_px: int


@dataclass
class SidecarConfig:
    base: str
    prompt: str
    primary_model_id: str
    fallback_model_id: str
    primary_model_path: str
    fallback_model_path: str
    runtime_device: str
    longest_side_px: int
    image_seq_len: int
    disable_image_splitting: bool
    max_new_tokens: int
    auto_fallback_latency_ms: int
    auto_fallback_cooldown_ms: int
    max_consecutive_failures: int


@dataclass
class RuntimeState:
    config: SidecarConfig
    lock: threading.Lock = field(default_factory=threading.Lock)
    loaded: Optional[LoadedRuntime] = None
    runtime_device_override_until_ms: int = 0
    fallback_until_ms: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_latency_ms: Optional[float] = None
    last_model_alias: str = ""
    last_model_path: str = ""
    last_fallback_reason: str = ""
    runtime_device_override: str = ""
    runtime_device_override_reason: str = ""

    def health_payload(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        fallback_active = self.fallback_until_ms > now_ms
        effective_runtime_device = self.effective_runtime_device()
        runtime_device_override = self.runtime_device_override or None
        runtime_device_override_reason = self.runtime_device_override_reason or None
        return {
            "ok": True,
            "service": "edge_vlm_sidecar",
            "configured": bool(self.config.primary_model_path),
            "primary_model_id": self.config.primary_model_id,
            "fallback_model_id": self.config.fallback_model_id,
            "active_model_id": self.last_model_alias or self.config.primary_model_id,
            "active_model_path": self.last_model_path,
            "runtime_device": effective_runtime_device,
            "runtime_device_requested": self.config.runtime_device,
            "runtime_device_override": runtime_device_override,
            "runtime_device_override_reason": runtime_device_override_reason,
            "fallback_active": fallback_active,
            "fallback_until_ms": self.fallback_until_ms if fallback_active else None,
            "fallback_reason": self.last_fallback_reason or None,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error or None,
            "edge_profile": {
                "disable_image_splitting": self.config.disable_image_splitting,
                "longest_side_px": self.config.longest_side_px,
                "image_seq_len": self.config.image_seq_len,
            },
        }

    def current_slot(self) -> Tuple[str, str, bool]:
        now_ms = int(time.time() * 1000)
        fallback_active = self.fallback_until_ms > now_ms
        if fallback_active and self.config.fallback_model_path:
            return self.config.fallback_model_id, self.config.fallback_model_path, True
        return self.config.primary_model_id, self.config.primary_model_path, False

    def activate_fallback(self, reason: str) -> None:
        if not self.config.fallback_model_path:
            return
        self.fallback_until_ms = int(time.time() * 1000) + self.config.auto_fallback_cooldown_ms
        self.last_fallback_reason = reason

    def effective_runtime_device(self) -> str:
        if self.runtime_device_override:
            now_ms = int(time.time() * 1000)
            if self.runtime_device_override_until_ms > now_ms:
                return self.runtime_device_override
            self.runtime_device_override = ""
            self.runtime_device_override_reason = ""
            self.runtime_device_override_until_ms = 0
        return self.config.runtime_device

    def override_runtime_device(self, device: str, reason: str) -> None:
        self.runtime_device_override = device
        self.runtime_device_override_reason = reason
        self.runtime_device_override_until_ms = int(time.time() * 1000) + int(
            self.config.auto_fallback_cooldown_ms
        )


def load_runtime_modules() -> Tuple[Any, Any, Any, Any]:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    ensure_torchvision_import_compat(torch)
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    return torch, Image, AutoConfig, (AutoProcessor, AutoModelForImageTextToText)


def decode_pil_image(image_b64: str, image_cls: Any) -> Any:
    raw = base64.b64decode(image_b64.encode("utf-8"))
    from io import BytesIO

    return image_cls.open(BytesIO(raw)).convert("RGB")


def load_model(state: RuntimeState, model_alias: str, model_path: str) -> LoadedRuntime:
    torch, _, auto_config_cls, auto_classes = load_runtime_modules()
    auto_processor_cls, auto_model_cls = auto_classes
    device = resolve_torch_device(torch, state.effective_runtime_device())
    if (
        state.loaded
        and state.loaded.model_alias == model_alias
        and state.loaded.model_path == model_path
        and state.loaded.device == device
    ):
        return state.loaded

    release_loaded_runtime(state)
    dtype = torch_dtype_for_device(torch, device)

    config = auto_config_cls.from_pretrained(model_path)
    processor_call_kwargs: Dict[str, Any] = {}
    if getattr(config, "model_type", "") == "smolvlm":
        from transformers import (  # noqa: PLC0415
            SmolVLMForConditionalGeneration,
            SmolVLMProcessor,
        )

        processor = SmolVLMProcessor.from_pretrained(model_path)
        if device == "cuda":
            processor.image_seq_len = int(state.config.image_seq_len)
            processor_call_kwargs = {
                "images_kwargs": {
                    "do_image_splitting": not state.config.disable_image_splitting,
                    "size": {"longest_edge": int(state.config.longest_side_px)},
                    "max_image_size": {"longest_edge": int(state.config.longest_side_px)},
                }
            }
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    else:
        processor = auto_processor_cls.from_pretrained(model_path)
        model = auto_model_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    model.to(device)
    model.eval()
    state.loaded = LoadedRuntime(
        model_alias=model_alias,
        model_path=model_path,
        device=device,
        dtype_name=str(dtype).replace("torch.", ""),
        processor=processor,
        model=model,
        processor_call_kwargs=processor_call_kwargs,
        image_seq_len=int(state.config.image_seq_len),
        longest_side_px=int(state.config.longest_side_px),
    )
    return state.loaded


def release_loaded_runtime(state: RuntimeState) -> None:
    loaded = state.loaded
    if not loaded:
        return
    try:
        del loaded.model
    except Exception:
        pass
    state.loaded = None
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def infer_once(state: RuntimeState, payload: Dict[str, Any], model_alias: str, model_path: str) -> Dict[str, Any]:
    torch, image_cls, _, _ = load_runtime_modules()
    runtime = load_model(state, model_alias, model_path)
    image = decode_pil_image(str(payload["image_jpeg_b64"]), image_cls)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "inline.jpg"},
                {"type": "text", "text": state.config.prompt},
            ],
        }
    ]
    prompt = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = runtime.processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        **runtime.processor_call_kwargs,
    )
    inputs = move_batch_to_device(
        torch,
        inputs,
        device=runtime.device,
        dtype=torch_dtype_for_device(torch, runtime.device) if runtime.device == "cuda" else None,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max(state.config.max_new_tokens, 8),
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    generated_ids = output_ids[:, prompt_tokens:]
    raw_text = runtime.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption = normalize_caption(raw_text)
    derived_tags, derived_objects = caption_keywords(caption)
    action_guess = infer_action_from_caption(caption)
    return {
        "ok": True,
        "caption": caption,
        "tags": derived_tags,
        "objects": derived_objects,
        "action_guess": action_guess,
        "latency_ms": round(elapsed_ms, 3),
        "model_id": model_alias,
        "model_path": model_path,
        "runtime_profile": {
            "device": runtime.device,
            "torch_dtype": runtime.dtype_name,
            "disable_image_splitting": state.config.disable_image_splitting,
            "longest_side_px": runtime.longest_side_px,
            "image_seq_len": runtime.image_seq_len,
        },
    }


def infer_with_fallback(state: RuntimeState, payload: Dict[str, Any]) -> Dict[str, Any]:
    model_alias, model_path, fallback_active = state.current_slot()
    degraded_reasons: list[str] = []
    try:
        result = infer_once(state, payload, model_alias, model_path)
        state.consecutive_failures = 0
        state.last_error = ""
        state.last_latency_ms = float(result["latency_ms"])
        state.last_model_alias = model_alias
        state.last_model_path = model_path
        if (
            not fallback_active
            and state.config.fallback_model_path
            and result["latency_ms"] > state.config.auto_fallback_latency_ms
        ):
            state.activate_fallback(
                f"primary_latency_ms>{state.config.auto_fallback_latency_ms}"
            )
            degraded_reasons.append("latency_threshold_exceeded")
        result["fallback_active"] = fallback_active or bool(degraded_reasons)
        result["degraded_reasons"] = degraded_reasons
        result["inference_source"] = "vlm_sidecar"
        return result
    except Exception as error:
        state.consecutive_failures += 1
        state.last_error = str(error)
        if is_cuda_runtime_error(error) and state.effective_runtime_device() != "cpu":
            release_loaded_runtime(state)
            state.override_runtime_device("cpu", f"cuda_runtime_error:{error}")
            try:
                retry = infer_once(state, payload, model_alias, model_path)
            except Exception as cpu_error:
                state.last_error = str(cpu_error)
            else:
                state.consecutive_failures = 0
                state.last_error = ""
                state.last_latency_ms = float(retry["latency_ms"])
                state.last_model_alias = model_alias
                state.last_model_path = model_path
                retry["fallback_active"] = True
                retry["degraded_reasons"] = retry.get("degraded_reasons", [])
                retry["degraded_reasons"].append("cuda_runtime_fallback_to_cpu")
                retry["inference_source"] = "vlm_sidecar"
                return retry
        if state.config.fallback_model_path and model_path != state.config.fallback_model_path:
            degraded_reasons.append("primary_runtime_error")
            release_loaded_runtime(state)
            if state.consecutive_failures >= state.config.max_consecutive_failures:
                state.activate_fallback(f"primary_runtime_error:{error}")
            else:
                degraded_reasons.append("fallback_retry_without_cooldown")
            fallback_alias = state.config.fallback_model_id
            fallback_path = state.config.fallback_model_path
            try:
                retry = infer_once(state, payload, fallback_alias, fallback_path)
            except Exception as fallback_error:
                state.last_error = str(fallback_error)
            else:
                retry["fallback_active"] = True
                retry["degraded_reasons"] = degraded_reasons
                retry["inference_source"] = "vlm_sidecar_fallback"
                state.last_latency_ms = float(retry["latency_ms"])
                state.last_model_alias = fallback_alias
                state.last_model_path = fallback_path
                state.last_error = ""
                return retry
        raise


class SidecarHandler(BaseHTTPRequestHandler):
    runtime_state: RuntimeState

    server_version = "edge_vlm_sidecar/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, self.runtime_state.health_payload())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/infer":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty_body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as error:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"invalid_json: {error}"},
            )
            return
        if not isinstance(payload, dict):
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "payload_not_object"})
            return
        if not isinstance(payload.get("image_jpeg_b64"), str) or not payload["image_jpeg_b64"].strip():
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "image_jpeg_b64_required"},
            )
            return

        with self.runtime_state.lock:
            try:
                result = infer_with_fallback(self.runtime_state, payload)
                self._write_json(HTTPStatus.OK, result)
            except Exception as error:
                self.runtime_state.last_error = str(error)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "ok": False,
                        "error": str(error),
                        "fallback_active": self.runtime_state.fallback_until_ms > int(time.time() * 1000),
                    },
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHEK edge VLM semantic sidecar")
    parser.add_argument("--base", default=non_empty_env("EDGE_VLM_SIDECAR_BASE", "http://127.0.0.1:3032"))
    parser.add_argument("--prompt", default=non_empty_env("EDGE_VLM_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--primary-model-id", default=non_empty_env("EDGE_VLM_MODEL_ID", "SmolVLM2-500M"))
    parser.add_argument("--fallback-model-id", default=non_empty_env("EDGE_VLM_FALLBACK_MODEL_ID", "SmolVLM2-256M"))
    parser.add_argument("--primary-model-path", default=non_empty_env("EDGE_VLM_PRIMARY_MODEL_PATH"))
    parser.add_argument("--fallback-model-path", default=non_empty_env("EDGE_VLM_FALLBACK_MODEL_PATH"))
    parser.add_argument("--runtime-device", default=non_empty_env("EDGE_VLM_RUNTIME_DEVICE", "auto"))
    parser.add_argument("--longest-side-px", type=int, default=env_int("EDGE_VLM_EDGE_LONGEST_SIDE_PX", 256))
    parser.add_argument("--image-seq-len", type=int, default=env_int("EDGE_VLM_EDGE_IMAGE_SEQ_LEN", 16))
    parser.add_argument(
        "--disable-image-splitting",
        action="store_true",
        default=env_bool("EDGE_VLM_DISABLE_IMAGE_SPLITTING", True),
    )
    parser.add_argument("--max-new-tokens", type=int, default=env_int("EDGE_VLM_MAX_NEW_TOKENS", 12))
    parser.add_argument(
        "--auto-fallback-latency-ms",
        type=int,
        default=env_int("EDGE_VLM_AUTO_FALLBACK_LATENCY_MS", 2200),
    )
    parser.add_argument(
        "--auto-fallback-cooldown-ms",
        type=int,
        default=env_int("EDGE_VLM_AUTO_FALLBACK_COOLDOWN_MS", 60000),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=env_int("EDGE_VLM_MAX_CONSECUTIVE_FAILURES", 2),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host, port = base_to_host_port(args.base)
    runtime_state = RuntimeState(
        config=SidecarConfig(
            base=args.base,
            prompt=args.prompt,
            primary_model_id=args.primary_model_id,
            fallback_model_id=args.fallback_model_id,
            primary_model_path=args.primary_model_path,
            fallback_model_path=args.fallback_model_path,
            runtime_device=args.runtime_device,
            longest_side_px=int(args.longest_side_px),
            image_seq_len=int(args.image_seq_len),
            disable_image_splitting=bool(args.disable_image_splitting),
            max_new_tokens=int(args.max_new_tokens),
            auto_fallback_latency_ms=int(args.auto_fallback_latency_ms),
            auto_fallback_cooldown_ms=int(args.auto_fallback_cooldown_ms),
            max_consecutive_failures=int(args.max_consecutive_failures),
        )
    )
    SidecarHandler.runtime_state = runtime_state
    server = ThreadingHTTPServer((host, port), SidecarHandler)
    print(
        json.dumps(
            {
                "ok": True,
                "service": "edge_vlm_sidecar",
                "host": host,
                "port": port,
                "primary_model_id": args.primary_model_id,
                "fallback_model_id": args.fallback_model_id,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
