import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field


parser = argparse.ArgumentParser(description="IndexTTS FastAPI Server")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=3002)
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--version", type=str, default="2.5", choices=["2", "2.5"], help="Model version to use")
parser.add_argument("--examples_dir", type=str, default="./examples")
parser.add_argument("--fp16", action="store_true", default=False, help="Use half precision for inference if available (FP16 on v2, BF16 on v2.5)")
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
parser.add_argument("--accel", action="store_true", default=False, help="Use GPT2 acceleration engine if available")
parser.add_argument("--torch_compile", action="store_true", default=False, help="Use torch.compile to optimize s2mel if available")
parser.add_argument("--qwen_emo", action="store_true", default=False, help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")
parser.add_argument("--mp3_bitrate", type=str, default="128k")
parser.add_argument("--n_timesteps", type=int, default=10, help="MLX: CFM solver steps per segment (upstream default 25; 10 is ~2.5x faster with little audible loss)")
parser.add_argument("--mlx", action=argparse.BooleanOptionalAction, default=None, help="v2.5: serve with index-tts-2.5-mlx. On by default for v2.5; pass --no-mlx for the PyTorch backend. Does not change webui.py.")
cmd_args, _ = parser.parse_known_args()

if cmd_args.mlx is None:
    cmd_args.mlx = cmd_args.version == "2.5"
if cmd_args.mlx and cmd_args.version != "2.5":
    parser.error("--mlx requires --version 2.5")


# Validate optional acceleration dependencies early, so missing extras fail
# at startup instead of halfway through inference.
def _require_optional_extra(flag_name, module_name, install_cmd):
    try:
        __import__(module_name)
    except ImportError:
        parser.error(
            f"--{flag_name} requires {module_name}, which is not installed. "
            f"Install it with: {install_cmd}"
        )


if cmd_args.accel:
    _require_optional_extra("accel", "flash_attn", "uv sync --extra accel")

if cmd_args.torch_compile:
    _require_optional_extra("torch_compile", "triton", "uv sync --extra torch_compile")

IS_V25 = cmd_args.version == "2.5"
USE_MLX = cmd_args.mlx
DEFAULT_N_TIMESTEPS = cmd_args.n_timesteps

if not USE_MLX:
    REQUIRED_FILES = {
        "2": ["bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"],
        "2.5": [
            "gpt.pth",
            "s2mel.pth",
            "codec.pth",
            "multilingual_zh_ja_yue_char_del.tiktoken",
            "wav2vec2bert_stats.pt",
        ],
    }
    MODEL_REPO = {
        "2": "IndexTeam/IndexTTS-2",
        "2.5": "IndexTeam/IndexTTS-2.5",
    }
    required_files = REQUIRED_FILES[cmd_args.version]
    missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
    if missing:
        print(
            f"Model directory {cmd_args.model_dir} is incomplete for v{cmd_args.version} "
            f"(missing: {', '.join(missing)}). Downloading {MODEL_REPO[cmd_args.version]}..."
        )
        from indextts.utils.model_download import snapshot_download
        try:
            snapshot_download(MODEL_REPO[cmd_args.version], local_dir=cmd_args.model_dir)
        except Exception as e:
            print(f"Failed to download model to {cmd_args.model_dir}: {e}")
            sys.exit(1)
        missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
        if missing:
            print(f"Failed to download model to {cmd_args.model_dir} (still missing: {', '.join(missing)}). Please download it manually.")
            sys.exit(1)
        print("Model downloaded successfully.")

    from indextts.utils.model_download import ensure_config_available
    try:
        ensure_config_available(cmd_args.model_dir, version=cmd_args.version)
    except Exception as e:
        print(f"Failed to download config.yaml: {e}")
        sys.exit(1)

    if IS_V25:
        from indextts.infer_v2_5 import IndexTTS2
    else:
        from indextts.infer_v2 import IndexTTS2

    LOW_VRAM_THRESHOLD_GB = 10.0

    def detect_vram_gb():
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    _vram_gb = detect_vram_gb()
    LOW_VRAM = _vram_gb is not None and _vram_gb < LOW_VRAM_THRESHOLD_GB

    HALF_PRECISION = cmd_args.fp16 or LOW_VRAM
    LOAD_QWEN_EMO = cmd_args.qwen_emo or not LOW_VRAM

    if LOW_VRAM:
        print(
            f">> {_vram_gb:.1f} GB VRAM detected (< {LOW_VRAM_THRESHOLD_GB:.0f} GB): "
            "enabling half precision"
            + ("" if LOAD_QWEN_EMO else " and skipping QwenEmotion")
        )
        if not LOAD_QWEN_EMO:
            print(">> Emotion control from a text description is unavailable; pass --qwen_emo to force it.")

    def build_tts(use_accel=False, use_torch_compile=False):
        """Build an IndexTTS2 instance with the requested acceleration options."""
        import torch

        kwargs = dict(
            model_dir=cmd_args.model_dir,
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            use_deepspeed=cmd_args.deepspeed,
            use_cuda_kernel=cmd_args.cuda_kernel,
            use_accel=use_accel,
            use_torch_compile=use_torch_compile,
            use_qwen_emo=LOAD_QWEN_EMO,
        )
        if IS_V25:
            use_bf16 = HALF_PRECISION and torch.cuda.is_bf16_supported()
            if HALF_PRECISION and not use_bf16:
                print(">> BF16 is not supported on this device, falling back to full precision.")
            kwargs["use_bf16"] = use_bf16
        else:
            kwargs["use_fp16"] = HALF_PRECISION
        return IndexTTS2(**kwargs)

    print(f">> loading IndexTTS {cmd_args.version} model...")
    tts = build_tts(use_accel=cmd_args.accel, use_torch_compile=cmd_args.torch_compile)
else:
    try:
        from index_tts_2_5_mlx import IndexTTS as MlxIndexTTS
    except ImportError:
        parser.error(
            "--mlx requires index-tts-2.5-mlx, which is not installed. "
            "Install it with: uv pip install index-tts-2.5-mlx"
        )
    print(">> loading IndexTTS 2.5 MLX model...")
    tts = MlxIndexTTS()
    _mlx_speakers = {}
    _mlx_beams_warned = False
    MLX_LANGS = {"ZH": "zh", "EN": "en", "JA": "ja", "YUE": "yue"}

    def mlx_speaker(prompt_path):
        key = (prompt_path, os.path.getmtime(prompt_path))
        spk = _mlx_speakers.get(key)
        if spk is None:
            spk = tts.build_speaker(prompt_path)
            _mlx_speakers[key] = spk
        return spk

    def _synthesize_mlx(prompt_path, text, wav_path, lang, duration_factor,
                        n_timesteps, max_text_tokens_per_segment):
        t0 = time.perf_counter()
        spk = mlx_speaker(prompt_path)
        t_spk = time.perf_counter() - t0

        t0 = time.perf_counter()
        pcm = tts.synthesize(
            text,
            lang=lang,
            spk=spk,
            duration_factor=duration_factor,
            n_timesteps=n_timesteps,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
        )
        t_syn = time.perf_counter() - t0

        t0 = time.perf_counter()
        tts.write_wav(wav_path, pcm)
        timings = dict(getattr(tts._pipe, "last_timings", {}))
        timings["spk"] = t_spk
        timings["wav"] = time.perf_counter() - t0
        # Whatever synthesize() spent outside the stages it reports (text
        # frontend, numpy concat, MLX graph sync).
        timings["other"] = t_syn - sum(
            v for k, v in timings.items() if k not in ("spk", "wav"))
        return timings

# inference is not thread-safe — serialize calls
infer_lock = asyncio.Lock()

app = FastAPI(title="IndexTTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v2.5 supported languages (same choices as webui.py)
LANGUAGES_V25 = ["ZH", "EN", "JA", "AR", "ES"]


class TTSRequest(BaseModel):
    agentName: str = Field(..., description="Voice prompt name; resolves to examples/{agentName}.wav")
    text: str = Field(..., min_length=1)
    lang: str = Field(default="ZH", description=f"Text language, one of {LANGUAGES_V25} (IndexTTS 2.5 only)")
    duration_factor: float = Field(default=1.0, description="Speech duration factor (IndexTTS 2.5 only)")
    # Beam search dominates latency on MPS. 1 is sampling (fast); 3 matches the WebUI quality default.
    num_beams: int = Field(default=1, ge=1, le=10)
    # MLX only. CFM solver steps: each step is one DiT call, so latency is
    # linear in this. 25 is the upstream default; 10 is near-indistinguishable.
    n_timesteps: int = Field(default=DEFAULT_N_TIMESTEPS, ge=1, le=50)
    # MLX only. Text is split into segments; each segment is a separate
    # GPT+CFM+vocoder pass, so smaller segments mean more total work.
    max_text_tokens_per_segment: int = Field(default=120, ge=20, le=400)


def wav_to_mp3(wav_path: str, bitrate: str) -> bytes:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", wav_path,
            "-vn", "-c:a", "libmp3lame", "-b:a", bitrate,
            "-f", "mp3", "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": cmd_args.version,
        "backend": "mlx" if USE_MLX else "pytorch",
        "model_version": "2.5-mlx" if USE_MLX else (tts.model_version or "1.0"),
    }


@app.post("/tts")
async def synthesize(req: TTSRequest):
    prompt_path = os.path.join(cmd_args.examples_dir, f"{req.agentName}.wav")
    if not os.path.isfile(prompt_path):
        raise HTTPException(status_code=404, detail=f"voice prompt not found: {prompt_path}")

    if USE_MLX:
        lang = (req.lang or "ZH").upper()
        mlx_lang = MLX_LANGS.get(lang)
        if mlx_lang is None:
            raise HTTPException(
                status_code=400,
                detail=f"MLX backend supports {', '.join(MLX_LANGS)} (got {req.lang})",
            )
        global _mlx_beams_warned
        if req.num_beams != 1 and not _mlx_beams_warned:
            print(f">> MLX has no beam search; ignoring num_beams={req.num_beams}")
            _mlx_beams_warned = True
    else:
        infer_kwargs = {"num_beams": req.num_beams}
        if IS_V25:
            lang = (req.lang or "ZH").upper()
            if lang not in LANGUAGES_V25:
                raise HTTPException(status_code=400, detail=f"unsupported lang: {req.lang} (choose from {LANGUAGES_V25})")
            infer_kwargs["lang"] = lang
            infer_kwargs["duration_factor"] = float(req.duration_factor)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    started = time.perf_counter()
    try:
        async with infer_lock:
            if USE_MLX:
                # MLX streams are thread-local. The model was loaded on this
                # thread; a worker thread raises "no Stream(cpu, 1)".
                timings = _synthesize_mlx(
                    prompt_path,
                    req.text,
                    wav_path,
                    mlx_lang,
                    float(req.duration_factor),
                    req.n_timesteps,
                    req.max_text_tokens_per_segment,
                )
            else:
                timings = None
                await asyncio.to_thread(
                    tts.infer,
                    prompt_path,
                    req.text,
                    wav_path,
                    **infer_kwargs,
                )
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise HTTPException(status_code=500, detail="synthesis produced no audio")

        t_mp3 = time.perf_counter()
        mp3_bytes = await asyncio.to_thread(wav_to_mp3, wav_path, cmd_args.mp3_bitrate)
        if timings:
            timings["mp3"] = time.perf_counter() - t_mp3

        detail = ""
        if timings:
            detail = " " + " ".join(f"{k}={v:.2f}" for k, v in timings.items() if v >= 0.01)
        print(
            f">> /tts agent={req.agentName} chars={len(req.text)} "
            f"n_timesteps={req.n_timesteps} {time.perf_counter() - started:.2f}s{detail}"
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return Response(content=mp3_bytes, media_type="audio/mpeg")


if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="info")
