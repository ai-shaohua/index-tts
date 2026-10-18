"""IndexTTS FastAPI server with sentence-level splitting (port 3003).

A copy of server.py that splits the incoming text at sentence ends and
semicolons, synthesizes each segment separately, and concatenates the pieces
with a fixed silence between them -- so a full stop gets a real 500ms pause
instead of whatever breath the model happened to generate.

Splitting happens BEFORE normalization (see PRDs/2026-10/sentence-split-tts-spec.md
section 1): the semicolon is only used to pick the cut points, and each raw
segment then goes through the model's own normalization untouched. The model
still collapses a trailing ';' to ',' internally -- that does not matter,
because the pause is silence spliced into the audio, not a symbol in the text.

server.py and webui.py are deliberately left alone; this is a new port.

Benchmarking / debugging note
----------------------------
Keep exactly ONE process with the model loaded. Measure latency by POSTing to
a running server's /tts and reading the per-segment lines it logs (one per
sentence, with that segment's audio length and wall time) -- do NOT start a
separate script that constructs its own IndexTTS() while this server is up.
Two resident models exhaust RAM and will crash the machine. Kill stale servers
first:

    pgrep -f "new-server.py|server.py" | xargs -r kill -9

Unlike server.py this does not report the gpt/cfm/bigvgan stage breakdown:
those timings are per-synthesize() and there is now one call per sentence.

Send requests sequentially; inference is serialized behind infer_lock anyway,
so concurrent requests only add memory pressure without improving throughput.
Note that background system load can easily double a measurement -- check
`uptime` before trusting absolute timings.
"""

import argparse
import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_split import concat_segments, split_sentences


parser = argparse.ArgumentParser(description="IndexTTS FastAPI Server")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=3003)
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--version", type=str, default="2.5", choices=["2", "2.5"], help="Model version to use")
parser.add_argument("--examples_dir", type=str, default="./wolf-audio")
parser.add_argument("--fp16", action="store_true", default=False, help="Use half precision for inference if available (FP16 on v2, BF16 on v2.5)")
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
parser.add_argument("--accel", action="store_true", default=False, help="Use GPT2 acceleration engine if available")
parser.add_argument("--torch_compile", action="store_true", default=False, help="Use torch.compile to optimize s2mel if available")
parser.add_argument("--qwen_emo", action="store_true", default=False, help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")
parser.add_argument("--mp3_bitrate", type=str, default="128k")
parser.add_argument("--n_timesteps", type=int, default=25, help="MLX: CFM solver steps per segment (upstream default 25; 10 is ~2.5x faster with little audible loss)")
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

    def _synthesize_mlx_pcm(spk, text, lang, duration_factor,
                            n_timesteps, max_text_tokens_per_segment):
        """One segment -> int16 PCM [N]. No wav file; the caller concatenates.

        interval_silence=0 because any split *inside* this segment is the
        model's own token-budget split, which must not get a 500/333ms pause
        (spec section 4.1). The speaker context is built once by the caller and
        reused for every segment.
        """
        return tts.synthesize(
            text,
            lang=lang,
            spk=spk,
            duration_factor=duration_factor,
            n_timesteps=n_timesteps,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            interval_silence=0,
        )

SAMPLING_RATE = 22050   # both backends (pipeline.py:19, infer_v2_5.py:737)


def normalize_for_display(text: str, lang: str) -> str:
    """Reproduce the model's normalization step, for the debug read-out.

    APPROXIMATE on purpose: this mirrors `char_rep_map` cleanup + WeTextProcessing,
    but not the `.lower()`, pronunciation-annotation expansion, `<|LANG|>` prefix
    or tokenizer split that follow it. Good enough to tell "the normalizer read
    this number wrong" from "the model mispronounced it"; not the final tokens.

    A trailing `;` comes back as `,` here -- that is what the model really sees
    (spec section 5.1). The pause is spliced into the audio, so it is unaffected.
    """
    lang = (lang or "zh").lower()
    try:
        if USE_MLX:
            from index_tts_2_5_mlx.core.frontend import CHAR_REP_MAP, CLEAN_PATTERN

            text = CLEAN_PATTERN.sub(lambda x: CHAR_REP_MAP[x.group()], text)
            fe = tts._pipe.frontend
            if fe.use_normalization and lang in ("zh", "zhen", "en"):
                text = fe.normalizer.normalize(text)
            return text
        normalizer = getattr(tts, "text_process", None) or tts.normalizer
        text = normalizer.clean_pattern.sub(
            lambda x: normalizer.char_rep_map[x.group()], text)
        if lang in ("zh", "zhen", "en"):
            text = normalizer.normalize(text)
        return text
    except Exception as e:
        # A debug read-out must never fail the request it is attached to.
        print(f">> normalize_for_display failed: {e}")
        return text


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
    agentName: str = Field(..., description="Voice prompt name; resolves to wolf-audio/{agentName}.wav")
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


def _write_wav(path: str, pcm: np.ndarray) -> None:
    """int16 PCM [N] -> mono 22050 Hz wav."""
    import soundfile as sf

    sf.write(path, pcm.astype(np.int16, copy=False), SAMPLING_RATE, subtype="PCM_16")


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


def _resolve_lang(req: TTSRequest) -> str:
    """Validate `lang` against the active backend; returns the backend's own code."""
    lang = (req.lang or "ZH").upper()
    if USE_MLX:
        mlx_lang = MLX_LANGS.get(lang)
        if mlx_lang is None:
            raise HTTPException(
                status_code=400,
                detail=f"MLX backend supports {', '.join(MLX_LANGS)} (got {req.lang})",
            )
        return mlx_lang
    if IS_V25 and lang not in LANGUAGES_V25:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported lang: {req.lang} (choose from {LANGUAGES_V25})",
        )
    return lang


def _split_payload(req: TTSRequest, lang: str):
    """-> (pairs, response body). Both endpoints go through here, so there is
    one split per request and one implementation of it.

    `min_length=1` on TTSRequest.text lets a whitespace-only string through, so
    re-check after strip() here (spec section 5.2).
    """
    pairs = split_sentences(req.text)
    if not pairs:
        raise HTTPException(status_code=400, detail="text is empty after stripping whitespace")

    sentences = [seg for seg, _ in pairs]
    return pairs, {
        "sentences": sentences,
        "sentences_normalized": [normalize_for_display(seg, lang) for seg in sentences],
        # silences_ms[i] is the pause AFTER sentences[i]. The last segment's
        # pause is always 0 and is not reported, so this is one shorter.
        "silences_ms": [ms for _, ms in pairs[:-1]],
    }


@app.post("/split")
async def split(req: TTSRequest):
    """Debug only. Shows the cut points and the normalized read-out without
    spending synthesis time -- use it to tell a normalization bug from a
    mispronunciation. NOT a prerequisite for /tts, which splits internally.
    """
    return _split_payload(req, _resolve_lang(req))[1]


@app.post("/tts")
async def synthesize(req: TTSRequest):
    prompt_path = os.path.join(cmd_args.examples_dir, f"{req.agentName}.wav")
    if not os.path.isfile(prompt_path):
        raise HTTPException(status_code=404, detail=f"voice prompt not found: {prompt_path}")

    lang = _resolve_lang(req)
    pairs, payload = _split_payload(req, lang)

    if USE_MLX:
        global _mlx_beams_warned
        if req.num_beams != 1 and not _mlx_beams_warned:
            print(f">> MLX has no beam search; ignoring num_beams={req.num_beams}")
            _mlx_beams_warned = True
    else:
        infer_kwargs = {"num_beams": req.num_beams, "interval_silence": 0}
        if IS_V25:
            infer_kwargs["lang"] = lang
            infer_kwargs["duration_factor"] = float(req.duration_factor)

    def _segment_pcm_torch(seg_text):
        result = tts.infer(prompt_path, seg_text, None, **infer_kwargs)
        if not isinstance(result, tuple):
            return None
        _sr, wav_data = result   # (samples, channels)
        wav = np.asarray(wav_data)
        if wav.ndim > 1:
            wav = wav[:, 0]
        return wav.astype(np.int16, copy=False)

    started = time.perf_counter()
    seg_pcms = []
    # Hold the lock across ALL segments: releasing between them would let
    # another request interleave its own segments into the model.
    async with infer_lock:
        spk = mlx_speaker(prompt_path) if USE_MLX else None
        for i, (seg_text, ms) in enumerate(pairs, start=1):
            t_seg = time.perf_counter()
            if USE_MLX:
                # MLX streams are thread-local. The model was loaded on this
                # thread; a worker thread raises "no Stream(cpu, 1)".
                pcm = _synthesize_mlx_pcm(
                    spk,
                    seg_text,
                    lang,
                    float(req.duration_factor),
                    req.n_timesteps,
                    req.max_text_tokens_per_segment,
                )
            else:
                pcm = await asyncio.to_thread(_segment_pcm_torch, seg_text)

            if pcm is None or len(pcm) == 0:
                # One bad segment fails the whole request; never return a
                # half-finished utterance.
                raise HTTPException(
                    status_code=500,
                    detail=f"synthesis produced no audio for segment {i}/{len(pairs)}",
                )

            tail = seg_text.rstrip()[-1:] or "-"
            print(
                f">>   seg {i}/{len(pairs)} tail={tail!r} chars={len(seg_text)} "
                f"audio={len(pcm) / SAMPLING_RATE:.2f}s pause={ms}ms "
                f"{time.perf_counter() - t_seg:.2f}s"
            )

            seg_pcms.append(pcm)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        # concat_segments applies the 20ms tail fade before each silence: the
        # stop token sometimes lands early, and that step would otherwise click
        # at every seam (spec section 4.3).
        wav = concat_segments(seg_pcms, [ms for _, ms in pairs], SAMPLING_RATE)
        await asyncio.to_thread(_write_wav, wav_path, wav)
        mp3_bytes = await asyncio.to_thread(wav_to_mp3, wav_path, cmd_args.mp3_bitrate)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    print(
        f">> /tts agent={req.agentName} chars={len(req.text)} "
        f"segments={len(pairs)} n_timesteps={req.n_timesteps} "
        f"audio={len(wav) / SAMPLING_RATE:.2f}s {time.perf_counter() - started:.2f}s"
    )

    return {
        **payload,
        "audio_base64": base64.b64encode(mp3_bytes).decode("ascii"),
        "media_type": "audio/mpeg",
    }


if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="info")
