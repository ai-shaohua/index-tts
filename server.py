import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
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

from indextts.infer_v2 import IndexTTS2


parser = argparse.ArgumentParser(description="IndexTTS FastAPI Server")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=3002)
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--examples_dir", type=str, default="./examples")
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
parser.add_argument("--mp3_bitrate", type=str, default="128k")
cmd_args, _ = parser.parse_known_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist.")
    sys.exit(1)

for f in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
    if not os.path.exists(os.path.join(cmd_args.model_dir, f)):
        print(f"Required file {f} missing in {cmd_args.model_dir}")
        sys.exit(1)

print(">> loading IndexTTS2 model...")
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

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


class TTSRequest(BaseModel):
    agentName: str = Field(..., description="Voice prompt name; resolves to examples/{agentName}.wav")
    text: str = Field(..., min_length=1)


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
    return {"status": "ok", "model_version": tts.model_version or "1.0"}


@app.post("/tts")
async def synthesize(req: TTSRequest):
    prompt_path = os.path.join(cmd_args.examples_dir, f"{req.agentName}.wav")
    if not os.path.isfile(prompt_path):
        raise HTTPException(status_code=404, detail=f"voice prompt not found: {prompt_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        async with infer_lock:
            await asyncio.to_thread(
                tts.infer,
                prompt_path,
                req.text,
                wav_path,
            )

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise HTTPException(status_code=500, detail="synthesis produced no audio")

        mp3_bytes = await asyncio.to_thread(wav_to_mp3, wav_path, cmd_args.mp3_bitrate)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return Response(content=mp3_bytes, media_type="audio/mpeg")


if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="info")
