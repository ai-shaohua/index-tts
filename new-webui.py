"""句末切分合成的 Gradio 界面（默认 7861）。

只通过 HTTP 调 `new-server.py`（默认 http://127.0.0.1:3003）。
**不 import indextts，不构造任何模型对象** —— 同一时刻只允许一个带模型的进程
（见 memory/tts-benchmark-single-process.md）。

点击生成只调一次 `POST /tts/rich`：切分、逐段合成、拼接都在服务器里，
响应已同时带回 sentences / sentences_normalized / silences_ms 和音频，
所以界面不需要再调 debug 用的 `/split`。
方案见 PRDs/2026-10/sentence-split-tts-spec.md 第 6 节。
"""

import argparse
import base64
import glob
import os
import tempfile

import gradio as gr
import requests

parser = argparse.ArgumentParser(
    description="IndexTTS sentence-split WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--port", type=int, default=7861, help="Port to run the web UI on")
parser.add_argument("--server", type=str, default="http://127.0.0.1:3003",
                    help="Base URL of new-server.py")
parser.add_argument("--examples_dir", type=str, default="./wolf-audio",
                    help="Voice prompt directory; the dropdown lists *.wav from here")
parser.add_argument("--timeout", type=float, default=600.0,
                    help="HTTP timeout in seconds; synthesizing a long speech is slow")
parser.add_argument("--share", action="store_true", default=False,
                    help="Enable Gradio sharing mode")
cmd_args = parser.parse_args()

LANGUAGES = ["ZH", "EN", "JA", "AR", "ES"]

DEFAULT_TEXT = (
    "6号开枪，铁狼王，我是真预言家。昨夜验12金水，今夜原定验7，可惜未能落验，"
    "故7绝非我的金水，别误传。3、7号首轮站6，但3号最终回票6，7号仍投我，"
    "7号狼面最高；我信逻辑，不信遗言污金水。"
)


def list_voices():
    """wolf-audio/*.wav -> 文件名去扩展名。启动时扫描一次。"""
    paths = sorted(glob.glob(os.path.join(cmd_args.examples_dir, "*.wav")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def render_segments(sentences, silences_ms):
    """切段展示：每段**原文**与其后的停顿毫秒。

    展示原文而不是归一化结果，因为原文才是逐字送进 infer() 的那串文本。
    """
    lines = []
    for i, seg in enumerate(sentences):
        pause = f"后面 {silences_ms[i]}ms" if i < len(silences_ms) else "后面没有额外停顿"
        lines.append(f"{i + 1}. {seg}　　**{pause}**")
    return "\n\n".join(lines)


def render_normalized(sentences_normalized):
    """折叠区里的读法，段序号与切段展示对齐。"""
    return "\n\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(sentences_normalized))


def generate(agent_name, lang, text):
    """调一次 POST /tts，用同一个响应填切段列表、折叠读法和播放器。

    返回 (切段 Markdown, 读法 Markdown, 音频路径, 错误 Markdown, 按钮状态)。
    失败时不动原文输入框。
    """
    def fail(msg):
        return "", "", None, f"❌ {msg}", gr.update(interactive=True)

    if not agent_name:
        return fail("请先选择音色。")
    if not text or not text.strip():
        return fail("请输入原文。")

    try:
        resp = requests.post(
            f"{cmd_args.server.rstrip('/')}/tts/rich",
            json={"agentName": agent_name, "text": text, "lang": lang},
            timeout=cmd_args.timeout,
        )
    except requests.RequestException as e:
        return fail(f"连不上 {cmd_args.server}：{e}（new-server.py 起了吗？）")

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return fail(f"服务器返回 {resp.status_code}：{detail}")

    data = resp.json()
    # gr.Audio 要文件路径，所以 base64 解码后落一个临时 mp3。
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(base64.b64decode(data["audio_base64"]))
        mp3_path = tmp.name

    return (
        render_segments(data["sentences"], data["silences_ms"]),
        render_normalized(data["sentences_normalized"]),
        mp3_path,
        "",
        gr.update(interactive=True),
    )


with gr.Blocks(title="IndexTTS 句末切分合成") as demo:
    gr.Markdown(
        "## IndexTTS 句末与分号切分合成\n"
        "按句末标点和分号切开，逐段合成后拼接：句末后面 500ms，分号后面 333ms。"
        f"合成由 `{cmd_args.server}` 完成，本页面不加载模型。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            voices = list_voices()
            voice = gr.Dropdown(
                choices=voices,
                value=voices[0] if voices else None,
                label="音色",
                info=f"来自 {cmd_args.examples_dir}/*.wav",
            )
            lang = gr.Dropdown(choices=LANGUAGES, value="ZH", label="语言")
        with gr.Column(scale=3):
            text = gr.Textbox(
                value=DEFAULT_TEXT,
                label="原文",
                lines=6,
                placeholder="粘贴发言原文，句末和分号处会被切开逐段合成",
            )

    run = gr.Button("生成", variant="primary")
    error = gr.Markdown("")
    audio = gr.Audio(label="音频", type="filepath")
    segments = gr.Markdown("")
    with gr.Accordion(label="读法（归一化后）", open=False):
        gr.Markdown(
            "_归一化这一步的近似结果，供排查某段数字读错是归一化的问题还是模型念错。_\n\n"
            "_没有复现其后的 `.lower()`、发音标注展开、`<|LANG|>` 前缀和 tokenizer 切分，"
            "不要当成模型最终看到的 token。段末的 `；` 在这里会显示成 `,`，模型内部确实这么收，"
            "不影响停顿 —— 停顿是拼在音频上的静音。_"
        )
        normalized = gr.Markdown("")

    run.click(
        # 点下去先置灰，避免重复提交；generate() 返回时恢复。
        fn=lambda: gr.update(interactive=False),
        outputs=[run],
    ).then(
        fn=generate,
        inputs=[voice, lang, text],
        outputs=[segments, normalized, audio, error, run],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=cmd_args.share,
    )
