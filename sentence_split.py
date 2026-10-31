"""句末 / 分号 / 冒号切分、短段合并与段尾淡出 —— new-server.py 的纯函数。

放在单独模块里只有一个原因：`new-server.py` 在 import 时就加载模型权重，
测试没法 import 它。这里不 import 任何模型相关的东西，
`tests/test_sentence_split.py` 直接 `import sentence_split` 即可。

`/split` 与 `/tts` 都调用这里的 `split_sentences`，不存在第二份实现。
方案见 PRDs/2026-10/sentence-split-tts-spec.md。
"""

import math
import re

import numpy as np

# 句末 / 分号 / 冒号。ASCII '.' 只在它是真小数点时不切：两侧都是数字才算小数
# （`0.7` 是小数，`.7` 不是）。归一化前的原文里 `3.5` 还没变成「三点五」，
# 按 '.' 无条件切会读成「三点」+「五」。
#
# 冒号反过来：只有后面跟数字才切（`验人结果：12金水`），`他说：我是预言家`
# 这种领起下文的冒号不切。前面也是数字时仍然不切 —— 那是 `12:30` 这种时间，
# 切开会读成「十二」+「三十」。
SPLIT_PATTERN = re.compile(
    r'(?<=[。！？；!?;])'      # 句末标点、分号
    r'|(?<=\.)(?!\d)'          # '.' 后面不是数字 -> 句号
    r'|(?<=(?<!\d)\.)(?=\d)'   # '.' 后面是数字但前面不是 -> 仍是句号（如 "出局.7号"）
    r'|(?<=(?<!\d)[:：])(?=\d)'  # 冒号后面是数字、前面不是 -> 切（如 "结论:3号是狼"）
)

# --- 停顿时长：调效果就改这两个常量，其余全部由它们推导 ---
SENTENCE_PAUSE_MS = 150                       # 句末停顿
# 分号与冒号共用同一个停顿时长。
SEMICOLON_PAUSE_MS = 150

SENTENCE_PUNCT = frozenset('.。!！?？')
SEMICOLON_PUNCT = frozenset(';；:：')

# --- 合并阈值：太短的片段单独合成音质差，所以先攒够字数再送 ---
# 效果递减的顺序是：长度适中 > 稍长 > 过短。所以宁可让 chunk 偏长，也不留短段。
# 当前 chunk 不足这么多字时，下一句并进来而不是另起一段。
MIN_CHUNK_CHARS = 40
# 最后一个 chunk 不足这么多字时并回前一个，绝不单独成段。
MIN_TAIL_CHUNK_CHARS = 30
# 并进来之后超过这个长度就放弃合并，让下一句另起一段 —— 偏长比过短好，
# 但也有限度。减 5 是为了把上界压在「并完刚好 80」之下。
# 末段例外：它只能选「超长」或「孤零零一个短段」，按上面的顺序宁可超长。
MAX_MERGED_CHARS = MIN_CHUNK_CHARS + MIN_TAIL_CHUNK_CHARS - 5


def pause_after(segment: str) -> int:
    """该片段后面应插入的静音毫秒数；没有句末标点、分号或冒号则为 0。"""
    tail = segment.rstrip()
    if not tail:
        return 0
    ch = tail[-1]
    if ch in SENTENCE_PUNCT:
        return SENTENCE_PAUSE_MS
    if ch in SEMICOLON_PUNCT:
        return SEMICOLON_PAUSE_MS
    return 0


def _chunk_len(chunk: str) -> int:
    """按 `strip()` 后的字符数算长度：首尾空白不该顶字数。"""
    return len(chunk.strip())


def merge_short_segments(
    segments: list[str],
    min_chars: int = MIN_CHUNK_CHARS,
    min_tail_chars: int = MIN_TAIL_CHUNK_CHARS,
    max_chars: int = MAX_MERGED_CHARS,
) -> list[str]:
    """把切出来的片段按字数攒成 chunk —— 短片段单独合成音质差。

    规则（顺序扫描，不回看）：
      1. 当前 chunk 不足 `min_chars` 字时，下一句并进来；够了才另起一段。
         所以一个 chunk 一旦超过阈值就封口，不会无限增长。
      2. 但并进来之后若超过 `max_chars`，就放弃这次合并、让下一句另起一段。
         宁可留一个偏短的 chunk，也不要一个过长的。
      3. 扫完之后，最后一个 chunk 不足 `min_tail_chars` 字就并回前一个 ——
         它没有「下一句」可以把自己撑起来，单独成段只会得到那段糟糕的音频。
         这一步不受 `max_chars` 限制：末段只能在「超长」和「过短」之间选，
         而超长的效果明显好于过短。

    片段原样拼接（不插空格、不补标点）：片段末尾的标点已经在文本里，
    合进来之后模型会在那里自然换气，只是不再有拼进音频的静音。
    """
    if not segments:
        return []

    chunks = [segments[0]]
    for seg in segments[1:]:
        if (_chunk_len(chunks[-1]) < min_chars
                and _chunk_len(chunks[-1] + seg) <= max_chars):
            chunks[-1] += seg
        else:
            chunks.append(seg)

    if len(chunks) > 1 and _chunk_len(chunks[-1]) < min_tail_chars:
        # 先 pop 再写：`chunks[-2] += chunks.pop()` 的下标在 pop 之后才生效，
        # 只有两段时会越界。
        tail = chunks.pop()
        chunks[-1] += tail

    return chunks


def split_sentences(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    min_tail_chars: int = MIN_TAIL_CHUNK_CHARS,
    max_chars: int = MAX_MERGED_CHARS,
) -> list[tuple[str, int]]:
    """原文 -> [(送去合成的原文片段, 该片段后面的停顿毫秒), ...]

    先按标点切，再按字数把太短的片段合并回去（见 `merge_short_segments`），
    所以返回的段数通常少于标点数。`min_chars` 与 `min_tail_chars` 都传 0
    就是「只切不合并」。

    最后一段的停顿恒为 0。片段按原顺序返回，标点保留在片段末尾
    （零宽后视切分不丢字符）。`strip()` 后为空的片段丢弃，所以
    连续的 `。。` / `；；` 不会产生空段。
    """
    if not text or not text.strip():
        return []

    segments = [seg for seg in SPLIT_PATTERN.split(text) if seg.strip()]
    if not segments:
        return []

    chunks = merge_short_segments(segments, min_chars, min_tail_chars, max_chars)

    # 停顿按合并后的段尾重算：被合进来的那些标点不再对应静音。
    pauses = [pause_after(chunk) for chunk in chunks]
    pauses[-1] = 0   # 最后一段后面不插停顿，无论它以什么标点结尾
    return list(zip(chunks, pauses))


# 与 indextts/utils/common.TAIL_FADE_MS 保持一致。
TAIL_FADE_MS = 20.0


def fade_out_tail_np(pcm: np.ndarray, sr: int = 22050,
                     fade_ms: float = TAIL_FADE_MS) -> np.ndarray:
    """把 int16 PCM [N] 末尾 fade_ms 斜坡降到零。不修改入参。

    `indextts.utils.common.fade_out_tail` 的 numpy 版。段尾的阶跃原本落在
    文件末尾（一声轻咔哒），现在每段后面紧跟数字静音，阶跃会落在段落接缝上，
    500ms 静音还会让它更显眼。MLX 路径全程 numpy，为一个 20ms 斜坡把 torch
    拽进来不划算，所以这里写一份；`common.py` 不动，等价性由测试钉住。

    升余弦而不是线性：到零时斜率也为零，淡出本身不引入新的导数不连续点。
    """
    if pcm.size == 0 or fade_ms <= 0:
        return pcm
    n = min(int(sr * fade_ms / 1000.0), pcm.shape[-1])
    if n <= 1:
        return pcm
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n)))
    out = pcm.copy()
    out[-n:] = (out[-n:].astype(np.float32) * ramp).astype(pcm.dtype)
    return out


def silence_pcm(ms: int, sr: int = 22050) -> np.ndarray:
    """ms 毫秒的 int16 静音。`ms == 0` 返回空数组（拼接时不追加任何采样）。"""
    return np.zeros(round(sr * ms / 1000), dtype=np.int16)


def concat_segments(segments: list[np.ndarray], pauses: list[int],
                    sr: int = 22050) -> np.ndarray:
    """逐段 PCM + 每段后面的停顿 -> 一条 int16 PCM。

    顺序是 `seg[0], sil[0], seg[1], sil[1], …, seg[n-1]`，尾部不追加静音
    （`pauses[-1]` 恒为 0，由 `split_sentences` 保证）。每段在追加静音之前
    先过 20ms 淡出，否则停止符偏早时那个阶跃会落在接缝上。

    Args:
        segments: 每段的 int16 PCM [N]，按原顺序。
        pauses: 与 `segments` 等长；`pauses[i]` 是 `segments[i]` 后面的毫秒数。
    """
    if len(segments) != len(pauses):
        raise ValueError(f"segments/pauses length mismatch: {len(segments)} vs {len(pauses)}")
    parts = []
    for i, (pcm, ms) in enumerate(zip(segments, pauses)):
        parts.append(fade_out_tail_np(pcm, sr))
        # 最后一段后面不插静音，无论 pauses[-1] 是什么。
        if ms and i < len(segments) - 1:
            parts.append(silence_pcm(ms, sr))
    return np.concatenate(parts)
