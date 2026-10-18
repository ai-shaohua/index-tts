"""`sentence_split` 的纯函数测试 —— 不需要模型权重。

覆盖 PRDs/2026-10/sentence-split-tts-spec.md 第 7 节的用例表：切点、停顿、
常量可调、小数点、以及段尾淡出与上游 torch 版的等价性。

运行方式：
    uv run --extra test pytest tests/test_sentence_split.py -v
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import sentence_split  # noqa: E402
from sentence_split import (  # noqa: E402
    SEMICOLON_PAUSE_MS,
    SENTENCE_PAUSE_MS,
    TAIL_FADE_MS,
    concat_segments,
    fade_out_tail_np,
    merge_short_segments,
    silence_pcm,
    split_sentences,
)

SAMPLE_RATE = 22050

# PRD 的验收文本。顿号留在第 3 段内，分号把「我信逻辑」切成第 4 段。
ACCEPTANCE_TEXT = (
    "6号开枪，铁狼王，我是真预言家。"
    "昨夜验12金水，今夜原定验7，可惜未能落验，故7绝非我的金水，别误传。"
    "3、7号首轮站6，但3号最终回票6，7号仍投我，7号狼面最高；"
    "我信逻辑，不信遗言污金水。"
)


def _cut(text):
    """只看切点：阈值归零关掉按字数的合并。"""
    return split_sentences(text, min_chars=0, min_tail_chars=0)


def _sentences(text):
    return [seg for seg, _ in _cut(text)]


def _pauses(text):
    """除最后一段外每段后面的停顿，即 `/split` 返回的 silences_ms。"""
    return [ms for _, ms in _cut(text)[:-1]]


# --- 切点 -------------------------------------------------------------

def test_acceptance_text_splits_into_four_segments():
    assert _sentences(ACCEPTANCE_TEXT) == [
        "6号开枪，铁狼王，我是真预言家。",
        "昨夜验12金水，今夜原定验7，可惜未能落验，故7绝非我的金水，别误传。",
        "3、7号首轮站6，但3号最终回票6，7号仍投我，7号狼面最高；",
        "我信逻辑，不信遗言污金水。",
    ]


def test_acceptance_text_pauses():
    assert _pauses(ACCEPTANCE_TEXT) == [
        SENTENCE_PAUSE_MS, SENTENCE_PAUSE_MS, SEMICOLON_PAUSE_MS,
    ]


def test_ideographic_comma_stays_inside_its_segment():
    """`、` 不切段，留在第 3 段内（归一化后它才变成逗号）。"""
    third = _sentences(ACCEPTANCE_TEXT)[2]
    assert third.startswith("3、7号")


def test_semicolon_is_kept_at_the_end_of_its_segment():
    segments = _cut(ACCEPTANCE_TEXT)
    seg, ms = segments[2]
    assert seg.endswith("；"), "分号必须留在段尾，不被丢弃"
    assert ms == SEMICOLON_PAUSE_MS


@pytest.mark.parametrize("text", [
    "他说：我是预言家。",        # 领起下文的冒号，后面不是数字
    "他说:我是预言家。",
    "现在 12:30 开会。",         # 时间，冒号前面也是数字
    "现在 12：30 开会。",
])
def test_colon_not_followed_by_a_digit_does_not_split(text):
    """冒号只有后跟数字才切；时间里的冒号前面也是数字，同样不切。"""
    assert _sentences(text) == [text]


def test_colon_followed_by_a_digit_splits_with_the_semicolon_pause():
    """`验人结果：12金水` 这种冒号切段，停顿与分号一致。"""
    assert _cut("验人结果：12号是金水。") == [
        ("验人结果：", SEMICOLON_PAUSE_MS),
        ("12号是金水。", 0),
    ]


def test_comma_and_ellipsis_do_not_split():
    assert _sentences("我想想，也许……他是狼。") == ["我想想，也许……他是狼。"]


def test_newline_stays_inside_the_segment():
    """换行留在段内，交给归一化处理成空格。"""
    assert _sentences("第一行\n第二行。") == ["第一行\n第二行。"]


# --- 停顿 -------------------------------------------------------------

def test_single_segment_without_punctuation():
    pairs = _cut("我是预言家")
    assert len(pairs) == 1
    assert _pauses("我是预言家") == []


def test_single_segment_with_punctuation_has_no_trailing_pause():
    """最后一段的停顿强制为 0，无论它以什么标点结尾。"""
    assert split_sentences("我是预言家。") == [("我是预言家。", 0)]
    assert split_sentences("我是预言家；") == [("我是预言家；", 0)]
    assert split_sentences("我是预言家：") == [("我是预言家：", 0)]


def test_last_segment_pause_is_zero():
    assert _cut(ACCEPTANCE_TEXT)[-1][1] == 0


def test_pauses_come_from_the_constants(monkeypatch):
    """钉住「代码里没有裸的毫秒数」：改常量，停顿跟着变。"""
    monkeypatch.setattr(sentence_split, "SENTENCE_PAUSE_MS", 800)
    monkeypatch.setattr(sentence_split, "SEMICOLON_PAUSE_MS", 533)

    text = "啊。好；行：1号。"
    assert [ms for _, ms in _cut(text)] == [800, 533, 533, 0]


def test_both_constants_zero_degenerates_to_split_only(monkeypatch):
    """两个常量都为 0 时退化成「只切分、不插静音」，拼接逻辑无需特判。"""
    monkeypatch.setattr(sentence_split, "SENTENCE_PAUSE_MS", 0)
    monkeypatch.setattr(sentence_split, "SEMICOLON_PAUSE_MS", 0)

    pairs = _cut(ACCEPTANCE_TEXT)
    assert len(pairs) == 4
    assert [ms for _, ms in pairs] == [0, 0, 0, 0]


# --- 空段 -------------------------------------------------------------

def test_repeated_full_stops_produce_no_empty_segment():
    assert _sentences("啊。。好。") == ["啊。", "。", "好。"]


def test_repeated_semicolons_produce_no_empty_segment():
    assert _sentences("啊；；好。") == ["啊；", "；", "好。"]


def test_punctuation_pile_up_produces_no_empty_segment():
    """连续切点不产生空段。冒号必须后跟数字，所以两个冒号无法相邻成为切点，
    这里用句号 + 冒号凑出相邻的两个切点。"""
    assert _sentences("啊。：1好。") == ["啊。", "：", "1好。"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "。", "；", "："])
def test_degenerate_input(text):
    """纯空白与空字符串返回 []；只有标点时仍是一段（没有空段）。"""
    pairs = _cut(text)
    if text.strip():
        assert pairs == [(text, 0)]
    else:
        assert pairs == []


# --- 按字数合并 -------------------------------------------------------

def _seg(n, tail="。"):
    """n 个字（含句末标点）的片段，内容无所谓，只有长度要紧。"""
    return "啊" * (n - len(tail)) + tail


def test_short_segments_merge_until_the_threshold():
    """不到 50 字就把下一句并进来；一旦够了就封口。"""
    assert merge_short_segments([_seg(20), _seg(20), _seg(20), _seg(40)]) == [
        _seg(20) + _seg(20) + _seg(20),   # 20 -> 40 -> 60，到 60 才够 50
        _seg(40),
    ]


def test_a_segment_at_the_threshold_is_not_extended():
    """恰好 50 字即满足阈值，下一句另起一段（前提是它自己够长）。"""
    assert merge_short_segments([_seg(50), _seg(40)]) == [_seg(50), _seg(40)]


def test_a_long_segment_is_never_split_further():
    """合并只会变长，不会把超长片段再切开。"""
    long_seg = _seg(500)
    assert merge_short_segments([long_seg]) == [long_seg]


def test_short_tail_merges_back_into_the_previous_chunk():
    """末段不足 30 字时并回前一段：它没有下一句能把自己撑起来。"""
    assert merge_short_segments([_seg(60), _seg(20)]) == [_seg(60) + _seg(20)]


def test_tail_at_the_tail_threshold_stays_on_its_own():
    assert merge_short_segments([_seg(60), _seg(30)]) == [_seg(60), _seg(30)]


def test_a_short_only_segment_is_kept_as_is():
    """只有一段时无处可并，再短也原样返回。"""
    assert merge_short_segments([_seg(5)]) == [_seg(5)]
    assert _sentences(_seg(5)) == [_seg(5)]


def test_merge_is_lossless():
    """合并只改分段，不动字符：拼起来必须和原文逐字一致。"""
    segments = [_seg(n) for n in (10, 5, 80, 20, 7)]
    assert "".join(merge_short_segments(segments)) == "".join(segments)


def test_pause_is_recomputed_after_merging():
    """被合进来的标点不再对应静音，停顿按合并后的段尾算。"""
    text = _seg(60, "；") + _seg(20)
    assert split_sentences(text) == [(text, 0)]   # 末段太短并回来，分号的停顿消失


def test_merge_thresholds_are_parameters_not_hardcoded():
    assert split_sentences("啊。好。", min_chars=0, min_tail_chars=0) == [
        ("啊。", SENTENCE_PAUSE_MS), ("好。", 0),
    ]


# --- ASCII '.' 的小数点例外 -------------------------------------------

@pytest.mark.parametrize("text", [
    "他有3.5秒。",       # 两侧都是数字 -> 小数
    "0.7秒结束。",       # 前导 0
    "版本v2.5发布了。",  # 版本号
    "1.2.3都对。",       # 多段小数
])
def test_decimal_point_does_not_split(text):
    assert _sentences(text) == [text]


def test_digit_then_real_full_stop_splits():
    """`.` 右边是汉字 -> 它是句号，不是小数点。"""
    assert _sentences("得分是3.他很强。") == ["得分是3.", "他很强。"]


def test_non_digit_then_digit_splits():
    """`.` 左边是汉字 -> `.7` 不是小数。"""
    assert _sentences("3号出局.7号跟上.") == ["3号出局.", "7号跟上."]


def test_leading_dot_splits():
    """左边没有数字 -> 仍按句号切。"""
    assert _sentences(".7开头") == [".", "7开头"]


def test_english_full_stops_split():
    assert _sentences("Hi. Bye.") == ["Hi.", " Bye."]


# --- 静音 -------------------------------------------------------------

def test_silence_lengths_match_the_spec():
    assert len(silence_pcm(SENTENCE_PAUSE_MS)) == round(SAMPLE_RATE * SENTENCE_PAUSE_MS / 1000)
    assert len(silence_pcm(SEMICOLON_PAUSE_MS)) == round(SAMPLE_RATE * SEMICOLON_PAUSE_MS / 1000)
    assert silence_pcm(0).size == 0


def test_silence_is_int16_zeros():
    sil = silence_pcm(SENTENCE_PAUSE_MS)
    assert sil.dtype == np.int16
    assert not sil.any()


# --- 拼接 -------------------------------------------------------------

def _tone(samples, value=20000):
    return np.full(samples, value, dtype=np.int16)


def test_concat_inserts_silence_between_segments_only():
    """`seg[0], sil[0], seg[1], sil[1], …, seg[n-1]`，尾部不追加静音。"""
    segs = [_tone(SAMPLE_RATE) for _ in range(4)]
    pauses = [500, 500, 333, 0]

    out = concat_segments(segs, pauses, SAMPLE_RATE)

    expected = 4 * SAMPLE_RATE + len(silence_pcm(500)) * 2 + len(silence_pcm(333))
    assert len(out) == expected


def test_concat_single_segment_adds_no_silence():
    seg = _tone(SAMPLE_RATE)

    assert len(concat_segments([seg], [0], SAMPLE_RATE)) == SAMPLE_RATE


def test_concat_never_appends_silence_after_the_last_segment():
    """即使最后一段的停顿非 0（调用方传错）也不追加 —— 尾部静音永远没有。"""
    segs = [_tone(100), _tone(100)]

    out = concat_segments(segs, [500, 500], SAMPLE_RATE)

    assert len(out) == 200 + len(silence_pcm(500))


def test_concat_silence_lands_exactly_between_the_segments():
    segs = [_tone(1000), _tone(1000)]
    sil = len(silence_pcm(SENTENCE_PAUSE_MS))

    out = concat_segments(segs, [SENTENCE_PAUSE_MS, 0], SAMPLE_RATE)

    assert not out[1000:1000 + sil].any(), "段间必须是数字静音"
    assert out[1000 + sil] != 0, "静音之后必须立刻是第二段"


def test_concat_fades_each_segment_tail():
    """每段的最后一个采样点都落到零，接缝上不留阶跃。"""
    segs = [_tone(SAMPLE_RATE), _tone(SAMPLE_RATE)]

    out = concat_segments(segs, [SENTENCE_PAUSE_MS, 0], SAMPLE_RATE)

    assert abs(int(out[SAMPLE_RATE - 1])) <= 1   # 第一段尾
    assert abs(int(out[-1])) <= 1                # 第二段尾


def test_concat_with_zero_pauses_is_a_plain_join():
    """两个常量都为 0 时退化成只拼接，不特判。"""
    segs = [_tone(100), _tone(100), _tone(100)]

    assert len(concat_segments(segs, [0, 0, 0], SAMPLE_RATE)) == 300


def test_concat_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        concat_segments([_tone(10)], [0, 0], SAMPLE_RATE)


def test_concat_output_is_int16():
    out = concat_segments([_tone(100), _tone(100)], [500, 0], SAMPLE_RATE)

    assert out.dtype == np.int16


def test_concat_matches_the_split_contract():
    """split_sentences 的输出可以直接喂给 concat_segments。"""
    pairs = split_sentences(ACCEPTANCE_TEXT)
    segs = [_tone(SAMPLE_RATE) for _ in pairs]

    out = concat_segments(segs, [ms for _, ms in pairs], SAMPLE_RATE)

    extra = sum(len(silence_pcm(ms, SAMPLE_RATE)) for _, ms in pairs)
    assert len(out) == len(pairs) * SAMPLE_RATE + extra


# --- 段尾淡出 ---------------------------------------------------------

def test_fade_lands_the_last_sample_on_zero():
    pcm = np.full(SAMPLE_RATE, 20000, dtype=np.int16)

    faded = fade_out_tail_np(pcm, SAMPLE_RATE)

    assert abs(int(faded[-1])) <= 1


def test_fade_only_touches_the_ramp_region():
    pcm = np.full(SAMPLE_RATE, 20000, dtype=np.int16)
    n = int(SAMPLE_RATE * TAIL_FADE_MS / 1000.0)

    faded = fade_out_tail_np(pcm, SAMPLE_RATE)

    assert np.array_equal(faded[:-n], pcm[:-n])
    assert faded.shape == pcm.shape
    assert faded.dtype == pcm.dtype


def test_fade_does_not_mutate_the_caller_array():
    pcm = np.full(SAMPLE_RATE, 20000, dtype=np.int16)
    before = pcm.copy()

    fade_out_tail_np(pcm, SAMPLE_RATE)

    assert np.array_equal(pcm, before)


def test_fade_is_bit_exact_on_a_silent_tail():
    """正常收尾的生成尾部本来就是数字静音，缩放零仍是零 —— 这正是无条件
    施加这个淡出的理由。"""
    pcm = np.concatenate([
        np.full(SAMPLE_RATE, 20000, dtype=np.int16),
        np.zeros(SAMPLE_RATE // 4, dtype=np.int16),
    ])

    assert np.array_equal(fade_out_tail_np(pcm, SAMPLE_RATE), pcm)


@pytest.mark.parametrize("samples", [0, 1, 2])
def test_fade_tolerates_degenerate_input(samples):
    pcm = np.ones(samples, dtype=np.int16)

    assert fade_out_tail_np(pcm, SAMPLE_RATE).shape == pcm.shape


def test_fade_is_a_no_op_when_disabled():
    pcm = np.full(SAMPLE_RATE, 20000, dtype=np.int16)

    assert np.array_equal(fade_out_tail_np(pcm, SAMPLE_RATE, fade_ms=0.0), pcm)


def test_fade_matches_the_upstream_torch_implementation():
    """钉住 numpy 版与 indextts/utils/common.fade_out_tail 不漂移。

    两份实现的唯一差异是 int16 取整舍入，所以逐采样差允许 1 个 LSB。
    """
    torch = pytest.importorskip("torch")
    from indextts.utils.common import fade_out_tail

    rng = np.random.default_rng(0)
    pcm = rng.integers(-32768, 32767, size=SAMPLE_RATE, dtype=np.int16)

    ours = fade_out_tail_np(pcm, SAMPLE_RATE)
    theirs = fade_out_tail(torch.from_numpy(pcm)[None], SAMPLE_RATE)[0].numpy()

    assert np.abs(ours.astype(np.int32) - theirs.astype(np.int32)).max() <= 1


# --- 归一化一致性回归 -------------------------------------------------

def test_per_segment_normalization_equals_whole_text_normalization():
    """前置切分的前提：逐段归一化拼接 == 整段归一化。

    WeTextProcessing 的数字读法不跨句末标点取上下文，所以在句末 / 分号处
    切开是安全的（方案 §1「已验证的前提」第 2 条）。
    """
    pytest.importorskip("wetext")
    from indextts.utils.front import TextNormalizer

    normalizer = TextNormalizer(enable_glossary=True)
    normalizer.load()

    per_seg = "".join(normalizer.normalize(seg) for seg in _sentences(ACCEPTANCE_TEXT))
    whole = normalizer.normalize(ACCEPTANCE_TEXT)

    assert per_seg == whole


def test_wetext_leaves_the_fullwidth_semicolon_alone():
    """分号是被其后的 char_rep_map 收成 `,` 的，不是 WeTextProcessing。
    所以切分发生在归一化之前时，分号一定还在（方案 §1 第 3 条）。"""
    pytest.importorskip("wetext")
    from indextts.utils.front import TextNormalizer

    normalizer = TextNormalizer(enable_glossary=True)
    normalizer.load()

    assert "；" in normalizer.zh_normalizer.normalize("七号狼面最高；")
