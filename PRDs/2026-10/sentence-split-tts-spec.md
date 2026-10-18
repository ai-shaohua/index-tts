# 技术方案：句末与分号切分合成

对应 PRD：`PRDs/2026-10/sentence-split-tts.md`
日期：2026-10-18
状态：待实现
改动文件：`new-server.py`（已存在，当前是 `server.py` 的逐字副本）、`new-webui.py`（当前空文件）
不改文件：`webui.py`、`server.py`、`indextts/utils/front.py`、`index_tts_2_5_mlx`

## 1. 与 PRD 的主要偏离：改为前置切分

PRD 第 41–58 节要求「先归一化、再切段」，并为此在 `new-server.py` 里复制一份改过的 `char_rep_map`、同时关掉 `infer()` 内部的文本清洗。本方案改成**先切分、再让每段各自走原有归一化**。

### 为什么改

归一化后切分要求新服务器自己复制 IndexTTS 的整条中文归一化链路（术语保护 → 拼音声调保护 → 人名保护 → WeTextProcessing → 还原 → 标点替换），并把已归一化文本以「禁用清洗」的方式送回模型。代价：

- 两条后端的清洗开关位置不同。PyTorch 侧是 `infer(text_normalization=False)`（`indextts/infer_v2_5.py:698` 的 `clean_pattern.sub` 在开关**之外**，无条件执行，仍会把 `;` 收成 `,`）；MLX 侧 `Frontend.prepare()`（`core/frontend.py:398`）同样无条件执行 `CLEAN_PATTERN`，且**没有**任何参数可以关掉它。也就是说 PRD 第 58 行设想的「关掉这条会改写分号的文本清洗」，在 MLX 后端（v2.5 默认后端）无法实现，除非 fork 上游包。
- 复制一份归一化链路后，上游每次改 `front.py` 都要同步，否则新旧路径读法会漂移。

前置切分不需要碰归一化的任何一环：分号只用于**决定切点**，切完就把原文片段原样交给现有 `infer()` / `synthesize()`，各段内部该怎么归一化就怎么归一化。模型内部仍会把段尾的 `；` 收成 `,`，这不影响结果——段尾停顿由服务器拼在音频上，不依赖模型看到的符号。

### 已验证的前提

1. **切点标点保留在段尾。** 用 `re.split(r'(?<=[...])', text)` 的零宽后视切分，标点留在前一段末尾，不丢字符。
2. **逐段归一化 == 整段归一化。** 以 PRD 验收文本实测，逐段归一化结果拼接与整段归一化结果逐字相同；每段归一化幂等。WeTextProcessing 的数字读法不跨句末标点取上下文，所以在句末/分号处切开是安全的。
   ```
   per-seg : ['六号开枪，铁狼王，我是真预言家。', '昨夜验十二金水，…', '三、七号首轮站六；', '我信逻辑。']
   whole   : '六号开枪，铁狼王，我是真预言家。昨夜验十二金水，…三、七号首轮站六；我信逻辑。'
   equal   : True
   ```
3. **WeTextProcessing 不动全角分号。** `；` 经 `zh_normalizer.normalize()` 后仍是 `；`，收成 `,` 的是其后的 `char_rep_map`。因此切分发生在归一化之前时，分号一定还在。

### 前置切分引入的唯一新风险：小数点

归一化前的原文里 ASCII `.` 可能是小数点。`他有3.5秒。` 若按 `.` 切，会切成 `他有3.` + `5秒。`，归一化后读成「他有三点」+「五秒」。归一化后切分不存在这个问题，因为那时 `3.5` 已变成 `三点五`。

对策见 §3 的 `SPLIT_PATTERN`：ASCII `.` 只有在**后面不是数字**时才作为切点。`。` `！` `？` `；` `!` `?` `;` 无此问题。

## 2. 架构

```
new-webui.py  (Gradio 7861, 不加载模型)
      │  HTTP
      ▼
new-server.py (FastAPI 3003, 唯一持有模型的进程)
      │
      ├── split_sentences(text)          # 纯函数，不依赖模型
      │     └─ [(raw_segment, trailing_punct), ...]
      │
      └── 逐段 synthesize（同一把 infer_lock）
            ├─ 段内 interval_silence = 0
            ├─ 段尾淡出（见 §4.3）
            └─ 段间插入 500ms / 333ms 静音 → 拼接 → mp3
```

`new-server.py` 以 `server.py` 为基线，只增不改：保留 `--version`/`--mlx`/`--n_timesteps` 等全部现有参数与模型加载逻辑，默认端口改为 `3003`。

## 3. 切分

```python
# 句末 / 分号。ASCII '.' 只在它是真小数点时不切：两侧都是数字才算小数（0.7 是，.7 不是）。
SPLIT_PATTERN = re.compile(
    r'(?<=[。！？；!?;])'      # 句末标点、分号
    r'|(?<=\.)(?!\d)'          # '.' 后面不是数字 -> 句号
    r'|(?<=(?<!\d)\.)(?=\d)'   # '.' 后面是数字但前面不是 -> 仍是句号（如 "出局.7号"）
)

# --- 停顿时长：调效果就改这两个常量，其余全部由它们推导 ---
SENTENCE_PAUSE_MS = 500                                      # 句末停顿
SEMICOLON_PAUSE_RATIO = 2 / 3                                # 分号停顿 = 句末 × 该比例
SEMICOLON_PAUSE_MS = round(SENTENCE_PAUSE_MS * SEMICOLON_PAUSE_RATIO)   # -> 333

PAUSE_MS = {
    '.': SENTENCE_PAUSE_MS, '。': SENTENCE_PAUSE_MS,
    '!': SENTENCE_PAUSE_MS, '！': SENTENCE_PAUSE_MS,
    '?': SENTENCE_PAUSE_MS, '？': SENTENCE_PAUSE_MS,
    ';': SEMICOLON_PAUSE_MS, '；': SEMICOLON_PAUSE_MS,
}

def split_sentences(text: str) -> list[tuple[str, int]]:
    """原文 -> [(送去合成的原文片段, 该片段后面的停顿毫秒), ...]

    最后一段的停顿恒为 0。片段按原顺序返回，标点保留在片段末尾。
    """
```

规则：

- 切在句末（`。` `！` `？` `.` `!` `?`）和分号（`；` `;`）**之后**，标点归入前一段。
- ASCII `.` 的额外条件：**只有两侧都是数字时才判定为小数点，不切**。`0.7` `3.5` `v2.5` `1.2.3` 不切；`3号出局.7号跟上.` 会切（`.` 左边是汉字「局」，不是数字，所以 `.7` 不是小数）；`得分是3.他很强。` 会切（`.` 右边是汉字）；`.7开头` 会切（左边没有数字）。
- 不切：`,` `，` `、` `：` `:` `…` `……` 换行。换行留在段内，交给归一化处理成空格。
- `strip()` 后为空、或只剩空白的片段丢弃，不发合成请求。连续 `。。` / `；；` 只产生一段（第二个标点并入同段尾部，不产生空段）。
- 每段的停顿只看该段 `rstrip()` 后的最后一个字符；不在 `PAUSE_MS` 里（例如没有句末标点的尾段）则为 0。
- 停顿时长集中在 `SENTENCE_PAUSE_MS` / `SEMICOLON_PAUSE_RATIO` 两个常量里，便于调效果。代码里不得出现裸的 `500` / `333`；`PAUSE_MS` 和 §4.2 的静音长度都从常量推导。两者都改为 0 时退化成「只切分、不插静音」，拼接逻辑无需特判。
- 最后一段的停顿强制为 0，无论它以什么标点结尾。
- 只有一段时不插入任何静音。

`/split` 与 `/tts` 调用同一个 `split_sentences`，无第二份实现。

### 与 PRD 验收示例的差异

PRD 第 90–97 行把切段结果写成归一化后的文本（`六号开枪,铁狼王,…`）。前置切分下，服务器返回并展示的是**原文片段**：

```text
1. 6号开枪，铁狼王，我是真预言家。                                后面 500ms
2. 昨夜验12金水，今夜原定验7，可惜未能落验，故7绝非我的金水，别误传。   后面 500ms
3. 3、7号首轮站6，但3号最终回票6，7号仍投我，7号狼面最高；            后面 333ms
4. 我信逻辑，不信遗言污金水。                                      后面没有额外停顿
```

段数、切点位置、停顿序列与 PRD 完全一致，只是展示的是归一化前的文本。这反而更满足 PRD 第 43 行「界面展示的段落必须和实际送去合成的段落是同一份结果」——展示的就是逐字送进 `infer()` 的那串文本。

`/split` 和 `/tts` 同时额外返回 `sentences_normalized`（见 §5.1），用服务器已加载的归一化器算出来，供排查读法。界面默认展示原文片段，读法放在折叠区域里（见 §6）。

注意 `sentences_normalized` 只是**近似**模型内部所见：它复现了归一化这一步，但没有复现其后的 `.lower()`、发音标注展开、`<|LANG|>` 前缀和 tokenizer 切分。当读法参考可以，不要当成「模型最终看到的 token」。

## 4. 合成与拼接

### 4.1 逐段合成

在**同一把** `infer_lock` 内按顺序跑完所有段，中途不释放锁，避免别的请求插进来抢模型。

- MLX：复用 `_synthesize_mlx` 的结构，但不逐段写 wav——`tts.synthesize()` 直接返回 int16 PCM，在内存里累积。speaker context 只 `mlx_speaker(prompt_path)` 一次，全部段落复用（缓存已按 `(path, mtime)` 命中，但显式只取一次更清楚）。
- PyTorch：`tts.infer(..., output_path=None)` 返回 `(sr, wav_data)`，`wav_data` 形状 `(samples, channels)`。

两条路径都必须显式传 **`interval_silence=0`**：MLX 的 `Pipeline.synthesize()` 默认 `interval_silence=200`（`core/pipeline.py:87`），PyTorch 的 `infer()` 默认也是 200。段**内部**若超过 `max_text_tokens_per_segment` 仍由模型自己切，那些内部切口不加任何静音，也不加 500/333。

`duration_factor` 对所有段使用同一个值，默认 `1.0`。

一段失败（抛异常或返回空 PCM）则整次请求失败，返回 500，不返回半成品。

日志每段一行：段序号 / 段总数 / 段尾标点 / 该段耗时 / 字符数。

### 4.2 静音与拼接

- 采样率 22050、单声道、int16，与两条后端一致（`pipeline.py:19`、`infer_v2_5.py:737`）。
- `silence = np.zeros(round(22050 * ms / 1000), dtype=np.int16)`，`ms` 取自 `split_sentences` 返回的每段停顿（即 §3 的常量）。按默认值：500ms → 11025 采样，333ms → 7343 采样。`ms == 0` 时不追加任何数组。
- 拼接顺序：`seg[0], sil[0], seg[1], sil[1], …, seg[n-1]`，尾部不追加静音。
- 合成后一次性写临时 wav，再走现有 `wav_to_mp3()` 转 mp3。

### 4.3 段尾淡出（建议，PRD 未覆盖）

**这是本方案建议新增的一条，因为前置/后置切分都会碰到它。**

IndexTTS 的波形长度由采样到的 `stop_mel_token` 下标推出，停止符偶尔落得偏早，一段的最后一个采样点可能离零很远（见 `indextts/utils/common.py:68` 的 `fade_out_tail` 注释）。原来整段生成时，这个阶跃在文件末尾，听感上是收尾一声轻咔哒；现在每段后面紧跟一段数字静音，阶跃落在**段落接缝**上，每个接缝都可能咔一下，而且 500ms 静音会让它更显眼。

PyTorch 路径的 `infer_generator` 在最终拼接后调用了 `fade_out_tail`（`infer_v2_5.py:868`），但那是对整条音频的末尾，段间接缝拿不到。MLX 路径（v2.5 默认）**完全没有**这个淡出。

做法：在每段 PCM 追加静音之前，对该段末尾施加 20ms 升余弦淡出。上游实测 112 条正常生成中 62 条逐比特未变、其余最大改动不超过 -42.8 dBFS（中位 -76.8 dBFS），所以无条件施加，不加检测阈值。

`indextts/utils/common.fade_out_tail` 收的是 `(channels, samples)` 的 torch 张量，而 MLX 路径全程 numpy。为一个 20ms 斜坡把 torch 拽进纯 numpy 流程不划算，`new-server.py` 内写一份 numpy 版（`common.py` 不动）：

```python
TAIL_FADE_MS = 20.0   # 与 indextts/utils/common.TAIL_FADE_MS 保持一致

def fade_out_tail_np(pcm, sr=22050, fade_ms=TAIL_FADE_MS):
    """把 int16 PCM [N] 末尾 fade_ms 斜坡降到零。不修改入参。

    升余弦而不是线性：到零时斜率也为零，淡出本身不引入新的导数不连续点。
    """
    n = min(int(sr * fade_ms / 1000.0), pcm.shape[-1])
    if n <= 1:
        return pcm
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n)))
    out = pcm.copy()
    out[-n:] = (out[-n:].astype(np.float32) * ramp).astype(pcm.dtype)
    return out
```

调用处（§4.2 的拼接循环里）只多一行：

```python
parts.append(fade_out_tail_np(seg_pcm))   # 原本是 parts.append(seg_pcm)
if ms:
    parts.append(np.zeros(round(22050 * ms / 1000), dtype=np.int16))
```

PyTorch 路径返回的 `wav_data` 形状是 `(samples, channels)`，取单声道展平成 `[N]` 后同样过这个函数；它原本在 `infer_generator` 里对整条音频做的那次淡出保持不变，重复施加一次 20ms 斜坡无害（落在已经接近零的尾部）。

已与上游 torch 版逐采样比对，最大差 1 个 LSB（int16 取整舍入），听感无差异。测试里用一条断言钉住这个等价关系，防止以后两份实现漂移（见 §7）。

## 5. 接口

全部沿用 `server.py` 的 `TTSRequest`，字段与默认值不变（`agentName` / `text` / `lang` / `duration_factor` / `num_beams` / `n_timesteps` / `max_text_tokens_per_segment`）。

### 5.1 `POST /split`

**纯 debug 接口。** 唯一用途是不花合成时间就能看到切点和归一化后的读法——某段数字读错时，先用它确认是归一化阶段就错了，还是模型念错了。

不是 `/tts` 的前置步骤：`/tts` 自身包含切分，正常调用方只调 `/tts`。界面也不调它（见 §6）。

只切分，不调用模型推理（归一化器已随模型进程常驻，算 `sentences_normalized` 不额外加载权重）。请求体与 `/tts` 同构，只读 `text`、`lang`。

```json
{
  "sentences": [
    "6号开枪，铁狼王，我是真预言家。",
    "昨夜验12金水，今夜原定验7，可惜未能落验，故7绝非我的金水，别误传。",
    "3、7号首轮站6，但3号最终回票6，7号仍投我，7号狼面最高；",
    "我信逻辑，不信遗言污金水。"
  ],
  "sentences_normalized": [
    "六号开枪,铁狼王,我是真预言家.",
    "昨夜验十二金水,今夜原定验七,可惜未能落验,故七绝非我的金水,别误传.",
    "三,七号首轮站六,但三号最终回票六,七号仍投我,七号狼面最高,",
    "我信逻辑,不信遗言污金水."
  ],
  "silences_ms": [500, 500, 333]
}
```

`silences_ms` 长度 = `len(sentences) - 1`；`sentences` 只有一段时为 `[]`。
`sentences_normalized[2]` 末尾是 `,` 而非 `;`——模型内部确实这么收，不是 bug。注意它与 `sentences` 等长（不像 `silences_ms` 少 1）。

### 5.2 `POST /tts`

响应从现有的裸 mp3 字节改为 JSON（PRD 第 143–152 行）：

```json
{
  "sentences": ["..."],
  "sentences_normalized": ["..."],
  "silences_ms": [500, 333],
  "audio_base64": "<mp3 bytes, base64>",
  "media_type": "audio/mpeg"
}
```

比 PRD 第 143–152 行多一个 `sentences_normalized`：界面的折叠读法区从这里取数，不必再去调 `/split`。字段名与 `/split` 保持一致，渲染逻辑可共用。

这与 `server.py` 的 `/tts` 返回 `Response(media_type="audio/mpeg")` 不兼容，但 `new-server.py` 是新端口新服务，现有调用方不受影响。

错误码沿用现有语义：音色文件不存在 404；`text` 为空或 `strip()` 后为空、`lang` 不在后端支持列表里 400；合成没有产出音频 500。注意现有 `TTSRequest.text` 已有 `min_length=1`，纯空白字符串会通过校验，需在 handler 里再判一次 `strip()`。

### 5.3 `GET /health`

沿用 `server.py` 现有实现（`status` / `version` / `backend` / `model_version`），无需改动，供界面确认服务已起。

## 6. 界面

`new-webui.py` 从零写，Gradio 5.45.0（已在 `pyproject.toml:78`），默认 `7861`，只通过 HTTP 调 `http://127.0.0.1:3003`。**不 import `indextts`，不构造任何模型对象。**

组件：

| 组件 | 说明 |
|---|---|
| 音色下拉 | `wolf-audio/*.wav` 文件名去扩展名，启动时扫描一次 |
| 语言 | 下拉，默认 `ZH` |
| 原文输入 | 多行 Textbox |
| 生成按钮 | 点击后 `interactive=False`，返回后恢复 |
| 切段展示 | Markdown，列出每段**原文**与其后的停顿毫秒 |
| 读法（折叠） | `gr.Accordion(label="读法（归一化后）", open=False)`，内含 `sentences_normalized` |
| 音频播放器 | `gr.Audio`，载入 mp3 |

切段展示用原文片段，因为那才是逐字送进 `infer()` 的文本（PRD 第 43 行）。读法默认折叠，展开后按同样的段序号列出 `sentences_normalized`，便于排查某段数字读错是归一化阶段的问题还是模型念错。折叠区标题注明它是归一化结果的近似（§3 末尾那条说明）。

默认展示（对应 §3 的示例）：

```text
1. 6号开枪，铁狼王，我是真预言家。                                后面 500ms
2. 昨夜验12金水，今夜原定验7，可惜未能落验，故7绝非我的金水，别误传。   后面 500ms
3. 3、7号首轮站6，但3号最终回票6，7号仍投我，7号狼面最高；            后面 333ms
4. 我信逻辑，不信遗言污金水。                                      后面没有额外停顿

▸ 读法（归一化后）
```

交互：**点击生成只调一次 `POST /tts`**。`/tts` 是完整接口，切分、逐段合成、拼接都在它里面，响应已同时带回 `sentences`、`sentences_normalized`、`silences_ms` 和音频——界面用同一个响应填切段列表、折叠读法和播放器，不需要再调 `/split`。失败时在错误区显示服务器返回的 `detail`，不清空原文输入框。

这与 PRD 第 38 行「界面先显示切段结果，再等待合成」有出入：单次 `/tts` 的响应是合成完成后一次性回来的，做不到分阶段显示。本方案取一次性显示——合成期间按钮置灰并显示等待状态，完成后切段列表、折叠读法、音频一起出现。

不为了分阶段显示而让界面先调 `/split`：那会让 `/tts` 内部把切分重算一遍，并把一个 debug 接口塞进正常合成路径。切段结果本来就在 `/tts` 的响应里。

注意 `/tts` 返回 base64，界面需 `base64.b64decode` 后写临时 mp3 文件再交给 `gr.Audio`。

## 7. 测试

纯函数部分（`split_sentences`）可脱离模型测，放 `tests/test_sentence_split.py`：

| 用例 | 断言 |
|---|---|
| PRD 验收文本 | 4 段；切点如 §3；停顿为 `[SENTENCE, SENTENCE, SEMICOLON]`（默认值即 `[500, 500, 333]`） |
| 顿号 | `、` 不切段，留在第 3 段内 |
| 分号 | `；` 留在第 3 段**末尾**不被丢弃，且该段停顿为 `SEMICOLON_PAUSE_MS` |
| 常量可调 | `monkeypatch` 把 `SENTENCE_PAUSE_MS` 改成 800，断言句末停顿变 800、分号停顿变 533（`round(800 * 2/3)`）——钉住「没有裸字面量」 |
| 冒号 | `：` 不切段 |
| 单段无标点 | `len(sentences) == 1`，`silences_ms == []` |
| 连续句号 `啊。。好。` | 不产生空段 |
| 连续分号 `啊；；好。` | 不产生空段 |
| 小数点 `他有3.5秒。` | 不在 `3.5` 中间切 |
| 小数点 `0.7秒结束。` | 不在 `0.7` 中间切 |
| 版本号 `版本v2.5发布了。` | 不在 `2.5` 中间切 |
| 多段小数 `1.2.3都对。` | 不切 |
| 数字后真句号 `得分是3.他很强。` | 切成两段（`.` 右边是汉字） |
| 汉字后数字开头 `3号出局.7号跟上.` | 切成两段（`.` 左边是汉字，`.7` 不是小数） |
| 无前导数字 `.7开头` | 在 `.` 后切 |
| 英文句点 `Hi. Bye.` | 按 `.` 切成两段 |
| 纯空白 / 空字符串 | 返回 `[]` |
| 尾段带句号 | 最后一段停顿为 0 |

淡出等价性（不需要模型权重，需要 torch）：随机 int16 PCM 上 `fade_out_tail_np(pcm)` 与上游 `fade_out_tail(torch.from_numpy(pcm)[None], 22050)[0]` 逐采样差 ≤ 1。钉住 §4.3 两份实现不漂移。

归一化一致性回归（需要 wetext，不需要模型权重）：逐段归一化拼接 == 整段归一化，用 PRD 验收文本。

端到端（手动，单进程）：起 `new-server.py`，`curl /split` 比对 JSON；`curl /tts` 存 mp3 听接缝与段尾；确认只启 `new-webui.py` 时 RSS 不出现模型级占用。

**同一时刻只允许一个带模型的进程**（见 `memory/tts-benchmark-single-process.md`）。测 `new-server.py` 前先 `ps aux | grep "[s]erver.py" | awk '{print $2}' | xargs -r kill -9`。

## 8. 验收对照

PRD §验收 的每一条：

- [x] 4 段、`、` 留在段内 — §7
- [x] 分号的停顿被正确处理 — 分号作为切点生效，段末保留原文的 `；`，该段后面拼 333ms。**已确认替代 PRD 第 52 行的字面要求**：不再要求 `;` 以 ASCII 形式送进 tokenizer（模型内部仍收成 `,`，见 §1）。验收口径改为「分号处切开 + 333ms 停顿」，由 §7 的分号用例和 §4.2 的拼接断言覆盖
- [x] 停顿 500 / 500 / 333，尾段无停顿 — §4.2
- [x] 单段不插停顿、只合成一次 — §3
- [x] 连续句号/分号不产生空段 — §3、§7
- [x] `：` 不切段 — §3
- [x] 404 / 400 — §5.2
- [x] 播放器能播 mp3 — §6
- [x] `webui.py` / `server.py` / 共享 `char_rep_map` 零改动 — 前置切分天然满足，不需要复制 `char_rep_map`
- [x] 只启 `new-webui.py` 不加载模型 — §6

## 9. 已确认的决定

全部待确认项已定，无遗留问题。

1. **前置切分**替代 PRD 的「先归一化再切段」（§1）。
2. **段尾淡出**采纳，`new-server.py` 内写 numpy 版，`common.py` 不动（§4.3）。
3. **界面**展示原文片段，读法放折叠区；点击生成只调一次 `/tts`，不分阶段显示（§6）。
4. **`/split`** 是纯 debug 接口，不在正常合成路径上（§5.1）。
5. **分号**：放弃 PRD 第 52 行「`;` 以 ASCII 送进模型」的字面要求，验收口径改为「分号处切开 + 后面 333ms」。模型内部仍把分号收成 `,`，不影响停顿——停顿是拼在音频上的静音（§8）。
