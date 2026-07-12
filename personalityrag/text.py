from __future__ import annotations

import re
import string
import warnings
from pathlib import Path

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    jieba = None  # type: ignore[assignment]
    JIEBA_AVAILABLE = False


JIEBA_RUNTIME_DISABLED = False


DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "自己",
        "自家",
        "咱",
        "咱们",
        "这",
        "那",
        "这个",
        "那个",
        "这些",
        "那些",
        "哪",
        "哪个",
        "哪些",
        "谁",
        "什么",
        "怎么",
        "怎样",
        "多少",
        "的",
        "了",
        "着",
        "过",
        "地",
        "得",
        "呢",
        "吗",
        "吧",
        "啊",
        "呀",
        "哇",
        "哦",
        "嗯",
        "啦",
        "嘛",
        "呗",
        "和",
        "与",
        "及",
        "以及",
        "或",
        "或者",
        "还是",
        "而",
        "且",
        "并",
        "但",
        "但是",
        "然而",
        "可是",
        "不过",
        "而且",
        "并且",
        "因此",
        "所以",
        "因为",
        "由于",
        "如果",
        "假如",
        "虽然",
        "尽管",
        "除非",
        "在",
        "从",
        "向",
        "往",
        "到",
        "由",
        "为",
        "对",
        "关于",
        "按照",
        "根据",
        "通过",
        "经过",
        "沿着",
        "朝",
        "朝着",
        "沿",
        "用",
        "以",
        "按",
        "依",
        "凭",
        "靠",
        "当",
        "于",
        "比",
        "很",
        "太",
        "非常",
        "极",
        "十分",
        "最",
        "更",
        "挺",
        "特别",
        "尤其",
        "都",
        "也",
        "还",
        "再",
        "又",
        "就",
        "才",
        "已",
        "曾",
        "已经",
        "正在",
        "将",
        "将要",
        "总是",
        "一直",
        "从来",
        "刚",
        "刚才",
        "马上",
        "立刻",
        "顿时",
        "忽然",
        "突然",
        "渐渐",
        "逐渐",
        "慢慢",
        "个",
        "只",
        "件",
        "条",
        "张",
        "把",
        "块",
        "片",
        "次",
        "遍",
        "些",
        "点",
        "下",
        "回",
        "趟",
        "番",
        "场",
        "阵",
        "样",
        "种",
        "哎",
        "唉",
        "哼",
        "嘿",
        "哈",
        "是",
        "有",
        "没",
        "没有",
        "不",
        "别",
        "莫",
        "勿",
        "非",
        "未",
        "无",
        "成",
        "做",
        "看",
        "说",
        "让",
        "给",
        "被",
        "能",
        "会",
        "要",
        "想",
        "之",
        "所",
        "其",
        "此",
        "该",
        "各",
        "每",
        "某",
        "另",
        "等",
        "等等",
        "如此",
        "这样",
        "那样",
        "如何",
        "多么",
        "一下",
        "一点",
        "一些",
        "一切",
        "一样",
        "一般",
        "一起",
        "一边",
        "上下",
        "左右",
        "前后",
        "里外",
        "东西",
        "方面",
        "时候",
        "地方",
        "样子",
        "起来",
        "出来",
        "进去",
        "过去",
        "过来",
        "下去",
        "上来",
        "、",
        "，",
        "。",
        "！",
        "？",
        "；",
        "：",
        "……",
        "—",
    }
)


CHINESE_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—"
    '‘’‛“”„‟…‧﹏'
    "·・•●○◉◎◇◆□■△▲▽▼☆★"
)


class TextProcessor:
    def __init__(self, stopwords_dir: Path | str | None = None):
        self.stopwords = set(DEFAULT_STOPWORDS)
        self.custom_words: set[str] = set()
        self.stopwords_dir = Path(stopwords_dir) if stopwords_dir else None
        if not JIEBA_AVAILABLE:
            warnings.warn(
                "jieba is not installed; Chinese tokenization will be limited.",
                UserWarning,
            )
        self._load_local_stopwords()

    def _load_local_stopwords(self) -> None:
        if not self.stopwords_dir or not self.stopwords_dir.exists():
            return
        for path in self.stopwords_dir.glob("*.txt"):
            try:
                self.stopwords.update(
                    line.strip()
                    for line in path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    if line.strip() and not line.startswith("#")
                )
            except OSError:
                continue

    def tokenize(self, text: str, remove_stopwords: bool = True) -> list[str]:
        if not text or not str(text).strip():
            return []
        cleaned = self._clean_text(str(text))
        if not cleaned:
            return []
        tokens = self._segment(cleaned)
        filtered: list[str] = []
        for token in tokens:
            if not token or token.isspace():
                continue
            if all(not char.isalnum() for char in token):
                continue
            if len(token) == 1 and token.isascii():
                continue
            if remove_stopwords and token in self.stopwords:
                continue
            filtered.append(token)
        return filtered

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.translate(str.maketrans("", "", CHINESE_PUNCTUATION))
        return " ".join(text.split()).strip()

    def _segment(self, text: str) -> list[str]:
        if not text:
            return []
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
        if has_chinese and JIEBA_AVAILABLE and not JIEBA_RUNTIME_DISABLED:
            try:
                return list(jieba.cut_for_search(text))  # type: ignore[union-attr]
            except Exception as exc:
                warnings.warn(
                    f"jieba tokenization failed; falling back to built-in segmentation: {exc}",
                    UserWarning,
                )
                self._disable_jieba_runtime()
        return self._fallback_segment(text)

    @staticmethod
    def _disable_jieba_runtime() -> None:
        global JIEBA_RUNTIME_DISABLED
        JIEBA_RUNTIME_DISABLED = True

    @staticmethod
    def _fallback_segment(text: str) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []

        def flush_buffer() -> None:
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                flush_buffer()
                tokens.append(char)
                continue
            if char.isspace():
                flush_buffer()
                continue
            buffer.append(char)

        flush_buffer()
        return tokens
