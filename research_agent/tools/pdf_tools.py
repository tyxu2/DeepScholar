import pdfplumber
import re
import logging


def _mute_pdfminer_fontbbox_warning():
    """
    pdfminer 在解析部分 PDF 字体描述时会反复打印 FontBBox 警告，
    该警告通常不影响正文提取，容易污染终端输出。
    这里仅定向抑制该类日志，不吞掉其他异常。
    """
    for name in ("pdfminer", "pdfminer.pdffont"):
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or logger.level < logging.ERROR:
            logger.setLevel(logging.ERROR)
        logger.propagate = False


_mute_pdfminer_fontbbox_warning()


def extract_text(pdf_path: str, max_chars: int = 12000) -> str:
    """提取 PDF 全文，按页拼接，超长截断。"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                if len(text) >= max_chars:
                    break
    except Exception:
        return ""
    return text[:max_chars]


def extract_sections(text: str) -> dict[str, str]:
    """
    启发式切分论文章节。
    返回：{section_name: content}
    """
    section_patterns = [
        r"\n(Abstract|Introduction|Related Work|Background|"
        r"Methodology|Method|Approach|Proposed Method|"
        r"Experiments?|Experimental Results?|Evaluation|"
        r"Results?|Discussion|Conclusion|References)\s*\n"
    ]
    pattern = "|".join(section_patterns)
    splits = re.split(pattern, text, flags=re.IGNORECASE)

    sections: dict[str, str] = {}
    if len(splits) < 2:
        sections["full"] = text
        return sections

    # splits 格式：[前缀, 章节名, 内容, 章节名, 内容, ...]
    sections["preamble"] = splits[0]
    i = 1
    while i + 1 < len(splits):
        name = splits[i].strip().lower() if splits[i] else f"section_{i}"
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        sections[name] = content[:3000]  # 每节最多 3000 字符
        i += 2

    return sections


def get_abstract(text: str) -> str:
    """单独提取摘要段落。"""
    match = re.search(
        r"Abstract\s*[\n:]\s*(.*?)(?=\n\s*(?:Introduction|1\s+Introduction|Keywords))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:1500]
    # fallback：取前 800 字符
    return text[:800]
