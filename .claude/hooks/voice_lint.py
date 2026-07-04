#!/usr/bin/env python3
"""
Voice lint hook (PostToolUse on Write). Reads the hook JSON on stdin, checks the
file that was just written for Joe's two hard "AI tell" rules, and if it finds
any, blocks with a message so the model fixes them before the user sees the draft.

Hard rules only (mechanical, no false positives on prose):
  1. No em-dashes (—).  Joe uses spaced hyphens, commas, or parens.
  2. No "X isn't just Y, it's Z" / "more than X, it's Y" contrarian construction.

Only lints prose-ish files (.md, .html, .txt). Skips code, SQL, notebooks, JSON.
Exit 0 = clean or not applicable. Exit 2 = violations (blocks, feeds reason back).
"""
import sys, json, re, pathlib

PROSE_EXT = {".md", ".markdown", ".html", ".htm", ".txt", ".mdx"}

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # can't parse → don't block

    fp = (payload.get("tool_input", {}) or {}).get("file_path", "")
    if not fp:
        sys.exit(0)
    ext = pathlib.Path(fp).suffix.lower()
    if ext not in PROSE_EXT:
        sys.exit(0)

    try:
        text = pathlib.Path(fp).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        sys.exit(0)

    problems = []

    # Rule 1: em-dashes
    if "—" in text:
        n = text.count("—")
        # show a couple of contexts
        ctx = []
        for m in re.finditer(r".{0,25}—.{0,25}", text):
            ctx.append("…" + m.group(0).replace("\n", " ").strip() + "…")
            if len(ctx) >= 3:
                break
        problems.append(f"{n} em-dash(es) (—). Joe never uses em-dashes; use a spaced hyphen ( - ), "
                        f"a comma, or parentheses. Examples: " + " | ".join(ctx))

    # Rule 2: the "isn't just X, it's Y" / "more than X, it's Y" contrarian flip
    flip_patterns = [
        r"\bis\s?n['’]?t just\b",
        r"\bare\s?n['’]?t just\b",
        r"\bnot just\b[^.\n]{0,60}\bit['’]?s\b",
        r"\bmore than (a|an|just)\b[^.\n]{0,60},?\s*it['’]?s\b",
        r"\bit['’]?s not (a|an|about)\b[^.\n]{0,50},?\s*it['’]?s\b",
    ]
    flip_hits = []
    for pat in flip_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            seg = text[max(0, m.start()-10):m.end()+40].replace("\n", " ").strip()
            flip_hits.append("…" + seg + "…")
    if flip_hits:
        problems.append("Contrarian 'isn't just X, it's Y' construction (banned in Joe's voice). "
                        "State the point plainly, don't reframe it. Found: " + " | ".join(flip_hits[:3]))

    if not problems:
        sys.exit(0)

    reason = ("Voice check failed on " + fp + ". Fix these before continuing:\n- " +
              "\n- ".join(problems))
    print(json.dumps({
        "decision": "block",
        "reason": reason,
        "systemMessage": "Voice lint: prose has AI tells (em-dash or 'isn't just' flip). Fixing."
    }))
    sys.exit(2)

if __name__ == "__main__":
    main()
