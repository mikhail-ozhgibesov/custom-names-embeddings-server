#!/usr/bin/env python3
"""Generate N multilingual names spanning Latin/Cyrillic/Arabic/CJK/Hangul scripts.

The mix is deliberately lopsided toward Latin (reflecting realistic name DBs)
but every script is represented so length-bucketing and the tokenizer's
multilingual path both get exercised. Output is a JSON list of strings.

Names are short (1-4 "words" of 2-8 chars each) so token lengths span ~3..40,
matching the names workload.
"""

import json
import random
import sys
from pathlib import Path

# Seed for reproducibility across runs.
SEED = 0xC0FFEE

# Per-script syllable pools (no diacritics that normalize to nothing).
LATIN = "a e i o u ah ba br ca da de ed el en er et ge gi ha il in is ka ke ki "
LATIN += "la li ma me na ne os ra re ri ro sa se ta te ti to va ve vi vo za"
LATIN = LATIN.split()

CYRILLIC = (
    "алекsey boris dmitri fedor igor ivan kirill leonid mikhail nikolai pavel sergei vadim yuri "
    "anna elena irina katya mariya natalya olga svetlana tatiana vera zoya"
).split()

ARABIC = (
    "ahmad ali amir fatima hassan hussein ibrahim jamal karim layla mariam "
    "mohammad omar rashid salim samir yusuf zahra"
).split()

# CJK: common surnames + given-name chars.
CJK_SUR = "李 王 张 刘 陈 杨 黄 赵 周 吴 徐 孙 朱 马 胡 郭 林 何 高 罗".replace(" ", "")
CJK_GIVEN = "明 华 建 文 龙 志 强 伟 芳 敏 静 丽 娟 兰 燕 伟 军 涛".replace(" ", "")

HANGUL = "minjun seo yeon jun ho eun ji woo young soo min ji eun ah ram yuna".split()


def make_name(rng: random.Random) -> str:
    """One short name; script is chosen per-name."""
    script = rng.choices(
        ["latin", "cyrillic", "arabic", "cjk", "hangul"],
        weights=[0.55, 0.15, 0.10, 0.12, 0.08],
        k=1,
    )[0]
    if script == "latin":
        n_words = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        return " ".join(rng.choice(LATIN) + rng.choice(LATIN) for _ in range(n_words)).title()
    if script == "cyrillic":
        n_words = rng.choices([1, 2], weights=[0.4, 0.6])[0]
        return " ".join(rng.choice(CYRILLIC) for _ in range(n_words)).title()
    if script == "arabic":
        n_words = rng.choices([1, 2], weights=[0.6, 0.4])[0]
        return " ".join(rng.choice(ARABIC) for _ in range(n_words))
    if script == "cjk":
        # Surname (1 char) + 1-2 given-name chars.
        given_len = rng.choices([1, 2], weights=[0.55, 0.45])[0]
        return rng.choice(CJK_SUR) + "".join(rng.choice(CJK_GIVEN) for _ in range(given_len))
    # hangul
    n_words = rng.choices([1, 2], weights=[0.5, 0.5])[0]
    return " ".join(rng.choice(HANGUL) for _ in range(n_words)).title()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("names_5k.json")
    rng = random.Random(SEED)
    names = [make_name(rng) for _ in range(n)]
    out.write_text(json.dumps(names, ensure_ascii=False))
    # Quick stats so the user can see the mix.
    lengths = [len(n) for n in names]
    print(f"wrote {len(names)} names to {out}")
    print(f"  char len: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")
    print(f"  sample: {names[:3]}")


if __name__ == "__main__":
    main()
