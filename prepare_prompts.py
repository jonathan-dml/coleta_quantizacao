"""Gera o conjunto fixo de prompts do experimento a partir de datasets públicos.

Baixa os arquivos parquet do Hugging Face Hub, sorteia N itens por categoria com
semente fixa e grava prompts/prompts.jsonl. O arquivo gerado deve ser versionado
junto com os resultados: o benchmark lê apenas ele, sem acesso à internet.

Uso:
    python prepare_prompts.py [--per-category 20] [--seed 42] [--out prompts/prompts.jsonl]
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from huggingface_hub import hf_hub_download

SOURCES = {
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
    "humanevalfix": ("bigcode/humanevalpack", "python/test-00000-of-00001.parquet"),
}

LETTERS = "ABCD"

MMLU_TEMPLATE = """The following is a multiple choice question about {subject}.

{question}
{choices}

Answer with only the letter (A, B, C or D) of the correct option."""

GSM8K_TEMPLATE = """Solve the following math problem step by step. At the end, write the final answer on its own line in the format "Final answer: <number>".

{question}"""

HUMANEVAL_TEMPLATE = """Complete the following Python function. Return the complete function implementation (including the signature and any needed imports) in a single ```python code block.

```python
{prompt}```"""

HUMANEVALFIX_TEMPLATE = """The following Python function contains a bug. Fix the bug and return the complete corrected function (including the signature and any needed imports) in a single ```python code block.

```python
{code}
{example_test}```

Fix bugs in {entry_point}."""


def load(name):
    repo, filename = SOURCES[name]
    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"))


def import_header(code):
    """Linhas de import do enunciado, reaproveitadas na execução dos testes."""
    return "\n".join(
        line for line in code.splitlines() if line.startswith(("import ", "from "))
    )


def user_msg(text):
    return [{"role": "user", "content": text}]


def build_mmlu(df, n, rng):
    # Uma questão de cada uma de n matérias distintas, para cobrir domínios variados.
    subjects = rng.sample(sorted(df["subject"].unique()), n)
    items = []
    for subject in subjects:
        rows = df[df["subject"] == subject]
        idx = rng.choice(list(rows.index))
        row = df.loc[idx]
        choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        text = MMLU_TEMPLATE.format(
            subject=subject.replace("_", " "), question=row["question"].strip(), choices=choices
        )
        items.append({
            "category": "conhecimentos_gerais",
            "source": "cais/mmlu",
            "source_id": f"{subject}/{idx}",
            "messages": user_msg(text),
            "grader": "multiple_choice",
            "reference": {"answer": LETTERS[int(row["answer"])]},
        })
    return items


def build_gsm8k(df, n, rng):
    items = []
    for idx in sorted(rng.sample(range(len(df)), n)):
        row = df.iloc[idx]
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "category": "matematica",
            "source": "openai/gsm8k",
            "source_id": f"test/{idx}",
            "messages": user_msg(GSM8K_TEMPLATE.format(question=row["question"].strip())),
            "grader": "numeric",
            "reference": {"answer": answer},
        })
    return items


def build_humaneval(df, indices):
    items = []
    for idx in indices:
        row = df.iloc[idx]
        items.append({
            "category": "programacao",
            "source": "openai/openai_humaneval",
            "source_id": row["task_id"],
            "messages": user_msg(HUMANEVAL_TEMPLATE.format(prompt=row["prompt"])),
            "grader": "python_tests",
            "reference": {
                "entry_point": row["entry_point"],
                "header": import_header(row["prompt"]),
                "prompt": row["prompt"],
                "test": row["test"],
                "canonical": row["prompt"] + row["canonical_solution"],
            },
        })
    return items


def build_humanevalfix(df, indices):
    items = []
    for idx in indices:
        row = df.iloc[idx]
        header = "\n".join(filter(None, [row["import"].strip(), import_header(row["declaration"])]))
        text = HUMANEVALFIX_TEMPLATE.format(
            code=row["declaration"] + row["buggy_solution"],
            example_test=row["example_test"],
            entry_point=row["entry_point"],
        )
        items.append({
            "category": "correcao_bugs",
            "source": "bigcode/humanevalpack",
            "source_id": row["task_id"],
            "messages": user_msg(text),
            "grader": "python_tests",
            "reference": {
                "entry_point": row["entry_point"],
                "header": header,
                "prompt": row["declaration"],
                "test": row["test_setup"] + row["test"],
                "canonical": row["declaration"] + row["canonical_solution"],
            },
        })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="prompts/prompts.jsonl")
    args = ap.parse_args()

    n = args.per_category
    rng = random.Random(args.seed)

    items = build_mmlu(load("mmlu"), n, rng)
    items += build_gsm8k(load("gsm8k"), n, rng)

    # HumanEval e HumanEvalFix derivam dos mesmos 164 problemas:
    # sorteia 2n índices e usa conjuntos disjuntos em cada categoria.
    he = load("humaneval")
    code_idx = rng.sample(range(len(he)), 2 * n)
    items += build_humaneval(he, sorted(code_idx[:n]))
    items += build_humanevalfix(load("humanevalfix"), sorted(code_idx[n:]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            item = {"id": f"{item['category']}_{i % n:02d}", **item}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = pd.Series([it["category"] for it in items]).value_counts()
    print(f"{len(items)} prompts gravados em {out}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
