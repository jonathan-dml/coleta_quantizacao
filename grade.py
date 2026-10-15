"""Correção automática das respostas coletadas pelo benchmark.

Executada depois da coleta, para que a correção (incluindo a execução de código
gerado) não interfira nas medições de desempenho.

Uso:
    python grade.py results/<run_id>      # gera results/<run_id>/graded.jsonl
    python grade.py --selftest            # valida o avaliador com as respostas de referência
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent


# ---------------------------------------------------------------- múltipla escolha

def extract_choice(text):
    text = text.strip()
    # A letra é sensível a maiúsculas para não confundir o artigo "a" com a opção A.
    m = re.search(r"(?i:answer)\s*(?i:is|:)?\s*\(?\**([A-D])\b", text)  # "Answer: B", "answer is (B)"
    if m:
        return m.group(1)
    # Letras no início de linha ("B." / "**B**"). Se houver mais de uma distinta, o
    # modelo está apenas listando as alternativas: resposta ambígua, não um acerto.
    starts = set(re.findall(r"^\(?\**([A-D])\**\)?(?:[.):\s]|$)", text, re.MULTILINE))
    if starts:
        return starts.pop() if len(starts) == 1 else None
    # Último recurso: uma única letra isolada distinta no texto.
    letters = set(re.findall(r"\b([A-D])\b", text))
    return letters.pop() if len(letters) == 1 else None


# ---------------------------------------------------------------- numérico

NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def normalize_number(s):
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text):
    # Tolera markdown entre o rótulo e o número: "**Final Answer:**\n8000", "Final answer: $18".
    m = re.findall(rf"final answer[\W_]*?({NUMBER})", text, re.IGNORECASE)
    candidates = m or re.findall(NUMBER, text)
    return normalize_number(candidates[-1]) if candidates else None


# ---------------------------------------------------------------- código

def extract_code(text, entry_point):
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        # Bloco não fechado (resposta truncada) ou código sem cercas.
        m = re.search(r"```(?:python|py)?[^\n]*\n(.*)", text, re.DOTALL)
        blocks = [m.group(1)] if m else [text]
    for block in blocks:
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
            return block
    return blocks[0]


def build_program(code, ref):
    entry = ref["entry_point"]
    if not re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        # O modelo devolveu apenas o corpo da função: completa o enunciado.
        code = ref["prompt"] + code
    program = f"{ref['header']}\n\n{code}\n\n{ref['test']}\n"
    if not re.search(rf"^check\(\s*{re.escape(entry)}\s*\)", ref["test"], re.MULTILINE):
        program += f"\ncheck({entry})\n"
    return program


def run_python(program, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)], cwd=tmp, capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, ""
    lines = proc.stderr.strip().splitlines()
    return False, lines[-1][:200] if lines else f"exit {proc.returncode}"


# ---------------------------------------------------------------- despacho

class Grader:
    def __init__(self, code_timeout=10):
        self.code_timeout = code_timeout
        self._cache = {}  # respostas idênticas (comuns no greedy) são executadas uma vez

    def grade(self, prompt, response):
        kind, ref = prompt["grader"], prompt["reference"]
        if kind == "multiple_choice":
            got = extract_choice(response)
            return {"extracted": got, "correct": got == ref["answer"], "error": ""}
        if kind == "numeric":
            got = extract_number(response)
            expected = normalize_number(ref["answer"])
            ok = got is not None and abs(got - expected) < 1e-6
            return {"extracted": got, "correct": ok, "error": ""}
        if kind == "python_tests":
            program = build_program(extract_code(response, ref["entry_point"]), ref)
            key = hashlib.sha256(program.encode()).hexdigest()
            if key not in self._cache:
                self._cache[key] = run_python(program, self.code_timeout)
            ok, err = self._cache[key]
            return {"extracted": None, "correct": ok, "error": err}
        raise ValueError(f"grader desconhecido: {kind}")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config():
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def grade_run(run_dir, grader):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    prompts = {p["id"]: p for p in load_jsonl(ROOT / manifest["prompts_file"])}
    rows = [r for r in load_jsonl(run_dir / "raw.jsonl") if r.get("error") is None]

    out_path = run_dir / "graded.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            result = grader.grade(prompts[row["prompt_id"]], row["response_text"])
            keys = ("quant", "rep", "is_warmup", "prompt_id", "category", "finish_reason")
            f.write(json.dumps({**{k: row[k] for k in keys}, **result}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(rows)} corrigidas")
    print(f"{len(rows)} respostas corrigidas -> {out_path}")


def selftest(grader):
    """Respostas de referência devem acertar 100%; respostas vazias devem errar."""
    prompts = load_jsonl(ROOT / load_config()["experiment"]["prompts_file"])
    failures = []
    for p in prompts:
        ref = p["reference"]
        if p["grader"] == "multiple_choice":
            good = [ref["answer"], f"The answer is {ref['answer']}.", f"**{ref['answer']}**"]
            bad = ["I don't know.", "Options:\nA. one\nB. two\nC. three\nD. four"]
        elif p["grader"] == "numeric":
            good = [f"Some steps: 3 + 4 = 7\nFinal answer: {ref['answer']}",
                    f"Step 1 costs 10.\n**Final Answer**: {ref['answer']}",
                    f"Step 1 costs 10.\n**Final Answer:**\n{ref['answer']}",
                    f"So the result is {ref['answer']}."]
            bad = ["Final answer: 123456789"]
        else:
            good = [f"Here it is:\n```python\n{ref['canonical']}\n```"]
            bad = [f"```python\n{ref['prompt']}    pass\n```"]
        for resp in good:
            r = grader.grade(p, resp)
            if not r["correct"]:
                failures.append((p["id"], "deveria acertar", resp[:80], r))
        for resp in bad:
            r = grader.grade(p, resp)
            if r["correct"]:
                failures.append((p["id"], "deveria errar", resp[:80], r))
    for f in failures:
        print("FALHA:", *f)
    print(f"selftest: {len(prompts)} prompts, {len(failures)} falhas")
    return not failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    grader = Grader(load_config()["grading"]["code_timeout_s"])
    if args.selftest:
        sys.exit(0 if selftest(grader) else 1)
    if not args.run_dir:
        ap.error("informe o diretório do run ou --selftest")
    grade_run(args.run_dir, grader)


if __name__ == "__main__":
    main()
