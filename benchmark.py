"""Benchmark de níveis de quantização de LLMs GGUF com llama.cpp.

Para cada nível de quantização: carrega o modelo no llama-server, executa todas
as baterias de prompts (aquecimento + repetições medidas) e registra, por prompt,
as métricas de prefill/decode reportadas pelo llama.cpp, além de VRAM, RAM,
temperatura e clock da GPU amostrados durante toda a execução.

Uso:
    python benchmark.py                               # matriz completa do config.toml
    python benchmark.py --models Q4_K_M --limit 2 --reps 1   # teste rápido
    python benchmark.py --resume 20261014-153000      # retoma um run interrompido
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import tomllib
import winreg
from collections import Counter
from datetime import datetime
from pathlib import Path

import psutil

from llama_server import LlamaServer, ServerError
from monitor import GPU, ResourceMonitor

ROOT = Path(__file__).parent


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def cpu_name():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        return platform.processor()


def llama_version(binary):
    out = subprocess.run([binary, "--version"], capture_output=True, text=True)
    lines = (out.stdout + out.stderr).splitlines()
    return " | ".join(l.strip() for l in lines if l.startswith(("version:", "built with")))


def environment(cfg, gpu):
    return {
        "llama_cpp": llama_version(cfg["server"]["binary"]),
        "gpu": gpu.info(),
        "cpu": cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def wait_cooldown(gpu, cfg):
    limit, max_wait = cfg["cooldown"]["max_temp_c"], cfg["cooldown"]["max_wait_s"]
    t0 = time.perf_counter()
    while (temp := gpu.snapshot()["gpu_temp_c"]) > limit and time.perf_counter() - t0 < max_wait:
        print(f"  aguardando resfriamento: {temp} °C > {limit} °C")
        time.sleep(10)
    return gpu.snapshot()["gpu_temp_c"]


def run_llama_bench(cfg, model_path, quant, run_dir):
    lb = cfg["llama_bench"]
    print(f"  llama-bench {' '.join(lb['args'])}")
    proc = subprocess.run([lb["binary"], "-m", str(model_path), *lb["args"], "-o", "jsonl"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  llama-bench falhou ({proc.returncode}): {proc.stderr.strip()[-300:]}")
        return
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            append_jsonl(run_dir / "llama_bench.jsonl", {"quant": quant, **json.loads(line)})


def run_quant(cfg, sel, quant, prompts, run_dir, gpu, done):
    model_path = Path(cfg["models"]["dir"]) / f"{cfg['models']['name']}-{quant}.gguf"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    n_reps = sel["warmup"] + sel["reps"]
    pending = [(rep, p) for rep in range(n_reps) for p in prompts if (quant, rep, p["id"]) not in done]
    if not pending:
        print(f"[{quant}] já concluído, pulando")
        return

    print(f"\n[{quant}] {model_path.name} — {len(pending)} execuções pendentes")
    temp_start = wait_cooldown(gpu, cfg)
    vram_baseline = gpu.snapshot()["vram_used_mb"]

    srv_cfg, gen = cfg["server"], cfg["generation"]
    server = LlamaServer(srv_cfg["binary"], model_path, srv_cfg["host"], srv_cfg["port"], srv_cfg["args"],
                         run_dir / f"server_{quant}.log", srv_cfg["load_timeout_s"])
    monitor = ResourceMonitor(gpu, cfg["monitor"]["interval_s"])
    server.start()
    try:
        monitor.start(server.pid)
        server.wait_ready()
        monitor.set_phase("idle")
        time.sleep(2)  # algumas amostras com o modelo carregado e ocioso
        meta = server.model_meta()
        print(f"  carregado em {server.load_time_s:.1f}s; VRAM {gpu.snapshot()['vram_used_mb']:.0f} MB")

        monitor.set_phase("battery")
        for i, (rep, prompt) in enumerate(pending, 1):
            before = gpu.snapshot()
            t0 = time.perf_counter()
            row = {"quant": quant, "rep": rep, "is_warmup": rep < sel["warmup"],
                   "prompt_id": prompt["id"], "category": prompt["category"],
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            try:
                resp = server.chat(prompt["messages"], gen, srv_cfg["request_timeout_s"])
            except Exception as e:  # registra a falha e segue; aborta se o servidor caiu
                append_jsonl(run_dir / "raw.jsonl", {**row, "error": repr(e)})
                if not server.alive():
                    raise ServerError(f"llama-server caiu:\n{server.log_tail()}") from e
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            after = gpu.snapshot()
            t, choice = resp["timings"], resp["choices"][0]
            row.update({
                "prompt_tokens": t["prompt_n"],
                "completion_tokens": t["predicted_n"],
                "prefill_ms": round(t["prompt_ms"], 3),
                "prefill_tps": round(t["prompt_per_second"], 3),
                "decode_ms": round(t["predicted_ms"], 3),
                "decode_tps": round(t["predicted_per_second"], 3),
                "latency_ms": round(latency_ms, 3),
                "cached_tokens": t.get("cache_n", 0),
                "finish_reason": choice["finish_reason"],
                "gpu_temp_start_c": before["gpu_temp_c"],
                "gpu_temp_end_c": after["gpu_temp_c"],
                "gpu_sm_clock_end_mhz": after["gpu_sm_clock_mhz"],
                "response_text": choice["message"].get("content") or "",
                "error": None,
            })
            append_jsonl(run_dir / "raw.jsonl", row)
            print(f"  [{quant}] rep {rep}{' (aquec.)' if row['is_warmup'] else ''} "
                  f"{i}/{len(pending)} {prompt['id']:<26} "
                  f"pp {row['prefill_tps']:7.1f} t/s  tg {row['decode_tps']:6.1f} t/s  "
                  f"{row['completion_tokens']:4d} tok  {row['finish_reason']}")
    finally:
        monitor.stop()
        server.stop()
        monitor.save_csv(run_dir / f"monitor_{quant}.csv")

    size = model_path.stat().st_size
    n_params = meta.get("n_params")
    summary = {
        "quant": quant,
        "model_file": model_path.name,
        "file_size_mb": round(size / 1024**2, 1),
        "n_params": n_params,
        "bpw": round(size * 8 / n_params, 3) if n_params else None,
        "load_time_s": server.load_time_s,
        "gpu_temp_start_c": temp_start,
        "vram_baseline_mb": vram_baseline,
        **monitor.summary(),
        "server_meta": meta,
    }
    append_jsonl(run_dir / "models.jsonl", summary)
    print(f"  resumo: VRAM pico {summary['vram_peak_mb']:.0f} MB, RAM pico {summary['ram_peak_mb']:.0f} MB, "
          f"temp máx {summary['gpu_temp_max_c']} °C")

    if sel["llama_bench"]:
        run_llama_bench(cfg, model_path, quant, run_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--models", help="lista de quantizações separadas por vírgula (padrão: todas do config)")
    ap.add_argument("--limit", type=int, help="máximo de prompts por categoria")
    ap.add_argument("--reps", type=int, help="repetições medidas (padrão: config)")
    ap.add_argument("--warmup", type=int, help="repetições de aquecimento (padrão: config)")
    ap.add_argument("--no-llama-bench", action="store_true")
    ap.add_argument("--resume", metavar="RUN_ID", help="retoma um run existente com a mesma configuração")
    args = ap.parse_args()

    if args.resume:
        run_dir = ROOT / "results" / args.resume
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        cfg, sel = manifest["config"], manifest["selection"]
        print(f"Retomando {args.resume} com a configuração gravada no manifesto")
    else:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)
        exp = cfg["experiment"]
        sel = {
            "quants": args.models.split(",") if args.models else cfg["models"]["quants"],
            "limit": args.limit,
            "reps": args.reps if args.reps is not None else exp["reps"],
            "warmup": args.warmup if args.warmup is not None else exp["warmup_reps"],
            "llama_bench": cfg["llama_bench"]["enabled"] and not args.no_llama_bench,
        }
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = ROOT / cfg["experiment"]["results_dir"] / run_id
        run_dir.mkdir(parents=True)
        manifest = None

    prompts_path = ROOT / cfg["experiment"]["prompts_file"]
    prompts = load_jsonl(prompts_path)
    if sel["limit"]:
        seen = Counter()
        selected = []
        for p in prompts:
            seen[p["category"]] += 1
            if seen[p["category"]] <= sel["limit"]:
                selected.append(p)
        prompts = selected
    prompts_sha = hashlib.sha256(prompts_path.read_bytes()).hexdigest()

    gpu = GPU(cfg["monitor"]["gpu_index"])
    if manifest is None:
        manifest = {
            "run_id": run_dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "prompts_file": cfg["experiment"]["prompts_file"],
            "prompts_sha256": prompts_sha,
            "n_prompts": len(prompts),
            "selection": sel,
            "config": cfg,
            "environment": environment(cfg, gpu),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    elif manifest["prompts_sha256"] != prompts_sha:
        sys.exit("prompts.jsonl mudou desde o início deste run; não é possível retomar")

    print(f"Run {run_dir.name}: {len(sel['quants'])} níveis × {len(prompts)} prompts × "
          f"({sel['warmup']} aquec. + {sel['reps']} reps)")
    print(f"llama.cpp: {manifest['environment']['llama_cpp']}")

    done = {(r["quant"], r["rep"], r["prompt_id"])
            for r in load_jsonl(run_dir / "raw.jsonl") if r.get("error") is None}
    t0 = time.perf_counter()
    for quant in sel["quants"]:
        run_quant(cfg, sel, quant, prompts, run_dir, gpu, done)
    print(f"\nConcluído em {(time.perf_counter() - t0) / 60:.1f} min. Resultados em {run_dir}")
    print(f"Próximos passos: python grade.py {run_dir}  &&  python analyze.py {run_dir}")


if __name__ == "__main__":
    main()
