"""Estatísticas descritivas, comparação com a linha de base, tabelas e gráficos.

Lê raw.jsonl, graded.jsonl, models.jsonl e llama_bench.jsonl de um run e grava:
    summary/perf_by_quant_category.csv  média, dp, mín, máx, mediana por quant × categoria
    summary/perf_by_quant.csv           vazão agregada por bateria (média/dp entre repetições)
    summary/quality_by_quant.csv        acurácia/pass@1 por quant × categoria
    summary/by_quant.csv                visão consolidada + variação relativa à linha de base
    summary/tabela_resultados.tex       tabela no formato da Tabela 2 da proposta
    figures/*.png|pdf                   gráficos comparativos

Uso:
    python analyze.py results/<run_id>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CATEGORIES = {
    "conhecimentos_gerais": "Conhecimentos gerais (MMLU)",
    "matematica": "Matemática (GSM8K)",
    "programacao": "Programação (HumanEval)",
    "correcao_bugs": "Correção de bugs (HumanEvalFix)",
}

# Paleta de referência validada (modo claro); ordem fixa por categoria.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
METRICS = ["prefill_tps", "decode_tps", "latency_ms", "prompt_tokens", "completion_tokens"]


def load_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def describe(df, by, cols):
    stats = df.groupby(by, sort=False)[cols].agg(["mean", "std", "min", "max", "median", "count"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    return stats.reset_index()


def battery_throughput(measured):
    """Vazão de cada bateria completa: tokens totais / tempo total da fase.

    Ponderar pelo número de tokens evita que respostas de 1–2 tokens (MMLU)
    distorçam a média de tokens/s do decode.
    """
    per_rep = measured.groupby(["quant", "rep"], sort=False).agg(
        prompt_tokens=("prompt_tokens", "sum"), prefill_ms=("prefill_ms", "sum"),
        completion_tokens=("completion_tokens", "sum"), decode_ms=("decode_ms", "sum"),
        battery_s=("latency_ms", lambda x: x.sum() / 1000),
    ).reset_index()
    per_rep["prefill_tps"] = per_rep["prompt_tokens"] / per_rep["prefill_ms"] * 1000
    per_rep["decode_tps"] = per_rep["completion_tokens"] / per_rep["decode_ms"] * 1000
    return describe(per_rep, "quant", ["prefill_tps", "decode_tps", "battery_s"])


def quality(graded, raw_measured, quants):
    measured = graded[~graded["is_warmup"]]
    by_cat = (measured.groupby(["quant", "category"], sort=False)["correct"].mean() * 100).unstack()
    by_cat = by_cat.reindex(index=quants, columns=list(CATEGORIES))
    by_cat["geral"] = measured.groupby("quant")["correct"].mean().reindex(quants) * 100
    # Determinismo: fração de prompts com resposta idêntica em todas as repetições.
    distinct = raw_measured.groupby(["quant", "prompt_id"])["response_text"].nunique()
    by_cat["respostas_identicas_pct"] = (distinct == 1).groupby("quant").mean().reindex(quants) * 100
    trunc = raw_measured.groupby("quant")["finish_reason"].apply(lambda s: (s == "length").mean() * 100)
    by_cat["truncadas_pct"] = trunc.reindex(quants)
    return by_cat.reset_index().rename(columns={"index": "quant"})


def llama_bench_table(lb):
    if lb.empty:
        return pd.DataFrame()
    lb = lb.assign(test=lb.apply(lambda r: f"pp{r.n_prompt}" if r.n_gen == 0 else f"tg{r.n_gen}", axis=1))
    wide = lb.pivot_table(index="quant", columns="test", values=["avg_ts", "stddev_ts"], sort=False)
    wide.columns = [f"bench_{test}_{'tps' if v == 'avg_ts' else 'std'}" for v, test in wide.columns]
    return wide.reset_index()


def consolidate(models, perf, qual, bench, baseline):
    df = models.copy()
    df["vram_model_mb"] = df["vram_loaded_mb"] - df["vram_baseline_mb"]
    df = df.merge(perf[["quant", "prefill_tps_mean", "prefill_tps_std", "decode_tps_mean",
                        "decode_tps_std", "battery_s_mean"]], on="quant", how="left")
    df = df.merge(qual, on="quant", how="left")
    if not bench.empty:
        df = df.merge(bench, on="quant", how="left")
    if baseline in set(df["quant"]):
        base = df.set_index("quant").loc[baseline]
        for col in ["file_size_mb", "vram_model_mb", "prefill_tps_mean", "decode_tps_mean", "load_time_s"]:
            df[f"{col}_rel_pct"] = (df[col] / base[col] - 1) * 100
        for col in [*CATEGORIES, "geral"]:
            df[f"{col}_delta_pp"] = df[col] - base[col]  # diferença em pontos percentuais
    return df.drop(columns=["server_meta"], errors="ignore")


def fmt(x, nd=1):
    return "--" if pd.isna(x) else f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def latex_table(df, baseline):
    rows = []
    for _, r in df.iterrows():
        name = f"{r.quant} (base)" if r.quant == baseline else r.quant
        rows.append(
            f"{name.replace('_', chr(92) + '_')} & {fmt(r.bpw, 2)} & {fmt(r.file_size_mb, 0)} & "
            f"{fmt(r.vram_model_mb, 0)} & {fmt(r.prefill_tps_mean, 0)} $\\pm$ {fmt(r.prefill_tps_std, 0)} & "
            f"{fmt(r.decode_tps_mean, 1)} $\\pm$ {fmt(r.decode_tps_std, 1)} & {fmt(r.geral, 1)} \\\\"
        )
    return "\n".join([
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados por nível de quantização}",
        "\\label{tab:resultados}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "\\textbf{Nível} & \\textbf{bpw} & \\textbf{Tam. (MB)} & \\textbf{VRAM (MB)} & "
        "\\textbf{\\textit{Prefill} (t/s)} & \\textbf{\\textit{Decode} (t/s)} & \\textbf{Qualidade (\\%)} \\\\",
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\\\[2pt]{\\footnotesize \\textbf{Fonte: Dados da Pesquisa.}}",
        "\\end{table}",
        "",
    ])


# ---------------------------------------------------------------- gráficos

def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel(ylabel, color=INK_2)


def new_fig(width=7.5, height=4):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return fig, ax


def save(fig, out_dir, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def bar_chart(df, col, std_col, ylabel, title, out_dir, name, nd=0):
    fig, ax = new_fig()
    x = range(len(df))
    ax.bar(x, df[col], width=0.6, color=SERIES[0], yerr=df[std_col] if std_col else None,
           error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for i, v in zip(x, df[col]):
        ax.annotate(fmt(v, nd), (i, v), xytext=(0, 6 if not std_col else 10), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(list(x), df["quant"])
    style_axes(ax, ylabel)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    save(fig, out_dir, name)


def memory_chart(df, out_dir):
    fig, ax = new_fig()
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], df["file_size_mb"], width=w, color=SERIES[0], label="Arquivo em disco")
    ax.bar([i + w / 2 + 0.01 for i in x], df["vram_model_mb"], width=w, color=SERIES[1],
           label="VRAM ocupada (modelo + KV cache + buffers)")
    ax.set_xticks(list(x), df["quant"])
    style_axes(ax, "MB")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")
    ax.set_title("Tamanho em disco e VRAM por nível de quantização", color=INK, loc="left", fontsize=11)
    save(fig, out_dir, "memoria")


def quality_chart(qual, out_dir):
    # Pequenos múltiplos (um painel por categoria, mesma escala): as categorias
    # frequentemente empatam, e linhas sobrepostas num único eixo se esconderiam.
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharey=True, facecolor=SURFACE)
    x = list(range(len(qual)))
    for ax, (cat, label) in zip(axes.flat, CATEGORIES.items()):
        ax.plot(x, qual[cat], color=SERIES[0], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for xi, v in zip(x, qual[cat]):
            ax.annotate(fmt(v, 0), (xi, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=INK_2)
        ax.set_xticks(x, qual["quant"], fontsize=7, rotation=45)
        ax.set_ylim(-5, 115)
        style_axes(ax, "Acertos (%)" if ax in axes[:, 0] else "")
        ax.set_title(label, color=INK, loc="left", fontsize=9)
    fig.suptitle("Qualidade das respostas por categoria", color=INK, x=0.02, ha="left", fontsize=11)
    save(fig, out_dir, "qualidade_por_categoria")


def tradeoff_chart(df, out_dir):
    fig, ax = new_fig(6.5, 4.5)
    ax.scatter(df["decode_tps_mean"], df["geral"], s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
               zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"{r.quant}\n{fmt(r.file_size_mb, 0)} MB", (r.decode_tps_mean, r.geral), xytext=(7, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)
    style_axes(ax, "Qualidade geral (% de acertos)")
    ax.set_ylim(-5, 110)
    ax.margins(x=0.15)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Velocidade de decode (tokens/s)", color=INK_2)
    ax.set_title("Compromisso entre velocidade e qualidade", color=INK, loc="left", fontsize=11)
    save(fig, out_dir, "tradeoff_velocidade_qualidade")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = manifest["config"]["experiment"]["baseline"]

    raw = load_jsonl(run_dir / "raw.jsonl")
    raw = raw[raw["error"].isna()]
    measured = raw[~raw["is_warmup"]]
    quants = [q for q in manifest["selection"]["quants"] if q in set(measured["quant"])]
    models = load_jsonl(run_dir / "models.jsonl").drop_duplicates("quant", keep="last")
    models = models.set_index("quant").reindex(quants).reset_index()

    graded_path = run_dir / "graded.jsonl"
    if not graded_path.exists():
        raise SystemExit(f"{graded_path} não encontrado; execute antes: python grade.py {run_dir}")
    graded = load_jsonl(graded_path)

    summary_dir, fig_dir = run_dir / "summary", run_dir / "figures"
    summary_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    describe(measured, ["quant", "category"], METRICS).to_csv(
        summary_dir / "perf_by_quant_category.csv", index=False)
    perf = battery_throughput(measured)
    perf.to_csv(summary_dir / "perf_by_quant.csv", index=False)
    qual = quality(graded, measured, quants)
    qual.to_csv(summary_dir / "quality_by_quant.csv", index=False)
    bench = llama_bench_table(load_jsonl(run_dir / "llama_bench.jsonl"))
    if not bench.empty:
        bench.to_csv(summary_dir / "llama_bench.csv", index=False)

    df = consolidate(models, perf, qual, bench, baseline)
    df.to_csv(summary_dir / "by_quant.csv", index=False)
    (summary_dir / "tabela_resultados.tex").write_text(latex_table(df, baseline), encoding="utf-8")

    bar_chart(df, "decode_tps_mean", "decode_tps_std", "tokens/s",
              "Velocidade de geração (decode)", fig_dir, "decode_tps")
    bar_chart(df, "prefill_tps_mean", "prefill_tps_std", "tokens/s",
              "Velocidade de processamento do prompt (prefill)", fig_dir, "prefill_tps")
    bar_chart(df, "load_time_s", None, "segundos", "Tempo de carregamento do modelo", fig_dir, "carga", nd=1)
    memory_chart(df, fig_dir)
    quality_chart(qual, fig_dir)
    tradeoff_chart(df, fig_dir)

    cols = ["quant", "bpw", "file_size_mb", "vram_model_mb", "prefill_tps_mean", "decode_tps_mean", "geral"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"\nTabelas em {summary_dir}\nGráficos em {fig_dir}")


if __name__ == "__main__":
    main()
