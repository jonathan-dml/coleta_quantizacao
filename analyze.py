"""Estatísticas descritivas, comparação com a linha de base, tabelas e gráficos.

Lê raw.jsonl, graded.jsonl, models.jsonl e llama_bench.jsonl de um run e grava:
    summary/perf_by_quant_category.csv  média, dp, mín, máx, mediana por quant × categoria
    summary/perf_by_quant.csv           vazão agregada por bateria (média/dp entre repetições)
    summary/quality_by_quant.csv        acurácia/pass@1 por quant × categoria
    summary/by_quant.csv                visão consolidada + variação relativa à linha de base,
                                        IC de Wilson, McNemar e respostas idênticas à linha de base
    summary/tabela_desempenho.tex       tabela de desempenho (formato da Tabela 3 do artigo)
    summary/tabela_qualidade.tex        tabela de qualidade (formato da Tabela 4 do artigo)
    figures/*.png|pdf                   gráficos comparativos

Uso:
    python analyze.py results/<run_id>
"""

import argparse
import json
import math
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


def wilson_ci(k, n, z=1.96):
    """Intervalo de confiança de Wilson (95%) para uma proporção, em %."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (center - half) * 100, (center + half) * 100


def mcnemar_exact(b, c):
    """Teste exato de McNemar (binomial bicaudal sobre os pares discordantes)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def first_measured(df):
    """Linhas da primeira repetição medida de cada nível (o greedy é determinístico)."""
    df = df[~df["is_warmup"]]
    return df[df["rep"] == df.groupby("quant")["rep"].transform("min")]


def paired_stats(graded, raw_measured, quants, baseline):
    """Qualidade por prompt comparada à linha de base: IC de Wilson, McNemar,
    novos erros/acertos e proporção de respostas idênticas às da linha de base."""
    correct = first_measured(graded).set_index(["quant", "prompt_id"])["correct"].astype(bool)
    text = first_measured(raw_measured).set_index(["quant", "prompt_id"])["response_text"]
    has_base = baseline in correct.index.get_level_values("quant")
    rows = []
    for q in quants:
        c = correct.loc[q]
        lo, hi = wilson_ci(int(c.sum()), len(c))
        row = {"quant": q, "acertos": int(c.sum()), "n_prompts": len(c), "geral_ic95_inf": lo, "geral_ic95_sup": hi}
        if has_base:
            base = correct.loc[baseline].reindex(c.index).astype(bool)
            novos_erros, novos_acertos = int((base & ~c).sum()), int((~base & c).sum())
            same = text.loc[q] == text.loc[baseline].reindex(text.loc[q].index)
            row.update({
                "novos_erros": novos_erros,
                "novos_acertos": novos_acertos,
                "p_mcnemar": math.nan if q == baseline else mcnemar_exact(novos_erros, novos_acertos),
                "identicas_base_pct": same.mean() * 100,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def llama_bench_table(lb):
    if lb.empty:
        return pd.DataFrame()
    lb = lb.assign(test=lb.apply(lambda r: f"pp{r.n_prompt}" if r.n_gen == 0 else f"tg{r.n_gen}", axis=1))
    wide = lb.pivot_table(index="quant", columns="test", values=["avg_ts", "stddev_ts"], sort=False)
    wide.columns = [f"bench_{test}_{'tps' if v == 'avg_ts' else 'std'}" for v, test in wide.columns]
    return wide.reset_index()


def consolidate(models, perf, qual, paired, bench, baseline):
    df = models.copy()
    df["vram_model_mb"] = df["vram_loaded_mb"] - df["vram_baseline_mb"]
    df = df.merge(perf[["quant", "prefill_tps_mean", "prefill_tps_std", "decode_tps_mean",
                        "decode_tps_std", "battery_s_mean"]], on="quant", how="left")
    df = df.merge(qual, on="quant", how="left")
    df = df.merge(paired, on="quant", how="left")
    if not bench.empty:
        df = df.merge(bench, on="quant", how="left")
    if baseline in set(df["quant"]):
        base = df.set_index("quant").loc[baseline]
        for col in ["file_size_mb", "vram_model_mb", "prefill_tps_mean", "decode_tps_mean", "load_time_s",
                    "battery_s_mean"]:
            df[f"{col}_rel_pct"] = (df[col] / base[col] - 1) * 100
        for col in [*CATEGORIES, "geral"]:
            df[f"{col}_delta_pp"] = df[col] - base[col]  # diferença em pontos percentuais
    return df.drop(columns=["server_meta"], errors="ignore")


def fmt(x, nd=1):
    return "--" if pd.isna(x) else f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_signed(x, nd=1):
    if pd.isna(x):
        return "--"
    sign = "+" if x > 0 else "$-$" if x < 0 else ""
    return sign + fmt(abs(x), nd)


def fmt_p(p):
    return "--" if pd.isna(p) else "$<$0,001" if p < 0.001 else fmt(p, 3)


def is_offload(value):
    """True se o nível rodou com offload parcial (aceita bool do Python ou do numpy;
    runs anteriores ao suporte a offload não têm a coluna)."""
    return pd.api.types.is_bool(value) and bool(value)


def tex_name(quant):
    return quant.replace("_", "\\_")


def latex_wrap(caption, label, colspec, header, rows, notes):
    return "\n".join([
        "\\begin{table}[htb]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        header + " \\\\",
        "\\hline",
        *[r + " \\\\" for r in rows],
        "\\hline",
        "\\end{tabular}",
        "\\\\[2pt]{\\footnotesize \\textbf{Fonte: Dados da pesquisa.}}",
        *(["\\\\[2pt]{\\footnotesize " + notes + "}"] if notes else []),
        "\\end{table}",
        "",
    ])


def latex_performance(df, model, baseline):
    """Tabela de desempenho no formato da Tabela 3 do artigo."""
    rows, offloaded = [], []
    for _, r in df.iterrows():
        vram = fmt(r.vram_model_mb, 0)
        if is_offload(r.get("offload")):
            vram += "$^\\dagger$"
            offloaded.append(f"{tex_name(r.quant)} com {r.gpu_layers} camadas na GPU e as demais na CPU")
        rows.append(" & ".join([
            tex_name(r.quant), vram,
            "--" if r.quant == baseline else fmt_signed(r.get("vram_model_mb_rel_pct"), 1),
            fmt(r.load_time_s, 1),
            f"{fmt(r.prefill_tps_mean, 0)} $\\pm$ {fmt(r.prefill_tps_std, 0)}",
            f"{fmt(r.decode_tps_mean, 1)} $\\pm$ {fmt(r.decode_tps_std, 1)}",
            "--" if r.quant == baseline else fmt_signed(r.get("decode_tps_mean_rel_pct"), 1),
            fmt(r.get("bench_tg128_tps"), 1),
            fmt(r.battery_s_mean, 1),
        ]))
    header = ("\\textbf{Nível} & \\textbf{VRAM (MB)} & \\textbf{$\\Delta$VRAM (\\%)} & \\textbf{Carga (s)} & "
              "\\textbf{\\textit{Prefill} (t/s)} & \\textbf{\\textit{Decode} (t/s)} & "
              "\\textbf{$\\Delta$\\textit{Decode} (\\%)} & \\textbf{tg128 (t/s)} & \\textbf{Bateria (s)}")
    notes = ("VRAM: memória atribuída ao modelo carregado; tg128: geração de 128 \\textit{tokens} no "
             "llama-bench; Bateria: tempo total de uma bateria de prompts.")
    if offloaded:
        notes += " $^\\dagger$ \\textit{Offload} parcial: " + "; ".join(offloaded) + "."
    return latex_wrap(
        f"Métricas de desempenho do {model} por nível de quantização "
        "(média $\\pm$ desvio padrão entre as baterias medidas)",
        f"tab:desempenho-{model}", "lrrrrrrrr", header, rows, notes)


def latex_quality(df, model, baseline):
    """Tabela de qualidade no formato da Tabela 4 do artigo."""
    rows = []
    for _, r in df.iterrows():
        is_base = r.quant == baseline
        rows.append(" & ".join([
            tex_name(r.quant),
            *[fmt(r[c], 0) for c in CATEGORIES],
            fmt(r.geral, 2),
            "--" if is_base else fmt_signed(r.get("geral_delta_pp"), 2),
            "--" if is_base else fmt_p(r.get("p_mcnemar")),
            fmt(r.truncadas_pct, 2),
            fmt(r.get("identicas_base_pct"), 2),
        ]))
    header = ("\\textbf{Nível} & \\textbf{MMLU} & \\textbf{GSM8K} & \\textbf{HumanEval} & \\textbf{HEFix} & "
              "\\textbf{Geral} & \\textbf{$\\Delta$ (p.p.)} & \\textbf{\\textit{p}} & \\textbf{Trunc.} & "
              "\\textbf{Idênt.}")
    notes = (f"MMLU, GSM8K, HumanEval e HEFix (HumanEvalFix): percentual de acertos em cada categoria. "
             f"Geral: percentual de acertos em todos os prompts. $\\Delta$ (p.p.): diferença do acerto geral "
             f"em relação ao {tex_name(baseline)}. \\textit{{p}}: teste exato de McNemar em relação ao "
             f"{tex_name(baseline)}. Trunc.: percentual de respostas interrompidas pelo limite de "
             f"\\textit{{tokens}}. Idênt.: percentual de respostas idênticas às do {tex_name(baseline)}.")
    return latex_wrap(f"Qualidade das respostas do {model} por nível de quantização",
                      f"tab:qualidade-{model}", "lrrrrrrrrr", header, rows, notes)


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


def memory_chart(df, model, out_dir):
    fig, ax = new_fig()
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], df["file_size_mb"], width=w, color=SERIES[0], label="Arquivo em disco")
    ax.bar([i + w / 2 + 0.01 for i in x], df["vram_model_mb"], width=w, color=SERIES[1],
           label="VRAM ocupada (modelo + KV cache + buffers)")
    # Com offload parcial parte dos pesos fica na RAM: a VRAM fica abaixo do arquivo.
    for i, r in zip(x, df.itertuples()):
        if is_offload(getattr(r, "offload", None)):
            ax.annotate(f"offload\n{r.gpu_layers} camadas\nna GPU", (i + w / 2, r.vram_model_mb), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(list(x), df["quant"])
    style_axes(ax, "MB")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")
    ax.set_title(f"Tamanho em disco e VRAM por nível de quantização — {model}", color=INK, loc="left",
                 fontsize=11)
    save(fig, out_dir, "memoria")


def label_points(ax, xs, ys, labels, color=INK_2):
    """Rótulos ao lado de cada ponto; pontos muito próximos alternam o lado do
    rótulo para que os textos não se sobreponham."""
    x_span = (max(xs) - min(xs)) or 1
    placed = []
    for x, y, text in sorted(zip(xs, ys, labels)):
        crowded = any(abs(x - px) < 0.08 * x_span and abs(y - py) < 8 for px, py, side in placed if side == "r")
        side = "l" if crowded else "r"
        ax.annotate(text, (x, y), xytext=(-7, -4) if side == "l" else (7, 4), textcoords="offset points",
                    ha="right" if side == "l" else "left", va="top" if side == "l" else "bottom",
                    fontsize=8, color=color)
        placed.append((x, y, side))


def quality_chart(qual, model, out_dir):
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
    fig.suptitle(f"Qualidade das respostas por categoria — {model}", color=INK, x=0.02, ha="left", fontsize=11)
    save(fig, out_dir, "qualidade_por_categoria")


def tradeoff_chart(df, model, out_dir):
    fig, ax = new_fig(6.5, 4.5)
    ax.scatter(df["decode_tps_mean"], df["geral"], s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
               zorder=3)
    label_points(ax, df["decode_tps_mean"].tolist(), df["geral"].tolist(),
                 [f"{r.quant}\n{fmt(r.file_size_mb, 0)} MB" for r in df.itertuples()])
    style_axes(ax, "Qualidade geral (% de acertos)")
    ax.set_ylim(-5, 110)
    ax.margins(x=0.15)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Velocidade de decode (tokens/s)", color=INK_2)
    ax.set_title(f"Compromisso entre velocidade e qualidade — {model}", color=INK, loc="left", fontsize=11)
    save(fig, out_dir, "tradeoff_velocidade_qualidade")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = manifest["config"]["experiment"]["baseline"]
    model = manifest["config"]["models"]["name"]

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

    paired = paired_stats(graded, measured, quants, baseline)
    df = consolidate(models, perf, qual, paired, bench, baseline)
    df.to_csv(summary_dir / "by_quant.csv", index=False)
    (summary_dir / "tabela_desempenho.tex").write_text(latex_performance(df, model, baseline), encoding="utf-8")
    (summary_dir / "tabela_qualidade.tex").write_text(latex_quality(df, model, baseline), encoding="utf-8")

    bar_chart(df, "decode_tps_mean", "decode_tps_std", "tokens/s",
              f"Velocidade de geração (decode) — {model}", fig_dir, "decode_tps")
    bar_chart(df, "prefill_tps_mean", "prefill_tps_std", "tokens/s",
              f"Velocidade de processamento do prompt (prefill) — {model}", fig_dir, "prefill_tps")
    bar_chart(df, "load_time_s", None, "segundos", f"Tempo de carregamento do modelo — {model}",
              fig_dir, "carga", nd=1)
    memory_chart(df, model, fig_dir)
    quality_chart(qual, model, fig_dir)
    tradeoff_chart(df, model, fig_dir)

    cols = ["quant", "bpw", "file_size_mb", "vram_model_mb", "prefill_tps_mean", "decode_tps_mean", "geral",
            "geral_ic95_inf", "geral_ic95_sup", "p_mcnemar", "identicas_base_pct"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"\nTabelas em {summary_dir}\nGráficos em {fig_dir}")


if __name__ == "__main__":
    main()
