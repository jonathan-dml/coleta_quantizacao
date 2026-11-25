"""Comparação entre modelos: junta os resultados de dois ou mais runs.

Cada run precisa ter sido analisado antes (python analyze.py results/<run_id>).
Os runs devem usar o mesmo conjunto de prompts e os mesmos parâmetros de geração.

Gera em --out:
    comparacao.csv                      modelo × nível: memória, velocidade e qualidade
    tabela_comparacao.tex               variação relativa à linha de base de cada modelo
    figures/decode_comparacao.*         velocidade de decode por nível
    figures/delta_acuracia_comparacao.* perda de acurácia relativa ao próprio F16
    figures/memoria_comparacao.*        tamanho dos arquivos × VRAM total da GPU
    figures/tradeoff_comparacao.*       velocidade × qualidade dos modelos

Uso:
    python compare.py results/<run_2B> results/<run_4B> --out results/comparacao
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze import (INK, INK_2, MUTED, SERIES, SURFACE, fmt, fmt_signed, is_offload, label_points,
                     latex_wrap, new_fig, save, style_axes, tex_name)

COLUMNS = ["quant", "bpw", "file_size_mb", "vram_model_mb", "offload", "gpu_layers", "prefill_tps_mean",
           "decode_tps_mean", "decode_tps_std", "decode_tps_mean_rel_pct", "geral", "geral_delta_pp",
           "p_mcnemar", "identicas_base_pct", "truncadas_pct", "battery_s_mean"]
MARKERS = ["o", "s", "^"]


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = run_dir / "summary" / "by_quant.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} não encontrado; execute antes: python analyze.py {run_dir}")
    df = pd.read_csv(summary)
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    if "offload" not in df:  # runs anteriores ao suporte a offload
        df["offload"], df["gpu_layers"] = False, "all"
    df.insert(0, "model", manifest["config"]["models"]["name"])
    return manifest, df


def check_comparable(manifests):
    """Os runs só são comparáveis com os mesmos prompts e parâmetros de geração."""
    ref = manifests[0]
    for m in manifests[1:]:
        for what, a, b in [
            ("prompts (SHA-256)", ref["prompts_sha256"], m["prompts_sha256"]),
            ("número de prompts", ref["n_prompts"], m["n_prompts"]),
            ("parâmetros de geração", ref["config"]["generation"], m["config"]["generation"]),
            ("linha de base", ref["config"]["experiment"]["baseline"], m["config"]["experiment"]["baseline"]),
        ]:
            if a != b:
                raise SystemExit(f"runs não comparáveis: {what} diferem entre {ref['run_id']} e {m['run_id']}")


def latex_comparison(df, models, quants, baseline):
    rows = []
    for q in quants:
        cells = [tex_name(q)]
        for m in models:
            r = df[(df.model == m) & (df.quant == q)]
            if r.empty:
                cells += ["--"] * 5
                continue
            r = r.iloc[0]
            base = q == baseline
            vram = fmt(r.vram_model_mb, 0) + ("$^\\dagger$" if is_offload(r.offload) else "")
            cells += [vram, fmt(r.decode_tps_mean, 1),
                      "--" if base else fmt_signed(r.decode_tps_mean_rel_pct, 1),
                      fmt(r.geral, 2), "--" if base else fmt_signed(r.geral_delta_pp, 2)
                      + ("*" if r.p_mcnemar < 0.05 else "")]
        rows.append(" & ".join(cells))
    groups = " & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{m}}}}}" for m in models)
    sub = " & ".join(["\\textbf{VRAM} & \\textbf{\\textit{Decode}} & \\textbf{$\\Delta$\\%} & "
                      "\\textbf{Geral} & \\textbf{$\\Delta$ p.p.}"] * len(models))
    header = f" & {groups} \\\\\n\\textbf{{Nível}} & {sub}"
    offloaded = [f"{r.model} {tex_name(r.quant)}: {r.gpu_layers} camadas na GPU"
                 for r in df.itertuples() if is_offload(r.offload)]
    notes = (f"VRAM em MB; \\textit{{Decode}} em \\textit{{tokens}} por segundo; $\\Delta$\\%: variação do "
             f"\\textit{{decode}} em relação ao {tex_name(baseline)} do mesmo modelo; Geral: percentual de acertos; "
             f"$\\Delta$ p.p.: diferença do acerto geral em relação ao {tex_name(baseline)} do mesmo modelo; "
             f"* diferença significativa (McNemar, $p < 0{{,}}05$).")
    if offloaded:
        notes += " $^\\dagger$ \\textit{Offload} parcial (" + "; ".join(offloaded) + ")."
    return latex_wrap("Comparação entre os modelos por nível de quantização, em valores absolutos e relativos "
                      "à linha de base de cada modelo", "tab:comparacao",
                      "l" + "rrrrr" * len(models), header, rows, notes)


# ---------------------------------------------------------------- gráficos

def grouped_bars(ax, df, models, quants, col):
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        d = df[df.model == m].set_index("quant").reindex(quants)
        xs = [j + (i - (len(models) - 1) / 2) * width for j in range(len(quants))]
        ax.bar(xs, d[col], width=width - 0.02, color=SERIES[i], label=m)
        yield m, xs, d
    ax.set_xticks(range(len(quants)), quants)


def decode_chart(df, models, quants, out_dir):
    fig, ax = new_fig(8, 4)
    for m, xs, d in grouped_bars(ax, df, models, quants, "decode_tps_mean"):
        for x, v in zip(xs, d["decode_tps_mean"]):
            ax.annotate(fmt(v, 0), (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7, color=INK_2)
    style_axes(ax, "tokens/s")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")
    ax.set_title("Velocidade de geração (decode) por modelo e nível de quantização", color=INK, loc="left",
                 fontsize=11)
    save(fig, out_dir, "decode_comparacao")


def delta_accuracy_chart(df, models, quants, baseline, out_dir):
    fig, ax = new_fig(8, 4.2)
    x = list(range(len(quants)))
    for i, m in enumerate(models):
        d = df[df.model == m].set_index("quant").reindex(quants)
        ax.plot(x, d["geral_delta_pp"], color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m)
        for xi, (q, r) in zip(x, d.iterrows()):
            if q == baseline or pd.isna(r.geral_delta_pp):
                continue
            star = "*" if r.p_mcnemar < 0.05 else ""
            ax.annotate(fmt_signed(r.geral_delta_pp, 1).replace("$-$", "−") + star, (xi, r.geral_delta_pp),
                        xytext=(0, 8 if i == 0 else -14), textcoords="offset points", ha="center", fontsize=7,
                        color=INK_2)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(x, quants)
    style_axes(ax, f"Δ acerto geral vs. {baseline} (p.p.)")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="lower left")
    ax.set_title(f"Variação da qualidade em relação ao {baseline} de cada modelo", color=INK, loc="left",
                 fontsize=11)
    fig.text(0.01, 0.01, "* diferença significativa (McNemar, p < 0,05)", fontsize=7, color=MUTED)
    save(fig, out_dir, "delta_acuracia_comparacao")


def memory_chart(df, models, quants, vram_total_mb, out_dir):
    fig, ax = new_fig(8, 4.2)
    for m, xs, d in grouped_bars(ax, df, models, quants, "file_size_mb"):
        for x, r in zip(xs, d.itertuples()):
            label = fmt(r.file_size_mb, 0) + ("\noffload" if is_offload(r.offload) else "")
            ax.annotate(label, (x, r.file_size_mb), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=6.5, color=INK_2)
    ax.axhline(vram_total_mb, color=INK_2, linewidth=1, linestyle="--")
    ax.text(0.99, vram_total_mb, f"VRAM total da GPU ({fmt(vram_total_mb / 1024, 0)} GB)",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(df["file_size_mb"].max(), vram_total_mb) * 1.15)
    style_axes(ax, "Tamanho do arquivo (MB)")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.72))
    ax.set_title("Tamanho dos arquivos em relação à memória de vídeo disponível", color=INK, loc="left",
                 fontsize=11)
    save(fig, out_dir, "memoria_comparacao")


def tradeoff_chart(df, models, out_dir):
    fig, ax = new_fig(7, 4.8)
    for i, m in enumerate(models):
        d = df[df.model == m]
        ax.scatter(d["decode_tps_mean"], d["geral"], s=70, color=SERIES[i], marker=MARKERS[i],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=m)
        label_points(ax, d["decode_tps_mean"].tolist(), d["geral"].tolist(), d["quant"].tolist())
    style_axes(ax, "Qualidade geral (% de acertos)")
    ax.set_ylim(-5, 110)
    ax.margins(x=0.12)
    ax.set_xlabel("Velocidade de decode (tokens/s)", color=INK_2)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")
    ax.set_title("Compromisso entre velocidade e qualidade por modelo", color=INK, loc="left", fontsize=11)
    save(fig, out_dir, "tradeoff_comparacao")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="diretórios dos runs, na ordem desejada (ex.: 2B e depois 4B)")
    ap.add_argument("--out", default="results/comparacao")
    args = ap.parse_args()
    if len(args.runs) > len(MARKERS):
        raise SystemExit(f"no máximo {len(MARKERS)} runs por comparação")

    loaded = [load_run(r) for r in args.runs]
    manifests = [m for m, _ in loaded]
    check_comparable(manifests)
    df = pd.concat([d for _, d in loaded], ignore_index=True)
    models = list(dict.fromkeys(df["model"]))
    if len(models) != len(loaded):
        raise SystemExit("cada run deve ser de um modelo diferente")
    quants = [q for q in manifests[0]["selection"]["quants"] if q in set(df["quant"])]
    baseline = manifests[0]["config"]["experiment"]["baseline"]
    vram_total = manifests[0]["environment"]["gpu"]["vram_total_mb"]

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparacao.csv", index=False)
    (out_dir / "tabela_comparacao.tex").write_text(latex_comparison(df, models, quants, baseline),
                                                   encoding="utf-8")

    decode_chart(df, models, quants, fig_dir)
    delta_accuracy_chart(df, models, quants, baseline, fig_dir)
    memory_chart(df, models, quants, vram_total, fig_dir)
    tradeoff_chart(df, models, fig_dir)

    cols = ["model", "quant", "vram_model_mb", "decode_tps_mean", "decode_tps_mean_rel_pct", "geral",
            "geral_delta_pp", "p_mcnemar"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(df[cols].to_string(index=False))
    print(f"\nResultados em {out_dir}")


if __name__ == "__main__":
    main()
