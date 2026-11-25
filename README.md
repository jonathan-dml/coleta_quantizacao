# Benchmark de quantização de LLMs com llama.cpp

Script de coleta do TCC *Análise do Impacto da Quantização no Desempenho de LLMs Open Weight
Executados Localmente* (Sistemas de Informação, PUC Minas São Gabriel).

Para cada nível de quantização de um modelo GGUF, o script carrega o modelo no `llama-server`,
executa uma bateria de prompts organizados por categoria (aquecimento + repetições medidas),
registra as métricas de desempenho de cada execução e corrige automaticamente as respostas.

## Requisitos

- llama.cpp com `llama-server` e `llama-bench` no PATH (testado na build 11149, commit d2e54583c, CUDA)
- GPU NVIDIA (o monitoramento usa NVML)
- Python 3.11+

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Uso

```powershell
# 1. Gera o conjunto fixo de prompts (já versionado em prompts/prompts.jsonl)
python prepare_prompts.py

# 2. Valida o avaliador: respostas de referência devem acertar 100%
python grade.py --selftest

# 3. Executa a matriz experimental definida em config.toml
python benchmark.py                                      # modelo padrão (Qwen3.5-2B)
python benchmark.py --model Qwen3.5-4B                   # outro modelo (prefixo dos arquivos GGUF)
python benchmark.py --models Q4_K_M --limit 2 --reps 1   # teste rápido
python benchmark.py --resume <run_id>                    # retoma um run interrompido

# 4. Corrige as respostas e gera estatísticas, tabelas e gráficos de cada run
python grade.py results\<run_id>
python analyze.py results\<run_id>

# 5. Compara os modelos (cada run precisa ter passado pelo passo 4)
python compare.py results\<run_2B> results\<run_4B> --out results\comparacao
```

Feche aplicativos que usam a GPU antes da execução: a VRAM ocupada por outros processos é
medida como linha de base e descontada, mas a concorrência afeta os tempos.

## Modelos maiores que a VRAM (offload parcial)

Arquivos que não cabem inteiros na GPU, como o F16 do Qwen3.5-4B (8,3 GB numa GPU de 8 GB),
rodam com parte das camadas na CPU. O número de camadas na GPU é fixado por arquivo em
`[models.gpu_layers]` do `config.toml` e vale tanto para o `llama-server` quanto para o
`llama-bench`. Para obter o valor, com os demais aplicativos fechados:

```powershell
llama-fit-params -m C:\LLMs\models\Qwen3.5-4B-F16.gguf -c 4096 -fa on -np 1 -b 2048 -ub 512
```

O benchmark aborta antes de carregar um arquivo sem essa entrada que não caiba na VRAM livre
e avisa quando, após a carga, sobram menos de 300 MB livres (no Windows o driver pode então
usar memória compartilhada, o que distorce os tempos).

## Carga de trabalho

20 prompts por categoria, sorteados com semente 42 (`prepare_prompts.py`):

| Categoria | Dataset | Métrica de qualidade |
|---|---|---|
| Conhecimentos gerais | MMLU (`cais/mmlu`, test), 20 matérias distintas | acurácia (letra extraída) |
| Matemática | GSM8K (`openai/gsm8k`, test) | acurácia (resposta numérica final) |
| Programação | HumanEval (`openai/openai_humaneval`) | pass@1 (executa os testes) |
| Correção de bugs | HumanEvalFix (`bigcode/humanevalpack`, python), problemas disjuntos dos do HumanEval | pass@1 |

O código gerado pelo modelo é executado localmente em subprocesso isolado (`python -I`), em
diretório temporário, com timeout de 10 s.

## Parâmetros fixos

Definidos em `config.toml` e gravados no `manifest.json` de cada run:
todas as camadas na GPU (`-ngl all`, `--fit off`; exceto as exceções de `[models.gpu_layers]`),
contexto 4096, flash attention ligado,
um slot, 8 threads, cache de prompt desligado, raciocínio (thinking) desligado,
decodificação greedy (`temperature 0`), semente 42, máximo de 1024 tokens gerados.

## Saídas (`results/<run_id>/`)

| Arquivo | Conteúdo |
|---|---|
| `manifest.json` | configuração, versão do llama.cpp, hardware, SHA-256 dos prompts |
| `raw.jsonl` | uma linha por quant × repetição × prompt: tokens, tempos de prefill/decode, tokens/s, latência, motivo de parada, temperatura da GPU, resposta |
| `models.jsonl` | por quant: tamanho, bpw, camadas na GPU, tempo de carga, VRAM (base/carregado/pico/livre), RAM pico, temperatura máx., clock e potência |
| `monitor_<quant>.csv` | série temporal de VRAM, utilização, temperatura, clock, potência e RAM (250 ms) |
| `llama_bench.jsonl` | medição sintética padronizada (pp512 / tg128) |
| `server_<quant>.log` | log do llama-server |
| `graded.jsonl` | correção de cada resposta |
| `summary/*.csv` | estatísticas descritivas (média, dp, mín., máx., mediana) e comparação com a linha de base |
| `summary/tabela_desempenho.tex` | VRAM, carga, prefill, decode, tg128 e tempo de bateria por nível |
| `summary/tabela_qualidade.tex` | acertos por categoria, Δ e McNemar vs. linha de base, truncadas e idênticas |
| `figures/*.png, *.pdf` | gráficos |

A vazão por nível (`perf_by_quant.csv`) é calculada por bateria como tokens totais ÷ tempo total
da fase, para que respostas muito curtas (MMLU) não distorçam a média de tokens/s.

Como a decodificação greedy é determinística, a qualidade é comparada prompt a prompt com a linha
de base usando a primeira repetição medida: intervalo de confiança de Wilson (95%) da acurácia,
teste exato de McNemar, novos erros/acertos e percentual de respostas idênticas.

O `compare.py` só aceita runs com os mesmos prompts e parâmetros de geração e grava em `--out`
`comparacao.csv`, `tabela_comparacao.tex` e gráficos de decode, variação da acurácia relativa ao
F16 de cada modelo, tamanho dos arquivos frente à VRAM e velocidade × qualidade.
