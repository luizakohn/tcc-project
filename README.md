# RAG Semantic Retrieval Benchmark

Experimento científico para comparar métricas de similaridade semântica na etapa de retrieve de sistemas RAG.

## Stack

- Python 3.12+
- PostgreSQL + pgvector
- OpenAI Embedding API (`text-embedding-3-small`)

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar PostgreSQL com pgvector

```sql
CREATE DATABASE rag_benchmark;
\c rag_benchmark
CREATE EXTENSION vector;
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Editar .env com suas credenciais
```

### 4. Executar o experimento

Edite `experiment_runner.py` com seus chunks e perguntas, depois:

```bash
python experiment_runner.py
```

## Arquitetura

```
ingestion/
  embedder.py    — Wrapper da OpenAI Embedding API
  store.py       — Criação de tabelas e inserção no Postgres

retrieval/
  query_engine.py — Executa as 4 formas de consulta (cosine, euclidean, dot product, sequential scan)
  cache.py        — Limpeza de cache entre consultas

evaluation/
  metrics.py     — Kendall's Tau e Overlap@K
  exporter.py    — Geração dos CSVs de resultado
```

## Métricas de consulta

| Métrica | Operador pgvector | Descrição |
|---------|-------------------|-----------|
| Cosine similarity | `<=>` | Similaridade angular |
| Euclidean distance | `<->` | Distância L2 |
| Dot product | `<#>` | Produto interno negativo |
| Sequential scan | `<=>` + `SET enable_indexscan = off` | Ground truth sem índice |

## Métricas de avaliação

- **Kendall's Tau**: Correlação de ranking entre resultado aproximado e ground truth
- **Overlap@K**: Fração de elementos em comum nos top-K resultados

## Saídas

Os resultados são exportados em `results/`:

- `timings_{dims}d.csv` — Tempos de execução por pergunta e métrica
- `results_{dims}d.csv` — IDs retornados, Kendall's Tau e Overlap@K
- `summary.csv` — Médias agregadas por dimensão e métrica

## Bases

| Base | Dimensões | Tabela |
|------|-----------|--------|
| A | 768 | `base_768d` |
| B | 1024 | `base_1024d` |
| C | 1536 | `base_1536d` |
