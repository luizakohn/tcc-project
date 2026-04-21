import random

from datasets import load_dataset

TARGET_TOTAL = 5_000
RANDOM_SEED = 42


def load_queries() -> list[str]:
    """Retorna apenas as queries de teste do Quati."""
    test_topics_ds = load_dataset(
        "unicamp-dl/quati", "quati_test_topics", split="quati_test_topics",
        trust_remote_code=True,
    )
    return [row["query"] for row in test_topics_ds]


def load_passages(target_total: int = TARGET_TOTAL, seed: int = RANDOM_SEED) -> list[str]:
    """Retorna os passages prontos para embed_and_store."""
    random.seed(seed)

    passages_ds = load_dataset(
        "unicamp-dl/quati", "quati_1M_passages", split="quati_1M_passages",
        trust_remote_code=True,
    )
    qrels_ds = load_dataset(
        "unicamp-dl/quati", "quati_1M_qrels", split="quati_1M_qrels",
        trust_remote_code=True,
    )

    id_para_passage = {row["passage_id"]: row["passage"] for row in passages_ds}

    passage_ids_nos_qrels = {row["passage_id"] for row in qrels_ds}

    chunks_qrels = [
        id_para_passage[pid]
        for pid in passage_ids_nos_qrels
        if pid in id_para_passage
    ]

    ids_extras = random.sample(
        [row["passage_id"] for row in passages_ds if row["passage_id"] not in passage_ids_nos_qrels],
        max(0, target_total - len(chunks_qrels)),
    )
    chunks_extras = [id_para_passage[pid] for pid in ids_extras]

    return chunks_qrels + chunks_extras


def load_quati(target_total: int = TARGET_TOTAL, seed: int = RANDOM_SEED):
    """Retorna (passages, queries).

    passages : list[str] — textos prontos para embed_and_store
    queries  : list[str] — queries de teste para run_experiment
    """
    return load_passages(target_total, seed), load_queries()

if __name__ == "__main__":
    qrels_ds = load_dataset(
        "unicamp-dl/quati", "quati_1M_qrels", split="quati_1M_qrels",
        trust_remote_code=True,
    )
    test_topics_ds = load_dataset(
        "unicamp-dl/quati", "quati_test_topics", split="quati_test_topics",
        trust_remote_code=True,
    )

    test_query_ids = {row["query_id"] for row in test_topics_ds}

    for row in qrels_ds:
        if row["query_id"] in test_query_ids:
            print(row)
