import random

from datasets import load_dataset

TARGET_TOTAL = 5_000
RANDOM_SEED = 42


def load_queries() -> list[tuple[str, str]]:
    """Retorna as queries de teste do Quati como lista de (query_id, query_text)."""
    test_topics_ds = load_dataset(
        "unicamp-dl/quati", "quati_test_topics", split="quati_test_topics",
        trust_remote_code=True,
    )
    return [(row["query_id"], row["query"]) for row in test_topics_ds]


def load_qrels() -> dict[str, set[str]]:
    """Retorna os qrels de teste: query_id → conjunto de passage_ids relevantes (score >= 1)."""
    qrels_ds = load_dataset(
        "unicamp-dl/quati", "quati_1M_qrels", split="quati_1M_qrels",
        trust_remote_code=True,
    )
    test_topics_ds = load_dataset(
        "unicamp-dl/quati", "quati_test_topics", split="quati_test_topics",
        trust_remote_code=True,
    )
    test_query_ids = {row["query_id"] for row in test_topics_ds}

    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        qid = row["query_id"]
        if qid not in test_query_ids:
            continue
        if int(row["score"]) < 1:
            continue
        qrels.setdefault(qid, set()).add(row["passage_id"])

    return qrels


def load_passages(target_total: int = TARGET_TOTAL, seed: int = RANDOM_SEED) -> list[tuple[str, str]]:
    """Retorna os passages prontos para embed_and_store.

    Returns:
        list de tuplas (passage_id, text), onde passage_id é o identificador
        original do Quati — necessário para comparação posterior com os qrels.
    """
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
        (pid, id_para_passage[pid])
        for pid in passage_ids_nos_qrels
        if pid in id_para_passage
    ]

    ids_extras = random.sample(
        [row["passage_id"] for row in passages_ds if row["passage_id"] not in passage_ids_nos_qrels],
        max(0, target_total - len(chunks_qrels)),
    )
    chunks_extras = [(pid, id_para_passage[pid]) for pid in ids_extras]

    return chunks_qrels + chunks_extras


def load_quati(target_total: int = TARGET_TOTAL, seed: int = RANDOM_SEED):
    """Retorna (passages, queries).

    passages : list[tuple[str, str]] — (passage_id, text) prontos para embed_and_store
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
