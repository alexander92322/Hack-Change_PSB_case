"""Офлайн-проверка поиска (без API-ключа): находит ли retrieve() нужные пункты и не тащит ли лишние.

Без API эмбеддинги и реранкер недоступны - проверяется BM25 + точные правила + выбор документов.
Если нужный пункт не найден здесь, на платформе его спасает только векторный поиск.

    python scripts/check_retrieval.py                          # контрольный набор (32 вопроса)
    python scripts/check_retrieval.py . scripts/retrieval_tests/holdout_questions.csv scripts/retrieval_tests/holdout_expect.json
    python scripts/check_retrieval.py . data/questions.csv scripts/retrieval_tests/demo_expect.json

Ожидания: {"id": [["часть имени файла", "метка пункта или пусто"], ["!файл, которого быть не должно", ""]]}
"""
import json, os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "retrieval_tests")
root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."))
qpath = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(T, "control_questions.csv"))
epath = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.join(T, "control_expect.json"))
sys.path.insert(0, root); os.chdir(root)
import rag_kb as kb, agent
exp = json.load(open(epath, encoding="utf-8"))
qs = {r["id"]: r["question"] for r in csv.DictReader(open(qpath, encoding="utf-8-sig"))}
meta, vecs = kb.load_index()
if hasattr(agent, "sanitize_meta"): agent.sanitize_meta(meta)
ok = 0
for qid, want in exp.items():
    cur, hist = agent.parse_history(qs[qid])
    rq = "\n".join([f"Контекст {m['role']}: {m['content']}" for m in hist[-4:]] + [f"Текущий вопрос: {cur}"])
    if hasattr(agent, "_bare_alfa") and agent._bare_alfa(cur, hist):
        hits = agent.retrieve_both_alfa(rq, meta, vecs)
    else:
        hits = agent.retrieve(rq, meta, vecs)
    got = [(h["doc"]["file"], str(h["chunk"].get("label") or "")) for h in hits]
    bad = []
    for f, l in want:
        if f.startswith("!"):
            n = sum(f[1:] in g for g, _ in got)
            if n: bad.append(f"лишний {f[1:]} x{n}")
        elif not any(f in g and (not l or gl == l or gl.startswith(l + ",") or gl.endswith(", " + l)) for g, gl in got):
            bad.append(f"нет {f} {l}")
    ok += not bad
    print(f"[{'OK ' if not bad else 'BAD'}] {qid}: {'; '.join(bad)}" + ("" if not bad else f"   top: {[(g.split('__')[1][:22], gl) for g, gl in got[:6]]}"))
print(f"Итого {ok}/{len(exp)}")
