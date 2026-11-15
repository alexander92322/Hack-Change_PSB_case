"""Сквозной тест постобработки без API: модель-заглушка возвращает эталонный ответ организаторов
(или эталон с подменённым числом/ссылкой), смотрим, что дойдёт до выхода.

1. Эталоны должны пройти без изменений - иначе проверки портят правильные ответы.
2. Испорченные эталоны должны быть пойманы - иначе выдуманный факт уйдёт судье (оценка 0).

    python scripts/check_answers.py        # нужны data/questions.csv и data/demo_output.csv
"""
import csv, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."))
sys.path.insert(0, root); os.chdir(root)
import agent, llm
qs = {r["id"]: r["question"] for r in csv.DictReader(open("data/questions.csv", encoding="utf-8-sig"))}
ref = {r["id"]: r["answer"] for r in csv.DictReader(open("data/demo_output.csv", encoding="utf-8-sig"))}
STATUS = {"3": "refused_no_data", "13": "partial", "17": "partial"}
a = agent.LegalAgent()
cur = {}
llm.chat = lambda messages, max_tokens=None: json.dumps({"answer": cur["a"], "status": cur["s"]}, ensure_ascii=False)
print("== 1. Эталоны должны пройти без изменений")
same = 0
for i, q in qs.items():
    cur["a"], cur["s"] = ref[i], STATUS.get(i, "answered")
    question, history = agent.parse_history(q)
    r = a.answer(question, history)
    ok = r["answer"] == ref[i]
    same += ok
    if not ok:
        print(f"  Q{i} ИЗМЕНЁН [{r['status']}]: {r['answer'][:160]}")
print(f"  без изменений: {same}/{len(qs)}")
print("== 2. Испорченные эталоны должны быть пойманы")
CORRUPT = {
    "1": ("450 000", "480 000"), "2": ("шести месяцев", "9 месяцев"), "5": ("10 000 000", "25 000 000"),
    "6": ("5 рабочих дней", "10 рабочих дней"), "9": ("45 календарных", "90 календарных"),
    "11": ("трёх рабочих дней", "14 рабочих дней"), "7": ("п. 4.3", "п. 4.7"),
    "12": ("ст. 70", "ст. 72"), "16": ("п. 2.1", "п. 9.9"), "18": ("трёх рабочих дней", "5 рабочих дней"),
}
caught = 0
for i, (old, new) in CORRUPT.items():
    bad = ref[i].replace(old, new)
    assert bad != ref[i], i
    cur["a"], cur["s"] = bad, "answered"
    question, history = agent.parse_history(qs[i])
    r = a.answer(question, history)
    got = new.split()[0] not in r["answer"] or new not in r["answer"]
    caught += got
    print(f"  Q{i} {old!r}->{new!r}: {'ПОЙМАН' if got else 'ПРОПУЩЕН'} [{r['status']}] {r['answer'][:110]}")
print(f"  поймано: {caught}/{len(CORRUPT)}")
os._exit(0)
