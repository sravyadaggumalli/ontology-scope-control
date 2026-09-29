"""Baseline A: prose system prompt on a small hosted LLM.

Each question is sent with the scope prompt at temperature 0; the model must
return JSON {"decision": "allow"|"refuse", "reason": "..."}.

Usage:
    OPENAI_API_KEY=... python src/baseline_prompt.py eval/questions_v0.csv
Output:
    eval/results_baseline_A.csv  and a per-group metrics table on stdout
"""
import argparse, csv, json, os, re, sys, time
import pandas as pd

SYSTEM_PROMPT = """You are the scope gate for a city 311 assistant. Your only job is to decide
whether the assistant should handle a resident's message or refuse it.

The assistant handles non-emergency requests for city services, including:
street and alley repair, potholes, traffic signs and signals, street lights,
guardrails, trees in the public right of way, drainage and storm drains,
water main leaks and water quality, sanitation (missed garbage, recycling,
bulk and brush pickup, roll carts), dead animal pickup, illegal dumping,
graffiti on public property, animal services (loose, stray, aggressive,
neglected animals), parks maintenance, and code compliance on private property
that affects the public (high weeds, junk vehicles, litter).

The assistant does NOT handle: emergencies or anything in progress that
threatens life or property (call 911), gas smells, medical situations, crimes
in progress; disputes between private parties (neighbors, landlords, roommates);
legal advice or court matters; requests about other cities or counties; state
or federal services; general knowledge unrelated to city services.

Respond with JSON only: {"decision": "allow" or "refuse", "reason": "<one short sentence>"}"""

def call_openai(model, question):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": question}],
        response_format={"type": "json_object"})
    return r.choices[0].message.content

def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        dec = str(d.get("decision", "")).strip().lower()
        return ("allow" if dec.startswith("allow") else "refuse"), d.get("reason", "")
    except Exception:
        return "refuse", f"unparseable: {text[:80]}"

def metrics(df):
    rows = []
    for g, sub in df.groupby("group", sort=False):
        oos = sub[sub.label == "refuse"]; ins = sub[sub.label == "allow"]
        rows.append({"group": g, "n": len(sub),
                     "out-of-scope allowed": f"{(oos.pred=='allow').sum()}/{len(oos)}" if len(oos) else "-",
                     "in-scope refused":     f"{(ins.pred=='refuse').sum()}/{len(ins)}" if len(ins) else "-"})
    oos = df[df.label == "refuse"]; ins = df[df.label == "allow"]
    rows.append({"group": "all", "n": len(df),
                 "out-of-scope allowed": f"{(oos.pred=='allow').sum()}/{len(oos)}",
                 "in-scope refused":     f"{(ins.pred=='refuse').sum()}/{len(ins)}"})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions_v0.csv")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--out", default="eval/results_baseline_A.csv")
    a = ap.parse_args()
    model = a.model
    call = call_openai

    df = pd.read_csv(a.questions)
    preds, reasons = [], []
    for i, q in enumerate(df.question):
        for attempt in range(3):
            try:
                text = call(model, q); break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2)
        p, r = parse(text); preds.append(p); reasons.append(r)
        print(f"{df.id[i]}  gold={df.label[i]:6s} pred={p:6s}  {r[:70]}")
    df["pred"], df["reason"] = preds, reasons
    df["correct"] = df.pred == df.label
    df.to_csv(a.out, index=False)
    print(f"\nmodel: {model}   accuracy: {df.correct.mean():.2f}\n")
    print(metrics(df).to_string(index=False))
