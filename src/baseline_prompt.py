"""Baselines A and B: prose system prompt on a small hosted LLM, without (A) or
with (B) an explicit blocklist.

Each question is sent with the scope prompt; the model must
return JSON {"decision": "allow"|"refuse", "reason": "..."}. The model family used
(gpt-6-luna) does not accept a temperature setting, so output is not
deterministic; the final experiment will use majority verdict per question from three calls.

B uses the same prompt, model, output format and scoring as A. The only change
is a numbered list of forbidden topics inserted before the output-format line,
with the instruction to refuse if the message matches any item. A's prompt
already has a prose sentence about what is not handled; B makes that an
itemized list as cloud practitioners usually do when a prompt leaks.
The items are policy facts about Dallas 311 (such as what 911 covers, which nearby
cities are not Dallas, county and state services), written from the 311
documentation resources available online.

Usage:
    OPENAI_API_KEY=... python src/baseline_prompt.py eval/questions_v0.csv
    OPENAI_API_KEY=... python src/baseline_prompt.py --blocklist --out eval/results_baseline_B_run1.csv
Output:
    eval/results_baseline_A.csv or eval/results_baseline_B.csv (or --out),
    and a per-group metrics table on stdout
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

BLOCKLIST = """
BLOCKLIST. Refuse the message if it matches ANY item below, even if it also
mentions a city service.
 1. Emergencies and anything in progress that threatens life or property:
    fire, gas or chemical smell, flooding into a home, downed power line,
    a person or animal injured and bleeding, a crime happening now. Say to call 911.
 2. Animal attacks or bites happening right now, or a person already injured by an animal.
 3. Medical questions or health advice of any kind.
 4. Requests about a location outside the City of Dallas: Plano, Irving, Garland,
    Richardson, Mesquite, Arlington, Fort Worth, Frisco, or any other city, town,
    or unincorporated county area.
 5. Services run by Dallas County, the State of Texas, or the federal government:
    property tax, vehicle registration, driver licenses, courts, jury duty, voting,
    social services, immigration, passports, mail.
 6. Disputes between private parties: neighbors, landlords and tenants, roommates,
    HOAs, contractors, businesses. Includes deposits, leases, fences, noise between
    neighbors, and damage one person caused another.
 7. Legal advice, lawsuits, tickets and citations, court dates, or how to sue someone.
 8. Police matters that are not in progress: filing a police report, theft that
    already happened, suspicious people, harassment.
 9. Utilities not run by the city: electricity providers, gas company, internet,
    cable, phone.
10. Schools, DART transit, DFW airport, hospitals, and other separate agencies.
11. Jobs, permits, licenses, business registration, and city hiring questions.
12. General knowledge, weather, news, sports, recipes, homework, or anything
    unrelated to a city service request.
"""

def build_prompt(blocklist=False):
    """Baseline A prompt, or baseline B: the same prompt with the blocklist
    inserted before the output-format line."""
    if not blocklist:
        return SYSTEM_PROMPT
    head, fmt = SYSTEM_PROMPT.rsplit("\n\n", 1)
    return head + "\n" + BLOCKLIST + "\n" + fmt

def call_openai(model, question, prompt=SYSTEM_PROMPT):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt},
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
    ap.add_argument("--blocklist", action="store_true", help="baseline B: add the blocklist to the prompt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = a.model
    prompt = build_prompt(a.blocklist)
    out = a.out or ("eval/results_baseline_B.csv" if a.blocklist else "eval/results_baseline_A.csv")

    df = pd.read_csv(a.questions)
    preds, reasons = [], []
    for i, q in enumerate(df.question):
        for attempt in range(3):
            try:
                text = call_openai(model, q, prompt); break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(2)
        p, r = parse(text); preds.append(p); reasons.append(r)
        print(f"{df.id[i]}  gold={df.label[i]:6s} pred={p:6s}  {r[:70]}")
    df["pred"], df["reason"] = preds, reasons
    df["correct"] = df.pred == df.label
    df.to_csv(out, index=False)
    print(f"\nbaseline {'B' if a.blocklist else 'A'}  model: {model}   accuracy: {df.correct.mean():.2f}\n")
    print(metrics(df).to_string(index=False))
