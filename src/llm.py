"""
LLM client & ticket classifier for Vireo Audio (optional AI layer).

Reads credentials from .env (see .env.example) and talks to any OpenAI-compatible
endpoint. Tickets are classified in batches and cached by ticket_id in
cache/llm_labels.jsonl, so each ticket is paid for once; re-runs are free.
Without a configured key the rest of the pipeline keeps working on keyword rules.

Smoke test:
    python -m src.llm            # ping + classify 3 real tickets
"""

import os
import sys
import json
import re

from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT_DIR, "cache", "llm_labels.jsonl")
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# v1 (discarded): 14 broad themes + resolved yes/no. Too coarse: "Return or replacement
# request" swallowed pickup failures and refund chasers, GST invoices fell into "Other",
# and "resolved" could not tell a refund from a real fix.
# v2 -> v3: 40-ticket review found refund chasers after a pickup labelled "Double charge",
# missed return pickups labelled "Delivery", and no home for wrong-variant deliveries.
# See docs/prompt_log.md.
PROMPT_VERSION = "v3"

# theme -> what belongs in it (shown to the model; also the digest's vocabulary)
THEME_GUIDE = {
    "Delivery delayed / not delivered": "order not arrived, tracking stuck, courier says delivered but not received",
    "Address change / wrong address": "moved house, wrong pincode, redirect a shipment",
    "Damaged in transit / dead on arrival": "arrived cracked/broken, dead out of the box, missing parts",
    "Wrong item / variant received": "different product, wrong colour or model delivered",
    "Return pickup not done": "courier never came to COLLECT a return/replacement (customer waiting at home for pickup)",
    "Refund delayed / not received": "money not back after a return pickup, cancellation or approved refund",
    "Double charge / payment failed": "charged twice for one order, money deducted but no order created, gateway failure (NOT waiting for a refund)",
    "Coupon / discount / price": "promo code not working, discount not applied, price drop",
    "Invoice / GST": "invoice download, GST bill, GSTIN correction",
    "Order cancellation": "wants to cancel before dispatch, cancel button not working",
    "Pairing & connection drops": "cannot pair, not discoverable, disconnects, audio stutters from signal",
    "Battery drain / not charging": "short battery life, case or device not charging, charger/cable faults",
    "Audio fault (one side, distortion, mic)": "one earbud silent, crackling, buzzing, low mic",
    "App / firmware update failure": "app crashes or won't open, firmware update stuck, device bricked after update",
    "Screen / touch / hardware fault": "watch screen unresponsive, buttons, physical fault not covered above",
    "Warranty claim / repair status": "chasing an RMA, repair or warranty replacement already raised",
    "Account / OTP login": "cannot log in, OTP not received, account locked",
    "Product / compatibility question": "pre-sales or how-to question, compatibility with a phone/TV",
    "Other": "none of the above",
}
THEMES = list(THEME_GUIDE)
OUTCOMES = ["fixed", "refund", "replacement_or_reship", "escalated", "pending_or_deferred", "unclear"]

SYSTEM_PROMPT = f"""You label customer-support tickets for Vireo Audio, an Indian consumer-audio brand
(earbuds, headphones, speakers, smartwatches, chargers).
Each ticket has the customer's opening message and the agent's closing note. Messages can have
typos, Hinglish or be IVR transcripts; notes use shorthand (cx=customer, rslvd=resolved,
rplc=replacement, pkp=pickup, crr=courier, fw=firmware, pg=payment gateway).

Themes (pick exactly one, for the customer's MAIN problem; use the note if the message is vague):
{json.dumps(THEME_GUIDE, indent=1)}

Return ONLY a JSON array, one object per ticket, same order, no prose:
{{"ticket_id": str,
  "theme": one of the theme names above, spelled exactly,
  "is_repeat_claim": true if the customer OR the note says this was raised before
                     ("already told", "third time", "was told it was resolved", "repeat contact"),
  "outcome": one of {json.dumps(OUTCOMES)} - what the note says happened;
             "unclear" if the note is empty or says nothing specific ("done", "see prev", "cx ok")}}"""


def is_configured():
    return bool(os.getenv("LLM_API_KEY"))


def get_client():
    from openai import OpenAI
    if not is_configured():
        raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL") or None)


def get_model():
    return os.getenv("LLM_MODEL", "qwen/qwen3.8-max:free")  # same as .env.example


def chat(messages, client=None, **kwargs):
    """Single chat completion. Returns (text, usage)."""
    client = client or get_client()
    res = client.chat.completions.create(model=get_model(), messages=messages, **kwargs)
    return res.choices[0].message.content, res.usage


def _parse_json_array(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])


def load_cache(path=CACHE_PATH, prompt_version=PROMPT_VERSION):
    """Labels from earlier runs. Rows from other prompt versions are ignored, so a new prompt re-labels."""
    labels = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("prompt_version") == prompt_version:
                    labels[row["ticket_id"]] = row
    return labels


def _label_batch(batch, client):
    """One API call for one batch. Returns (valid_rows, usage) and raises on transport/parse errors."""
    payload = [
        {"ticket_id": r.ticket_id,
         "customer_message": str(r.customer_message or "")[:1200],
         "agent_notes": str(r.agent_notes or "")[:600]}
        for r in batch.itertuples()
    ]
    text, usage = chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        client=client, temperature=0,
    )
    pending_ids = set(batch["ticket_id"])
    valid = []
    for row in _parse_json_array(text):
        if row.get("ticket_id") in pending_ids and row.get("theme") in THEMES:
            pending_ids.discard(row["ticket_id"])  # also ignores duplicate rows
            valid.append({
                "ticket_id": row["ticket_id"],
                "theme": row["theme"],
                "is_repeat_claim": bool(row.get("is_repeat_claim")),
                "outcome": row.get("outcome") if row.get("outcome") in OUTCOMES else "unclear",
                "model": get_model(),
                "prompt_version": PROMPT_VERSION,
            })
    return valid, usage


def classify_tickets(df, batch_size=20, max_batches=None, cache_path=CACHE_PATH, client=None,
                     max_workers=1, progress=False):
    """
    Labels tickets in df (needs ticket_id, customer_message, agent_notes).
    Only tickets missing from the cache are sent; failures are left for the next run.
    Returns (labels_dict, stats).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time

    cache = load_cache(cache_path)
    todo = df[~df["ticket_id"].isin(cache.keys())]
    stats = {"cached": len(df) - len(todo), "sent": 0, "failed": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "seconds": 0.0}
    if len(todo) == 0:
        return cache, stats

    client = client or get_client()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]

    lock = threading.Lock()
    started = time.time()
    with open(cache_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_label_batch, b, client): b for b in batches}
        for done, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            with lock:
                stats["calls"] += 1
                try:
                    rows, usage = fut.result()
                except Exception as e:
                    print(f"  batch failed ({type(e).__name__}: {str(e)[:150]})", file=sys.stderr)
                    stats["failed"] += len(batch)
                    continue
                for row in rows:
                    cache[row["ticket_id"]] = row
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats["sent"] += len(rows)
                stats["failed"] += len(batch) - len(rows)
                if usage:
                    stats["prompt_tokens"] += usage.prompt_tokens or 0
                    stats["completion_tokens"] += usage.completion_tokens or 0
                if progress and (done % 10 == 0 or done == len(batches)):
                    print(f"  {done}/{len(batches)} batches | labelled {stats['sent']:,} | "
                          f"failed {stats['failed']:,} | {time.time() - started:.0f}s", flush=True)
    stats["seconds"] = round(time.time() - started, 1)
    return cache, stats


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.path.insert(0, ROOT_DIR)
    import argparse
    parser = argparse.ArgumentParser(description="LLM smoke test / bulk ticket labelling")
    parser.add_argument("--all", action="store_true", help="Label every uncached ticket (resumable)")
    parser.add_argument("--weeks", type=int, default=None, help="Only label the latest N ISO weeks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(f"Endpoint: {os.getenv('LLM_BASE_URL')} | model: {get_model()} | key set: {is_configured()} "
          f"| prompt {PROMPT_VERSION}")

    if args.all or args.weeks:
        from src.clean import clean_tickets_data
        df = clean_tickets_data()
        if args.weeks:
            df = df[df["year_week"].isin(sorted(df["year_week"].unique())[-args.weeks:])]
        print(f"Labelling {len(df):,} tickets (cached ones are skipped)...")
        labels, stats = classify_tickets(df, batch_size=args.batch_size, max_workers=args.workers, progress=True)
        n = stats["sent"] or 1
        print(f"Done: {stats}")
        print(f"Per ticket: {stats['prompt_tokens'] / n:.0f} prompt + {stats['completion_tokens'] / n:.0f} "
              f"completion tokens | {stats['seconds'] / n:.2f}s wall-clock")
        sys.exit(0)

    print("\n1) Ping")
    reply, usage = chat([{"role": "user", "content": "Reply with exactly: pong"}], temperature=0)
    print(f"   reply={reply!r} usage={usage}")

    print("\n2) Classify 3 real tickets")
    from src.clean import clean_tickets_data
    sample = clean_tickets_data().sample(3, random_state=7)
    labels, stats = classify_tickets(sample, batch_size=3)
    for r in sample.itertuples():
        lab = labels.get(r.ticket_id, {})
        print(f"   {r.ticket_id} | bot tag: {r.category:<20} | LLM: {lab.get('theme')} "
              f"| repeat_claim={lab.get('is_repeat_claim')} outcome={lab.get('outcome')}")
        print(f"      msg:  {str(r.customer_message)[:100]!r}")
        print(f"      note: {str(r.agent_notes)[:100]!r}")
    print(f"   stats: {stats}")
