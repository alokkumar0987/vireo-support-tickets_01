"""
Complaint themes for every ticket: cached LLM label > confident TF-IDF (src/cascade.py) >
live LLM (optional) > keyword rules. See attach_themes().

attach_themes(df) adds:
  theme              one of src.llm.THEMES
  theme_source       'llm', 'tfidf' or 'keyword'
  route_reason       TF-IDF routing: confident / low_confidence / novel / predicted_other
  tfidf_confidence   TF-IDF top-class probability
  keyword_theme      the offline rule-based theme (always present; used as fallback and baseline)
  is_repeat_claim    customer or note says this was raised before
  note_outcome       what the agent note says happened (src.llm.OUTCOMES)

The keyword rules replace digest.COMPLAINT_TAXONOMY's first-substring-wins matching, which
sent "charged twice" to Battery (via "charge") and "repair" to Bluetooth (via "pair").
Rules use word boundaries and run from most to least specific.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import THEMES, OUTCOMES, load_cache
from src.clean import add_repeat_contact_features

# Themes that are one customer problem seen at different stages: a refund chased after a
# cancellation or double charge is the same issue coming back, not a new one (policy §10).
ISSUE_FAMILY = {
    "Order cancellation": "Money back", "Double charge / payment failed": "Money back",
    "Coupon / discount / price": "Money back", "Refund delayed / not received": "Money back",
    "Return pickup not done": "Return & replacement", "Wrong item / variant received": "Return & replacement",
    "Damaged in transit / dead on arrival": "Return & replacement",
}
REPEAT_COLUMNS_NEEDED = {"customer_id", "product_sku", "created_at_dt", "resolved_at_dt", "status",
                         "contact_cost_inr", "category"}

# (theme, regex) in priority order: first match wins, so specific money/logistics
# themes sit above the generic hardware words that also appear in them.
KEYWORD_RULES = [
    ("Invoice / GST", r"\binvoice|\bgst|tax bill"),
    ("Wrong item / variant received", r"wrong (colou?r|variant|model|item|product|size)|different (thing|product|item|model)|ordered \w+, got \w+"),
    ("Return pickup not done", r"pick ?up (has not|hasn'?t|not|never|missed|did not|didn'?t)|nobody came|no one came|waiting for (your|the) courier|\bpkp missed|pickup not done"),
    ("Refund delayed / not received", r"refund (not|pending|delay|still|was promised|hasn'?t)|money (hasn'?t|has not|not|is not|isn'?t) (come|back|credited)|amount is nowhere|waiting for (my|the) refund|not (been )?credited"),
    # "two entries"/"went to you"/"no orders" found by embedding clustering (analysis/cluster_themes.py, cluster 30)
    ("Double charge / payment failed", r"charged (twice|two times|double)|double (charge|payment|debit)|deducted|payment (failed|went through|debited)|no order (id|was created|created)|two entries|went to you|(site|app) says i have no orders|\bupi\b"),
    ("Coupon / discount / price", r"coupon|promo|discount|\d+ ?% off|full price|price (drop|match|adjust)"),
    ("Order cancellation", r"\bcancel"),
    ("Address change / wrong address", r"\baddress|pin ?code|moved (house|houses|flat)|old flat"),
    ("Damaged in transit / dead on arrival", r"crack|broken|damaged|crushed|dead on arrival|\bdoa\b|missing (part|piece|cable)|before i (even )?switched"),
    ("App / firmware update failure", r"firmware|\bfw\b|update (failed|stuck|prompt)|after (the )?update|\bapp (not|crash|won'?t|keeps|is not)|loading screen|spinning circle|won'?t turn on"),
    ("Warranty claim / repair status", r"\brma\d*|\brepair|warranty|service cent|sent the unit|claim number"),
    ("Product / compatibility question", r"compatib|can i connect|before i buy|will (this|it) (work|talk|run)|does (my|the|this|it) .{0,30}work with|work with (my )?iphone"),
    ("Pairing & connection drops", r"\bpair|bluetooth|disconnect|discoverable|stutter|device list|keeps losing|connects for a second"),
    ("Battery drain / not charging", r"batter|\bcharg|drain|backup|\bdies\b|full to empty|charge it twice"),
    ("Audio fault (one side, distortion, mic)", r"\bsound|audio|one (ear|side)|(left|right) (side|one|bud|earbud|ear)|buzz|crackl|distort|\bmic\b|volume|hiss|static"),
    ("Screen / touch / hardware fault", r"screen|touch|display|button|unresponsive"),
    ("Account / OTP login", r"\botp\b|log ?in|locked out|password|my account"),
    ("Delivery delayed / not delivered", r"deliver|tracking|courier|shipment|dispatch|parcel|not (yet )?received|haven'?t received|not arrived"),
]
_COMPILED_RULES = [(theme, re.compile(pattern)) for theme, pattern in KEYWORD_RULES]

REPEAT_CLAIM_PATTERN = re.compile(
    r"already|again|third time|second time|was told|last time|previous|following up|follow up on|"
    r"raised this|supposedly sorted|emailed twice|called (before|twice)|repeat contact|fix did not hold"
)

LAZY_NOTE_PATTERN = re.compile(r"^\W*(done|sorted|closed|cx ok|ok|as discuss?e?d|see prev|resolved on call|-)?\W*$")
OUTCOME_RULES = [
    ("replacement_or_reship", r"replac|rplc|re-?ship|new unit"),
    ("refund", r"refund|rfnd|store credit"),
    ("escalated", r"escalat|transferr|\bxfer|\btransfer"),
    ("pending_or_deferred", r"pending|reschedul|\beta\b|follow(ed)?[- ]up w|awaiting|will (call|update|check)"),
    ("fixed", r"resolv|rslvd|fixed|successful|answered|updated|unlocked|satisfied|confirmed ok|regenerated|shared|emailed|paired ok"),
]
_COMPILED_OUTCOMES = [(o, re.compile(p)) for o, p in OUTCOME_RULES]

assert all(theme in THEMES for theme, _ in KEYWORD_RULES)
assert all(o in OUTCOMES for o, _ in OUTCOME_RULES)


def keyword_theme(message, note=""):
    """Rule-based theme from the customer message; the agent note breaks ties when the message is vague."""
    for text in (str(message or "").lower(), str(note or "").lower()):
        for theme, rx in _COMPILED_RULES:
            if rx.search(text):
                return theme
    return "Other"


def keyword_repeat_claim(message, note=""):
    return bool(REPEAT_CLAIM_PATTERN.search(f"{message or ''} {note or ''}".lower()))


def keyword_outcome(note):
    text = re.sub(r"~\w+|\(sop [\d.]+\)|\[closed\]", "", str(note or "").lower()).strip()
    if LAZY_NOTE_PATTERN.match(text):
        return "unclear"
    for outcome, rx in _COMPILED_OUTCOMES:
        if rx.search(text):
            return outcome
    return "unclear"


def attach_themes(df, use_llm=True, labels=None, use_model=True, call_llm=False, model=None, client=None,
                  llm_cache_path=None):
    """
    Returns a copy of df with theme columns. Per ticket, the first available source wins:
      1. 'llm'      an LLM label already in cache/llm_labels.jsonl (paid for once, reused free)
      2. 'tfidf'    src.cascade model, only when confident, familiar and not "Other"
      3. 'llm'      live LLM call for the remaining tickets, only if call_llm=True and a key is set
      4. 'keyword'  offline rules (always works)
    use_llm=False ignores cached LLM labels; use_model=False skips the TF-IDF layer.
    """
    from src import cascade, llm as llm_mod

    df = df.copy()
    msgs, notes = df["customer_message"], df["agent_notes"]
    df["keyword_theme"] = [keyword_theme(m, n) for m, n in zip(msgs, notes)]
    kw_repeat = pd.Series([keyword_repeat_claim(m, n) for m, n in zip(msgs, notes)], index=df.index)
    kw_outcome = pd.Series([keyword_outcome(n) for n in notes], index=df.index)

    if labels is None:
        labels = load_cache() if use_llm else {}
    df["theme"], df["theme_source"] = df["keyword_theme"], "keyword"
    df["is_repeat_claim"], df["note_outcome"] = kw_repeat, kw_outcome
    df["route_reason"], df["tfidf_confidence"] = "", np.nan

    # 2. TF-IDF layer (needs enough cached LLM labels to train on)
    if use_model:
        try:
            model = model or cascade.get_model(df, labels=load_cache())
            pred = cascade.predict(model, df)
            df["route_reason"], df["tfidf_confidence"] = pred["route_reason"], pred["tfidf_confidence"]
            confident = pred["route_reason"] == "confident"
            df.loc[confident, "theme"] = pred.loc[confident, "tfidf_theme"]
            df.loc[confident, "theme_source"] = "tfidf"
        except (ValueError, FileNotFoundError) as e:
            print(f"  TF-IDF layer skipped ({e}); using keyword rules", file=sys.stderr)

    # 3. live LLM for what the model could not decide
    if call_llm and llm_mod.is_configured():
        hard = df[(df["theme_source"] == "keyword") & ~df["ticket_id"].isin(labels)]
        if len(hard):
            new_labels, stats = llm_mod.classify_tickets(hard, client=client, max_workers=4,
                                                          cache_path=llm_cache_path or llm_mod.CACHE_PATH)
            print(f"  LLM labelled {stats['sent']:,} hard tickets ({stats['failed']:,} failed -> keyword)",
                  file=sys.stderr)
            labels = {**labels, **new_labels}

    # 1 & 3. LLM labels (cached or just fetched) override everything else
    llm_rows = df["ticket_id"].map(labels)
    has_llm = llm_rows.notna()
    if has_llm.any():
        df.loc[has_llm, "theme"] = [r["theme"] for r in llm_rows[has_llm]]
        df.loc[has_llm, "theme_source"] = "llm"
        df.loc[has_llm, "is_repeat_claim"] = [bool(r.get("is_repeat_claim")) for r in llm_rows[has_llm]]
        df.loc[has_llm, "note_outcome"] = [r.get("outcome", "unclear") for r in llm_rows[has_llm]]

    # Policy §10 "same issue": now that every ticket has a reliable theme, repeats are recomputed on
    # customer + product + issue family (was customer + product only, which also counted e.g. a
    # delivery ticket followed by a battery ticket).
    df["issue_family"] = df["theme"].map(lambda t: ISSUE_FAMILY.get(t, t))
    if REPEAT_COLUMNS_NEEDED.issubset(df.columns):
        df = add_repeat_contact_features(df, issue_col="issue_family")
    return df


def ensure_themes(df):
    """Themes + the policy §10 issue-level repeat columns; a no-op if attach_themes already ran.

    Every report must go through this: without it, repeat/FCR columns silently fall back to the
    product-level definition from clean.py (29.8% instead of 12.9%)."""
    return df if "issue_family" in df.columns else attach_themes(df)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from src.clean import clean_tickets_data
    d = attach_themes(clean_tickets_data())
    print(f"Theme source: {d['theme_source'].value_counts().to_dict()}")
    print(d["theme"].value_counts().to_string())
    both = d[d["theme_source"] == "llm"]
    if len(both):
        agree = (both["theme"] == both["keyword_theme"]).mean()
        print(f"\nKeyword rules agree with LLM on {agree:.1%} of {len(both):,} LLM-labelled tickets")
