"""
Tests for the hybrid classifier (src/cascade.py + themes.attach_themes routing).
Uses a small synthetic ticket set and a fake LLM client: no network, no real cache writes.
"""

import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cascade, llm as llm_mod
from src.themes import attach_themes

TEMPLATES = {
    "Battery drain / not charging": ["battery drains fast, lasts {n} hours", "case not charging at all {n}",
                                     "battery dies by lunch day {n}"],
    "Refund delayed / not received": ["refund not received after return {n} days", "money not back after pickup {n}",
                                      "still waiting for my refund {n}"],
    "Invoice / GST": ["need gst invoice for order {n}", "invoice not downloading {n}", "tax bill for accounts {n}"],
}


def _synthetic(n_per_template=8):
    rows, labels = [], {}
    for theme, templates in TEMPLATES.items():
        for t_i, tpl in enumerate(templates):
            for i in range(n_per_template):
                tid = f"{theme[:3]}-{t_i}-{i}"
                rows.append({"ticket_id": tid, "customer_message": tpl.format(n=i), "agent_notes": "resolved"})
                labels[tid] = {"theme": theme, "is_repeat_claim": False, "outcome": "fixed"}
    return pd.DataFrame(rows), labels


@pytest.fixture(scope="module")
def trained():
    df, labels = _synthetic()
    return cascade.train_model(df, labels, exclude_ids=set()), labels


def _new(messages):
    return pd.DataFrame({"ticket_id": [f"NEW-{i}" for i in range(len(messages))],
                         "customer_message": messages, "agent_notes": [""] * len(messages)})


def test_clear_ticket_is_confident_and_correct(trained):
    model, _ = trained
    pred = cascade.predict(model, _new(["my battery drains fast, lasts 2 hours"]))
    assert pred["tfidf_theme"].iloc[0] == "Battery drain / not charging"
    assert pred["route_reason"].iloc[0] == "confident"


def test_unseen_kind_of_ticket_is_flagged_novel(trained):
    model, _ = trained
    pred = cascade.predict(model, _new(["zqxv wklm pqrs tuvw"]))
    assert pred["is_novel"].iloc[0] and pred["route_reason"].iloc[0] == "novel"


def test_check_set_tickets_are_never_trained_on():
    df, labels = _synthetic()
    held_out = set(df["ticket_id"].iloc[:5])
    model = cascade.train_model(df, labels, exclude_ids=held_out)
    assert model["n_train"] == len(df) - 5


def test_training_needs_two_themes():
    df, labels = _synthetic()
    one_theme = {t: v for t, v in labels.items() if v["theme"] == "Invoice / GST"}
    with pytest.raises(ValueError):
        cascade.train_model(df, one_theme, exclude_ids=set())


def test_routing_without_key_falls_back_to_keywords(trained, monkeypatch):
    model, _ = trained
    monkeypatch.setattr(llm_mod, "is_configured", lambda: False)
    # 2nd ticket: a theme the model never trained on -> novel. (A never-seen theme that SHARES words
    # with a known one, e.g. "courier never came for the pickup" vs trained "money not back after
    # pickup", can instead be confidently wrong; that limitation is documented in docs/prompt_log.md.)
    out = attach_themes(_new(["battery drains fast, lasts 3 hours", "otp not coming on my phone"]),
                        labels={}, model=model, call_llm=True)
    assert out["theme_source"].tolist()[0] == "tfidf"
    assert out["theme_source"].tolist()[1] == "keyword", "novel + no key -> keyword rules"
    assert out["theme"].tolist()[1] == "Account / OTP login"


def test_routing_sends_only_hard_tickets_to_llm(trained, monkeypatch, tmp_path):
    model, _ = trained
    sent = []

    def create(model, messages, **kwargs):
        tickets = json.loads(messages[-1]["content"])
        sent.extend(t["ticket_id"] for t in tickets)
        rows = [{"ticket_id": t["ticket_id"], "theme": "Other", "is_repeat_claim": True, "outcome": "unclear"}
                for t in tickets]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(rows)))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_mod, "is_configured", lambda: True)
    out = attach_themes(_new(["need gst invoice for order 7", "zqxv wklm pqrs tuvw"]),
                        labels={}, model=model, call_llm=True, client=client,
                        llm_cache_path=str(tmp_path / "labels.jsonl"))
    assert sent == ["NEW-1"], "only the novel ticket may reach the LLM"
    assert out["theme_source"].tolist() == ["tfidf", "llm"]
    assert bool(out["is_repeat_claim"].iloc[1]) is True


def test_cached_llm_label_wins_over_model(trained):
    model, _ = trained
    labels = {"NEW-0": {"theme": "Invoice / GST", "is_repeat_claim": False, "outcome": "fixed"}}
    out = attach_themes(_new(["battery drains fast, lasts 3 hours"]), labels=labels, model=model)
    assert out["theme_source"].iloc[0] == "llm" and out["theme"].iloc[0] == "Invoice / GST"
