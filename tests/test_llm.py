"""
Offline tests for the LLM layer (src/llm.py). Uses a fake client, so no network or key needed.
"""

import os
import sys
import json
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import classify_tickets, load_cache, _parse_json_array


class FakeClient:
    """Mimics client.chat.completions.create and echoes a label per ticket."""
    def __init__(self, reply_fn):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._reply_fn = reply_fn

    def _create(self, model, messages, **kwargs):
        self.calls += 1
        tickets = json.loads(messages[-1]["content"])
        content = self._reply_fn(tickets)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def _tickets(n):
    return pd.DataFrame({
        "ticket_id": [f"TK-{i}" for i in range(n)],
        "customer_message": ["my earbuds will not charge"] * n,
        "agent_notes": ["advised reset, resolved"] * n,
    })


def _good_reply(tickets):
    rows = [{"ticket_id": t["ticket_id"], "theme": "Battery drain / not charging",
             "is_repeat_claim": False, "outcome": "fixed"} for t in tickets]
    return "```json\n" + json.dumps(rows) + "\n```"


def test_parse_json_array_strips_code_fences():
    assert _parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert _parse_json_array('Here you go: [{"a": 1}] thanks') == [{"a": 1}]


def test_classify_uses_cache_on_second_run(tmp_path):
    cache = str(tmp_path / "labels.jsonl")
    client = FakeClient(_good_reply)
    df = _tickets(5)

    labels, stats = classify_tickets(df, batch_size=2, cache_path=cache, client=client)
    assert stats["sent"] == 5 and stats["failed"] == 0
    assert client.calls == 3  # 2 + 2 + 1

    labels, stats = classify_tickets(df, batch_size=2, cache_path=cache, client=client)
    assert stats["cached"] == 5 and stats["sent"] == 0
    assert client.calls == 3, "Cached tickets must not be re-sent (re-runs are free)"


def test_classify_rejects_invented_themes_and_ids(tmp_path):
    def bad_reply(tickets):
        return json.dumps([
            {"ticket_id": tickets[0]["ticket_id"], "theme": "Made-up theme"},
            {"ticket_id": "TK-NOT-SENT", "theme": "Other"},
        ])
    labels, stats = classify_tickets(_tickets(2), batch_size=2,
                                     cache_path=str(tmp_path / "l.jsonl"), client=FakeClient(bad_reply))
    assert stats["sent"] == 0 and stats["failed"] == 2
    assert "TK-NOT-SENT" not in labels


def test_classify_survives_unparseable_reply(tmp_path):
    labels, stats = classify_tickets(_tickets(3), batch_size=3,
                                     cache_path=str(tmp_path / "l.jsonl"),
                                     client=FakeClient(lambda t: "sorry, I can't do that"))
    assert stats["failed"] == 3 and stats["sent"] == 0


def test_unknown_outcome_is_coerced_to_unclear(tmp_path):
    reply = lambda t: json.dumps([{"ticket_id": t[0]["ticket_id"], "theme": "Other", "outcome": "magic"}])
    labels, stats = classify_tickets(_tickets(1), cache_path=str(tmp_path / "l.jsonl"), client=FakeClient(reply))
    assert labels["TK-0"]["outcome"] == "unclear" and stats["sent"] == 1


def test_new_prompt_version_ignores_old_cache(tmp_path):
    cache = tmp_path / "l.jsonl"
    cache.write_text(json.dumps({"ticket_id": "TK-0", "theme": "Other", "prompt_version": "v1"}) + "\n",
                     encoding="utf-8")
    assert load_cache(str(cache)) == {}, "v1 labels must not be reused by the v2 prompt"
    assert "TK-0" in load_cache(str(cache), prompt_version="v1")


def test_parallel_workers_label_everything_once(tmp_path):
    client = FakeClient(_good_reply)
    labels, stats = classify_tickets(_tickets(45), batch_size=10, max_workers=4,
                                     cache_path=str(tmp_path / "l.jsonl"), client=client)
    assert stats["sent"] == 45 and client.calls == 5
    assert len(load_cache(str(tmp_path / "l.jsonl"))) == 45
