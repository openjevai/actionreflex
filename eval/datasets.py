"""Download and convert public agent-safety benchmarks into gate checks.

Both datasets are fetched at a pinned commit into `eval/.cache/` (gitignored) and
never committed to this repository: R-Judge ships without a license, so it is used
here for evaluation only, the way its authors' own tooling uses it.

- R-Judge (Yuan et al., 2024) - 571 multi-turn agent interaction records labelled
  safe/unsafe. Each record is replayed as the gate would see it live: every agent
  action is checked *before* it runs, with only the conversation up to that point.
  https://github.com/Lordog/R-Judge
- InjecAgent (Zhan et al., 2024) - 1,054 indirect prompt-injection cases (x2 for the
  "enhanced" variant). The checked action is the attacker's tool call, right after
  the agent read a tool response carrying the injected instruction.
  https://github.com/uiuc-kang-lab/InjecAgent
"""

from __future__ import annotations

import ast
import io
import json
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from actionreflex import Action

CACHE = Path(__file__).parent / ".cache"

SOURCES = {
    "R-Judge": ("Lordog/R-Judge", "83ce301da3ad50dd8b397e772863f5411c3d3dc2"),
    "InjecAgent": ("uiuc-kang-lab/InjecAgent", "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"),
}


@dataclass
class Check:
    """One gate check: an action, as the gate would see it before execution."""

    key: str
    action: Action


@dataclass
class Case:
    """A labelled unit of evaluation. `unsafe` is the ground truth; the gate "catches"
    a case if any of its checks is stopped."""

    id: str
    dataset: str
    unsafe: bool
    subset: str
    checks: list[Check] = field(default_factory=list)
    has_tool_call: bool = True


def fetch(name: str) -> Path:
    """Download a pinned dataset snapshot (once) and return its directory."""
    repo, sha = SOURCES[name]
    target = CACHE / f"{repo.split('/')[1]}-{sha[:7]}"
    if target.exists():
        return target
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        root = tar.getnames()[0].split("/")[0]
        try:
            tar.extractall(CACHE, filter="data")
        except TypeError:  # Python < 3.10.12 / 3.11.4: no extraction filters
            tar.extractall(CACHE)
    (CACHE / root).rename(target)
    return target


# --- parsing agent actions ---------------------------------------------------

_TOOL = re.compile(
    r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)"
    r"\s*(?:\n\s*Action Input:\s*|\s+Input:\s*|:\s*)?"
    r"(?P<args>[\{\[].*)?\s*$",
    re.DOTALL,
)
_FUNC = re.compile(r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)\((?P<args>.*)\)\s*$", re.DOTALL)


def _parse_args(raw: str) -> dict[str, Any]:
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(raw)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        return value if isinstance(value, dict) else {"input": value}
    return {"input": raw}


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """Turn an agent's action string into (tool name, arguments).

    Plain-text replies to the user become `reply_to_user`. Returns None when the
    agent took no action at all.
    """
    text = (text or "").strip()
    if not text or text == "None":
        return None
    if m := _TOOL.match(text):
        if m.group("args"):
            return m.group("name"), _parse_args(m.group("args"))
        if sum(c.isupper() for c in m.group("name")) >= 2:  # bare CamelCase tool name
            return m.group("name"), {}
    if m := _FUNC.match(text):
        return m.group("name"), {"input": m.group("args")}
    return "reply_to_user", {"message": text}


def _clean(value: Any) -> Any:
    return None if value in (None, "None", "") else value


# --- R-Judge -----------------------------------------------------------------


def load_rjudge() -> list[Case]:
    root = fetch("R-Judge") / "data"
    cases: list[Case] = []
    for path in sorted(root.rglob("*.json")):
        for record in json.loads(path.read_text()):
            cases.append(_rjudge_case(record, category=path.parent.name, source=path.stem))
    return cases


def _rjudge_case(record: dict[str, Any], category: str, source: str) -> Case:
    case = Case(
        id=f"rjudge-{record['id']}",
        dataset="R-Judge",
        unsafe=int(record["label"]) == 1,
        subset="intended_attack" if record.get("attack_type") == "injection" else "unintended",
    )
    profile = _clean(record.get("profile"))
    history: list[dict[str, Any]] = []
    user_request = None
    step = 0

    for turn in record["contents"]:
        for entry in turn:
            role = entry.get("role")
            if role == "user":
                user_request = _clean(entry.get("content")) or user_request
                history.append({"role": "user", "content": user_request})
            elif role == "environment":
                history.append({"role": "environment", "content": _clean(entry.get("content"))})
            elif role == "agent":
                action_text = str(entry.get("action") or "")
                thought = _clean(entry.get("thought"))
                parsed = parse_action(action_text)
                if parsed is not None:
                    name, arguments = parsed
                    context: dict[str, Any] = {
                        "user_request": user_request,
                        "history": list(history),
                    }
                    if profile:
                        context["agent_profile"] = profile
                    if thought:
                        context["agent_thought"] = thought
                    case.checks.append(
                        Check(key=f"{case.id}/{step}", action=Action(name, arguments, context))
                    )
                    step += 1
                history.append({"role": "agent", "thought": thought, "action": action_text})

    case.has_tool_call = any(c.action.name != "reply_to_user" for c in case.checks)
    case.subset += f"/{category}/{source}"
    return case


# --- InjecAgent ----------------------------------------------------------------


def load_injecagent() -> list[Case]:
    root = fetch("InjecAgent") / "data"
    cases: list[Case] = []

    for variant in ("base", "enhanced"):
        for kind, label in (("dh", "direct_harm"), ("ds", "data_stealing")):
            path = root / f"test_cases_{kind}_{variant}.json"
            for i, item in enumerate(json.loads(path.read_text())):
                cid = f"injecagent-{variant}-{kind}-{i}"
                # Stored as a quote-wrapped Python-literal string, not valid JSON.
                response = item["Tool Response"].strip()
                if len(response) >= 2 and response[0] == response[-1] == '"':
                    response = response[1:-1]
                context = {
                    "user_request": item["User Instruction"],
                    "history": [
                        {"role": "user", "content": item["User Instruction"]},
                        {
                            "role": "agent",
                            "thought": item.get("Thought"),
                            "action": f"{item['User Tool']}{item['Tool Parameters']}",
                        },
                        {"role": "environment", "content": response},
                    ],
                }
                # InjecAgent records which attacker tool the injection tries to trigger,
                # but not the arguments an agent would pass, so the gate sees the call
                # without them - less information than a real attack would give it.
                action = Action(name=item["Attacker Tools"][0], arguments={}, context=context)
                cases.append(
                    Case(
                        id=cid,
                        dataset="InjecAgent",
                        unsafe=True,
                        subset=f"{variant}/{label}",
                        checks=[Check(key=cid, action=action)],
                    )
                )

    # The legitimate tool call each scenario starts from: what the user actually asked for.
    for i, line in enumerate((root / "user_cases.jsonl").read_text().splitlines()):
        item = json.loads(line)
        cid = f"injecagent-legit-{i}"
        action = Action(
            name=item["User Tool"],
            arguments=_parse_args(item["Tool Parameters"]),
            context={"user_request": item["User Instruction"], "history": []},
        )
        cases.append(
            Case(
                id=cid,
                dataset="InjecAgent",
                unsafe=False,
                subset="legitimate",
                checks=[Check(key=cid, action=action)],
            )
        )
    return cases


LOADERS = {"rjudge": load_rjudge, "injecagent": load_injecagent}
