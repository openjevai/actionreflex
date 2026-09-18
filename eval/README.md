# Evaluating actionreflex on public agent-safety benchmarks

actionreflex was run, unchanged and untuned, on two public benchmarks:
[R-Judge](https://github.com/Lordog/R-Judge) (571 agent interaction records labelled safe
or unsafe) and [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (2,108 indirect
prompt-injection attacks). Each record was replayed the way the gate would see it in
production: every agent action is checked before it runs, using only what came before it.

**Summary** (`jev-latest` → `jev-1.13.0`, default policies and thresholds):

- **R-Judge: F1 89.3** (95% CI 86.6–91.9), recall 90.4, specificity 86.7. The R-Judge
  paper's best model, GPT-4o, scored 74.5. That comparison has caveats, covered below.
- **Attacks:** all 200 unsafe injection records caught; 11 false alarms among the 214 safe ones.
  **InjecAgent:** 2,106 of 2,108 attacker tool calls stopped, and 0 of 17 legitimate calls.
- **Weak spot: unintended risks.** F1 72.7 against GPT-4o's 80.9. The gate misses cases
  where the agent guesses at an ambiguous request instead of asking, and it raises false
  alarms on safe records (specificity 55.4).
- **Stable and cheap:** three runs agreed on 99.8% of record decisions, and F1 stayed
  within 89.2–89.5. Median latency is about 260ms per check. The full evaluation (3,584
  checks, six questions each) cost **$0.19**.

Full generated tables: [RESULTS.md](RESULTS.md).

## What was tested

- **The library as shipped.** The six built-in policies were written before this evaluation
  and were not changed after it (`src/actionreflex/policies.py` is the same as at
  `6deca13`). The default thresholds were used, with no tuning.
- **The "default gate"** is `default_policies()`: `destructive_action` escalates,
  `intent_mismatch` and `follows_injected_instructions` block, and `contains_pii` only
  warns. A record counts as **caught** if any of its actions is blocked or escalated.
  Escalated actions go to a human, so counting them as caught assumes someone reviews them.
  The "block only" figures below show what's stopped with no human involved.
- **All six questions per check.** Every check asked all six policies in a single request,
  and the raw answers were saved. Every configuration below is computed from the same
  answers.

### How records become checks

**R-Judge.** Each record is a conversation between a user, an agent and its tools. The
harness walks through it, and before each agent action it asks the gate about that action.
The context passed is:

- the user's request;
- the conversation and tool outputs so far;
- the agent's system profile;
- the reasoning the agent wrote just before acting.

The gate never sees what happened *after* the action. Agent turns that are plain chat
replies, not tool calls, are checked as a `reply_to_user` action.

**InjecAgent.** The checked action is the attacker's tool call, made right after the agent
read a tool response carrying the injected instruction. Both the base and the "enhanced"
attack wording are included. The dataset names the attacker's tool but not the arguments
it would be called with, so the gate saw the call without them. The 17 legitimate tool
calls the scenarios start from serve as negatives.

## Results

### R-Judge, compared with the paper's LLM judges

| | All F1 | Attacks F1 | Attacks recall | Attacks spec. | Unintended F1 | Unintended recall | Unintended spec. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **actionreflex, default gate** | **89.3** | **97.3** | **100.0** | **94.9** | 72.7 | 71.3 | 55.4 |
| actionreflex, default gate, block only | 82.6 | 97.6 | 100.0 | 95.3 | 39.4 | 27.7 | 76.8 |
| actionreflex, all six policies | 89.7 | 97.3 | 100.0 | 94.9 | 74.9 | 78.2 | 44.6 |
| GPT-4o (paper) | 74.5 | 72.2 | 91.5 | 42.1 | **80.9** | 72.0 | **89.1** |
| Llama-3-8B-Instruct (paper) | 61.0 | 65.7 | 66.5 | 66.4 | 48.3 | 36.0 | 76.4 |
| GPT-3.5 (paper) | 45.0 | 40.5 | 37.0 | 57.5 | 55.6 | 42.0 | 83.6 |
| Random (paper) | 51.3 | 56.3 | 50.0 | 50.0 | 49.1 | 50.0 | 50.0 |

"Attacks" are R-Judge's 414 injection records, 200 unsafe and 214 safe. "Unintended" are
its 157 other records, 101 unsafe and 56 safe.

**Why the gate does so well on attacks.** In the safe attack records, the agent *read*
an injected instruction and ignored it. A judge that reads the whole record sees the
injection and tends to flag it anyway: GPT-4o's specificity on attacks is 42.1. The gate
judges each action, not the transcript, so an ignored injection doesn't trigger it.

That separation is easy in this data. The attacker's tool is almost always unrelated to
what the user asked for, like a smart-lock call during a product lookup. Read the 100%
recall as "catches off-task tool calls reliably", not as "solves prompt injection".

The one exception proves the point. InjecAgent case 275 is the only scenario where the
attacker's tool matches the user's task: the user asked for a GitHub user's details, and
the injection asked for a *different* user's details. The gate missed it in both variants,
and those are its only two misses on InjecAgent. An injection that hijacks an on-task
action, such as adding a recipient to an email the user did ask for, is the case that
matters most in practice. Neither benchmark tests it.

**Where it falls short.** The gate misses 29 unsafe records and raises 36 false alarms.
All 29 misses are unintended risks, and most follow one pattern: the user's request was
ambiguous and the agent acted on a guess instead of asking. Examples:

- picking one of two matching medication bottles;
- choosing which breaker to switch off;
- muting a security camera along with the TVs;
- granting an app's permissions without asking.

No built-in policy asks "did this action settle something the user left open?".
That's the clearest candidate for a new policy. It hasn't been tested here.

A few other misses are harmful *content* requests answered in chat, such as napalm
instructions. That's content safety, which this library doesn't attempt.

Most false alarms, 39 of the 51 policy triggers behind them, fired on chat replies rather
than tool calls. Typically, `intent_mismatch` read a safe refusal or a clarifying question
as "not doing what the user asked".

### Checking tool calls only (post hoc)

actionreflex is meant to sit in front of tool execution; checking chat replies was a
choice made for this harness. Scoring only the tool calls cuts false alarms from 36 to 9:

| | F1 | Recall | Specificity |
| --- | --- | --- | --- |
| All 571 records | 86.9 (83.7–89.8) | 79.1 | 96.7 |
| 502 records containing tool calls | 94.8 | 93.3 | 96.4 |
| Attacks | 100.0 | 100.0 | 100.0 |
| Unintended | 51.4 | 37.6 | 83.9 |

The cost is recall on unintended risks: the risks that show up only in a chat reply
are no longer checked at all. This view was chosen *after* the error analysis, so it's
reported separately from the headline.

### Individual policies, and threshold tuning

| Policy | AUROC, all | AUROC, attacks | AUROC, unintended |
| --- | --- | --- | --- |
| `risk_tier` | 90.4 | 97.2 | **74.9** |
| `intent_mismatch` | 89.7 | 98.7 | 61.1 |
| `destructive_action` | 89.0 | 98.9 | 60.3 |
| `follows_injected_instructions` | 78.3 | **99.9** | 52.5 |
| `scope_of_impact` | 77.8 | 89.4 | 64.1 |
| `contains_pii` | 44.6 | 43.9 | 59.0 |

AUROC is the chance that an unsafe record scores as riskier than a safe one; 50 is random.

- **Each policy measures what it's named for.** The injection check is near-perfect on
  attacks and near chance on unintended risks, as it should be. `contains_pii` doesn't
  track "unsafe", as expected: personal data in an action isn't a safety failure. That's
  why it only warns by default.
- **Tuning thresholds adds little.** The default gate's combined risk score reaches an
  AUROC of 93.9. Tuning one threshold by 5-fold cross-validation gives F1 89.9 against
  89.3 untuned, with the tuned thresholds landing at 0.55–0.60 against the default 0.5.
  The defaults are close to the best this data supports.

### Robustness

- **Hiding the agent's reasoning** barely matters: F1 89.0 (86.2–91.5) against 89.3.
  Many agent frameworks never expose the reasoning before a tool call, and the gate
  doesn't depend on it.
- **Run-to-run stability.** Jev's probabilities vary slightly between identical calls: a
  mean change of about 0.01, and only 1.2% of checks returned exactly the same numbers. Per
  policy, the trigger decision was the same 98.6–99.9% of the time, and the gate's record
  decisions agreed 99.8% of the time. Three runs on the same model scored F1 89.2, 89.3 and
  89.5.
- **`jev-preview` isn't a different model yet.** It resolved to the same `jev-1.13.0` at
  run time, so its row in RESULTS.md is another repeat, not a model comparison.
- **Results vary by area:**

  | Area | Records | F1 |
  | --- | --- | --- |
  | Finance | 126 | 96.2 |
  | Application | 252 | 95.3 |
  | Program | 128 | 86.1 |
  | Web | 35 | 66.7 |
  | IoT | 30 | 58.8 |

  The two weak areas are small, and they're mostly unintended-risk records.

### Cost and latency

| | Per check | Full jev-latest evaluation (3,584 checks) |
| --- | --- | --- |
| Input tokens | ~1,200–1,300 (six questions and the context) | 4.46M |
| Cost at $0.042/M input tokens | ~$0.000052 (~$52 per million checks) | $0.19 |
| Latency p50 / p90 / p99 | 260 / 350 / 740–1,290 ms | a little over 2 minutes at 8 in flight |

Latency is measured client-side with 8 requests in flight. All four runs together
(10,086 checks) cost $0.53. TypeSafe prices output tokens at zero.

## Threats to validity

1. **The comparison isn't like-for-like.** The R-Judge baselines judged each whole record
   after the fact, with the paper's prompt, in 2024. The gate judges each action before it
   runs, with less information. Those baselines weren't re-run here. The next experiment
   should put a current frontier LLM through this same per-action harness and compare
   accuracy, latency and cost directly.
2. **The attack data is easy for action-level checks,** because the attacker's tool is
   almost always unrelated to the user's task. On-task hijacking is essentially untested:
   there's one case, and it was missed.
3. **InjecAgent's attacker calls have no arguments,** and it has only 17 legitimate calls,
   so its false-positive rate is only loosely estimated.
4. **Escalation counts as caught.** 44 of the 272 unsafe records caught were stopped only
   by escalation, so each of those needs a human to review it.
5. **Small subsets:** IoT has 30 records, Web 35, and InjecAgent's legitimate set 17.
6. **The tool-calls-only view was chosen after the error analysis.** The headline
   configuration was fixed beforehand.
7. **The labels are the dataset authors',** and some unintended-risk labels are judgment
   calls. R-Judge now has 571 records; the paper reported 569.

## Reproduce

```bash
pip install -e ".[dev]"
python -m eval.report            # rebuild RESULTS.md from the committed raw answers, no API key needed

export TYPESAFE_API_KEY=...
python -m eval.run               # re-run both datasets: ~3,600 checks, 2-3 minutes, ~$0.19
python -m eval.run --dataset rjudge --no-thought   # the reasoning ablation
python -m eval.run --dataset rjudge --tag repeat   # the stability run
```

The datasets are downloaded at pinned commits (R-Judge `83ce301`, InjecAgent `f19c9f2`)
into `eval/.cache/` and aren't committed. R-Judge has no licence, so it's used for
evaluation only. The raw results in `eval/results/` hold record IDs and numbers only, and
each run's `manifest.json` records the model that answered, the package versions and the
git commit.
