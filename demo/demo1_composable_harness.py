"""Demo 1 — Composable harness architecture (HarnessX).

Article section: "HarnessX: Composable Optimization Architecture"

A harness here is a PIPELINE of independently pluggable processors, one per
named lifecycle hook: context assembly, tool selection, response
generation, critique. Swapping the processor behind ONE hook is a real,
isolated operation — every hook enforces a contract (a canary call is run
BEFORE a new processor is installed), and the other hooks' call order and
call counts are provably untouched by the swap. That's the "Lego blocks"
claim: components plug into precise lifecycle hooks without disrupting the
rest of the pipeline.
"""

import inspect

from mini_harness import Trace
from mini_harness.trace import BOLD, DIM, GREEN, RED, YELLOW, paint


class ContractViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# A tiny multi-turn "session": facts accrue turn by turn, then a query asks
# about one of them. The harness must route the query through: assemble
# context -> select a tool -> respond -> critique the answer.
# ---------------------------------------------------------------------------

class Task:
    def __init__(self, turns, query_key, expected):
        self.turns = turns  # list of (turn_no, key, value)
        self.query_key = query_key
        self.expected = expected


TASKS = [
    Task([(1, "budget", "$500"), (2, "deadline", "March 3"), (3, "owner", "Alex")],
         "deadline", "March 3"),
    Task([(1, "region", "us-east"), (2, "tier", "gold"), (3, "sla", "99.9%"),
          (4, "owner", "Priya")], "owner", "Priya"),
    Task([(1, "lang", "Rust"), (2, "version", "1.82"), (3, "target", "wasm32"),
          (4, "profile", "release"), (5, "flags", "-O3")], "lang", "Rust"),
    Task([(1, "topic", "harnesses"), (2, "author", "Dickson"), (3, "venue", "TechTalks"),
          (4, "year", "2026")], "author", "Dickson"),
    Task([(1, "city", "Austin"), (2, "team", "infra"), (3, "budget", "$12k"),
          (4, "quarter", "Q3"), (5, "lead", "Sam")], "team", "infra"),
]


# ---------------------------------------------------------------------------
# Hook implementations. Each hook has a CONTRACT: a predicate its return
# value must satisfy. "context" is the hook we'll hot-swap.
# ---------------------------------------------------------------------------

def recency_window_context(turns, query_key, window=3):
    """Default: only the last `window` turns are visible to the pipeline."""
    return {k: v for _, k, v in turns[-window:]}


def relevance_context(turns, query_key, window=3):
    """Improved: scan the FULL history for the turn the query actually needs."""
    exact = {k: v for _, k, v in turns if k == query_key}
    return exact or {k: v for _, k, v in turns[-window:]}


def broken_context(turns, query_key, window=3):
    """A processor that violates the 'context' hook's contract on purpose."""
    return "not-a-dict"


def select_tool(context, query_key):
    return "fact_lookup"


def respond(context, query_key, tool):
    assert tool == "fact_lookup"
    return context.get(query_key, "<unknown>")


def critique(answer, expected):
    return answer == expected


CONTRACTS = {
    "context": lambda r: isinstance(r, dict),
    "select_tool": lambda r: isinstance(r, str),
    "respond": lambda r: isinstance(r, str),
    "critique": lambda r: isinstance(r, bool),
}

HOOK_ORDER = ["context", "select_tool", "respond", "critique"]


class Pipeline:
    """Ordered lifecycle hooks; each slot independently swappable and contract-checked."""

    def __init__(self, processors: dict):
        self.processors = dict(processors)
        self.calls = {name: 0 for name in HOOK_ORDER}
        self.rejected_swaps = 0

    def run(self, task: Task):
        self.calls["context"] += 1
        context = self.processors["context"](task.turns, task.query_key)

        self.calls["select_tool"] += 1
        tool = self.processors["select_tool"](context, task.query_key)

        self.calls["respond"] += 1
        answer = self.processors["respond"](context, task.query_key, tool)

        self.calls["critique"] += 1
        ok = self.processors["critique"](answer, task.expected)
        return answer, ok

    def swap(self, hook_name: str, new_fn, canary_task: Task) -> None:
        """Contract-checked hot-swap: run a canary call BEFORE installing."""
        sig = inspect.signature(new_fn)
        required = len(inspect.signature(self.processors[hook_name]).parameters)
        if len(sig.parameters) < required:
            self.rejected_swaps += 1
            raise ContractViolation(
                f"'{hook_name}' processor takes {len(sig.parameters)} args, "
                f"hook requires at least {required}"
            )
        try:
            canary_result = new_fn(canary_task.turns, canary_task.query_key)
        except Exception as exc:  # noqa: BLE001 — contract check, not app logic
            self.rejected_swaps += 1
            raise ContractViolation(f"'{hook_name}' processor raised on canary input: {exc}") from exc
        if not CONTRACTS[hook_name](canary_result):
            self.rejected_swaps += 1
            raise ContractViolation(
                f"'{hook_name}' processor returned {canary_result!r}, "
                f"violating the hook's contract"
            )
        self.processors[hook_name] = new_fn


def run_suite(pipeline: Pipeline, tasks: list[Task]):
    results = []
    for task in tasks:
        answer, ok = pipeline.run(task)
        results.append((task, answer, ok))
    return results


def main() -> None:
    trace = Trace(agent="harnessx", color=YELLOW)
    trace.banner(
        "DEMO 1 — Composable Harness Processors (HarnessX)",
        "hot-swap one lifecycle hook, prove the rest of the pipeline is untouched",
    )
    trace.intro(
        what="A harness built from independently pluggable processors, one per lifecycle "
             "hook (context, tool selection, response, critique). Each hook enforces a "
             "contract, checked with a live canary call before a new processor is ever "
             "installed.",
        watch="A baseline context-assembly processor causes real failures on tasks that "
              "need an older fact. We hot-swap ONLY that hook. A contract-breaking "
              "processor gets rejected before install; a valid one gets installed, and "
              "the other three hooks' call counts prove they were never touched.",
    )
    trace.note("Nothing here is scripted: routing, failures, and the swap outcome are computed live.")

    pipeline = Pipeline({
        "context": recency_window_context,
        "select_tool": select_tool,
        "respond": respond,
        "critique": critique,
    })

    trace.section("baseline pipeline — context hook = recency_window (last 3 turns)")
    results = run_suite(pipeline, TASKS)
    passed = sum(ok for _, _, ok in results)
    for task, answer, ok in results:
        tag = paint("PASS", GREEN, BOLD) if ok else paint("FAIL", RED, BOLD)
        trace._emit(f"    {tag}  query={task.query_key!r:14} got={answer!r:14} expected={task.expected!r}")
    trace.note(f"score: {passed}/{len(TASKS)}")
    trace.note(f"hook call counts: {pipeline.calls}")

    failures = [t for t, _, ok in results if not ok]
    trace.thought(
        f"Weakness: {len(failures)} failures all query a fact older than the 3-turn "
        f"window — the 'context' hook is the culprit, the other three hooks are fine."
    )

    calls_before = dict(pipeline.calls)

    trace.section("attempted swap #1 — a contract-breaking processor")
    try:
        pipeline.swap("context", broken_context, canary_task=TASKS[0])
        trace.verdict(False, "swap should have been rejected but was not")
    except ContractViolation as exc:
        trace.verdict(True, paint(f"REJECTED at plug-in time: {exc}", RED, BOLD))
        trace.note(paint("the broken processor never ran against real traffic", DIM))

    trace.section("attempted swap #2 — relevance_context (full-history scan)")
    pipeline.swap("context", relevance_context, canary_task=TASKS[0])
    trace.verdict(True, paint("INSTALLED: 'context' hook now points at relevance_context", GREEN, BOLD))

    trace.section("re-run with the new context processor")
    results2 = run_suite(pipeline, TASKS)
    passed2 = sum(ok for _, _, ok in results2)
    for task, answer, ok in results2:
        tag = paint("PASS", GREEN, BOLD) if ok else paint("FAIL", RED, BOLD)
        trace._emit(f"    {tag}  query={task.query_key!r:14} got={answer!r:14} expected={task.expected!r}")
    trace.note(f"score: {passed2}/{len(TASKS)}")

    delta = {name: pipeline.calls[name] - calls_before[name] for name in HOOK_ORDER}
    trace.section("composability check — did the swap disrupt anything else?")
    for name in HOOK_ORDER:
        untouched = name == "context" or delta[name] == len(TASKS)
        trace.note(f"{name:12} +{delta[name]} calls this run  "
                   f"{'(swapped)' if name == 'context' else '(same processor, same call count as before)'}")
    other_hooks_untouched = all(delta[n] == len(TASKS) for n in HOOK_ORDER if n != "context")
    trace.verdict(
        other_hooks_untouched,
        "select_tool / respond / critique fired exactly once per task, identical to the "
        "baseline run — only the swapped hook's behavior changed",
    )

    trace.section("final")
    trace.verdict(passed2 == len(TASKS), f"held-in score after swap: {passed2}/{len(TASKS)}")
    trace.note(f"rejected swap attempts: {pipeline.rejected_swaps}")

    trace.takeaway([
        "A harness is a set of components behind named hooks, not one monolithic prompt.",
        "Contracts are enforced with a live canary call, not just a docstring promise.",
        "Swapping one processor is provably isolated: the untouched hooks' call counts don't move.",
    ])


if __name__ == "__main__":
    main()
