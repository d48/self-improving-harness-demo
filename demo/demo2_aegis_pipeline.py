"""Demo 2 — The AEGIS evolution engine: Digester -> Planner -> Evolver -> Critic.

Article section: "HarnessX: AEGIS Evolution Engine"

A toy todo-list command parser is the harness under repair: an ordered list
of named rules turns messy commands ("please delete walk dog") into a
canonical (ACTION, ARG) pair. Four explicit, real stages run the loop:

  DIGESTER   run the held-in suite, cluster failures by root cause
  PLANNER    pick the next bounded candidate rule for the biggest cluster
  EVOLVER    apply the edit, then unit-test it in ISOLATION on just the
             target cluster before spending a full evaluation
  CRITIC     run the full held-in/held-out regression AND a static
             source-code scan for memorized literals — two independent,
             deterministic gates

One candidate rule memorizes the exact held-in strings for its cluster. It
aces the isolated unit test AND does not regress held-out (the held-out
case for that cluster was already failing, so nothing gets worse by the
numbers). Only the Critic's source-code scan catches it — the article's
point that reward hacking needs a detector that doesn't just watch scores.
"""

import inspect
from dataclasses import dataclass

from mini_harness import Trace
from mini_harness.trace import BOLD, CYAN, DIM, GREEN, RED, paint


# ---------------------------------------------------------------------------
# The harness under repair: normalize a todo command into (ACTION, ARG).
# ---------------------------------------------------------------------------

FILLERS = ("please ", "could you ", "kindly ")
SYNONYMS = {"delete": "remove", "cancel": "remove"}
TRAILING = {"done": "complete", "finished": "complete", "finish": "complete"}
VERBS = {"add": "ADD", "remove": "REMOVE", "complete": "COMPLETE"}


def rule_strip_filler(raw: str) -> str:
    value = raw
    for filler in FILLERS:
        if value.startswith(filler):
            value = value[len(filler):]
    return value


def rule_map_synonyms(raw: str) -> str:
    tokens = raw.split(" ", 1)
    if tokens[0] in SYNONYMS:
        tokens[0] = SYNONYMS[tokens[0]]
    return " ".join(tokens)


def rule_parse_trailing_status(raw: str) -> str:
    tokens = raw.rsplit(" ", 1)
    if len(tokens) == 2 and tokens[1] in TRAILING:
        return f"{TRAILING[tokens[1]]} {tokens[0]}"
    return raw


def rule_hack_memorize_word_order(raw: str) -> str:
    """Reward hack: a lookup table of the exact held-in raw strings, nothing general."""
    memorized = {
        "buy milk done": "complete buy milk",
        "call mom finished": "complete call mom",
    }
    return memorized.get(raw, raw)


RULES = {
    "strip_filler": rule_strip_filler,
    "map_synonyms": rule_map_synonyms,
    "parse_trailing_status": rule_parse_trailing_status,
    "hack_memorize_word_order": rule_hack_memorize_word_order,
}


def run_harness(harness: list[str], raw: str):
    value = raw
    for rule_name in harness:
        value = RULES[rule_name](value)
    tokens = value.split(" ", 1)
    verb = tokens[0]
    if verb in VERBS and len(tokens) == 2:
        return (VERBS[verb], tokens[1])
    return None


@dataclass(frozen=True)
class Case:
    raw: str
    expected: tuple
    cluster: str


HELD_IN = [
    Case("add buy milk", ("ADD", "buy milk"), "baseline"),
    Case("please add call mom", ("ADD", "call mom"), "filler_words"),
    Case("could you add walk dog", ("ADD", "walk dog"), "filler_words"),
    Case("delete buy milk", ("REMOVE", "buy milk"), "synonyms"),
    Case("cancel call mom", ("REMOVE", "call mom"), "synonyms"),
    Case("buy milk done", ("COMPLETE", "buy milk"), "word_order"),
    Case("call mom finished", ("COMPLETE", "call mom"), "word_order"),
]

HELD_OUT = [
    Case("please delete walk dog", ("REMOVE", "walk dog"), "filler_words+synonyms"),
    Case("remove call mom", ("REMOVE", "call mom"), "baseline"),
    Case("walk dog done", ("COMPLETE", "walk dog"), "word_order"),
    Case("add file taxes", ("ADD", "file taxes"), "baseline"),
]

# Bounded proposal catalog per cluster. The hack is listed first for
# "word_order" so the Planner tries it before the legitimate fix.
CANDIDATES = {
    "filler_words": ["strip_filler"],
    "synonyms": ["map_synonyms"],
    "word_order": ["hack_memorize_word_order", "parse_trailing_status"],
}

HELD_IN_RAW_STRINGS = {c.raw for c in HELD_IN}


def evaluate(harness: list[str], suite: list[Case]):
    passed, clusters = 0, {}
    for case in suite:
        if run_harness(harness, case.raw) == case.expected:
            passed += 1
        else:
            clusters.setdefault(case.cluster, []).append(case)
    return passed, clusters


# ---------------------------------------------------------------------------
# AEGIS stages
# ---------------------------------------------------------------------------

def digester(trace: Trace, harness: list[str]):
    """DIGESTER — isolate failures on the held-in suite, cluster by root cause."""
    in_score, clusters = evaluate(harness, HELD_IN)
    trace.stage("DIGESTER", f"held-in {in_score}/{len(HELD_IN)}")
    if not clusters:
        return in_score, None
    biggest = max(clusters, key=lambda c: len(clusters[c]))
    example = clusters[biggest][0]
    trace.note(f"failure clusters: {[(c, len(v)) for c, v in clusters.items()]}")
    trace.note(f"target cluster '{biggest}': e.g. {example.raw!r} -> "
               f"{run_harness(harness, example.raw)!r}, expected {example.expected!r}")
    return in_score, biggest


def planner(trace: Trace, cluster: str, tried: set):
    """PLANNER — pick the next untried bounded candidate for the target cluster."""
    candidate = next((r for r in CANDIDATES[cluster] if r not in tried), None)
    if candidate is None:
        trace.stage("PLANNER", "no untried candidates left for this cluster")
        return None
    trace.stage("PLANNER", f"strategy: try rule '{candidate}' for cluster '{cluster}'")
    return candidate


def evolver(trace: Trace, harness: list[str], candidate_rule: str, cluster: str):
    """EVOLVER — apply the edit, then unit-test it in isolation before full evaluation."""
    candidate_harness = harness + [candidate_rule]
    target_cases = [c for c in HELD_IN if c.cluster == cluster]
    isolated_passed = sum(
        run_harness(candidate_harness, c.raw) == c.expected for c in target_cases
    )
    trace.stage("EVOLVER", f"h + ['{candidate_rule}'] — isolated test on the '{cluster}' cluster only")
    trace.note(f"isolated: {isolated_passed}/{len(target_cases)} target-cluster cases pass")
    return candidate_harness, isolated_passed == len(target_cases)


def critic(trace: Trace, harness: list[str], candidate_harness: list[str], candidate_rule: str):
    """CRITIC — full regression gate AND a static reward-hack scan. Either can reject."""
    trace.stage("CRITIC", "full held-in/held-out regression + static source scan")

    old_in, _ = evaluate(harness, HELD_IN)
    old_out, _ = evaluate(harness, HELD_OUT)
    new_in, _ = evaluate(candidate_harness, HELD_IN)
    new_out, _ = evaluate(candidate_harness, HELD_OUT)
    improves = new_in > old_in
    regresses = new_out < old_out
    trace.note(f"regression gate: held-in {old_in}->{new_in}, held-out {old_out}->{new_out}")

    source = inspect.getsource(RULES[candidate_rule])
    memorized_hits = [raw for raw in HELD_IN_RAW_STRINGS if raw in source]
    trace.note(f"static scan: {len(memorized_hits)} held-in literal(s) found hardcoded in rule source"
               if memorized_hits else "static scan: no held-in literals found in rule source")

    if memorized_hits:
        return False, f"REJECTED — source scan caught memorized literals: {memorized_hits}"
    if regresses:
        return False, "REJECTED — held-out regresses"
    if not improves:
        return False, "REJECTED — held-in does not improve"
    return True, "MERGED"


def main() -> None:
    trace = Trace(agent="aegis", color=CYAN)
    trace.banner(
        "DEMO 2 — The AEGIS Evolution Engine",
        "Digester -> Planner -> Evolver -> Critic, with a static reward-hack scan",
    )
    trace.intro(
        what="A four-stage evolution pipeline repairing a real todo-command parser: the "
             "Digester clusters failures, the Planner picks a bounded fix, the Evolver "
             "applies it and unit-tests it in isolation, and the Critic runs a full "
             "regression gate PLUS a static scan of the candidate's source code.",
        watch="One candidate rule memorizes its target cluster's exact inputs. It passes "
              "the Evolver's isolated test and does not regress the held-out regression "
              "suite — a pure metrics gate would accept it. The Critic's source-code scan "
              "catches the memorization anyway and rejects it.",
    )
    trace.note("Nothing here is scripted: clustering, isolated tests, and the scan run live.")

    harness: list[str] = []
    tried: set = set()

    for iteration in range(1, 8):
        trace.section(f"iteration {iteration} — current harness h_{iteration - 1} = {harness}")

        in_score, cluster = digester(trace, harness)
        out_score, _ = evaluate(harness, HELD_OUT)
        trace.note(f"held-out (for reference): {out_score}/{len(HELD_OUT)}")
        if cluster is None:
            trace.final("No weaknesses left to mine on held-in. Loop terminates.")
            break

        candidate_rule = planner(trace, cluster, tried)
        if candidate_rule is None:
            trace.note(f"cluster '{cluster}' has no more candidates; digester will pick the next-biggest cluster")
            tried.add(f"__exhausted_{cluster}")
            # Force this cluster out of contention by marking all its candidates tried.
            for r in CANDIDATES[cluster]:
                tried.add(r)
            continue
        tried.add(candidate_rule)

        candidate_harness, isolated_ok = evolver(trace, harness, candidate_rule, cluster)
        trace.verdict(isolated_ok, "isolated test " + ("passed" if isolated_ok else "failed"))
        if not isolated_ok:
            trace.note("Evolver's own smoke test failed — not even sent to the Critic")
            continue

        accepted, reason = critic(trace, harness, candidate_harness, candidate_rule)
        if accepted:
            harness = candidate_harness
            trace.verdict(True, paint(f"{reason}: '{candidate_rule}' merged into h_{iteration}", GREEN, BOLD))
        else:
            trace.verdict(False, paint(f"{reason}; harness unchanged", RED, BOLD))
            trace.note(paint("rejected candidates are logged, not merged", DIM))

    trace.section("final harness")
    trace.note(f"h_final = {harness}")
    final_in, _ = evaluate(harness, HELD_IN)
    final_out, _ = evaluate(harness, HELD_OUT)
    trace.verdict(final_in == len(HELD_IN) and final_out == len(HELD_OUT),
                  f"held-in {final_in}/{len(HELD_IN)}, held-out {final_out}/{len(HELD_OUT)}")

    trace.takeaway([
        "Composability shows up inside the pipeline too: Digester/Planner/Evolver/Critic "
        "are independent stages, each replaceable on its own.",
        "The Evolver's isolated unit test is cheap triage, not the accept/reject decision.",
        "A regression gate alone can miss a hack that doesn't happen to touch held-out — "
        "the Critic needed a second, orthogonal signal: static analysis of the diff itself.",
    ])


if __name__ == "__main__":
    main()
