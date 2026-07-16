"""Demo 3 — Harness-model co-evolution via GRPO-style group-relative scoring.

Article section: "Harness-Model Co-Evolution"

Two independent learners share ONE replay buffer:

  HARNESS track   picks the context-assembly strategy: window3 / window6 / relevance
  MODEL track     picks the response style: terse / verbose / structured

Each round both tracks sample a GROUP of joint configs. Every sampled config
is scored for real against a task suite (genuine context lookup + string
formatting + correctness check, no lookup table of "right answers"). Both
tracks then update from the SAME per-sample advantage — reward minus the
group's own mean, i.e. Group Relative Policy Optimization — so a track never
sees an absolute reward, only how each of its samples did relative to the
group it was sampled alongside. Neither policy is ever told the ground-truth
best gene directly; both discover it purely from relative standing.
"""

import math
import random

from mini_harness import Trace
from mini_harness.trace import BOLD, DIM, GREEN, YELLOW, paint


class Task:
    def __init__(self, turns, query_key, expected, requires_structured):
        self.turns = turns
        self.query_key = query_key
        self.expected = expected
        self.requires_structured = requires_structured


TRAIN_TASKS = [
    Task([(1, "budget", "$500"), (2, "deadline", "March 3"), (3, "owner", "Alex")],
         "deadline", "March 3", requires_structured=False),
    Task([(1, "region", "us-east"), (2, "tier", "gold"), (3, "sla", "99.9%"), (4, "owner", "Priya")],
         "owner", "Priya", requires_structured=True),
    Task([(1, "lang", "Rust"), (2, "v", "1.82"), (3, "t", "wasm32"), (4, "p", "release"),
          (5, "f", "-O3"), (6, "os", "linux"), (7, "arch", "x86_64"), (8, "opt", "size")],
         "lang", "Rust", requires_structured=False),
    Task([(1, "topic", "harnesses"), (2, "author", "Dickson"), (3, "venue", "TechTalks"),
          (4, "year", "2026"), (5, "words", "1800"), (6, "edits", "3"), (7, "editor", "Sam"),
          (8, "status", "published")],
         "topic", "harnesses", requires_structured=True),
    Task([(1, "team", "infra"), (2, "city", "Austin"), (3, "budget", "$12k"), (4, "quarter", "Q3"),
          (5, "lead", "Sam")],
         "team", "infra", requires_structured=False),
    Task([(1, "city", "Austin"), (2, "team", "infra"), (3, "budget", "$12k"), (4, "quarter", "Q3"),
          (5, "lead", "Sam")],
         "city", "Austin", requires_structured=True),
]

HELD_OUT_TASKS = [
    Task([(1, "milestone", "beta"), (2, "risk", "low"), (3, "eta", "Aug"), (4, "cost", "$3k"),
          (5, "owner", "Lee"), (6, "status", "green"), (7, "notes", "ok")],
         "milestone", "beta", requires_structured=True),
    Task([(1, "region", "eu-west"), (2, "tier", "silver"), (3, "sla", "99%"), (4, "status", "open")],
         "status", "open", requires_structured=False),
    Task([(1, "channel", "slack"), (2, "assignee", "Kim"), (3, "priority", "P1"), (4, "eta", "Fri"),
          (5, "cost", "$1k"), (6, "reviewer", "Al"), (7, "phase", "beta"), (8, "tag", "infra"),
          (9, "owner", "Kim")],
         "channel", "slack", requires_structured=True),
    Task([(1, "lead", "Sam"), (2, "team", "infra"), (3, "budget", "$12k"), (4, "quarter", "Q3"),
          (5, "assignee", "Kim")],
         "assignee", "Kim", requires_structured=False),
]


# ---------------------------------------------------------------------------
# The two gene spaces. Each combination is a real, runnable joint config.
# ---------------------------------------------------------------------------

def context_window(turns, key, window):
    return {k: v for _, k, v in turns[-window:]}


def context_relevance(turns, key, window=3):
    exact = {k: v for _, k, v in turns if k == key}
    return exact or {k: v for _, k, v in turns[-window:]}


HARNESS_GENES = {
    "window3": lambda turns, key: context_window(turns, key, 3),
    "window6": lambda turns, key: context_window(turns, key, 6),
    "relevance": lambda turns, key: context_relevance(turns, key),
}

MODEL_GENES = {
    "terse": lambda key, value: value,
    "verbose": lambda key, value: f"The value of {key} is {value}.",
    "structured": lambda key, value: f"{key.upper()}: {value}",
}


def evaluate_config(harness_gene: str, model_gene: str, tasks: list[Task]) -> float:
    """The real, external reward: fraction of tasks answered correctly."""
    passed = 0
    for task in tasks:
        context = HARNESS_GENES[harness_gene](task.turns, task.query_key)
        value = context.get(task.query_key, "<unknown>")
        produced = MODEL_GENES[model_gene](task.query_key, value)
        if task.requires_structured:
            ok = produced == f"{task.query_key.upper()}: {task.expected}"
        else:
            ok = task.expected in produced
        passed += ok
    return passed / len(tasks)


# ---------------------------------------------------------------------------
# GRPO-style policy: a softmax over gene choices, updated from group-relative
# advantage (reward minus the sampled group's own mean reward).
# ---------------------------------------------------------------------------

def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Policy:
    def __init__(self, genes: list[str]):
        self.genes = genes
        self.logits = [0.0] * len(genes)

    def probs(self) -> list[float]:
        return softmax(self.logits)

    def sample(self, rng: random.Random) -> str:
        probs = self.probs()
        r = rng.random()
        cum = 0.0
        for gene, p in zip(self.genes, probs):
            cum += p
            if r <= cum:
                return gene
        return self.genes[-1]

    def update(self, gene_advantages: list[tuple], lr: float) -> None:
        grad = [0.0] * len(self.genes)
        counts = [0] * len(self.genes)
        for gene, advantage in gene_advantages:
            idx = self.genes.index(gene)
            grad[idx] += advantage
            counts[idx] += 1
        for i in range(len(self.genes)):
            if counts[i]:
                self.logits[i] += lr * (grad[i] / counts[i])

    def as_str(self) -> str:
        return "  ".join(f"{g}={p:.2f}" for g, p in zip(self.genes, self.probs()))


def main() -> None:
    trace = Trace(agent="co-evolve", color=GREEN)
    trace.banner(
        "DEMO 3 — Harness-Model Co-Evolution (GRPO)",
        "two policies, one shared replay buffer, updates from group-relative advantage only",
    )
    trace.intro(
        what="A HARNESS track (context strategy) and a MODEL track (response style) each "
             "sample a group of joint configs per round. Every config is scored for real "
             "against a task suite. Both tracks update from the same per-sample advantage "
             "-- reward minus the group's own mean -- and log every sample to one shared "
             "replay buffer.",
        watch="Neither track is ever told which gene is 'correct'. Watch the softmax "
              "probabilities drift, live, purely from relative standing inside sampled "
              "groups, until the joint config that generalizes best on held-out tasks "
              "dominates both policies.",
    )
    trace.note("Nothing here is scripted: sampling, scoring, and every policy update run live.")

    SEED = 7
    GROUP_SIZE = 6
    ITERATIONS = 24
    LR = 0.8

    rng = random.Random(SEED)
    policy_h = Policy(list(HARNESS_GENES))
    policy_m = Policy(list(MODEL_GENES))
    replay_buffer: list[dict] = []

    trace.section("uniform baseline — expected reward before any training")
    all_combo_rewards = [
        evaluate_config(hg, mg, HELD_OUT_TASKS)
        for hg in HARNESS_GENES
        for mg in MODEL_GENES
    ]
    baseline_reward = sum(all_combo_rewards) / len(all_combo_rewards)
    trace.note(f"uniform-random policy on held-out tasks: mean reward = {baseline_reward:.3f} "
               f"(averaged over all {len(all_combo_rewards)} joint configs)")

    trace.section(f"training — {ITERATIONS} rounds, group size {GROUP_SIZE}")
    for iteration in range(1, ITERATIONS + 1):
        samples = []
        for _ in range(GROUP_SIZE):
            hg = policy_h.sample(rng)
            mg = policy_m.sample(rng)
            reward = evaluate_config(hg, mg, TRAIN_TASKS)
            samples.append((hg, mg, reward))

        group_mean = sum(r for _, _, r in samples) / len(samples)
        h_updates, m_updates = [], []
        for hg, mg, reward in samples:
            advantage = reward - group_mean
            h_updates.append((hg, advantage))
            m_updates.append((mg, advantage))
            replay_buffer.append({
                "iter": iteration, "harness_gene": hg, "model_gene": mg,
                "reward": reward, "advantage": advantage,
            })

        policy_h.update(h_updates, lr=LR)
        policy_m.update(m_updates, lr=LR)

        if iteration == 1 or iteration % 6 == 0:
            trace._emit(
                f"    round {iteration:>2}  group-mean reward {group_mean:.3f}   "
                f"harness[{policy_h.as_str()}]   model[{policy_m.as_str()}]"
            )
            trace.pause()

    trace.section("shared replay buffer — both tracks read the same entries")
    trace.note(f"{len(replay_buffer)} entries logged; both tracks pulled their advantage from the same rows")
    for entry in replay_buffer[-4:]:
        trace.note(f"  iter={entry['iter']:<3} harness={entry['harness_gene']:<10} "
                   f"model={entry['model_gene']:<10} reward={entry['reward']:.2f} "
                   f"advantage={entry['advantage']:+.2f}")

    best_h = policy_h.genes[max(range(3), key=lambda i: policy_h.logits[i])]
    best_m = policy_m.genes[max(range(3), key=lambda i: policy_m.logits[i])]

    trace.section("converged policy vs. uniform baseline, evaluated on HELD-OUT tasks")
    trace.note(f"final harness policy:  {policy_h.as_str()}")
    trace.note(f"final model policy:    {policy_m.as_str()}")
    trained_reward = evaluate_config(best_h, best_m, HELD_OUT_TASKS)
    trace._emit(f"    argmax joint config: harness={best_h!r} model={best_m!r}")
    trace.verdict(
        trained_reward > baseline_reward,
        f"held-out reward: uniform baseline {baseline_reward:.3f}  ->  trained policy {trained_reward:.3f}",
    )

    trace.takeaway([
        "Co-evolution means two learners updating from one shared signal, not two isolated loops.",
        "GRPO needs no absolute reward model: reward minus the sampled group's own mean is enough.",
        "The winning joint config (best context strategy + best response style) only emerges "
        "because both tracks train against the SAME sampled groups, round after round.",
    ])


if __name__ == "__main__":
    main()
