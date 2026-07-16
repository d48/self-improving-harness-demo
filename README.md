# Self-Improving Agent Harnesses — Demos & Slides

A slide deck and three runnable demos teaching the concepts from
**Ben Dickson, ["A Primer on Self-Improving Agent Harnesses"](https://bdtechtalks.substack.com/p/a-primer-on-self-improving-agent-harnesses) (TechTalks, Jul 2026)** —
how AI agents can automate the optimization of their own harness (scaffolding)
by analyzing their execution traces and rewriting targeted code and prompts,
instead of relying on expensive model retraining.

This repo is a companion to
**[harness-engineering-demo](https://github.com/d48/harness-engineering-demo)**,
which covers the foundational "what is a harness" material and the classic
Self-Harness propose → evaluate → accept loop. This repo goes further into the
newer material: composable harness architecture, the four-stage AEGIS
evolution engine, and harness-model co-evolution via GRPO.

## Quick start

Requires only Python 3.10+ (stdlib only — no packages, no API keys, no network).
`npm install` is not required — `package.json` just wraps the Python entry point
for convenience.

```bash
python3 demo/run.py            # interactive menu
python3 demo/run.py 1          # run one demo
python3 demo/run.py all        # run everything
python3 demo/run.py all --fast # skip the dramatic pauses (smoke test)
```

or, equivalently, via npm:

```bash
npm run demo         # interactive menu
npm run demo:1        # run demo 1 (demo:2, demo:3 also available)
npm run demo:all      # run everything
npm run demo:all -- --fast   # extra flags go after `--`
```

Open `slides/self-improving-harness-slides.html` in a browser for the
presentation (arrow keys to navigate, `N` toggles speaker notes, `O` shows an
overview grid), or launch it with:

```bash
npm run slides
```

### Public URL (GitHub Pages)

A workflow at `.github/workflows/pages.yml` deploys the `slides/` folder to
GitHub Pages on every push to `main`. One-time setup (not scriptable from
outside the GitHub UI): in the repo, go to **Settings → Pages → Build and
deployment → Source**, and select **GitHub Actions**. After that, every push
to `main` that touches `slides/` redeploys automatically, and the deck is
available at:

```
https://<owner>.github.io/<repo>/
```

(`slides/index.html` redirects the bare URL straight to
`self-improving-harness-slides.html`.)

## The demos

Each demo is a small, **real** mechanism — genuine pipeline dispatch, genuine
clustering and static source-code analysis, genuine policy-gradient
optimization — not a script that prints narration. Any "model" or data source
inside a demo is deterministic (so the walkthrough is reproducible live on
stage), but everything around it — the pipeline, the gates, the math — is
real and computed live.

| # | Demo | Article concept | What actually happens |
|---|---|---|---|
| 1 | `demo1_composable_harness.py` | HarnessX: composable optimization architecture | A 4-hook pipeline (context, tool-select, respond, critique) hot-swaps its context-assembly processor; a contract-breaking processor is rejected via a live canary call; call-count instrumentation proves the untouched hooks never moved |
| 2 | `demo2_aegis_pipeline.py` | HarnessX: the AEGIS evolution engine | Digester → Planner → Evolver → Critic repairs a real command parser; a memorized "fix" passes its isolated unit test and doesn't regress held-out — only the Critic's static source-code scan catches it |
| 3 | `demo3_coevolution_grpo.py` | Harness-model co-evolution | A harness-track and a model-track policy share one replay buffer and update from group-relative advantage (GRPO) alone; the converged joint config lifts held-out reward from 0.56 to 1.00 |

## Repo layout

```
demo/
  mini_harness/         # shared teaching library
    trace.py            # terminal rendering: THOUGHT / TOOL / OBSERVE / verdicts
  demo1..demo3_*.py      # the three demos
  run.py                # menu / runner
slides/
  self-improving-harness-slides.html  # self-contained deck with diagrams (no dependencies)
scripts/
  open-slides.js         # cross-platform "open the deck in a browser" for `npm run slides`
```

## Presenting this

Suggested flow for a ~20-minute engineering talk:

1. Slides 1–5: the thesis, the base harness architecture, and the classic
   Self-Harness loop (weakness mining → proposal → validation) as background —
   point at the sibling repo's demo 4 for a live version of that exact loop.
2. Slides 6–8: the three demos in this repo — composable architecture, AEGIS,
   co-evolution — running each demo as you reach its slide.
3. Slides 9–10: the paradigm shift from per-session loop engineering to
   meta-runtime continual learning, then takeaways.

Every demo ends with a `TAKEAWAY` block that matches the corresponding slide's
speaker notes, so the deck and the terminal reinforce each other.
