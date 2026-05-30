hn# FDE Take-Home Playbook (24h, mid-level, 30-min follow-up)

A working agreement for the 24-hour Forward Deployed Engineer take-home.
Core principle: **I must own and be able to defend every architectural choice** —
the 30-min follow-up tests whether I can *think and talk* about the system, not
whether I memorized the code. Build incrementally, understand as we go.

---

## Part 1 — Kickoff prompt (paste into a fresh session with the spec)

```
I'm doing a 24-hour FDE take-home. Role: mid-level Forward Deployed Engineer.
There's a 30-min follow-up interview where I demo it and defend my decisions,
so I must own and understand every architectural choice — do NOT just build it for me.

Here is the task spec:
<paste spec>

Work with me in this order. Do NOT write feature code until I approve the design.

1. ANALYZE: Restate the real problem in your words. Who is the "customer" and
   what do they actually need? List every ambiguity / under-specified part, and
   for each, propose a default assumption + why. Flag what you think is secretly
   being tested.

2. DESIGN: Propose 1–2 architecture options (lean toward the simplest that fully
   solves it). For each: data model, key components, external APIs/SDKs, and the
   explicit SCOPE CUT-LINE (what we deliberately won't build). Recommend one and
   say why. Wait for my approval.

3. PLAN: Once I approve, give a file/module build order that gets a working
   end-to-end skeleton ASAP, then layers in quality. List files in build sequence.

4. BUILD: One module at a time. After each, stop and explain what it does and the
   one decision/trade-off in it, so I can review before we continue.

Keep a running DECISIONS.md as we go: each entry = decision, alternatives, trade-off.
That file becomes my README and interview prep. Ask me questions whenever the spec
is ambiguous instead of guessing silently.
```

---

## Part 2 — Phase checklist (control panel)

**Phase 0 — Solo (no AI), ~30 min**
- [ ] Read spec twice yourself first. Don't let AI anchor your thinking.
- [ ] Note your own gut take on the problem + 3 likely "hidden" eval criteria.

**Phase 1 — Analyze & Design (~1.5h)**
- [ ] AI restates problem; confirm it matches the spec's intent.
- [ ] Ambiguities listed → assumptions chosen (these go in the README).
- [ ] Architecture approved by *me*, scope cut-line agreed.

**Phase 2 — Skeleton first (next few hours)**
- [ ] Working end-to-end happy path exists, even if ugly. Run it.
- [ ] Commit. (Clean incremental commits = good story.)

**Phase 3 — Build out (bulk of time)**
- [ ] One module at a time, reviewed *as written* — never "understand later."
- [ ] Run frequently. Don't stockpile untested code.

**Phase 4 — Harden (last ~20%)**
- [ ] Tests on the critical path (signals seniority more than extra features).
- [ ] Obvious edge cases + error handling.
- [ ] README: how to run, assumptions, trade-offs, "with more time…".

**Phase 5 — Demo prep (last ~1–2h)**
- [ ] Rehearse the 2-min happy-path demo once, for real.
- [ ] Fill the interview one-pager (below).
- [ ] Leave a buffer — demos break on the day.

---

## Part 3 — Interview one-pager (fill as I build)

- **2-min demo script:** the happy path, rehearsed, that never crashes
- **Top 3–5 decisions + trade-off each:** "I did X because client need Y; trade-off is Z"
- **Assumptions I made:** the FDE money move — "spec didn't say X, I assumed Y because Z"
- **What I'd do with more time:** proves I scoped on purpose
- **One curveball rehearsed:** "if data 10×'d / went real-time / multi-tenant, I'd change ___"

---

## What an FDE task is really testing
- **Pragmatism / shipping** — works end-to-end and solves the real problem.
- **Ambiguity handling** — the assumptions I make and document *are* the test.
- **Communication** — can explain to a non-engineer stakeholder.
- **Integration glue** — wiring APIs/SDKs/data cleanly, not building a framework.

## When time runs low
Cut **scope**, not the harden/demo phases. A smaller thing I can demo and defend
flawlessly beats a bigger thing I can't.

---

## Architecture & scaling decision (build simple, design swappable)

**The rule:** ship the *simple* version, *design* the scalable one. Building the
distributed version (gateway + RabbitMQ + autoscaling workers + DB) for a take-home
is over-engineering — it's a NEGATIVE FDE signal, eats the time budget, and makes
the demo fragile. The maturity signal is "I knew not to, and I can say exactly when
I would."

**Default for HTTP-heavy / I/O-bound work:**
- Thread pool of workers (`ThreadPoolExecutor`) — publisher = pool, consumer = worker.
- Thread-safe in-memory store (dict + lock).
- Single process, runs with ONE command. Never breaks in the demo.
- (Python note: threads release the GIL during network I/O, so this is genuinely
  CORRECT for HTTP fan-out, not just a shortcut. CPU-bound → processes instead.)

**Make it swappable with 2–3 interfaces (no more — over-abstraction is its own trap):**

| Interface   | Methods                     | In-mem impl (ship this) | Prod swap (describe only) |
|-------------|-----------------------------|-------------------------|---------------------------|
| `TaskQueue` | enqueue / dequeue           | `queue.Queue`           | RabbitMQ                  |
| `Repository`| save / get / list           | dict + lock             | Postgres                  |
| `Worker/Pool`| submit / process           | `ThreadPoolExecutor`    | stateless worker procs    |

Business logic depends on the INTERFACE, never the concrete class. Only the in-mem
impls get wired up and tested.

**Bank the credit in the README** (worth more than 6h of Docker plumbing):
> Runs single-process with a thread pool + in-memory thread-safe store; satisfies the
> task and runs with one command. Built against `TaskQueue` / `Repository` interfaces,
> so production is a drop-in swap: `InMemoryQueue → RabbitMQ`, `InMemoryRepo → Postgres`,
> workers as stateless deployments behind a gateway, autoscaling on queue depth. Scope
> kept deliberately to a runnable prototype.

**Only build the scalable version if** the spec EXPLICITLY requires horizontal scaling
or persistence-across-restarts as acceptance criteria. Then it's the requirement, not
over-engineering.

**If I finish early:** harden → test → polish demo FIRST. Adding a second `Repository`
impl is the LAST bonus, never the priority.