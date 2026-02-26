# AI Agent Guide — Reject Inference Project

> **Purpose**: This is the single source of truth for any AI agent (or human) working on this project.  
> Read this file **first** in every new conversation.

---

## 1. Project Identity

| Field | Value |
|-------|-------|
| **Title** | Reject Inference Methodological Comparison |
| **Type** | Deep Learning MBA Final Exercise (academic / learning) |
| **Dataset** | Kowope Mart — Nigerian Credit Risk (DSN 2020) |
| **Goal** | Compare VIME-based hybrid reject inference (Route A) vs classical CatBoost baseline (Route B) |
| **Owner** | Student (final approver of all major decisions) |

---

## 2. Approval Gates — MANDATORY

> [!CAUTION]
> **No agent may change any of the following without explicit written approval from the project owner.**

| Gate | What requires approval |
|------|----------------------|
| **Architecture** | Any change to model architecture (layer sizes, heads, loss functions) |
| **Hyperparameters** | Changes to `config.py` values (learning rates, epochs, thresholds) |
| **Data pipeline** | Changes to feature engineering, imputation strategy, or encoding |
| **Evaluation** | Adding/removing metrics, changing hold-out split, changing comparison logic |
| **Dependencies** | Adding new packages to `requirements.txt` |
| **ALL NEW CODE** | ⚠️ **PAIR-PROGRAMMING REQUIRED**: See Section 2.1 below. |

---

## 2.1 Pair-Programming Protocol (MANDATORY)

> [!CAUTION]
> **Every line of code** written in this project must follow this protocol.

### The Process

For **each line or block of code**:

1. **EXPLAIN** — Agent describes what the code does and why it's needed
2. **DISCUSS** — Owner asks questions or suggests alternatives
3. **APPROVE** — Owner explicitly approves (e.g., "approved", "yes", "ok", "go ahead")
4. **WRITE** — Only then does the agent write the code to the file

### Why This Matters

This is a **learning exercise**. The goal is not just working code, but the owner's deep understanding of every component. Auto-generated code without explanation defeats the purpose.

### Documentation Per Code Block

```python
# Purpose: [What this block accomplishes]
# Why: [Why this approach was chosen]
# Approved: [Date]
```

### What Counts as "Approval"

| Approved | Not Approved |
|----------|--------------|
| "yes" | (no response) |
| "approved" | "maybe" |
| "ok, go ahead" | "I'm not sure" |
| "looks good" | "let me think" |

**When in doubt, ask again.**

---

### How to request approval

1. Write a brief **Proposal** in `docs/proposals/YYYY-MM-DD_<topic>.md`
2. Include: *What*, *Why*, *Alternatives considered*, *Impact*
3. Tag with `STATUS: PENDING` → Owner reviews → `STATUS: APPROVED` or `STATUS: REJECTED`

---

## 3. Multi-Agent Parallel Work Protocol

This project is designed for 3–5 agents to work in parallel from day 1.

### Work Domains (parallelizable)

| Domain | Scope | Can work independently? |
|--------|-------|------------------------|
| **A: Preprocessing** | `src/preprocessing/`, `data/` | ✅ Yes |
| **B: VIME / DL** | `src/models/vime/` | ✅ Yes (after preprocessing API is stable) |
| **C: Route A Hybrid** | `src/models/route_a_hybrid.py` | ✅ Yes (after VIME embeddings API is stable) |
| **D: Route B Baseline** | `src/models/route_b_classical.py` | ✅ Yes (after preprocessing API is stable) |
| **E: Evaluation** | `src/evaluation/`, `run_experiment.py` | ✅ Yes (mock data for dev) |

### Agent Rules

1. **Read `AI_GUIDE.md` first** — always, every conversation
2. **Read `docs/WORKING_PLAN.md`** — find your current phase and next step
3. **Check `docs/decision_log.md`** — know what's been decided
4. **Never modify files outside your domain** without coordination
5. **Log every decision** in `docs/decision_log.md`
6. **Create handover doc** before ending any conversation (see §5)

---

## 4. Documentation Rituals

### Code Documentation Standard
- Every `.py` file: module docstring explaining purpose and roadmap step
- Every function: Google-style docstring with Args, Returns, Examples
- Every non-obvious line: inline comment explaining *why*, not *what*
- Type hints on all function signatures

### Decision Log (`docs/decision_log.md`)
After every significant choice, append:
```markdown
## [DATE] — [DECISION TITLE]
**Context:** Why this came up
**Decision:** What was chosen
**Alternatives:** What else was considered
**Rationale:** Why this option won
**Approved by:** Owner / Agent
```

### Phase Completion Reports (`docs/reports/phaseXX_report.md`)
After each phase in the working plan:
- Summary of what was done
- Key metrics / validation results
- Lessons learned
- Blockers for next phase

---

## 5. Auto-Summarization & Handover Protocol

> [!IMPORTANT]
> **Every conversation MUST end with a handover document** if work was done.

### When to stop and create handover
1. **Major milestone reached** — a phase or sub-step is complete
2. **Conversation is getting long** — approaching ~30+ back-and-forth exchanges
3. **Major improvement achieved** — model metric improved significantly
4. **Blocked** — waiting for owner approval or data

### Handover Document Template (`docs/handovers/YYYY-MM-DD_handover.md`)
```markdown
# Handover — [Date] — [Topic]

## What was accomplished
- Bullet list of completed items

## Current state
- What phase/step we're at in WORKING_PLAN.md
- What files were modified (with brief description)

## Test results
- Any validation/QA results

## Next steps (for the next agent/conversation)
1. Exact next action
2. ...

## Open questions / blockers
- Items needing owner decision

## Key files to read first
- List of files the next agent must read
```

---

## 6. Project Folder Structure

```
final_ex/
├── AI_GUIDE.md                    ← YOU ARE HERE (read first!)
├── README.md
├── requirements.txt
├── config.py
├── run_experiment.py
│
├── docs/
│   ├── WORKING_PLAN.md            # Detailed step-by-step plan
│   ├── decision_log.md            # All decisions and rationale
│   ├── proposals/                 # Pending approval requests
│   ├── reports/                   # Phase completion reports
│   └── handovers/                 # Conversation handover docs
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   │   └── vime/
│   └── evaluation/
│
├── tests/                         # Unit & integration tests
│
└── outputs/
    ├── models/
    ├── figures/
    └── results/
```

---

## 7. Quality Standards

- **Reproducibility**: Every experiment must be reproducible via `SEED` in `config.py`
- **No magic numbers**: All parameters live in `config.py`
- **Tests**: Every module has corresponding tests in `tests/`
- **Academic rigor**: Methods must be justified with references where possible
