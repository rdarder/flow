# Brainstorm → intent.md Workflow

You are helping the user distill a brainstorming session into a `intent.md` document.

**Context:** The user just finished a brainstorming conversation (with you or another agent). The output is raw: mental models, explored options, false starts, detailed specs, and tentative decisions. Your job is to extract the **actionable core** without losing hard-won insights.

---

## Goal

Produce a `intent.md` that:
- Describes **what behavior** the user wants (not how to implement it)
- Captures **constraints** discovered during brainstorm (hard limits, not preferences)
- Records **decisions** made (rejected/adopted ideas with rationale)
- Marks **unknowns** (questions to answer through implementation)
- Contains **no project task ordering** (no phases, no implementation sequences)
- **Keeps algorithmic sequences** (e.g., "train generalist before specialists" — inherent to the technique)

---

## Process

### 1. Read the Brainstorm

Read the full brainstorming transcript. Identify:
- The **core wish** (what problem are we solving?)
- **Constraints** (hard limits: hardware, math, compatibility)
- **Decisions** (what was explicitly adopted/rejected/deferred)
- **Unknowns** (what's still unclear, needs experimentation)
- **Implementation details** (specific structures, protocols, file changes)

### 2. Ask Clarifying Questions

Before drafting, ask the user:

**Priority Questions (always ask):**
1. *"What's the single sentence goal? If we achieved only one thing, what would it be?"*
2. *"Are there any hard constraints I should preserve? (e.g., hardware limits, compatibility requirements)"*
3. *"What decisions from the brainstorm are final vs. tentative?"*
4. *"What's still unknown or needs experimentation?"*

**Optional Questions (if brainstorm was complex):**
5. *"Should I split this into multiple intent.md files? (e.g., 'MoE architecture' + 'MoE training' as separate epics)"*
6. *"Is there a detailed spec that should stay in a brainstorm/ file for reference?"*

### 3. Draft intent.md

Use this structure:

```markdown
# Changes: [Epic Name]

## Goal
[1-3 sentences: what behavior we want, not how]

## Core Tension
[Optional: why this is hard — conflicting requirements, trade-offs]

## The Bet
[Optional: why we think this approach will work — 1-2 sentences]

## Constraints
- [Hard limit 1: e.g., "$10 NPU budget"]
- [Hard limit 2: e.g., "No transformers"]
- [Implementation constraint: e.g., "DW2 no activation — breaks subtraction alignment"]

## Decisions
| Idea | Status | Why |
|------|--------|-----|
| [Idea name] | ✅ Adopted / ❌ Rejected / ⚠️ Deferred | [1-line rationale] |

## Unknowns / Ablation Questions
- [Question 1: e.g., "Optimal expert count: 16 vs 32 vs 64"]
- [Question 2: e.g., "Diversity loss weight: 0.01 vs 0.1"]

## Key Insights
[Optional: mathematical/algorithmic justification — 2-4 bullet points]

## Numbers to Preserve
- [Fixed parameter: e.g., "16D embeddings"]
- [Fixed parameter: e.g., "3 pyramid levels"]

## References
- `brainstorm/{topic}.md` — Full mental model and detailed spec (optional)
```

### 4. Review with User

Present the draft. Ask:
- *"Did I miss any constraints or decisions?"*
- *"Is anything framed as a plan that should be a behavior?"*
- *"What should I cut or expand?"*

Iterate until the user approves.

---

## Rules

### ✅ Include in intent.md

- **Behavioral goals** ("embeddings that adapt to scene changes")
- **Hard constraints** ("$10 NPU budget", "no padding")
- **Decisions with rationale** ("Rank-1 experts adopted — 75% parameter reduction")
- **Unknowns** ("optimal expert count: will ablate 16/32/64")
- **Key insights** ("sum of Rank-1 matrices can achieve full rank")

### ❌ Exclude from intent.md

- **Project task ordering** ("First implement X, then Y", "Phase 1 → Phase 2")
- **File-by-file implementation plans** ("Add MoESettings to settings.py")
- **Current state descriptions** (that's `progress.md`)
- **Ablation priorities** (discovered during sessions, not pre-planned)
- **Step-by-step protocols** ("Initialize via SVD: 1. Decompose, 2. Extract...") — unless algorithmically required

### ⚠️ Reframe

| Instead of... | Write... |
|---------------|----------|
| "First build infrastructure, then integrate" | (remove — project planning) |
| "Training Protocol: Phase A → Phase B" | "Training has two stages: generalist warmup, then specialist divergence" (keep — algorithmic) |
| "Add GatedPointwiseMoE to model.py" | "Pointwise layers use Mixture of Experts with Rank-1 factorization" |
| "DW2 initialized as Gaussian kernel" | "DW2: Gaussian initialization (learnable)" |

---

## Splitting Heuristic

If the brainstorm covers multiple **independent** changes, suggest splitting:

**Split when:**
- Two changes can be implemented in any order
- One change doesn't depend on the other's behavior
- They affect different subsystems

**Keep together when:**
- Changes are tightly coupled (e.g., MoE architecture + MoE training)
- One change's behavior depends on the other
- They share the same success metric

---