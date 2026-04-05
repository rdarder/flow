# Memento Workflow

You are an AI coding assistant. This document defines how we collaborate on software changes.

---

## Concept

We build software through **isolated sessions**. Each session is a complete work cycle: you wake up with no memory of prior sessions, we make progress on a coherent slice of changes, and we commit. Then the session ends—hard cut.

We maintain two documents on this workflow:
- **Changes** — Ideas for how the system should behave (not tasks, not a plan)
- **Progress** — Current system state after all sessions (rewritten each time, not a changelog)

**Core principles:**

1. **No upfront planning** — Decisions are made with maximum context, right before implementation. We don't prepare for future work; we do the mainline behavior now.

2. **Operational changes only** — Every session must change system behavior. This includes "sideways" refactors that restructure the system while remaining functional. Avoid "prep work" — infrastructure that doesn't change behavior yet.

3. **State over history** — Progress describes what the system does now, not what changed. If behavior evolved X→Y→Z, only Z is documented. This forces reading both files to understand what's remaining.

4. **Bounded scope** — Vertical slices push toward large scope; large scope fails. Aggressively cut optionality while keeping the slice operational. Small enough to complete cleanly, large enough to matter.

---

## Recipe

### 1. Understand

Read `ARCHITECTURE.md`, `changes.md`, `progress.md`, `NAVIGATION.md` and relevant code. Understand the gap between current state and desired state. Ask questions if anything is unclear—update docs if needed.

### 2. Scope

We want to decide what's the smallest change we can make that affects the system behavior in the direction of the expected changes in `changes.md`. 

- The change must not be elusive or just preparatory. Ideally it changes behavior.
- We don't propose changes that result in dead code, waiting for some future change to integrate it. Always integrated changes, working on mainline.
- Changes must be complete in one session. We can decide what to leave out, but the changes need to be consequential.

There are two driving forces: a behavior change will often lead to changes too large. When forced to chose a small change one often selects an inconsequential change. Choose the behavioral change and chop down the requirements until it's very small. When in doubt, you're doing too much in a single session, so reduce the scope while maintaining some significance.


Discuss and adjust with me. I'll let you know when we're done scoping the change.

### 3. Rehearse

Here we think about how would we go about writing the changes. Mostly 
- where are the affected components, 
- how would we test the changes
- do we want need to reorganize some of these componengs because of the upcoming changes? 
- imagine how the code would look like.

Here we're rehearsing what we'll face when you actually implement the changes. We want to anticipate some of the problems and decisions you'll face. We want to have enough anticipation here so that when you're out there writing the changes there's little surprise. Sometimes this rehearsing leads to us changing the scope of what we're changing (because it was too complex or because we'll be better off splitting it differently).

Clarify implementation details: edge cases, failure modes, integration points. Adjust scope if needed. Agree on what "done" looks like.

### 4. Implement

- Write code, docs, tests
- User reviews, you iterate
- Update `progress.md` to reflect new state (edit as a whole, don't append)
- Session ends with a git commit
- Repeat the git commit summary in your response so I know what was done right away.

---

## Conventions

### Git Commit Messages
- Single sentence title: behavior-focused, third person
- Optional bullets: behavior details, implementation notes
- Avoid lists of synthetic artifact changes: "created module something.py; added two tests". 
- If relevant, mention how to experience the changes (running with a certain configuration, or how to tell that something changed by using the software)

### Progress.md
- Describe behavior, not artifacts
- Rewrite to reflect current state (anti-changelog)
- The goal of `progress.md` is that in a new session, you can read the code, changes.md and progress.md and determine what is still remaining from changes.md

