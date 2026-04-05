# Architecture Interview Session

You are a senior engineer joining this project to continue implementation. You need to understand the system deeply to make good decisions going forward.

## Your Task

Interview the project author to create an **ARCHITECTURE.md** document that will serve future implementors (including yourself).

## Document Goals

This document is **agent-first** but also useful for humans. It should:

1. **Minimize perplexity** - A new agent reading this should understand what this project is about and why it's built this way
2. **Be timeless** - Focus on decisions that won't change frequently. It's okay if slightly outdated; should not be detrimental
3. **Capture rationale** - Why we chose X over Y, what problems we're solving, what constraints shaped the design
4. **Include entry points** - How to run the system (training, inference, tests)

## What to Ask About

### Big Picture
- What problem is this solving? What's the core approach?
- What are the main components and how do they connect?
- What's implemented vs. planned vs. abandoned?

### Design Decisions
- Why this architecture? What alternatives were considered?
- What constraints shaped the design (performance, hardware, algorithmic)?
- What are the key invariants or non-obvious couplings?
- What's the data flow during training/inference?

### Implementation
- What are the entry points? (CLI commands, main functions)
- How do you run training? How do you run inference?
- How do you run tests?
- What's the directory structure and what lives where?

### Esoteric Details
- Any "we do it this way because..." moments?
- Known limitations or fragile parts?
- Hyperparameters that matter vs. ones that don't?
- Anything that's not obvious from reading the code?

## Process

1. **Ask questions** - Start broad, then drill into specifics. Don't rush.
2. **Take notes** - Keep track of key points as you go.
3. **Synthesize** - Once you have the full picture, compile into a coherent document.
4. **Validate** - Confirm with the author that the document captures the essence correctly.

## Document Structure (Suggested)

```markdown
# Project Name: Architecture

## Problem Statement
What we're solving and why.

## Approach
High-level strategy and key ideas.

## System Overview
Main components and how they connect.

## Design Decisions
Key choices and rationale (the "why" behind the code).

## Components
- Component A: What it does, why it exists
- Component B: What it does, why it exists

## Entry Points
How to run everything (training, inference, tests).

## Known Limitations
What doesn't work, what's fragile, what's deferred.
```

## Notes

- Don't document what's obvious from code alone (class names, function signatures, simple mechanics)
- Do document what code can't express (motivation, trade-offs, constraints)
- Mechanics ARE appropriate when describing algorithms or model behavior that's not self-evident from the code structure
- Keep it concise but complete. No fluff.
- If something is "we might do this later," mark it as future work, not current architecture

Start by introducing yourself and asking the first big-picture question.
