# Brainstorm: Structured Logging with Dashboard Presentation

## The Wish

**Effortless observability** — emit rich structured data at the source, have it present appropriately depending on context (console, web dashboard, etc.).

Current pain point: TensorBoard + line-by-line console logging makes it hard to:
- See current state at a glance
- Follow task lifecycles (start → progress → done/error)
- Keep important metrics visible while other logs scroll by

---

## Core Tension

**Push-based logging** (emit data, forget about it) vs. **Pull-based dashboards** (data routed to specific UI locations)

Want both:
1. Simplicity of `log("train_step", {...})` — decoupled, just dump structured data
2. Control of "this number goes in this box" — routing to specific UI real estate

**Resolution:** Separate **Producer API** from **Presenter API** via a limited presentation model.

---

## Mental Model

```
Producer → Events (tagged, structured) → Presentation Model → Presenter (TUI/Web/Console)
```

**Key insight:** The producer knows nothing about how data is displayed. The presenter knows nothing about the business logic. They meet at a structured event contract.

---

## Producer API Patterns

### Pattern 1: Simple Structured Log
```python
log("train_step", step=..., loss=..., elapsed=...)
```

### Pattern 2: Task Lifecycle (context manager)
```python
with task("running validation") as t:
    t.progress("loading data")  # intermediate state
    # auto-finishes with ✓ or ✗ on exit
```

### Pattern 3: Latest Value (gauge-style)
```python
log("metrics/val_loss", value=0.42)  # overwrites previous
```

---

## Event Structure

Events have:
- **Tag/path:** `"training/metrics/loss"` (hierarchical)
- **Payload:** `dict` of values
- **Timestamp/step:** implicit or explicit
- **Type hint (optional):** `gauge | counter | stream | status`

---

## Presentation Model

The contract between producer and presenter:

- **Declarative layout spec** — defines screen regions (like flexbox)
- **Bindings** — map event tags to regions
- **Component types:**
  - Singleton number/text (updates in place)
  - Scroll stream (appending log lines)
  - Status indicator (✓/✗/⏳)
  - Alert/notification area

---

## Presenter Implementations

### Console (TUI)
- Default: nice line-oriented formatting (colors, alignment)
- Advanced: panel-based TUI with regions
- Backend: likely `textual` or `rich` (not raw ANSI)

### Web
- Same event stream, different renderer
- Additional: plotting/analytics capabilities

---

## Dashboard Builder (Nice-to-Have)

Interactive TUI component for defining layouts on the fly:
- Keystrokes to move/resize regions
- Toggle component styles
- Save layout for reuse

**Question:** Where does layout get saved?
- Config file?
- Per-project? Per-run?
- Wild idea: embedded in log stream itself?

---

## Constraints Discovered

1. **Terminal limitations:** Fixed character grid, no true regions (must redraw whole screen), limited colors
2. **Concurrency:** Training loops may have parallel workers — need to handle concurrent updates to same panel
3. **Backpressure:** What if events arrive faster than render? (Drop? Buffer? Throttle?)

---

## Unknowns

1. **Event routing granularity:** Does producer know about tags, or is there indirection layer?
2. **Multi-run support:** Should API support `run_id` from start, or defer to "analytics chapter"?
3. **Optimal event cardinality:** What's the right balance between "everything logged" vs. "only important things"?

---

## Explored Options

| Idea | Status | Why |
|------|--------|-----|
| Line-only logger with nice formatting | ✅ Core | Good enough for quick tasks, works everywhere |
| Panel-based TUI | ✅ Core | Solves "at a glance" monitoring |
| Dashboard builder UI | ⚠️ Deferred | Nice-to-have, depends on basics working |
| Web presenter | ⚠️ Deferred | "Another chapter" — analytics focus |
| Raw ANSI codes | ❌ Rejected | Too fragile; use `textual`/`rich` |
| Imperative presenter API | ❌ Rejected | Declarative is more composable |

---

## False Starts

- **Persistent panels in console:** Initially thought this was the main feature, but realized console is for quick tasks. Real value is in the separation of concerns (producer vs. presenter).
- **Post-run analytics:** Tempting to design for this from start, but scope creep. Focus on real-time observation first.

---

## Next Steps (If Continued)

1. Define minimal event structure
2. Implement line-oriented console presenter
3. Add TUI presenter with declarative layout
4. Test with real training loops
5. Iterate on dashboard builder

---

## Related

This could improve observability in `barevision.embeddings.training` and `barevision.checkpointer` — both currently use basic logging/TensorBoard.
