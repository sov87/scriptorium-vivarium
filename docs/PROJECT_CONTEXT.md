# SCRIPTORIUM — Project Context (Living Document)
Last updated: 2026-03-01

## Core goal
Local-first, reproducible pipeline that ingests historical text corpora into a structured
SQLite database with:
- Canonical archival JSONL intermediate
- FTS + vector retrieval
- Optional LLM answer/gloss generation with audit-traceable artifacts
- Defensible provenance/rights discipline

Long-term scope (planned):
- Old English / Anglo-Saxon (current)
- Latin (Perseus / Open Greek & Latin where openly licensed)
- Ancient Greek (canonical-greekLit / Open Greek & Latin where openly licensed)
- Patristic / liturgical public-domain corpora (e.g., CCEL subsets)

Vivarium long-term goal: open-ended, dynamic "fly-on-the-wall" historical city simulation
(Rome first) powered entirely by primary sources. Users can walk through ancient Rome in
present-tense prose. Every concrete factual claim traces to a real source segment.
Similar to a MUD but non-gamey, non-scripted, grounded in evidence.

---

## Non-negotiables
- Local-first/offline-capable: core build/search runs without cloud dependencies.
- Canonical intermediate: archival JSONL is the "source of truth" (no silent mutation).
- Provenance and rights: every corpus must have explicit rights/provenance notes.
- Audit chain for AI: answer-search produces cites; answer-show resolves passages.
- CI must not call an LLM (seeded artifacts only).

---

## Environment (known working)
- Windows 11
- Scriptorium project root: F:\Books\as_project
- Primary venv: F:\Books\as_project\.venv_clean\
  Activate: .\.venv_clean\Scripts\Activate.ps1
- Vivarium repo root: F:\Books\scriptorium-vivarium
  Activate: .\.venv\Scripts\Activate.ps1
- Local LLM server: llama-server.exe (NOT LM Studio — double-loading issue)
  Endpoint: http://localhost:1234/v1
  Model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
  Path: C:\Users\ethan\.lmstudio\models\lmstudio-community\
        Qwen3-30B-A3B-Instruct-2507-GGUF\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
  Launch flags: --ctx-size 32768 --n-gpu-layers 99 --port 1234
  Hardware: RTX 5090 (32GB VRAM)
- Auto-start batch: F:\Books\scriptorium-vivarium\tools\start_vivarium.bat
  Starts llama-server, waits 30s, activates .venv, launches CLI.
- IMPORTANT: Qwen3 runs in "thinking" mode by default. Mitigated by prepending
  /no_think to every user message. Do NOT fine-tune to fix schema compliance.

---

## Key configs and DBs
- Rome core DB: F:\Books\as_project\db\vivarium_rome_core.sqlite (~200MB+)
- Rome core config: configs\vivarium_rome_core.toml
- Rome core registry: docs\registry_rome_core.generated.json
- Broad DB: db\vivarium_rome.sqlite (too broad; retrieval drift; avoid)

---

## Vivarium engine architecture (current — v11)

The engine is split into THREE files that must all be in the same directory (tools\):

### 1. rome_experience_cli_final.py (main engine)
Python logic: retrieval, validation, navigation, game loop.
Does NOT contain world data or prompts (externalized).

Key features:
- Two-pass LLM: Plan pass (physics/state) → Render pass (prose)
- Mechanical citation gate: claims must match literal substring in full_text
- Optional strict auditor pass (--strict-audit flag)
- Quota-based retrieval: topography / daily life / politics buckets
- Jaccard + TF-IDF hybrid reranking (weights in RetrievalConfig dataclass)
- Graph navigation with aliases + Haversine fallback via Pleiades
- Era system: republic/augustan/flavian/late_empire
- ERA FLEXIBILITY rule: model uses broad spans, not pinpoint dates
- Era gate on navigation: era_unlocked checked BEFORE state mutation
- Atomic transactions: state_snapshot + rollback on any failure
- Repair loop: assistant→user with /no_think; includes evidence packet
- Stochastic evidence spawner: every 5 turns injects random evidence as world event
- Neighbor fallback: 0 FTS hits → broadens to connected graph nodes
- Weather system: cycles every 12 turns; terms injected into FTS queries
- Long-term memory: dropped history turns summarized and appended to running_summary
  (injected into system_prompt; hard cap 300 words)
- Telemetry DB: per-turn profiling written to telemetry.sqlite (--debug only)

Config loading uses __file__-relative paths (not cwd-relative):
```python
_dir = os.path.dirname(os.path.abspath(__file__))
WORLD_DATA = load_config(os.path.join(_dir, "world_config.json"))
PROMPTS_DATA = load_config(os.path.join(_dir, "prompts.json"))
```

### 2. world_config.json
Contains: WORLDS dict, WEATHER_STATES, WEATHER_TERMS, TIME_TERMS.
Edit for new locations, eras, graph nodes, bias terms — no Python changes needed.

### 3. prompts.json
Contains: system_prompt, plan_prompt, render_prompt, auditor_prompt,
repair_prompt, suggest_prompt, summarize_prompt.
Uses {placeholder} syntax. Edit for model behavior tuning — no Python changes needed.

---

## How to run Vivarium
```
cd F:\Books\scriptorium-vivarium\tools
.\.venv\Scripts\Activate.ps1
python rome_experience_cli_final.py --db "F:\Books\as_project\db\vivarium_rome_core.sqlite" --base-url "http://localhost:1234/v1" --model "qwen3-30b-a3b-instruct-2507" --max-tokens 16000 --debug
```

---

## How to rebuild the Rome core DB
```
cd F:\Books\as_project
python -m scriptorium db-build --config configs\vivarium_rome_core.toml --registry-override docs\registry_rome_core.generated.json --overwrite
```

---

## Current Rome core DB sources
- PerseusDL canonical-latinLit (Livy, Cicero, Tacitus, Pliny, Martial, Juvenal,
  Vitruvius, Varro, Suetonius + more; CC BY-SA 4.0)
- Smith's Dictionaries (Antiquities, Geography, Biography; CC BY-SA 4.0)
- Platner & Ashby, Topographical Dictionary of Ancient Rome (1929; public domain)
  1841 entries. Every named place, building, monument, road.
- Pleiades gazetteer (EXCLUDED from narrative retrieval; used for :place only)

---

## Rome graph (world_config.json) — 28 nodes
Era-locked nodes (era_unlocked field):
- imperial fora: [augustan, flavian, late_empire]
- pantheon: [augustan, flavian, late_empire]
- colosseum: [flavian, late_empire]
- baths of caracalla: [late_empire]
- aurelian walls: [late_empire]

---

## In-game commands
:help, :era <n>, :goto <place>, :place <n>, :suggest [N],
:audit <id>, :time <n>, :state, :scope, :reset, :quit

---

## Current capabilities (milestones reached)
- Full pipeline: ingest → DB build → FTS/vector retrieval → AI answer with audit chain
- Registry override: --registry-override; docs/corpora.json never mutated
- Platner & Ashby 1841 entries confirmed working in evidence packets
- Vivarium v11: two-pass LLM, citation gate, era system, weather, long-term memory,
  externalized config/prompts, telemetry, era-locked navigation
- Session persistence with legacy save compatibility (setdefault guards)
- Repair prompt receives evidence packet (citation gate retries now viable)
- Verified working: fisher-boy at Tiber, quest chain at Forum, NPC dialogue gate

---

## Known gaps / next priorities

1. SOURCE INGESTION (highest value — binding constraint on quality)
   Priority order:
   - Tacitus Annals — Julio-Claudian/Flavian; senate + street level
   - Cicero letters (Ad Atticum, Ad Familiares) — Republic daily life firsthand
   - Sallust (Bellum Catilinae) — Republic political atmosphere 60-30 BC
   - Pliny the Younger — Flavian/Trajanic social texture
   - Livy — Republican institutions and buildings
   - Suetonius Twelve Caesars — all four eras, personal detail
   Note: Platner & Ashby tells you what buildings existed. Literary sources tell
   you what it felt like to be inside them.

2. :suggest ERA FILTER BUG
   Fallback path (no last_packet) samples world_anchors without checking era_unlocked.
   In Republic era suggests "take me to the baths of caracalla."
   Fix: filter world_anchors against current era's era_unlocked before sampling.

3. SENSORY OPENING REPETITION
   Model opens with light/sky/sun despite prompt instruction. Prompt rule added to
   prompts.json. Monitor to confirm effectiveness.

4. VECTOR RETRIEVAL
   FTS ceiling real for inflected Latin/Greek. Vec index exists in pipeline.
   Promote to active use when FTS becomes bottleneck.

5. ANGLO-SAXON WINCHESTER
   Second world config in world_config.json. Untested. Needs sources before quality.

---

## What NOT to do (learned the hard way)
- Do NOT swap docs/corpora.json for subset builds — use --registry-override
- Do NOT AND location tokens into topography FTS query
- Do NOT include Pleiades in narrative retrieval
- Do NOT rely on --max-tokens alone for Qwen3 — use /no_think
- Do NOT fine-tune the model to fix schema compliance
- Do NOT use LM Studio to serve model — use llama-server direct (double-load bug)
- Do NOT hardcode relative config paths — use __file__-relative resolution
- Do NOT solely rely on Gemini written scripts in one shot — verify with:
    head -3 file.py  (must start with #!/usr/bin/env python3)
    tail -5 file.py  (must end cleanly, no stray prose or quotes)
    grep -c "def " file.py  (check function count vs expected)
    Check for duplicate function definitions after every Gemini file
- Do NOT trust file size as quality metric — audit function presence directly
- Do NOT deploy Gemini output without additional review

---

## AI-assisted development protocol
- End of session: update PROJECT_CONTEXT.md, commit
- New chat: paste PROJECT_CONTEXT.md, say "Continue from next priorities.
  Do not assume anything about file contents — ask for files before changing them."
- Engine changes: give Gemini a scoped instruction → upload result to Claude for
  line-by-line audit → deploy only after audit passes
- World/prompt changes: edit world_config.json or prompts.json directly
- Commit after every working state

---

## Current status quick check
```
python -m scriptorium check-ai-fts --config configs\window_0597_0865.toml --json
```
