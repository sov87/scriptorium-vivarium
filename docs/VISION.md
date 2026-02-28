# Vivarium & Scriptorium: Project Vision

## What We're Building

Vivarium is an attempt to do something that has never quite been done before: a historically grounded, open-ended simulation of an ancient city that you can walk through as a first-person observer. Not a game with quests and health bars. Not a textbook with hyperlinks. Something closer to time travel — the ability to stand in the Forum Romanum on a morning in 30 BC, watch a freedman argue over a scroll near the Temple of Saturn, follow a beggar into the Subura, and have every concrete detail of what you see trace back to a real ancient source.

The core bet is this: if you feed a language model not just its training data but a carefully curated, locally-hosted database of primary sources — Livy, Tacitus, Martial, Juvenal, Pliny, Vitruvius, Platner & Ashby's topographical dictionary, Pleiades geographic data — and then enforce that every factual claim it makes must cite a real passage from that database, you get something meaningfully different from asking ChatGPT about ancient Rome. You get a simulation that fails gracefully rather than confabulating confidently. You get a world bounded by evidence.

The infrastructure behind this is Scriptorium: a local-first archival pipeline that ingests historical texts from open sources (PerseusDL, Internet Archive, Open Greek and Latin), converts them to canonical JSONL with provenance records, indexes them in SQLite with full-text search, and serves them to Vivarium's retrieval engine. Every corpus has rights documentation. Every ingested file has a SHA256 hash. Nothing goes into the database that can't be traced back to its source.

Vivarium consumes this database and wraps it in a two-pass generation architecture: first a planning pass that selects evidence and maps out narrative beats, then a rendering pass that turns those beats into present-tense prose. The citation gate is mechanical — if a factual claim can't be matched as a literal substring in a retrieved segment, it fails. This is what separates the project from a fancy hallucination machine.

The long-term vision is a fully navigable ancient Rome — every hill, every street, every named building, every social class represented — where you can wander for hours and have the experience feel genuinely inhabited. Where talking to a merchant in the Macellum surfaces real prices from Diocletian's Edict. Where the smell of the Cloaca Maxima is grounded in Pliny's complaints about it. Where a senator you overhear near the Rostra is arguing about a real piece of legislation. Beyond Rome, the architecture should eventually support other cities and periods: Carthage during the Punic Wars, Constantinople in late antiquity, Anglo-Saxon Winchester.

---

## Challenges Faced

**The retrieval ceiling.** Full-text search on Latin and Greek is fundamentally limited because inflected languages defeat keyword matching. "Forum" matches "forum" but not "fori." "Tiberius" won't find "Tiberii." The FTS5 tokenizer with diacritic removal helps, but it's a partial solution. Vector retrieval (semantic search) is the real fix, and the infrastructure for it exists in Scriptorium — it just hasn't been promoted to active use yet.

**Pleiades drowning everything.** The Pleiades gazetteer contains millions of geographic place name entries, including modern streets and administrative regions. When included in naive full-text search, it dominates every query. The fix was to filter it out of narrative retrieval entirely and use it only for location lookups — but discovering this required debugging why "street crowd market" returned zero Latin literature hits and instead surfaced "F Street" and "Dere Street."

**The thinking token problem.** Qwen3 models run an internal reasoning process before generating output. This reasoning is invisible to the user but consumes thousands of tokens, leaving almost none for the actual JSON response. At "8192 max tokens," the model was effectively working with fewer than 2000 tokens of output budget. The fix — prepending `/no_think` to every user message — was not documented anywhere obvious and required searching to confirm.

**Schema validation wars.** The generation pipeline validates the LLM's output against a strict JSON schema. The schema was originally calibrated for a larger model with more consistent output. With a local 30B model, validation failures burned all four retry attempts on outputs that were substantively correct but had one list with five items instead of six. Relaxing the minimums from "must have 6-12 items" to "must have 3-12 items" fixed most failures without compromising quality.

**The registry swap antipattern.** For over a year, building a subset database (like the Rome core DB) required temporarily replacing `docs/corpora.json` with a subset registry, running the build, then restoring the original. A crash mid-build would corrupt the repository state and leave it in an unknown configuration. Implementing `--registry-override` as a proper CLI argument eliminated this entirely, but it required understanding the full call chain from `__main__.py` through `build_sqlite_db.py` before touching anything.

**Source density as the hard ceiling.** The simulation can only be as good as the evidence it can retrieve. Platner & Ashby's 1841 entries transformed location-specific narration — the Subura scene that mentions the Spino brook, the Clivus Suburanus, and the domus of C. Sestius by name draws directly on entries that didn't exist in the database the week before. But daily life texture — what people said to each other, what they ate, what they worried about — requires different sources: Martial's epigrams, Petronius, Apuleius, the papyri. The model's tendency to generate atmospheric loops ("the morning sun...") rather than advancing scenes is partly a prompt engineering failure and partly a retrieval failure. When there's no evidence for what the user asked about, the model recycles what it has.

**NPC interaction without a drama engine.** Asking to "talk to the beggar" generates more environmental description instead of dialogue. This is because the retrieval system knows how to find topographic evidence but has no concept of social scripts, conversational register, or what a beggar in the Subura would actually say. The fix requires both better sources (social history, comedy, epistolary literature) and better prompt engineering to distinguish between "describe this place" and "this person is speaking to you."

**Vibe-coded infrastructure risk.** The entire project was built without formal software development experience, using AI assistance for code generation. This produces surprisingly capable code — the provenance architecture and citation gate in particular are better than most RAG demos built by professional developers — but it also produces subtle bugs that compound silently across long sessions. A dead backfill loop, a validator that rejected valid JSON because the LLM added a "reasoning" key, a missing `return` statement after a fatal error call. The mitigation is discipline: short sessions, read before touching, commit after every working state.

---

## Where It Stands

The foundation is solid. The pipeline is reproducible. The citation architecture is defensible. The simulation produces genuine atmosphere grounded in real sources — a beggar muttering in the Subura while a child presses against a wall and someone whispers "Argentariae" from the alley above is not a bad approximation of what a fly on the wall in ancient Rome might have witnessed, and it cites Platner & Ashby for the street name.

The gap between "works" and "feels like open-world time travel" is still substantial. But the path is clear: more sources, better NPC dialogue prompting, persistent world state, and eventually vector retrieval to handle the morphological complexity of the languages. None of those are architecture changes. The hard problem — grounding a dynamic simulation in primary sources with mechanical citation enforcement — is already solved.
