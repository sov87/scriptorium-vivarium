#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sqlite3
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# =============================================================================
# Configuration & World State Rules
# =============================================================================

COOLDOWN_SIZE = 220
MAX_EVIDENCE = 12
MIN_PACKET_LOCAL = 12
PROMPT_EXCERPT_CHARS = 420

TURNPLAN_RETRIES = 4
RENDER_RETRIES = 4

PLEIADES_CORPUS_ID = "viv_pleiades_places"

# Hard Location Graph: Enforces navigation bounds before RAG fallback
ROME_GRAPH = {
    "forum romanum": {"connections": ["via sacra", "palatine", "comitium", "capitoline", "subura", "tiber"]},
    "palatine": {"connections": ["forum romanum", "circus maximus", "via sacra"]},
    "capitoline": {"connections": ["forum romanum", "campus martius"]},
    "comitium": {"connections": ["forum romanum", "curia", "rostra"]},
    "curia": {"connections": ["comitium", "forum romanum"]},
    "rostra": {"connections": ["comitium", "forum romanum"]},
    "subura": {"connections": ["forum romanum", "macellum"]},
    "macellum": {"connections": ["subura"]},
    "via sacra": {"connections": ["forum romanum", "palatine"]},
    "circus maximus": {"connections": ["palatine", "tiber"]},
    "tiber": {"connections": ["forum romanum", "circus maximus", "campus martius", "transtiberim"]},
    "campus martius": {"connections": ["capitoline", "tiber", "thermae"]},
    "transtiberim": {"connections": ["tiber"]},
    "thermae": {"connections": ["campus martius"]}
}

ROME_CORE_ANCHORS = list(ROME_GRAPH.keys())

SYNONYMS: Dict[str, List[str]] = {
    "barracks": ["castra", "quarters", "camp", "praetorian", "cohort", "legion", "praetorium", "castra praetoria"],
    "legionary": ["legion", "cohort", "century", "centurion", "castra", "contubernium", "miles"],
    "soldier": ["miles", "legionary", "centurion", "cohort", "castra"],
    "market": ["macellum", "tabernae", "shops", "corn", "annona", "emporium"],
    "forum": ["forum", "comitium", "rostra", "curia"],
    "basilica": ["basilica", "tribunal", "court", "hall"],
    "temple": ["templum", "aedes", "shrine", "sanctuary", "altar"],
    "bath": ["thermae", "balneum", "baths", "frigidarium", "tepidarium", "caldarium"],
    "house": ["domus", "insula", "villa", "atrium", "peristyle"],
    "street": ["via", "vicus", "clivus", "street", "road", "path"],
}

STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "over", "under", "during", "time", "city",
    "show", "around", "take", "bring", "walk", "me", "you", "us", "to", "in", "of", "a", "an",
    "is", "are", "was", "were", "what", "where", "how", "tell", "about", "then", "now",
    "roman", "rome"
}

GENERIC_TOUR_RE = re.compile(
    r"^(show me around|show me the city|give me a tour|walk me around|take me around|tour|around|look|explore|wait|stay)\.?$",
    re.IGNORECASE
)

AUTO_GOTO_RE = re.compile(
    r"^(take me to|go to|bring me to|walk to|walk me to|take me into|head to|enter|travel to)\s+(the\s+)?(.+)$",
    re.IGNORECASE
)

CLICHE_OPENINGS = [
    r"^(the\s+morning\s+sun\s+\w+[^.]{0,60}[,.]?\s+)",
    r"^(the\s+(morning|afternoon|evening|dawn|noon)\s+sun\s+)",
    r"^(in\s+the\s+(morning|afternoon|evening|night)\s+light[, ]+)",
    r"^(the\s+morning\s+light\s+spills\s+across\s+)",
    r"^(morning\s+light\s+spills\s+across\s+)",
    r"^(as\s+you\s+(stand|step|walk)\s+into\s+[^,]+,\s+)",
    r"^(the\s+air\s+is\s+thick\s+with\s+)",
    r"^(you\s+find\s+yourself\s+in\s+)",
]

# =============================================================================
# Bulletproof Utilities
# =============================================================================

def die(msg: str, code: int = 1) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(code)

def clamp(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def strip_json_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    return s.strip()

def try_json(s: str) -> Tuple[Optional[Any], Optional[str]]:
    """Aggressive JSON parser to survive chatty Local LLMs."""
    s_clean = strip_json_fences(s)
    try:
        return json.loads(s_clean), None
    except Exception as e:
        # Fallback: hunt for the first JSON object block in the raw string
        m = re.search(r"(\{.*\})", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)), None
            except Exception as e2:
                return None, str(e2)
        return None, str(e)

def de_cliche_opening(text: str, loc_label: str, time_of_day: str) -> str:
    s = (text or "").strip()
    for pat in CLICHE_OPENINGS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
    if s:
        s = s[0].upper() + s[1:]
    head = s[:120].lower()
    if loc_label and loc_label.lower() not in head:
        s = f"{loc_label} ({time_of_day}). " + s
    return s

def minutes_to_time_of_day(minutes: int) -> str:
    m = minutes % 1440
    if m < 300: return "night"
    if m < 420: return "dawn"
    if m < 720: return "morning"
    if m < 840: return "noon"
    if m < 1080: return "afternoon"
    if m < 1200: return "evening"
    return "night"

# =============================================================================
# Tokenization & FTS Query Building
# =============================================================================

def intent_tokens(text: str) -> List[str]:
    t = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    t = [x for x in t if len(x) >= 3 and not x.isdigit() and x not in STOPWORDS]
    seen: Set[str] = set()
    out: List[str] = []
    for x in t:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:16]

def location_tokens(loc_label: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9]+", (loc_label or "").lower())
    raw = [x for x in raw if len(x) >= 3 and not x.isdigit()]
    seen: Set[str] = set()
    out: List[str] = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            out.append(x)

    if any(x in out for x in ["rome", "roman", "romanum", "roma"]):
        if "rome" not in out: out.insert(0, "rome")
        if "roma" not in out: out.append("roma")

    if not out: out = ["rome"]
    return out[:10]

def era_bias_terms(user_text: str) -> List[str]:
    u = (user_text or "").lower()
    out: List[str] = []
    if any(k in u for k in ["roman empire", "empire", "imperial", "principate", "emperor", "tiberius", "augustus", "claudius", "nero"]):
        out += ["imperial", "emperor", "augustus", "tiberius", "a.d.", "ad"]
    if any(k in u for k in ["republic", "late republic", "caesar", "cicero", "pompey"]):
        out += ["republic", "senate", "consul", "tribune", "b.c.", "bc", "caesar", "cicero"]
    
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:12]

def _fts_term(t: str) -> str:
    t = (t or "").strip()
    if not t: return ""
    if " " in t or "-" in t or "." in t:
        return '"' + t.replace('"', "") + '"'
    return t

def build_or_query(tokens: Sequence[str]) -> str:
    tokens = [tok for tok in tokens if tok]
    if not tokens: return ""
    return " OR ".join(_fts_term(t) for t in tokens)

def build_and_with_syn_groups(tokens: Sequence[str], extra_or_terms: List[str] = None) -> str:
    groups: List[str] = []
    for tok in tokens:
        tok = (tok or "").strip()
        if not tok: continue
        syns = SYNONYMS.get(tok, [])
        if syns:
            grp_terms = [_fts_term(tok)] + [_fts_term(s) for s in syns if s and s != tok]
            groups.append("(" + " OR ".join(grp_terms) + ")")
        else:
            groups.append(_fts_term(tok))
    base = " AND ".join(groups)
    
    if extra_or_terms:
        extra_q = build_or_query(extra_or_terms)
        if base and extra_q: return f"({base}) AND ({extra_q})"
        if extra_q: return extra_q
    return base

# =============================================================================
# DB Access & Pleiades Parsing
# =============================================================================

class EvidenceStore:
    def __init__(self, db_path: str):
        if not os.path.exists(db_path): die(f"DB not found: {db_path}")
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        if not self._table("segments_fts"): die("Missing segments_fts")

        self.fts_cols = [r[1] for r in self.con.execute("PRAGMA table_info(segments_fts)").fetchall()]

        def pick(cols: List[str], cands: List[str]) -> Optional[str]:
            s = set(cols)
            for c in cands:
                if c in s: return c
            return None

        self.fts_text = pick(self.fts_cols, ["txt", "text", "content", "segment_text", "body"])
        self.fts_corpus = pick(self.fts_cols, ["corpus_id", "src"])
        self.fts_sid = pick(self.fts_cols, ["segment_id", "id"])
        if not self.fts_text or not self.fts_sid:
            die(f"Could not detect required columns in segments_fts. cols={self.fts_cols}")

    def _table(self, name: str) -> bool:
        return self.con.execute("select 1 from sqlite_master where type in ('table','view') and name=?", (name,)).fetchone() is not None

    def close(self) -> None:
        try: self.con.close()
        except: pass

    def search(self, q: str, corpora: Optional[List[str]], limit: int) -> List[Dict[str, str]]:
        q = (q or "").strip()
        if not q: return []

        where_c = ""
        params: List[Any] = [q]
        if corpora and self.fts_corpus:
            where_c = " AND " + self.fts_corpus + " IN (" + ",".join(["?"] * len(corpora)) + ")"
            params += corpora
        params.append(limit)

        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text"
        if self.fts_corpus: cols += f", {self.fts_corpus} as corpus_id"

        sql = f"select {cols} from segments_fts where segments_fts match ? {where_c} order by bm25(segments_fts) limit ?"
        try:
            rows = self.con.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            print(f"[WARN] FTS error on {q!r}: {e}", file=sys.stderr)
            return []

        out: List[Dict[str, str]] = []
        for r in rows:
            out.append({
                "segment_id": str(r["segment_id"]),
                "corpus_id": str(r["corpus_id"]) if "corpus_id" in r.keys() else "",
                "excerpt": clamp(norm_ws(str(r["text"])), PROMPT_EXCERPT_CHARS),
            })
        return out

    def audit(self, segment_id: str) -> Optional[Dict[str, str]]:
        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text"
        if self.fts_corpus: cols += f", {self.fts_corpus} as corpus_id"
        try:
            row = self.con.execute(f"select {cols} from segments_fts where {self.fts_sid}=? limit 1", (segment_id,)).fetchone()
        except sqlite3.OperationalError: return None
        if not row: return None
        return {
            "segment_id": str(row["segment_id"]),
            "corpus_id": str(row["corpus_id"]) if "corpus_id" in row.keys() else "",
            "text": norm_ws(str(row["text"])),
        }

def parse_pleiades_title(excerpt: str) -> str:
    m = re.search(r"Title:\s*([^|]+)", excerpt or "")
    return m.group(1).strip() if m else ""

def parse_pleiades_id(excerpt: str) -> str:
    m = re.search(r"PleiadesID:\s*(\d+)", excerpt or "")
    return m.group(1).strip() if m else ""

def parse_coords(excerpt: str) -> Optional[Tuple[float, float]]:
    s = excerpt or ""
    m = re.search(r"(lat|latitude)\s*[:=]\s*([-]?\d+(?:\.\d+)?)", s, re.I)
    n = re.search(r"(lon|long|longitude)\s*[:=]\s*([-]?\d+(?:\.\d+)?)", s, re.I)
    if m and n:
        try: return float(m.group(2)), float(n.group(2))
        except: return None
    m = re.search(r"([-]?\d+(?:\.\d+)?)\s*,\s*([-]?\d+(?:\.\d+)?)", s)
    if m:
        try: return float(m.group(1)), float(m.group(2))
        except: return None
    return None

def title_match_confidence(target: str, title: str) -> float:
    tt = [x for x in re.findall(r"[A-Za-z0-9]+", (target or "").lower()) if len(x) >= 3]
    if not tt: return 0.0
    ti = set([x for x in re.findall(r"[A-Za-z0-9]+", (title or "").lower()) if len(x) >= 3])
    hit = sum(1 for x in tt if x in ti)
    return hit / max(1, len(tt))

# =============================================================================
# Retrieval (Quota + Scoring Merge)
# =============================================================================

def score_hit(excerpt: str, intent: List[str], loc: List[str], bias: List[str], scope: str) -> int:
    s = (excerpt or "").lower()
    sc = 0
    sc += 4 * sum(1 for t in intent if t in s)
    sc += 2 * sum(1 for t in bias if t in s)
    loc_hits = sum(1 for t in loc if t in s)
    sc += (4 * loc_hits) if scope == "local" else (1 * loc_hits)
    return sc

def retrieve_packet_with_quotas(
    store: EvidenceStore, user_text: str, state: Dict[str, Any], corpora_filter: List[str], cooldown_set: Set[str], debug: bool
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    
    scope = state.get("scope", "local")
    loc_label = str(state.get("loc_label", "Rome"))
    
    itoks = intent_tokens(user_text)
    ltoks = location_tokens(loc_label)
    btoks = era_bias_terms(user_text)

    # Tour Fallback Defense: If the user types "look around", `itoks` will be empty.
    # Seed it with their current location to force high-quality topography.
    if bool(GENERIC_TOUR_RE.match((user_text or "").strip())) and not itoks:
        itoks = [t for t in ltoks if t not in ["rome", "roma"]]
        if not itoks: itoks = random.sample(ROME_CORE_ANCHORS, 2)

    # 1. Topography / Core Location
    q_topo = build_and_with_syn_groups(itoks) if itoks else build_or_query(ltoks)
    # 2. Daily Life & Texture
    life_terms = ["food", "merchant", "smell", "street", "crowd", "tabernae", "dirt", "noise", "pleb", "cart"]
    q_life = build_or_query(life_terms + itoks)
    # 3. Context & Politics
    pol_terms = ["law", "senate", "soldier", "guard", "rumor", "augustus", "emperor", "patrician", "slaves"]
    q_pol = build_or_query(pol_terms + itoks)

    queries = [(q_topo, 12), (q_life, 8), (q_pol, 8)]
    
    all_hits: List[Dict[str, str]] = []
    for q, limit in queries:
        if q:
            all_hits.extend(store.search(q, corpora_filter if corpora_filter else None, limit=limit * 3))

    # Strip Gazetteer out of narrative engine
    all_hits = [h for h in all_hits if h.get("corpus_id") != PLEIADES_CORPUS_ID]

    seen: Set[str] = set()
    uniq: List[Dict[str, str]] = []
    for h in all_hits:
        sid = h["segment_id"]
        if sid not in seen:
            seen.add(sid)
            uniq.append(h)

    scored: List[Tuple[int, int, Dict[str, str]]] = []
    for idx, h in enumerate(uniq):
        sc = score_hit(h.get("excerpt", ""), itoks, ltoks, btoks, scope)
        scored.append((sc, idx, h))
    scored.sort(key=lambda x: (-x[0], x[1]))

    packet: List[Dict[str, str]] = []
    for sc, _, h in scored:
        if h["segment_id"] in cooldown_set: continue
        packet.append(h)
        if len(packet) >= MAX_EVIDENCE: break

    loc_count = sum(1 for h in packet if any(t in (h.get("excerpt", "") or "").lower() for t in ltoks))

    # Starvation Backfill Logic
    if scope == "local" and len(packet) < MIN_PACKET_LOCAL:
        already: Set[str] = {h["segment_id"] for h in packet}
        for sc, _, h in scored:
            if len(packet) >= MIN_PACKET_LOCAL: break
            if h["segment_id"] not in already:
                packet.append(h)
                already.add(h["segment_id"])

    meta = {
        "scope": scope,
        "loc_label": loc_label,
        "intent_tokens": itoks,
        "loc_tokens": ltoks,
        "era_bias": btoks,
        "packet_n": len(packet),
        "packet_loc_hits": loc_count,
    }

    if debug: print(f"[DEBUG] meta={meta}")
    return packet, meta

# =============================================================================
# LLM client (Retry / Backoff)
# =============================================================================

def openai_chat(
    base_url: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, api_key: str, retries: int = 4
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_err = None
    last_raw = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                last_raw = raw
        except urllib.error.URLError as e:
            # Fast fail on connection refused
            if isinstance(e.reason, ConnectionRefusedError):
                die(f"Connection Refused: Ensure LLM server is running at {base_url}")
            last_err = f"URLError: {e}"
        except urllib.error.HTTPError as e:
            try: last_raw = e.read().decode("utf-8", errors="replace")
            except Exception: last_raw = None
            last_err = f"HTTPError {e.code}: {e}"
        except Exception as e:
            last_err = str(e)
        else:
            try:
                j = json.loads(last_raw)
                return j["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = f"Response JSON parse failed: {e}"

        time.sleep(min(6.0, 0.5 * (2 ** (attempt - 1))))

    die(f"LLM request failed after retries. LastErr={last_err}. Raw(trunc)={clamp(last_raw or '', 900)}")
    return ""

# =============================================================================
# Plan/Render schemas + strict validation
# =============================================================================

PLAN_REQUIRED_KEYS = {"narrative_beats", "state_delta"}
RENDER_REQUIRED_KEYS = {"sensory_environment", "npc_activity", "observations", "claims"}

def validate_evidence_ids(evs: Any, packet_ids: Set[str]) -> Optional[str]:
    if not isinstance(evs, list) or not evs: return "evidence_ids empty"
    bad = [x for x in evs if x not in packet_ids]
    if bad: return f"unknown evidence_ids {bad[:4]}"
    return None

def validate_plan(obj: Any, packet_ids: Set[str]) -> List[str]:
    e: List[str] = []
    if not isinstance(obj, dict): return ["plan not an object"]

    for k in list(obj.keys()):
        if k not in PLAN_REQUIRED_KEYS: del obj[k]
    for k in PLAN_REQUIRED_KEYS:
        if k not in obj: e.append(f"missing {k}")

    beats = obj.get("narrative_beats")
    if not isinstance(beats, list) or not (1 <= len(beats) <= 5):
        e.append("narrative_beats must be list len 1-5")
    else:
        for i, it in enumerate(beats):
            if not isinstance(it, dict):
                e.append(f"narrative_beats[{i}] not object")
                continue
            b = it.get("beat")
            if not isinstance(b, str) or not b.strip():
                e.append(f"narrative_beats[{i}].beat empty")
            err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
            if err: e.append(f"narrative_beats[{i}].{err}")

    delta = obj.get("state_delta")
    if not isinstance(delta, dict):
        e.append("state_delta must be an object")
    else:
        if "time_advanced_minutes" in delta and not isinstance(delta["time_advanced_minutes"], int):
            e.append("state_delta.time_advanced_minutes must be int")
        if "fatigue_change" in delta and not isinstance(delta["fatigue_change"], int):
            e.append("state_delta.fatigue_change must be int")
        
        changes = delta.get("narrative_changes")
        if not isinstance(changes, list):
            e.append("state_delta.narrative_changes must be list")
        else:
            for i, it in enumerate(changes):
                if not isinstance(it, dict): continue
                err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
                if err: e.append(f"state_delta.narrative_changes[{i}].{err}")

    return e

def validate_render(obj: Any, packet_ids: Set[str]) -> List[str]:
    e: List[str] = []
    if not isinstance(obj, dict): return ["render not an object"]

    allowed = RENDER_REQUIRED_KEYS | {"limitations"}
    for k in list(obj.keys()):
        if k not in allowed: del obj[k]
    for k in RENDER_REQUIRED_KEYS:
        if k not in obj: e.append(f"missing {k}")

    se = obj.get("sensory_environment")
    if not isinstance(se, str) or len(se.strip()) < 100:
        e.append("sensory_environment too short or invalid")

    na = obj.get("npc_activity")
    if not isinstance(na, list) or not (3 <= len(na) <= 12):
        e.append("npc_activity must be list len 3-12")

    obs = obj.get("observations")
    if not isinstance(obs, list) or not (3 <= len(obs) <= 12):
        e.append("observations must be list len 3-12")
    else:
        for i, it in enumerate(obs):
            if not isinstance(it, dict): continue
            err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
            if err: e.append(f"observations[{i}].{err}")

    claims = obj.get("claims")
    if not isinstance(claims, list) or not (3 <= len(claims) <= 16):
        e.append("claims must be list len 3-16")
    else:
        for i, it in enumerate(claims):
            if not isinstance(it, dict): continue
            err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
            if err: e.append(f"claims[{i}].{err}")

    lim = obj.get("limitations")
    if lim is not None and lim != "" and not isinstance(lim, str):
        e.append("limitations must be string if present")

    return e

def system_prompt() -> str:
    return (
        "You are the Game Engine for a stateful, first-person historical simulation in Rome, 30 BC.\n"
        "Hard rules:\n"
        "1) Output JSON only.\n"
        "2) NEVER reset the atmosphere or repeat set-dressing from recent_history. Progress time and state forward.\n"
        "3) Player choices MUST have consequences (fatigue, time passing, NPC reactions).\n"
        "4) Every concrete factual claim must appear in claims[] and cite evidence_ids from the packet.\n"
        "5) Present tense, gritty, show-don’t-tell. Do NOT sound like a tour guide.\n"
        "6) If a system_constraint prevents an action, narrate the limitation organically.\n"
    )

def plan_prompt(state: Dict[str, Any], history: List[Dict[str, str]], user_text: str, system_constraint: str, packet: List[Dict[str, str]]) -> str:
    return json.dumps({
        "task": "Create a TurnPlan calculating state changes and narrative beats.",
        "state": state,
        "recent_history": history,
        "system_constraint": system_constraint,
        "user_input": user_text,
        "evidence_packet": packet,
        "output_schema": {
            "narrative_beats": [{"beat": "string", "evidence_ids": ["segment_id"]}],
            "state_delta": {
                "time_advanced_minutes": "integer",
                "fatigue_change": "integer",
                "reputation_change": {"faction": "string", "amount": "integer"},
                "npcs_present": ["string"],
                "narrative_changes": [{"change": "string", "evidence_ids": ["segment_id"]}]
            }
        },
        "requirements": [
            "JSON only; no extra keys.",
            "Each narrative_beat and narrative_change MUST cite evidence_ids.",
            "Account for the system_constraint if it is not empty."
        ]
    }, ensure_ascii=False)

def render_prompt(state: Dict[str, Any], history: List[Dict[str, str]], plan: Dict[str, Any], packet: List[Dict[str, str]]) -> str:
    return json.dumps({
        "task": "Render a present-tense scene using the updated state, plan, and evidence.",
        "state": state,
        "recent_history": history,
        "plan": plan,
        "evidence_packet": packet,
        "output_schema": {
            "sensory_environment": "string (2-4 paragraphs, present tense)",
            "npc_activity": ["string (one line each, 6-12 lines)"],
            "observations": [{"text": "string", "evidence_ids": ["segment_id"]}],
            "claims": [{"claim": "string", "evidence_ids": ["segment_id"]}],
            "limitations": "string (optional)"
        },
        "requirements": [
            "JSON only; no extra keys.",
            "sensory_environment must progress the action, NOT repeat history.",
            "Every concrete factual claim must appear in claims[] with evidence_ids.",
            "Avoid clichés and do not reset atmosphere.",
        ]
    }, ensure_ascii=False)

def repair_prompt(kind: str, bad: str, errors: List[str]) -> str:
    return json.dumps({
        "task": f"Repair invalid {kind} output into valid JSON for the required schema.",
        "validation_errors": errors,
        "invalid_output": clamp(bad, 2000),
        "rules": [
            "Return JSON only.",
            "Include all required keys and correct types.",
            "Remove extra keys.",
            "Fix quotes/brackets/commas.",
        ]
    }, ensure_ascii=False)

def suggest_prompt(state: Dict[str, Any], packet: List[Dict[str, str]], n: int) -> str:
    return json.dumps({
        "task": "Generate grounded suggestions for what the user could do next.",
        "state": state,
        "evidence_packet": packet,
        "output_schema": {"suggestions": ["string"]},
        "requirements": [
            "Return JSON only: {\"suggestions\":[...]}",
            f"Exactly {n} suggestions.",
            "Suggestions should be prompts the user can type next.",
            "Do not introduce new facts; phrase as actions/questions."
        ]
    }, ensure_ascii=False)

# =============================================================================
# Main CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--corpus", action="append", default=[], help="Optional: restrict retrieval to specific corpus_id")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=2800)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--show-observations", action="store_true", default=False)
    ap.add_argument("--show-claims", action="store_true", default=False)
    args = ap.parse_args()

    store = EvidenceStore(args.db)
    cooldown: collections.deque[str] = collections.deque(maxlen=COOLDOWN_SIZE)
    cooldown_set: Set[str] = set()
    history: collections.deque[Dict[str, Any]] = collections.deque(maxlen=3)

    state: Dict[str, Any] = {
        "loc_label": "Forum Romanum",
        "loc_id": None,
        "coords": None,
        "day": 1,
        "time_minutes": 480, # 8:00 AM
        "time_of_day": "morning",
        "scope": "local",
        "fatigue": 0,
        "reputation": {},
        "npcs_present": [],
        "turn_index": 0
    }

    last_packet: List[Dict[str, str]] = []

    print("[OK] Vivarium Rome Experience (Stateful Game Engine)")
    print(f"[OK] db={args.db}")
    print("Commands:")
    print("  :goto <place> | :time <dawn|morning|noon|afternoon|evening|night> | :scope <local|global>")
    print("  :place <name> | :audit <segment_id> | :state | :llmtest | :suggest [N] | :quit")
    print("Then type any prompt, e.g. 'walk to the subura' or 'look around'.")
    print()

    try:
        while True:
            # Recompute UI time and day constraints
            state["day"] = 1 + (state["time_minutes"] // 1440)
            state["time_of_day"] = minutes_to_time_of_day(state["time_minutes"])

            try:
                prompt_prefix = f"[{state['loc_label']} | {state['time_of_day'].title()} (Day {state['day']}) | Fatig:{state['fatigue']}]> "
                user_in = input(prompt_prefix).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_in:
                continue

            # ----- Commands -----
            if user_in.startswith(":"):
                parts = user_in[1:].split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("q", "quit", "exit"):
                    break
                if cmd == "state":
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    continue
                if cmd == "time":
                    time_map = {"dawn": 360, "morning": 480, "noon": 720, "afternoon": 900, "evening": 1140, "night": 1320}
                    if arg not in time_map:
                        print("[WARN] :time must be dawn|morning|noon|afternoon|evening|night")
                        continue
                    # Keep same day, just update absolute minutes for that day
                    state["time_minutes"] = ((state["day"] - 1) * 1440) + time_map[arg]
                    state["time_of_day"] = arg
                    print("[OK] time_of_day =", state["time_of_day"])
                    continue
                if cmd == "scope":
                    if arg.lower() not in ("local", "global"):
                        print("[WARN] :scope must be local|global")
                        continue
                    state["scope"] = arg.lower()
                    print("[OK] scope =", state["scope"])
                    continue
                if cmd == "goto":
                    if not arg:
                        print("[WARN] :goto needs a place label")
                        continue
                    state["loc_label"] = arg.title()
                    state["loc_id"] = None
                    state["coords"] = None
                    print("[OK] loc_label =", state["loc_label"])
                    continue
                if cmd == "llmtest":
                    msgs = [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}]
                    out = openai_chat(args.base_url, args.model, msgs, 0.0, 64, args.api_key)
                    print("[LLMTEST] raw:", clamp(out, 240))
                    continue
                if cmd == "audit":
                    if not arg:
                        print("[WARN] :audit needs a segment_id")
                        continue
                    row = store.audit(arg)
                    if not row:
                        print("[WARN] segment not found")
                        continue
                    print(f"[AUDIT] {row.get('corpus_id','')}:{row['segment_id']}")
                    print(textwrap.fill(clamp(row["text"], 4000), width=90))
                    continue
                if cmd == "place":
                    if not arg:
                        print("[WARN] :place needs a name")
                        continue
                    q = build_and_with_syn_groups(intent_tokens(arg)) or build_or_query([arg])
                    hits = store.search(q, [PLEIADES_CORPUS_ID], limit=5)
                    if not hits:
                        print("[PLACE] No matches found.")
                        continue
                    print(f"[PLACE] Top {len(hits)} matches (Pleiades):")
                    for i, h in enumerate(hits, start=1):
                        title = parse_pleiades_title(h["excerpt"]) or h["segment_id"]
                        pid = parse_pleiades_id(h["excerpt"]) or ""
                        tag = f"(id={pid})" if pid else ""
                        print(f"  {i}) {title} {tag} :: {clamp(h['excerpt'], 160)}")
                    sel = input("Pick a number to set location (Enter cancels): ").strip()
                    if sel.isdigit():
                        k = int(sel)
                        if 1 <= k <= len(hits):
                            chosen = hits[k - 1]
                            title = parse_pleiades_title(chosen["excerpt"]) or chosen["segment_id"]
                            pid = parse_pleiades_id(chosen["excerpt"]) or None
                            coords = parse_coords(chosen["excerpt"])
                            state["loc_label"] = title.title()
                            state["loc_id"] = pid
                            state["coords"] = coords
                            print(f"[OK] location set -> {state['loc_label']}" + (f" (pleiades={pid})" if pid else ""))
                    continue
                if cmd == "suggest":
                    n = max(3, min(12, int(arg))) if arg.isdigit() else 6
                    if not last_packet:
                        cands = random.sample(ROME_CORE_ANCHORS, min(n, len(ROME_CORE_ANCHORS)))
                        print("[SUGGEST] Try prompts like:")
                        for c in cands: print(f"  - take me to the {c}")
                        continue
                    msgs = [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": suggest_prompt(state, last_packet, n)},
                    ]
                    raw = openai_chat(args.base_url, args.model, msgs, args.temperature, 700, args.api_key)
                    obj, err = try_json(raw)
                    if err or not isinstance(obj, dict) or "suggestions" not in obj or not isinstance(obj["suggestions"], list):
                        print("[WARN] suggest parse failed; raw:", clamp(raw, 300))
                        continue
                    print("[SUGGEST]")
                    for i, s in enumerate(obj["suggestions"], start=1):
                        if isinstance(s, str) and s.strip():
                            print(f"  {i}) {s.strip()}")
                    continue

                print("[WARN] unknown command")
                continue

            # ----- Navigation Graph Logic -----
            if len(user_in) < 3:
                print("[WARN] Input too short.")
                continue

            system_constraint = ""
            m = AUTO_GOTO_RE.match(user_in)
            if m:
                target_raw = m.group(3).strip().lower().strip(".")
                current_loc = state["loc_label"].lower()
                
                # Token Intersection Graph Match (Bulletproof)
                tgt_toks = set(re.findall(r"\w+", target_raw))
                best_match = None
                
                # Fast exact match check
                best_overlap = 0
                if target_raw in ROME_GRAPH:
                    best_match = target_raw
                else:
                    for node in ROME_GRAPH.keys():
                        node_toks = set(re.findall(r"\w+", node))
                        overlap = len(tgt_toks & node_toks)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = node
                
                if best_match and best_overlap >= 1 and any(t not in STOPWORDS for t in tgt_toks & set(re.findall(r"\w+", best_match))):
                    if best_match == current_loc:
                        system_constraint = f"Player is already at {best_match.title()}."
                    elif best_match in ROME_GRAPH.get(current_loc, {}).get("connections", []):
                        state["loc_label"] = best_match.title()
                        state["loc_id"] = None
                        system_constraint = f"Player successfully moved to {best_match.title()}. It takes 15 minutes."
                        state["time_minutes"] += 15
                    else:
                        system_constraint = f"Player tried to go to {best_match.title()}, but it is not directly connected to {current_loc.title()}. Pathfinding required. They are still at {current_loc.title()}."
                else:
                    # Fallback to Original Pleiades Search
                    q = build_and_with_syn_groups(intent_tokens(target_raw)) or build_or_query([target_raw])
                    hits = store.search(q, [PLEIADES_CORPUS_ID], limit=5)
                    if hits:
                        t0 = parse_pleiades_title(hits[0]["excerpt"]) or target_raw
                        conf0 = title_match_confidence(target_raw, t0)
                        conf1 = 0.0
                        if len(hits) > 1:
                            t1 = parse_pleiades_title(hits[1]["excerpt"]) or ""
                            conf1 = title_match_confidence(target_raw, t1)
                        if conf0 >= 0.75 and (conf0 - conf1) >= 0.25:
                            state["loc_label"] = t0.title()
                            state["loc_id"] = parse_pleiades_id(hits[0]["excerpt"]) or None
                            state["coords"] = parse_coords(hits[0]["excerpt"])
                            system_constraint = f"Player traveled outside the immediate graph to {t0.title()}. It takes 30 minutes."
                            state["time_minutes"] += 30
                        else:
                            system_constraint = f"Player tried to travel to {target_raw.title()} but the location is ambiguous. Still at {current_loc.title()}."
                    else:
                        system_constraint = f"Player tried to travel to {target_raw.title()} but it does not exist on the map. Still at {current_loc.title()}."

            if args.debug and system_constraint:
                print(f"[DEBUG Engine] {system_constraint}")

            # ----- Core Engine Loop -----
            packet, meta = retrieve_packet_with_quotas(
                store=store,
                user_text=user_in,
                state=state,
                corpora_filter=args.corpus,
                cooldown_set=cooldown_set,
                debug=args.debug
            )

            if not packet:
                print("[WARN] No evidence returned. Try a more specific prompt or use :place / :goto.")
                continue

            last_packet = packet
            packet_ids = {e["segment_id"] for e in packet}

            # 1. Plan Pass
            plan_obj: Optional[Dict[str, Any]] = None
            plan_last = ""
            plan_errs: List[str] = []
            for attempt in range(1, TURNPLAN_RETRIES + 1):
                msgs = [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": "/no_think\n" + plan_prompt(state, list(history), user_in, system_constraint, packet)},
                ]
                if attempt > 1:
                    msgs.append({"role": "user", "content": repair_prompt("TurnPlan", plan_last, plan_errs)})
                out = openai_chat(args.base_url, args.model, msgs, args.temperature, args.max_tokens, args.api_key)
                plan_last = out
                obj, err = try_json(out)
                if err:
                    plan_errs = [f"JSON parse error: {err}"]
                    continue
                plan_errs = validate_plan(obj, packet_ids)
                if plan_errs:
                    if args.debug: print(f"[DEBUG] Plan invalid (try {attempt}): {plan_errs}")
                    continue
                plan_obj = obj
                break

            if plan_obj is None:
                print("[FAIL] TurnPlan validation failed.")
                if args.debug:
                    print(plan_errs[:20])
                    print(clamp(plan_last, 900))
                continue

            # Apply State Delta & Progress Engine variables safely
            delta = plan_obj.get("state_delta", {})
            try: state["time_minutes"] += max(0, int(delta.get("time_advanced_minutes", 0)))
            except: pass
            
            # Immediately refresh day/time so the LLM renderer sees if morning crossed into noon
            state["day"] = 1 + (state["time_minutes"] // 1440)
            state["time_of_day"] = minutes_to_time_of_day(state["time_minutes"])

            try: 
                f_delta = int(delta.get("fatigue_change", 0))
                state["fatigue"] = max(0, min(100, state["fatigue"] + f_delta))
            except: pass

            if isinstance(delta.get("reputation_change"), dict) and delta["reputation_change"].get("faction"):
                fac = delta["reputation_change"]["faction"].lower()
                try: state["reputation"][fac] = state["reputation"].get(fac, 0) + int(delta["reputation_change"].get("amount", 0))
                except: pass

            if isinstance(delta.get("npcs_present"), list):
                state["npcs_present"] = [str(x) for x in delta["npcs_present"]][:4]

            state["turn_index"] += 1

            # 2. Render Pass
            render_obj: Optional[Dict[str, Any]] = None
            render_last = ""
            render_errs: List[str] = []
            for attempt in range(1, RENDER_RETRIES + 1):
                msgs = [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": "/no_think\n" + render_prompt(state, list(history), plan_obj, packet)},
                ]
                if attempt > 1:
                    msgs.append({"role": "user", "content": repair_prompt("Render", render_last, render_errs)})
                out = openai_chat(args.base_url, args.model, msgs, args.temperature + 0.1, args.max_tokens, args.api_key)
                render_last = out
                obj, err = try_json(out)
                if err:
                    render_errs = [f"JSON parse error: {err}"]
                    continue
                render_errs = validate_render(obj, packet_ids)
                if render_errs:
                    if args.debug: print(f"[DEBUG] Render invalid (try {attempt}): {render_errs}")
                    continue
                render_obj = obj
                break

            if render_obj is None:
                print("[FAIL] Render validation failed.")
                if args.debug:
                    print(render_errs[:20])
                    print(clamp(render_last, 900))
                continue

            for e in packet:
                sid = e["segment_id"]
                if sid not in cooldown_set:
                    cooldown.append(sid)
            cooldown_set = set(cooldown)

            # Output Formatting
            sensory = de_cliche_opening(render_obj["sensory_environment"], state["loc_label"], state["time_of_day"])
            npc_lines = [de_cliche_opening(line, state["loc_label"], state["time_of_day"]) for line in render_obj["npc_activity"]]

            history.append({
                "turn": state["turn_index"],
                "player_action": user_in,
                "engine_narrative": sensory
            })

            print("\n" + "=" * 92)
            print(textwrap.fill(sensory.strip(), width=92))
            print("\n" + "[Ambient activity]")
            for line in npc_lines:
                print("- " + line.strip())

            lim = render_obj.get("limitations")
            if lim:
                print("\n[Limitations]")
                print(textwrap.fill(lim.strip(), width=92))

            if args.show_observations:
                print("\n[Observations]")
                for o in render_obj["observations"]:
                    print(f"- {o['text']} (source {','.join(o.get('evidence_ids', []))})")

            if args.show_claims:
                print("\n[Claims (auditable)]")
                for c in render_obj["claims"]:
                    print(f"- {c['claim']} (source {','.join(c.get('evidence_ids', []))})")

            if args.debug:
                print("\n[DEBUG] evidence packet ids:", ", ".join([e["segment_id"] for e in packet[:10]]))

            print("=" * 92 + "\n")

    finally:
        store.close()

if __name__ == "__main__":
    main()