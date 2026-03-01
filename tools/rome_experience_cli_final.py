#!/usr/bin/env python3
"""
===============================================================================
V I V A R I U M  /  S C R I P T O R I U M
Stateful Historical Simulation Engine
Version: 12.0.0 (Data-Decoupled Structural Refactor)
===============================================================================
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import re
import sqlite3
import string
import sys
import textwrap
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass

def die(msg: str, code: int = 1) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(code)

def load_config(filepath: str) -> dict:
    if not os.path.exists(filepath):
        die(f"Missing config file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        die(f"JSON format error in {filepath}: {e}")
    except Exception as e:
        die(f"Failed to load {filepath}: {e}")

# Load Configuration globally so they are available immediately (e.g., for Argparse choices)
_dir = os.path.dirname(os.path.abspath(__file__))
WORLD_DATA = load_config(os.path.join(_dir, "world_config.json"))
PROMPTS_DATA = load_config(os.path.join(_dir, "prompts.json"))

WORLDS = WORLD_DATA.get("WORLDS", {})
WEATHER_STATES = WORLD_DATA.get("WEATHER_STATES", [])
WEATHER_TERMS = WORLD_DATA.get("WEATHER_TERMS", {})
TIME_TERMS = WORLD_DATA.get("TIME_TERMS", {})

# =============================================================================
# Retrieval Configuration Parameters
# =============================================================================
@dataclass
class RetrievalConfig:
    vector_hit_boost: float = 15.0
    jaccard_intent_weight: float = 10.0
    jaccard_loc_weight: float = 5.0
    exact_phrase_weight: float = 4.0
    exact_bias_weight: float = 2.0
    loc_hit_social_weight: float = 1.0
    loc_hit_local_weight: float = 4.0
    loc_hit_global_weight: float = 1.0

RETRIEVAL_CONFIG = RetrievalConfig()

# =============================================================================
# Enterprise Telemetry & Diagnostics
# =============================================================================

class Diagnostics:
    """Profiles execution time for database queries, LLM generation, and validation."""
    def __init__(self, enabled: bool, db_path: str):
        self.enabled = enabled
        self.timers: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
        self._history: List[Dict[str, float]] = []
        self.telemetry_db = os.path.join(os.path.dirname(os.path.abspath(db_path)), "telemetry.sqlite")
        
        if self.enabled:
            self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.telemetry_db) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS engine_telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        turn_index INTEGER,
                        total_time REAL,
                        fts_time REAL,
                        vec_time REAL,
                        rerank_time REAL,
                        plan_time REAL,
                        render_time REAL,
                        audit_time REAL
                    )
                """)
        except Exception as e:
            print(f"[WARN] Failed to initialize telemetry DB: {e}")

    def start(self, key: str) -> None:
        if self.enabled:
            self.timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        if self.enabled and key in self.timers:
            elapsed = time.perf_counter() - self.timers.pop(key)
            self.metrics[key] = self.metrics.get(key, 0.0) + elapsed
            return elapsed
        return 0.0

    def record_turn(self, turn_index: int) -> None:
        if self.enabled:
            self._history.append(self.metrics.copy())
            total = sum(self.metrics.values())
            try:
                with sqlite3.connect(self.telemetry_db) as con:
                    con.execute("""
                        INSERT INTO engine_telemetry 
                        (turn_index, total_time, fts_time, vec_time, rerank_time, plan_time, render_time, audit_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        turn_index,
                        total,
                        self.metrics.get("Retrieval_FTS", 0.0),
                        self.metrics.get("Retrieval_Vector", 0.0),
                        self.metrics.get("Retrieval_Rerank", 0.0),
                        self.metrics.get("LLM_Plan_Pass", 0.0),
                        self.metrics.get("LLM_Render_Pass", 0.0),
                        self.metrics.get("LLM_Strict_Auditor", 0.0)
                    ))
            except Exception:
                pass
            self.metrics.clear()

    def print_report(self) -> None:
        if not self.enabled or not self._history:
            return
        latest = self._history[-1]
        print("\n" + "=" * 40 + " [ENGINE TELEMETRY] " + "=" * 40)
        total = sum(latest.values())
        for k, v in sorted(latest.items(), key=lambda item: item[1], reverse=True):
            pct = (v / total) * 100 if total > 0 else 0.0
            print(f"{k.ljust(25)} | {v:7.3f}s | {pct:5.1f}%")
        print("-" * 100)
        print(f"{'TOTAL TURN TIME'.ljust(25)} | {total:7.3f}s | 100.0%")
        print("=" * 100)


# =============================================================================
# Core Engine Constants (Physics & Rules)
# =============================================================================

COOLDOWN_SIZE = 220
MAX_EVIDENCE = 12
MIN_PACKET_LOCAL = 12
PROMPT_EXCERPT_CHARS = 1200 

TURNPLAN_RETRIES = 4
RENDER_RETRIES = 4
AUDIT_RETRIES = 2

STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "over", "under", "during", "time", "city",
    "show", "around", "take", "bring", "walk", "me", "you", "us", "to", "in", "of", "a", "an",
    "is", "are", "was", "were", "what", "where", "how", "tell", "about", "then", "now"
}

SOCIAL_VERBS = {"talk", "speak", "ask", "tell", "say", "greet", "buy", "sell", "listen", "hear", "haggle", "argue", "discuss", "trade"}

GENERIC_TOUR_RE = re.compile(
    r"^(show me around|show me the city|give me a tour|walk me around|take me around|tour|around|look|explore|wait|stay|continue|walk|wander|roam)\.?$",
    re.IGNORECASE
)

AUTO_GOTO_RE = re.compile(
    r"^(take me to|go to|bring me to|walk to|walk me to|take me into|head to|enter|travel to|explore|visit)\s+(the\s+)?(.+)$",
    re.IGNORECASE
)

GENERIC_MOVE_RE = re.compile(
    r"^(walk|go|head|travel|run|move)\s+(north|south|east|west|outside|inside|away|back|down|up|forward)", 
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
# Bulletproof Utilities, Math, & State Management
# =============================================================================

def clamp(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def clean_for_match(s: str) -> str:
    """Cryptographic-style normalizer for mechanical citation gating."""
    if not s: return ""
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r"\s+", " ", s).lower().strip()

def strip_json_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    return s.strip()

def try_json(s: str) -> Tuple[Optional[Any], Optional[str]]:
    s_clean = strip_json_fences(s)
    try:
        return json.loads(s_clean), None
    except Exception as e:
        m = re.search(r"(\{.*\})", s_clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)), None
            except Exception as e2:
                return None, str(e2)
        return None, str(e)

def de_cliche_opening(text: str, loc_label: str, time_of_day: str, weather: str, prepend_context: bool = True) -> str:
    s = (text or "").strip()
    for pat in CLICHE_OPENINGS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
    if s:
        s = s[0].upper() + s[1:]
    head = s[:120].lower()
    if prepend_context and loc_label and loc_label.lower() not in head:
        s = f"{loc_label} ({time_of_day}, {weather}). " + s
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

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_save_path(db_path: str, world_name: str) -> str:
    dir_name = os.path.dirname(os.path.abspath(db_path))
    base = os.path.basename(db_path)
    safe_world = world_name.replace(" ", "_").lower()
    return os.path.join(dir_name, f"{os.path.splitext(base)[0]}_{safe_world}_save.json")

def load_session(db_path: str, world_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[collections.deque], Optional[collections.deque]]:
    save_path = get_save_path(db_path, world_name)
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                state = data.get("state")
                hist = collections.deque(data.get("history", []), maxlen=5)
                cool = collections.deque(data.get("cooldown", []), maxlen=COOLDOWN_SIZE)
                return state, hist, cool
        except Exception as e:
            print(f"[WARN] Failed to load session save: {e}")
    return None, None, None

def save_session(db_path: str, world_name: str, state: Dict[str, Any], history: collections.deque, cooldown: collections.deque) -> None:
    save_path = get_save_path(db_path, world_name)
    data = {"state": state, "history": list(history), "cooldown": list(cooldown)}
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save session: {e}")

def print_help_ui(world_config: Dict[str, Any]) -> None:
    print("\n" + "="*80)
    print(" VIVARIUM ENGINE MANUAL ".center(80))
    print("="*80)
    print("Vivarium is an open-ended, mechanically cited historical simulation.")
    print("Everything you see is mathematically grounded in primary source evidence.")
    
    print("\n[NAVIGATION]")
    print("  You can move organically by typing commands like:")
    print("  > walk to the market")
    print("  > head to the senate house")
    print("  The engine resolves names via semantic aliases (e.g. 'baths' -> 'thermae').")
    print("  If you name a distant city, the engine uses Haversine math to travel there.")
    
    print("\n[INTERACTION & PHYSICS]")
    print("  > talk to the merchant")
    print("  > buy a loaf of bread")
    print("  Actions consume time and induce fatigue. Social actions force dialogue generation.")
    
    print("\n[DYNAMIC ERAS]")
    print(f"  Available Eras for {world_config['name']}:")
    for era, conf in world_config.get("eras", {}).items():
        print(f"    - {era.upper()}: {conf['system_context']}")
    print("  Change eras instantly by typing: :era <name>")
    
    print("\n[OOB COMMANDS]")
    print("  :suggest [N]  -> Generates N context-aware suggestions for what to do next.")
    print("  :place <name> -> Instantly teleports you to a specific Pleiades node.")
    print("  :audit <id>   -> Prints the raw database text for a given citation ID.")
    print("  :time <time>  -> Warps time (e.g. :time night).")
    print("  :state        -> Dumps the raw JSON engine state.")
    print("  :reset        -> Wipes your save file.")
    print("  :quit         -> Safely saves and exits.")
    print("="*80 + "\n")

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

def location_tokens(loc_label: str, world: Dict[str, Any]) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9]+", (loc_label or "").lower())
    raw = [x for x in raw if len(x) >= 3 and not x.isdigit()]
    seen: Set[str] = set()
    out: List[str] = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            out.append(x)

    core_kws = world.get("core_keywords", [])
    if any(x in out for x in core_kws):
        if core_kws and core_kws[0] not in out: 
            out.insert(0, core_kws[0])
            
    if not out and core_kws: 
        out = [core_kws[0]]
    return out[:10]

def era_bias_terms(user_text: str, era_config: Dict[str, Any]) -> List[str]:
    u = (user_text or "").lower()
    out: List[str] = []
    for group in era_config.get("bias_groups", []):
        if any(k in u for k in group["triggers"]):
            out += group["terms"]
            
    for group in era_config.get("bias_groups", []):
        out += group["terms"][:3]
        
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

def build_and_with_syn_groups(tokens: Sequence[str], world: Dict[str, Any], extra_or_terms: List[str] = None) -> str:
    groups: List[str] = []
    syn_map = world.get("synonyms", {})
    for tok in tokens:
        tok = (tok or "").strip()
        if not tok: continue
        syns = syn_map.get(tok, [])
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
# DB Access, Gazetteer Parsing & Vector Retrieval Setup
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
        if not self.fts_corpus:
            die("segments_fts is missing corpus_id/src column; required for --corpus and gazetteer logic.")
            
        self.has_vector_table = self._table("segments_vec")

    def _table(self, name: str) -> bool:
        return self.con.execute("select 1 from sqlite_master where type in ('table','view') and name=?", (name,)).fetchone() is not None

    def close(self) -> None:
        try: self.con.close()
        except: pass

    def search_fts(self, q: str, corpora: Optional[List[str]], limit: int) -> List[Dict[str, str]]:
        q = (q or "").strip()
        if not q: return []

        where_c = ""
        params: List[Any] = [q]
        if corpora and self.fts_corpus:
            where_c = " AND " + self.fts_corpus + " IN (" + ",".join(["?"] * len(corpora)) + ")"
            params += corpora
        params.append(limit)

        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text, {self.fts_corpus} as corpus_id"

        sql = f"select {cols} from segments_fts where segments_fts match ? {where_c} order by bm25(segments_fts) limit ?"
        try:
            rows = self.con.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            print(f"[WARN] FTS error on {q!r}: {e}", file=sys.stderr)
            return []

        out: List[Dict[str, str]] = []
        for r in rows:
            raw_text = norm_ws(str(r["text"]))
            out.append({
                "segment_id": str(r["segment_id"]),
                "corpus_id": str(r["corpus_id"]) if "corpus_id" in r.keys() else "",
                "excerpt": clamp(raw_text, PROMPT_EXCERPT_CHARS),
                "full_text": raw_text,
                "source_method": "FTS5"
            })
        return out

    def search_vector(self, query_vector: List[float], corpora: Optional[List[str]], limit: int) -> List[Dict[str, str]]:
        if not self.has_vector_table:
            return []
            
        vec_json = json.dumps(query_vector)
        
        sql = f"""
            SELECT v.segment_id, f.{self.fts_text} as text, f.{self.fts_corpus} as corpus_id
            FROM segments_vec v
            JOIN segments_fts f ON v.segment_id = f.{self.fts_sid}
            WHERE v.embedding MATCH ? 
            ORDER BY v.distance ASC 
            LIMIT ?
        """
        
        try:
            rows = self.con.execute(sql, (vec_json, limit)).fetchall()
        except sqlite3.OperationalError as e:
            return []

        out: List[Dict[str, str]] = []
        for r in rows:
            raw_text = norm_ws(str(r["text"]))
            cid = str(r["corpus_id"]) if "corpus_id" in r.keys() else ""
            if corpora and cid not in corpora:
                continue
                
            out.append({
                "segment_id": str(r["segment_id"]),
                "corpus_id": cid,
                "excerpt": clamp(raw_text, PROMPT_EXCERPT_CHARS),
                "full_text": raw_text,
                "source_method": "VECTOR"
            })
        return out

    def audit(self, segment_id: str) -> Optional[Dict[str, str]]:
        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text, {self.fts_corpus} as corpus_id"
        try:
            row = self.con.execute(f"select {cols} from segments_fts where {self.fts_sid}=? limit 1", (segment_id,)).fetchone()
        except sqlite3.OperationalError: return None
        if not row: return None
        return {
            "segment_id": str(row["segment_id"]),
            "corpus_id": str(row["corpus_id"]) if "corpus_id" in row.keys() else "",
            "text": norm_ws(str(row["text"])),
        }

def parse_gazetteer_title(excerpt: str) -> str:
    m = re.search(r"Title:\s*([^|]+)", excerpt or "")
    return m.group(1).strip() if m else ""

def parse_gazetteer_id(excerpt: str) -> str:
    m = re.search(r"(?:PleiadesID|GazetteerID|ID):\s*([a-zA-Z0-9_-]+)", excerpt or "", re.I)
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
# API Clients (LLM Chat & Embeddings)
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
            if isinstance(e.reason, ConnectionRefusedError):
                raise RuntimeError(f"Connection Refused: Ensure LLM server is running at {base_url}")
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

    raise RuntimeError(f"LLM request failed after retries. LastErr={last_err}. Raw(trunc)={clamp(last_raw or '', 900)}")

def openai_embedding(base_url: str, model: str, text: str, api_key: str) -> List[float]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": text}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            j = json.loads(raw)
            return j["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] Embedding generation failed: {e}")
        return []

# =============================================================================
# Retrieval (Hybrid Quota + Scoring Merge)
# =============================================================================

def compute_jaccard_similarity(query_tokens: Set[str], text_tokens: Set[str]) -> float:
    if not query_tokens or not text_tokens: return 0.0
    intersection = query_tokens.intersection(text_tokens)
    union = query_tokens.union(text_tokens)
    return len(intersection) / len(union)

def hybrid_score_hit(excerpt: str, intent: List[str], loc: List[str], bias: List[str], scope: str, source_method: str, is_social: bool, config: RetrievalConfig) -> float:
    s = (excerpt or "").lower()
    s_toks = set(re.findall(r"[a-z0-9]+", s))
    i_toks = set(intent)
    l_toks = set(loc)
    
    sc = 0.0
    
    if source_method == "VECTOR":
        sc += config.vector_hit_boost
        
    sc += config.jaccard_intent_weight * compute_jaccard_similarity(i_toks, s_toks)
    sc += config.jaccard_loc_weight * compute_jaccard_similarity(l_toks, s_toks)
    
    sc += config.exact_phrase_weight * sum(1 for t in intent if t in s)
    sc += config.exact_bias_weight * sum(1 for t in bias if t in s)
    
    loc_hits = sum(1 for t in loc if t in s)
    
    if is_social:
        sc += (config.loc_hit_social_weight * loc_hits)
    else:
        sc += (config.loc_hit_local_weight * loc_hits) if scope == "local" else (config.loc_hit_global_weight * loc_hits)
    
    return sc

def retrieve_packet_with_quotas(
    store: EvidenceStore, 
    user_text: str, 
    state: Dict[str, Any], 
    corpora_filter: List[str], 
    cooldown_set: Set[str], 
    world: Dict[str, Any],
    era_config: Dict[str, Any],
    debug: bool,
    diag: Diagnostics,
    vec_enabled: bool = False,
    vec_base_url: str = "",
    vec_model: str = "",
    vec_api_key: str = ""
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    
    scope = state.get("scope", "local")
    loc_label = str(state.get("loc_label", world["starting_location"]))
    time_of_day = str(state.get("time_of_day", "morning")).lower()
    weather = str(state.get("weather", "clear")).lower()
    
    itoks = intent_tokens(user_text)
    ltoks = location_tokens(loc_label, world)
    btoks = era_bias_terms(user_text, era_config)
    
    time_boost_terms = TIME_TERMS.get(time_of_day, [])
    weather_boost_terms = WEATHER_TERMS.get(weather, [])
    
    world_anchors = list(world.get("graph", {}).keys())

    if bool(GENERIC_TOUR_RE.match((user_text or "").strip())) and not itoks:
        itoks = [t for t in ltoks if t not in world.get("core_keywords", [])]
        if not itoks and world_anchors: 
            itoks = random.sample(world_anchors, min(2, len(world_anchors)))

    is_social = any(w in (user_text or "").lower() for w in SOCIAL_VERBS)

    q_topo = build_and_with_syn_groups(itoks, world) if itoks else build_or_query(ltoks)
    q_life = build_or_query(world.get("life_terms", []) + time_boost_terms + weather_boost_terms + itoks)
    q_pol = build_or_query(world.get("pol_terms", []) + itoks)

    if is_social:
        queries = [(q_topo, 10), (q_life, 24), (q_pol, 20)]
    else:
        queries = [(q_topo, 24), (q_life, 12), (q_pol, 12)]
    
    all_hits: List[Dict[str, str]] = []
    
    diag.start("Retrieval_FTS")
    for q, limit in queries:
        if q:
            all_hits.extend(store.search_fts(q, corpora_filter if corpora_filter else None, limit=limit))
    
    loc_hit_count = sum(1 for h in all_hits if any(t in (h.get("excerpt", "") or "").lower() for t in ltoks))
    if loc_hit_count == 0 and loc_label.lower() in world.get("graph", {}):
        neighbors = world["graph"][loc_label.lower()].get("connections", [])
        neighbor_names = list(neighbors)
        for n in neighbors:
            neighbor_names.extend(world["graph"].get(n, {}).get("aliases", []))
            
        if neighbor_names:
            if debug: print(f"[DEBUG] 0 Location Hits. Broadening to neighbors: {neighbor_names[:5]}...")
            neighbor_q = build_or_query(neighbor_names)
            all_hits.extend(store.search_fts(neighbor_q, corpora_filter if corpora_filter else None, limit=12))
            
    diag.stop("Retrieval_FTS")
    
    if vec_enabled and vec_model:
        diag.start("Retrieval_Vector")
        vec_query = f"{user_text} in {loc_label} during {time_of_day} weather {weather}"
        query_emb = openai_embedding(vec_base_url, vec_model, vec_query, vec_api_key)
        if query_emb:
            vec_limit = 20 if is_social else 10
            vec_hits = store.search_vector(query_emb, corpora_filter if corpora_filter else None, limit=vec_limit)
            all_hits.extend(vec_hits)
            if debug: print(f"[DEBUG] Vector Retrieval returned {len(vec_hits)} hits.")
        diag.stop("Retrieval_Vector")

    gaz_id = world.get("gazetteer_corpus_id")
    if gaz_id:
        all_hits = [h for h in all_hits if h.get("corpus_id") != gaz_id]

    seen: Set[str] = set()
    uniq: List[Dict[str, str]] = []
    for h in all_hits:
        sid = h["segment_id"]
        if sid not in seen:
            seen.add(sid)
            uniq.append(h)

    diag.start("Retrieval_Rerank")
    scored: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, h in enumerate(uniq):
        sc = hybrid_score_hit(h.get("full_text", ""), itoks, ltoks, btoks, scope, h.get("source_method", "FTS5"), is_social, RETRIEVAL_CONFIG)
        scored.append((sc, idx, h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    diag.stop("Retrieval_Rerank")

    packet: List[Dict[str, str]] = []
    for sc, _, h in scored:
        if h["segment_id"] in cooldown_set: continue
        packet.append(h)
        if len(packet) >= MAX_EVIDENCE: break

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
        "is_social": is_social,
        "packet_n": len(packet),
        "packet_vec_hits": sum(1 for h in packet if h.get("source_method") == "VECTOR"),
    }

    if debug: print(f"[DEBUG] meta={meta}")
    return packet, meta

# =============================================================================
# Plan/Render Schemas, Strict Validation Gates, and Prompt Injection
# =============================================================================

PLAN_REQUIRED_KEYS = {"action_evaluation", "narrative_beats", "state_delta"}
RENDER_REQUIRED_KEYS = {"sensory_environment", "direct_dialogue", "npc_activity", "interactive_opportunities", "observations", "claims"}

def validate_evidence_ids(evs: Any, packet_ids: Set[str]) -> Optional[str]:
    if not isinstance(evs, list): return "evidence_ids must be a list"
    if len(evs) == 0: return None
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

    ae = obj.get("action_evaluation")
    if not isinstance(ae, str) or len(ae.strip()) < 5:
        e.append("action_evaluation must be a string describing physics/social outcome")

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
        if "inventory_add" in delta and delta["inventory_add"] is not None and not isinstance(delta["inventory_add"], list):
            e.append("state_delta.inventory_add must be a list of strings")
        if "inventory_remove" in delta and delta["inventory_remove"] is not None and not isinstance(delta["inventory_remove"], list):
            e.append("state_delta.inventory_remove must be a list of strings")
        if "status_add" in delta and delta["status_add"] is not None and not isinstance(delta["status_add"], list):
            e.append("state_delta.status_add must be a list of strings")
        if "status_remove" in delta and delta["status_remove"] is not None and not isinstance(delta["status_remove"], list):
            e.append("state_delta.status_remove must be a list of strings")
        if "npcs_present" in delta and delta["npcs_present"] is not None and not isinstance(delta["npcs_present"], list):
            e.append("state_delta.npcs_present must be a list of strings")
        if "add_world_note" in delta and delta["add_world_note"] is not None and not isinstance(delta["add_world_note"], str):
            e.append("state_delta.add_world_note must be a string or null")
        
        changes = delta.get("narrative_changes")
        if not isinstance(changes, list):
            e.append("state_delta.narrative_changes must be list")
        else:
            for i, it in enumerate(changes):
                if not isinstance(it, dict): continue
                err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
                if err: e.append(f"state_delta.narrative_changes[{i}].{err}")

    return e

def validate_render(obj: Any, packet: List[Dict[str, str]], is_social: bool) -> List[str]:
    e: List[str] = []
    if not isinstance(obj, dict): return ["render not an object"]
    
    packet_map = {item["segment_id"]: item.get("full_text", item.get("excerpt", "")) for item in packet}
    packet_ids = set(packet_map.keys())

    allowed = RENDER_REQUIRED_KEYS | {"limitations"}
    for k in list(obj.keys()):
        if k not in allowed: del obj[k]
    for k in RENDER_REQUIRED_KEYS:
        if k not in obj: e.append(f"missing {k}")

    se = obj.get("sensory_environment")
    if not isinstance(se, str) or len(se.strip()) < 50:
        e.append("sensory_environment too short or invalid")

    dd = obj.get("direct_dialogue")
    if not isinstance(dd, list) or len(dd) > 15:
        e.append("direct_dialogue must be a list (can be empty, up to 15 items)")
    elif is_social and len(dd) == 0:
        e.append("MECHANICAL DIALOGUE GATE: User intent was social ('talk', 'ask'). You MUST provide 'direct_dialogue'.")

    io = obj.get("interactive_opportunities")
    if not isinstance(io, list) or not (1 <= len(io) <= 6):
        e.append("interactive_opportunities must be list len 1-6")

    na = obj.get("npc_activity")
    if not isinstance(na, list) or not (2 <= len(na) <= 12):
        e.append("npc_activity must be list len 2-12")

    obs = obj.get("observations")
    if not isinstance(obs, list) or not (1 <= len(obs) <= 12):
        e.append("observations must be list len 1-12")
    else:
        for i, it in enumerate(obs):
            if not isinstance(it, dict): continue
            err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
            if err: e.append(f"observations[{i}].{err}")

    claims = obj.get("claims")
    if not isinstance(claims, list) or len(claims) > 16:
        e.append("claims must be a list (0-16 items)")
    else:
        for i, it in enumerate(claims):
            if not isinstance(it, dict): continue
            
            err = validate_evidence_ids(it.get("evidence_ids"), packet_ids)
            if err: 
                e.append(f"claims[{i}].{err}")
                continue
            
            quote = it.get("quote")
            if not isinstance(quote, str) or len(quote.strip()) < 3:
                e.append(f"claims[{i}] missing or invalid 'quote' (must be a literal substring)")
                continue
            
            quote_clean = clean_for_match(quote)
            found = False
            for sid in it.get("evidence_ids", []):
                full_text_clean = clean_for_match(packet_map.get(sid, ""))
                if quote_clean in full_text_clean:
                    found = True
                    break
            
            if not found:
                e.append(f"claims[{i}] REJECTED BY CITATION GATE: quote '{clamp(quote, 35)}' not found literally in cited primary sources.")

    lim = obj.get("limitations")
    if lim is not None and lim != "" and not isinstance(lim, str):
        e.append("limitations must be string if present")

    return e

def build_system_prompt(era_config: Dict[str, Any], running_summary: str) -> str:
    text = PROMPTS_DATA["system_prompt"]
    text = text.replace("{system_context}", era_config['system_context'])
    text = text.replace("{architectural_notes}", era_config['architectural_notes'])
    text = text.replace("{running_summary}", running_summary if running_summary else "None yet.")
    return text

def build_plan_prompt(state: Dict[str, Any], history: List[Dict[str, str]], user_text: str, system_constraint: str, packet: List[Dict[str, str]]) -> str:
    d = copy.deepcopy(PROMPTS_DATA["plan_prompt"])
    d["player_state"] = state
    d["recent_history"] = history
    d["system_constraint"] = system_constraint
    d["user_action"] = user_text
    d["evidence_packet"] = [{"segment_id": p["segment_id"], "text": p["excerpt"]} for p in packet]
    return json.dumps(d, ensure_ascii=False)

def build_render_prompt(state: Dict[str, Any], history: List[Dict[str, str]], plan: Dict[str, Any], packet: List[Dict[str, str]]) -> str:
    d = copy.deepcopy(PROMPTS_DATA["render_prompt"])
    d["player_state"] = state
    d["recent_history"] = history
    d["engine_plan"] = plan
    d["evidence_packet"] = [{"segment_id": p["segment_id"], "text": p["excerpt"]} for p in packet]
    return json.dumps(d, ensure_ascii=False)

def build_auditor_prompt(scene_text: str, packet: List[Dict[str, str]]) -> str:
    d = copy.deepcopy(PROMPTS_DATA["auditor_prompt"])
    d["scene_text"] = scene_text
    d["evidence_packet"] = [{"id": p["segment_id"], "text": p["excerpt"]} for p in packet]
    return json.dumps(d, ensure_ascii=False)

def build_repair_prompt(kind: str, bad: str, errors: List[str], packet: List[Dict[str, str]] = None) -> str:
    d = copy.deepcopy(PROMPTS_DATA["repair_prompt"])
    d["task"] = d["task"].replace("{__kind__}", kind)
    d["validation_errors"] = errors
    d["invalid_output"] = clamp(bad, 2000)
    d["evidence_packet"] = [{"id": p["segment_id"], "text": p["excerpt"]} for p in (packet or [])]
    return json.dumps(d, ensure_ascii=False)

def build_suggest_prompt(state: Dict[str, Any], packet: List[Dict[str, str]], n: int, era_config: Dict[str, Any]) -> str:
    d = copy.deepcopy(PROMPTS_DATA["suggest_prompt"])
    d["state"] = state
    d["era_context"] = era_config.get("system_context", "")
    d["architectural_rules"] = era_config.get("architectural_notes", "")
    d["evidence_packet"] = [{"id": p["segment_id"], "text": p["excerpt"]} for p in packet]
    d["requirements"] = [req.replace("{__n__}", str(n)) for req in d["requirements"]]
    return json.dumps(d, ensure_ascii=False)


# =============================================================================
# Main CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Vivarium: Stateful Historical Simulation Engine")
    ap.add_argument("--db", required=True, help="Path to the Scriptorium SQLite DB")
    ap.add_argument("--base-url", default="http://localhost:1234/v1", help="LLM API Base URL")
    ap.add_argument("--model", required=True, help="LLM Model Name")
    ap.add_argument("--api-key", default="lm-studio", help="API Key")
    ap.add_argument("--world", default="rome", choices=list(WORLDS.keys()), help="Select the simulated world setting.")
    ap.add_argument("--era", type=str, default="", help="Starting era (e.g. 'augustan', 'flavian', 'late_empire')")
    ap.add_argument("--corpus", action="append", default=[], help="Optional: restrict retrieval to specific corpus_id")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=16000)
    
    ap.add_argument("--debug", action="store_true", help="Print backend engine metadata, telemetry, and validation traces")
    ap.add_argument("--strict-audit", action="store_true", help="Enable secondary LLM pass to ban orphan hallucinations entirely")
    ap.add_argument("--show-observations", action="store_true", default=False)
    ap.add_argument("--show-claims", action="store_true", default=False)
    
    ap.add_argument("--enable-vector", action="store_true", help="Enables Vector Search (requires sqlite-vec)")
    ap.add_argument("--vec-model", type=str, default="nomic-embed-text-v1.5", help="Embedding model")
    ap.add_argument("--vec-base-url", type=str, default="http://localhost:1234/v1", help="Base URL for embeddings")
    args = ap.parse_args()

    world_config = WORLDS[args.world]
    store = EvidenceStore(args.db)
    diag = Diagnostics(args.debug, args.db)
    
    # ---------------------------------------------------------
    # Persistent State Initialization
    # ---------------------------------------------------------
    saved_state, saved_history, saved_cooldown = load_session(args.db, world_config["name"])
    
    if saved_cooldown is not None:
        cooldown = saved_cooldown
        cooldown_set = set(cooldown)
    else:
        cooldown: collections.deque[str] = collections.deque(maxlen=COOLDOWN_SIZE)
        cooldown_set: Set[str] = set()
        
    if saved_history is not None:
        history = saved_history
    else:
        history: collections.deque[Dict[str, Any]] = collections.deque(maxlen=5)

    if saved_state is not None:
        state = saved_state
        state.setdefault("inventory", ["a few bronze coins"])
        state.setdefault("status_effects", [])
        state.setdefault("npcs_present", [])
        state.setdefault("world_notes", [])
        state.setdefault("running_summary", "")
        
        if world_config.get("eras"):
            state.setdefault("era", list(world_config["eras"].keys())[0])
        else:
            state.setdefault("era", "")
        state.setdefault("turn_index", 0)
        state.setdefault("fatigue", 0)
        state.setdefault("scope", "local")
        state.setdefault("time_minutes", 480)
        state.setdefault("day", 1)
        state.setdefault("weather", "clear")
        
        if args.era and args.era in world_config.get("eras", {}):
            state["era"] = args.era
            
        print(f"[OK] Resuming saved session at Turn {state.get('turn_index', 0)} in {state.get('loc_label', 'Unknown')}.")
    else:
        initial_era = args.era
        if not initial_era or initial_era not in world_config.get("eras", {}):
            initial_era = list(world_config.get("eras", {}).keys())[0] if world_config.get("eras") else ""
            
        state: Dict[str, Any] = {
            "loc_label": world_config["starting_location"],
            "loc_id": None,
            "coords": None,
            "era": initial_era,
            "day": 1,
            "time_minutes": 480, 
            "time_of_day": "morning",
            "weather": "clear",
            "scope": "local",
            "fatigue": 0,
            "inventory": ["a few bronze coins"],
            "status_effects": [],
            "npcs_present": [],
            "world_notes": [],
            "running_summary": "",
            "turn_index": 0
        }

    last_packet: List[Dict[str, str]] = []

    # ---------------------------------------------------------
    # Banner & UI
    # ---------------------------------------------------------
    print("=" * 92)
    print(f" V I V A R I U M   |   {world_config['name'].upper()} ")
    print(" Stateful Historical Simulation Engine ")
    print("=" * 92)
    print(f"[DB] Core loaded: {args.db}")
    if args.strict_audit:
        print("[AUDIT] STRICT AUDIT mode enabled.")
    if args.enable_vector:
        if store.has_vector_table:
            print(f"[VECTOR] Semantic Search ENABLED via {args.vec_model}")
        else:
            print("[WARN] Vector Search requested, but 'segments_vec' table missing. Falling back to FTS5-only.")
            args.enable_vector = False

    print("\nType :help for a manual, or just start typing.")
    print("=" * 92)

    try:
        while True:
            if state["turn_index"] > 0 and state["turn_index"] % 12 == 0:
                state["weather"] = random.choice(WEATHER_STATES)

            state["day"] = 1 + (state["time_minutes"] // 1440)
            state["time_of_day"] = minutes_to_time_of_day(state["time_minutes"])
            
            era_key = state.get("era", "")
            valid_eras = world_config.get("eras", {})
            if not valid_eras:
                era_config = {"system_context": world_config.get("system_context", ""), "architectural_notes": "", "bias_groups": []}
            elif era_key not in valid_eras:
                era_key = list(valid_eras.keys())[0]
                state["era"] = era_key
                era_config = valid_eras[era_key]
            else:
                era_config = valid_eras[era_key]

            try:
                inv_str = ", ".join(state.get("inventory", [])) if state.get("inventory") else "Empty"
                stat_str = ", ".join(state.get("status_effects", [])) if state.get("status_effects") else "Normal"
                
                print(f"\n[{state['loc_label']} | {state['time_of_day'].title()} | {state['weather'].title()} (Day {state['day']}) | Era: {state['era'].title()}]")
                print(f"[Fatigue: {state['fatigue']}% | Inv: {clamp(inv_str, 40)} | Stat: {stat_str}]")
                user_in = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[OK] Graceful exit triggered.")
                break

            if not user_in:
                continue

            # ----- Out-of-Band Commands -----
            if user_in.startswith(":"):
                parts = user_in[1:].split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("q", "quit", "exit"):
                    break 
                if cmd == "help":
                    print_help_ui(world_config)
                    continue
                if cmd == "state":
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    continue
                if cmd == "reset":
                    save_path = get_save_path(args.db, world_config["name"])
                    if os.path.exists(save_path):
                        os.remove(save_path)
                        print("[OK] Session save wiped. Restart script to begin fresh.")
                    else:
                        print("[OK] No save file found.")
                    continue
                if cmd == "time":
                    time_map = {"dawn": 360, "morning": 480, "noon": 720, "afternoon": 900, "evening": 1140, "night": 1320}
                    if arg not in time_map:
                        print("[WARN] :time must be dawn|morning|noon|afternoon|evening|night")
                        continue
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
                if cmd == "era":
                    if not arg or arg not in world_config.get("eras", {}):
                        print(f"[WARN] Invalid era. Choices: {list(world_config.get('eras', {}).keys())}")
                        continue
                    state["era"] = arg
                    state["npcs_present"] = [] 
                    print(f"\n[TIME TRAVEL] The world shimmers and shifts. You are now in the {arg.upper()} era.")
                    print(f"Context: {world_config['eras'][arg]['system_context']}")
                    continue
                if cmd == "goto":
                    if not arg:
                        print("[WARN] :goto needs a place label")
                        continue
                    state["loc_label"] = arg.title()
                    state["loc_id"] = None
                    state["coords"] = None
                    state["npcs_present"] = []
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
                    print(f"\n[AUDIT] {row.get('corpus_id','')}:{row['segment_id']}")
                    print("-" * 92)
                    print(textwrap.fill(clamp(row["text"], 4000), width=92))
                    print("-" * 92)
                    continue
                if cmd == "place":
                    if not arg:
                        print("[WARN] :place needs a name")
                        continue
                    gaz_id = world_config.get("gazetteer_corpus_id")
                    if not gaz_id:
                        print("[PLACE] No gazetteer corpus configured for this world.")
                        continue
                    q = build_and_with_syn_groups(intent_tokens(arg), world_config) or build_or_query(intent_tokens(arg)) or build_or_query(location_tokens(arg, world_config))
                    hits = store.search_fts(q, [gaz_id], limit=5)
                    if not hits:
                        print("[PLACE] No matches found in gazetteer.")
                        continue
                    print(f"\n[PLACE] Top {len(hits)} matches:")
                    for i, h in enumerate(hits, start=1):
                        title = parse_gazetteer_title(h["excerpt"]) or h["segment_id"]
                        pid = parse_gazetteer_id(h["excerpt"]) or ""
                        tag = f"(id={pid})" if pid else ""
                        print(f"  {i}) {title} {tag} :: {clamp(h['excerpt'], 160)}")
                    sel = input("\nPick a number to set location (Enter cancels): ").strip()
                    if sel.isdigit():
                        k = int(sel)
                        if 1 <= k <= len(hits):
                            chosen = hits[k - 1]
                            title = parse_gazetteer_title(chosen["excerpt"]) or chosen["segment_id"]
                            pid = parse_gazetteer_id(chosen["excerpt"]) or None
                            coords = parse_coords(chosen["excerpt"])
                            state["loc_label"] = title.title()
                            state["loc_id"] = pid
                            state["coords"] = coords
                            state["npcs_present"] = [] 
                            print(f"[OK] location set -> {state['loc_label']}" + (f" (gazetteer={pid})" if pid else ""))
                    continue
                if cmd == "suggest":
                    n = max(3, min(12, int(arg))) if arg.isdigit() else 6
                    world_anchors = list(world_config.get("graph", {}).keys())
                    if not last_packet:
                        cands = random.sample(world_anchors, min(n, len(world_anchors))) if world_anchors else ["the market"]
                        print("[SUGGEST] Try prompts like:")
                        for c in cands: print(f"  - take me to the {c}")
                        continue
                    
                    diag.start("LLM_Suggest")
                    msgs = [
                        {"role": "system", "content": build_system_prompt(era_config, state.get("running_summary", ""))},
                        {"role": "user", "content": "/no_think\n" + build_suggest_prompt(state, last_packet, n, era_config)},
                    ]
                    raw = openai_chat(args.base_url, args.model, msgs, args.temperature, 700, args.api_key)
                    diag.stop("LLM_Suggest")
                    
                    obj, err = try_json(raw)
                    if err or not isinstance(obj, dict) or "suggestions" not in obj or not isinstance(obj["suggestions"], list):
                        print("[WARN] suggest parse failed; raw:", clamp(raw, 300))
                        continue
                    print("\n[SUGGESTIONS]")
                    for i, s in enumerate(obj["suggestions"], start=1):
                        if isinstance(s, str) and s.strip():
                            print(f"  {i}) {s.strip()}")
                    continue

                print("[WARN] unknown command")
                continue

            # ----- Core Turn Validation & Bounds Checking -----
            if len(user_in) < 3:
                print("[WARN] Input too short.")
                continue

            state_snapshot = copy.deepcopy(state)

            try:
                system_constraint = ""
                pre_applied_time = 0
                is_social = any(w in user_in.lower() for w in SOCIAL_VERBS)
                
                # --- 1. Dynamic City-Scale Spatial Navigation ---
                diag.start("Graph_Navigation")
                m_goto = AUTO_GOTO_RE.match(user_in)
                m_move = GENERIC_MOVE_RE.match(user_in)
                
                if m_goto:
                    target_raw = m_goto.group(3).strip().lower().strip(".")
                    current_loc = state["loc_label"].lower()
                    world_graph = world_config.get("graph", {})
                    
                    tgt_toks = set(re.findall(r"\w+", target_raw))
                    best_match = None
                    best_overlap = 0
                    
                    for node_key, node_data in world_graph.items():
                        valid_names = [node_key] + node_data.get("aliases", [])
                        if target_raw in valid_names:
                            best_match = node_key
                            best_overlap = len(tgt_toks) or 1
                            break
                            
                    if not best_match:
                        for node_key, node_data in world_graph.items():
                            node_toks = set()
                            valid_names = [node_key] + node_data.get("aliases", [])
                            for name in valid_names:
                                node_toks.update(re.findall(r"\w+", name))
                            
                            overlap = len(tgt_toks & node_toks)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_match = node_key
                    
                    if best_match and best_overlap >= 1 and any(t not in STOPWORDS for t in tgt_toks & set(re.findall(r"\w+", best_match))):
                        best_match_title = best_match.title()
                        
                        node_era_reqs = world_graph[best_match].get("era_unlocked")
                        if node_era_reqs and state["era"] not in node_era_reqs:
                            system_constraint = f"Player tried to go to {best_match_title}, but it does not exist in the {state['era'].title()} era. They are still at {current_loc.title()}."
                        elif best_match == current_loc:
                            system_constraint = f"Player is already at {best_match_title}."
                        elif best_match in world_graph.get(current_loc, {}).get("connections", []):
                            state["loc_label"] = best_match_title
                            state["loc_id"] = None
                            state["npcs_present"] = [] 
                            system_constraint = f"Player has initiated local travel to {best_match_title}. It takes 15 minutes. Render the journey and arrival."
                            state["time_minutes"] += 15
                            pre_applied_time = 15
                        else:
                            system_constraint = f"Player tried to go to {best_match_title}, but it is not directly connected to {current_loc.title()}. They are still at {current_loc.title()}."
                    else:
                        gaz_id = world_config.get("gazetteer_corpus_id")
                        if gaz_id:
                            q = build_and_with_syn_groups(intent_tokens(target_raw), world_config) or build_or_query(intent_tokens(target_raw)) or build_or_query(location_tokens(target_raw, world_config))
                            hits = store.search_fts(q, [gaz_id], limit=5)
                            if hits:
                                t0 = parse_gazetteer_title(hits[0]["excerpt"]) or target_raw
                                conf0 = title_match_confidence(target_raw, t0)
                                conf1 = 0.0
                                if len(hits) > 1:
                                    t1 = parse_gazetteer_title(hits[1]["excerpt"]) or ""
                                    conf1 = title_match_confidence(target_raw, t1)
                                
                                if conf0 >= 0.75 and (conf0 - conf1) >= 0.25:
                                    target_coords = parse_coords(hits[0]["excerpt"])
                                    
                                    travel_time = 30 
                                    if state.get("coords") and target_coords:
                                        dist_km = haversine_distance(state["coords"][0], state["coords"][1], target_coords[0], target_coords[1])
                                        travel_time = max(5, int(dist_km * 12))
                                        
                                        if args.debug:
                                            print(f"[DEBUG Navigation] Distance to {t0.title()}: {dist_km:.2f} km. Est time: {travel_time}m")
                                    
                                    state["loc_label"] = t0.title()
                                    state["loc_id"] = parse_gazetteer_id(hits[0]["excerpt"]) or None
                                    state["coords"] = target_coords
                                    state["npcs_present"] = [] 
                                    system_constraint = f"Player traveled geographically to {t0.title()}. It takes {travel_time} minutes. Render the journey and arrival."
                                    state["time_minutes"] += travel_time
                                    pre_applied_time = travel_time
                                else:
                                    system_constraint = f"Player tried to travel to {target_raw.title()} but the location is ambiguous. Still at {current_loc.title()}."
                            else:
                                system_constraint = f"Player tried to travel to {target_raw.title()} but it does not exist on the map. Still at {current_loc.title()}."
                
                elif m_move:
                    direction = m_move.group(2).lower()
                    system_constraint = f"Player is moving directionally ({direction}) without a named destination. Update their location organically in the narrative."

                diag.stop("Graph_Navigation")
                
                if args.debug and system_constraint:
                    print(f"[DEBUG Engine Constraint] {system_constraint}")

                # --- 2. Evidence Retrieval ---
                packet, meta = retrieve_packet_with_quotas(
                    store=store,
                    user_text=user_in,
                    state=state,
                    corpora_filter=args.corpus,
                    cooldown_set=cooldown_set,
                    world=world_config,
                    era_config=era_config,
                    debug=args.debug,
                    diag=diag,
                    vec_enabled=args.enable_vector,
                    vec_base_url=args.vec_base_url,
                    vec_model=args.vec_model,
                    vec_api_key=args.api_key
                )

                if not packet:
                    print("[WARN] No evidence returned. Try a more specific prompt or use :place / :goto.")
                    state = state_snapshot # ROLLBACK
                    continue

                last_packet = packet
                packet_ids = {e["segment_id"] for e in packet}

                if state["turn_index"] > 0 and state["turn_index"] % 5 == 0 and packet:
                    event_seed = random.choice(packet)
                    system_constraint += f" DYNAMIC WORLD EVENT: Organically weave this specific historical detail into the background as an active, spontaneous event the player witnesses right now: '{clamp(event_seed.get('excerpt',''), 300)}'."

                # --- 3. Plan Pass ---
                diag.start("LLM_Plan_Pass")
                plan_obj: Optional[Dict[str, Any]] = None
                plan_last = ""
                plan_errs: List[str] = []
                for attempt in range(1, TURNPLAN_RETRIES + 1):
                    msgs = [
                        {"role": "system", "content": build_system_prompt(era_config, state.get("running_summary", ""))},
                        {"role": "user", "content": "/no_think\n" + build_plan_prompt(state, list(history), user_in, system_constraint, packet)},
                    ]
                    if attempt > 1:
                        msgs.append({"role": "assistant", "content": plan_last})
                        msgs.append({"role": "user", "content": "/no_think\n" + build_repair_prompt("TurnPlan", plan_last, plan_errs, packet)})
                    
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
                diag.stop("LLM_Plan_Pass")

                if plan_obj is None:
                    print("\n[FAIL] TurnPlan validation failed. Rolling back state.")
                    state = state_snapshot 
                    continue

                # --- 4. Apply State Deltas Dynamically ---
                diag.start("Engine_State_Delta")
                delta = plan_obj.get("state_delta", {})
                try: 
                    adv = max(0, int(delta.get("time_advanced_minutes", 0)))
                    adv = max(0, adv - pre_applied_time)
                    state["time_minutes"] += adv
                except: pass
                
                state["day"] = 1 + (state["time_minutes"] // 1440)
                state["time_of_day"] = minutes_to_time_of_day(state["time_minutes"])

                try: 
                    f_delta = int(delta.get("fatigue_change", 0))
                    state["fatigue"] = max(0, min(100, state["fatigue"] + f_delta))
                except: pass

                invalid_items = {"none", "null", "n/a", "nothing", ""}
                
                if "inventory_add" in delta and isinstance(delta["inventory_add"], list):
                    for item in delta["inventory_add"]:
                        if isinstance(item, str) and item.strip().lower() not in invalid_items:
                            if item.strip() not in state["inventory"]:
                                state["inventory"].append(item.strip())
                
                if "inventory_remove" in delta and isinstance(delta["inventory_remove"], list):
                    for item in delta["inventory_remove"]:
                        if item in state["inventory"]:
                            state["inventory"].remove(item)

                if "status_add" in delta and isinstance(delta["status_add"], list):
                    for status in delta["status_add"]:
                        if isinstance(status, str) and status.strip().lower() not in invalid_items:
                            if status.strip() not in state["status_effects"]:
                                state["status_effects"].append(status.strip())

                if "status_remove" in delta and isinstance(delta["status_remove"], list):
                    for status in delta["status_remove"]:
                        if status in state["status_effects"]:
                            state["status_effects"].remove(status)
                
                if "npcs_present" in delta and isinstance(delta["npcs_present"], list):
                    valid_npcs = [str(x).strip() for x in delta["npcs_present"] if str(x).strip().lower() not in invalid_items]
                    state["npcs_present"] = valid_npcs[:5]
                            
                if "add_world_note" in delta and isinstance(delta["add_world_note"], str):
                    note = delta["add_world_note"].strip()
                    if note.lower() not in invalid_items:
                        if note not in state["world_notes"]:
                            if len(state["world_notes"]) >= 15:
                                state["world_notes"].pop(0) 
                            state["world_notes"].append(note)

                state["turn_index"] += 1
                diag.stop("Engine_State_Delta")

                if args.debug:
                    print(f"[DEBUG Action Eval] {plan_obj.get('action_evaluation')}")

                # --- 5. Render Pass ---
                diag.start("LLM_Render_Pass")
                render_obj: Optional[Dict[str, Any]] = None
                render_last = ""
                render_errs: List[str] = []
                for attempt in range(1, RENDER_RETRIES + 1):
                    msgs = [
                        {"role": "system", "content": build_system_prompt(era_config, state.get("running_summary", ""))},
                        {"role": "user", "content": "/no_think\n" + build_render_prompt(state, list(history), plan_obj, packet)},
                    ]
                    if attempt > 1:
                        msgs.append({"role": "assistant", "content": render_last})
                        msgs.append({"role": "user", "content": "/no_think\n" + build_repair_prompt("Render", render_last, render_errs, packet)})
                    
                    out = openai_chat(args.base_url, args.model, msgs, args.temperature + 0.1, args.max_tokens, args.api_key)
                    render_last = out
                    obj, err = try_json(out)
                    if err:
                        render_errs = [f"JSON parse error: {err}"]
                        continue
                    
                    render_errs = validate_render(obj, packet, is_social)
                    
                    if not render_errs and args.strict_audit:
                        diag.start("LLM_Strict_Auditor")
                        audit_msgs = [
                            {"role": "user", "content": "/no_think\n" + build_auditor_prompt(obj.get("sensory_environment", "") + " " + " ".join(obj.get("npc_activity", [])), packet)}
                        ]
                        audit_out = openai_chat(args.base_url, args.model, audit_msgs, 0.1, 512, args.api_key)
                        diag.stop("LLM_Strict_Auditor")
                        
                        audit_obj, audit_err = try_json(audit_out)
                        if audit_obj and audit_obj.get("pass") is False:
                            unsupported = audit_obj.get("unsupported_claims", [])
                            render_errs.append(f"STRICT AUDITOR FAILED: Hallucinated details detected - {unsupported}")

                    if render_errs:
                        if args.debug: print(f"[DEBUG] Render invalid (try {attempt}): {render_errs}")
                        continue
                    render_obj = obj
                    break
                diag.stop("LLM_Render_Pass")

                if render_obj is None:
                    print("\n[FAIL] Render validation failed. Rolling back state.")
                    state = state_snapshot 
                    continue

                # =================================================================
                # TRANSACTION COMMIT & LONG-TERM MEMORY SUMMARIZATION
                # =================================================================
                diag.start("Engine_Commit")
                for e in packet:
                    sid = e["segment_id"]
                    if sid not in cooldown_set:
                        cooldown.append(sid)
                cooldown_set = set(cooldown)

                sensory = de_cliche_opening(render_obj["sensory_environment"], state["loc_label"], state["time_of_day"], state["weather"], prepend_context=True)
                npc_lines = [de_cliche_opening(line, state["loc_label"], state["time_of_day"], state["weather"], prepend_context=False) for line in render_obj["npc_activity"]]

                hist_entry = {
                    "turn": state["turn_index"],
                    "player_action": user_in,
                    "engine_narrative": sensory,
                    "action_evaluation": plan_obj.get("action_evaluation", "Success"),
                }
                if render_obj.get("direct_dialogue"):
                    hist_entry["dialogue_spoken"] = render_obj["direct_dialogue"]
                
                # Intercept Dropped History Item for Memory Summarization
                if len(history) == history.maxlen:
                    dropped_turn = history[0]
                    try:
                        dropped_str = json.dumps(dropped_turn, ensure_ascii=False)
                        summary_prompt_text = PROMPTS_DATA["summarize_prompt"].replace("{dropped_turn}", dropped_str)
                        sum_msgs = [
                            {"role": "system", "content": "You are a concise narrative memory summarizer."},
                            {"role": "user", "content": summary_prompt_text}
                        ]
                        # Use deterministic temp for summarizer
                        summary_out = openai_chat(args.base_url, args.model, sum_msgs, 0.1, 100, args.api_key)
                        
                        if state.get("running_summary"):
                            state["running_summary"] += " " + summary_out.strip()
                        else:
                            state["running_summary"] = summary_out.strip()
                        
                        # Cap the context block strictly at ~200-300 words to avoid explosion over long sessions
                        sum_words = state["running_summary"].split()
                        if len(sum_words) > 300:
                            state["running_summary"] = " ".join(sum_words[-300:])
                            
                    except Exception as e:
                        if args.debug:
                            print(f"[DEBUG] Memory Summarizer failed: {e}")

                history.append(hist_entry)

                # Output Formatting
                print("\n" + "=" * 92)
                print(textwrap.fill(sensory.strip(), width=92))
                
                if render_obj.get("direct_dialogue"):
                    print("\n[Dialogue]")
                    for line in render_obj["direct_dialogue"]:
                        print(textwrap.fill(f"\"{line.strip()}\"", width=92, initial_indent="  ", subsequent_indent="  "))

                print("\n[Ambient Activity]")
                for line in npc_lines:
                    print(textwrap.fill("- " + line.strip(), width=92))

                if render_obj.get("interactive_opportunities"):
                    print("\n[Opportunities]")
                    for opp in render_obj["interactive_opportunities"]:
                        print(textwrap.fill(f"> {opp.strip()}", width=92))

                lim = render_obj.get("limitations")
                if lim:
                    print("\n[Limitations]")
                    print(textwrap.fill(lim.strip(), width=92))

                if args.show_observations:
                    print("\n[Observations]")
                    for o in render_obj["observations"]:
                        print(f"- {o['text']} (source {','.join(o.get('evidence_ids', []))})")

                if args.show_claims and render_obj.get("claims"):
                    print("\n[Claims (Auditable & Mechanically Proven)]")
                    for c in render_obj["claims"]:
                        print(f"- {c['claim']} (quote: \"{c.get('quote')}\" | source {','.join(c.get('evidence_ids', []))})")
                
                diag.stop("Engine_Commit")
                
                diag.record_turn(state["turn_index"])
                diag.print_report()

            except KeyboardInterrupt:
                print("\n[WARN] Turn cancelled by user. State rolled back.")
                state = state_snapshot 
                continue
            except RuntimeError as e:
                print(f"\n[FATAL] Local API Error: {e}")
                state = state_snapshot 
                break 
            except Exception as e:
                print(f"\n[ERROR] Unexpected engine failure: {e}")
                state = state_snapshot 
                continue

    except Exception as e:
        print(f"\n[FATAL] Uncaught top-level error: {e}")
        raise
    finally:
        print("\n[OK] Engine shutting down. Auto-saving session...")
        save_session(args.db, world_config["name"], state, history, cooldown)

if __name__ == "__main__":
    main()