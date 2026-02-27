#!/usr/bin/env python3
from __future__ import annotations

import argparse, collections, json, os, re, sqlite3, sys, textwrap, urllib.request
from typing import Any, Dict, List, Optional, Tuple

COOLDOWN_SIZE = 200
MAX_EVIDENCE = 22
PROMPT_EXCERPT_CHARS = 320
TURNPLAN_RETRIES = 4
RENDER_RETRIES = 4
MIN_PACKET_LOCAL = 10

PLEIADES_CORPUS_ID = "viv_pleiades_places"  # from your ingester

ROME_CORE_ANCHORS = [
    "forum", "forum romanum", "palatine", "capitoline", "comitium", "rostra", "curia",
    "tiber", "subura", "campus martius", "via sacra", "circus maximus", "janiculum",
    "transtiberim", "temple", "basilica", "macellum", "tabernae"
]

SYNONYMS = {
    "barracks": ["castra", "quarters", "camp", "praetorian", "cohort", "legion", "praetorium", "castra praetoria"],
    "legionary": ["legion", "cohort", "century", "centurion", "castra", "contubernium", "miles"],
    "soldier": ["miles", "legionary", "centurion", "cohort", "castra"],
    "market": ["macellum", "tabernae", "shops", "corn", "annona"],
    "forum": ["forum", "comitium", "rostra", "curia"],
    "basilica": ["basilica", "tribunal", "court", "hall"],
    "temple": ["templum", "aedes", "shrine"],
    "bath": ["thermae", "balneum"],
}

STOPWORDS = {
    "the","and","for","with","from","into","onto","over","under","during","time","city",
    "show","around","take","bring","walk","me","you","us","to","in","of","a","an",
    "roman","rome"  # handled separately via loc_label
}

GENERIC_TOUR_RE = re.compile(
    r"^(show me around|show me the city|give me a tour|walk me around|take me around|tour|around)\.?$",
    re.IGNORECASE
)

AUTO_GOTO_RE = re.compile(r"^(take me to|go to|bring me to|walk me to|take me into)\s+(the\s+)?(.+)$", re.IGNORECASE)

def die(msg: str, code: int = 1) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(code)

def clamp(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def try_json(s: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

def fts_tokens(text: str) -> List[str]:
    t = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    t = [x for x in t if len(x) >= 3 and not x.isdigit() and x not in STOPWORDS]
    seen=set(); out=[]
    for x in t:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:18]

def expand_synonyms(tokens: List[str]) -> List[str]:
    out = list(tokens)
    s = set(out)
    for t in list(tokens):
        for syn in SYNONYMS.get(t, []):
            syn = syn.lower()
            if syn not in s:
                out.append(syn)
                s.add(syn)
    return out[:30]

def time_bias_terms(user_text: str) -> List[str]:
    u = (user_text or "").lower()
    out: List[str] = []
    if any(k in u for k in ["roman empire", "the empire", "imperial", "principate", "dominate", "emperor", "reign"]):
        out += ["imperial", "augustus", "tiberius", "claudius", "nero", "trajan", "hadrian", "severus", "a.d.", "ad"]
    if any(k in u for k in ["republic", "late republic", "early republic", "caesar", "cicero"]):
        out += ["republic", "consul", "senate", "tribune", "comitia", "praetor", "censor", "b.c.", "bc", "caesar", "cicero"]
    seen=set(); uniq=[]
    for x in out:
        x=x.strip()
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq[:12]

def fts_query(text: str) -> str:
    toks = expand_synonyms(fts_tokens(text))
    if not toks:
        return "rome OR forum OR palatine OR capitoline OR tiber OR subura"
    return " OR ".join(toks)

def openai_chat(base_url: str, model: str, messages: List[Dict[str,str]], temperature: float, max_tokens: int, api_key: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        die(f"LLM request failed: {e}")
    try:
        j = json.loads(raw)
        return j["choices"][0]["message"]["content"]
    except Exception:
        die(f"LLM response parse failed. Raw(trunc)={clamp(raw,800)}")

def de_cliche_opening(narr: str, loc_label: str, time_of_day: str) -> str:
    s = narr.strip()
    s = re.sub(r"^(in\s+the\s+morning\s+light[, ]+|morning\s+light\s+spills\s+across\s+|the\s+morning\s+light\s+spills\s+across\s+)", "", s, flags=re.IGNORECASE).strip()
    if not re.search(re.escape(loc_label), s, flags=re.IGNORECASE):
        s = f"{loc_label} ({time_of_day}). " + s
    return s

def loc_tokens(loc_label: str) -> List[str]:
    toks = re.findall(r"[A-Za-z]+", (loc_label or "").lower())
    toks = [t for t in toks if len(t) >= 4]
    if any(t in toks for t in ["rome","roman","romanum"]):
        for x in ["rome","roma"]:
            if x not in toks:
                toks.append(x)
    seen=set(); out=[]
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:10]

def is_loc_relevant(excerpt: str, tokens: List[str]) -> bool:
    if not tokens:
        return True
    s = (excerpt or "").lower()
    return any(t in s for t in tokens)

def parse_pleiades_title(excerpt: str) -> str:
    # our ingester writes "Title: X | PleiadesID: ..."
    m = re.search(r"Title:\s*([^|]+)", excerpt)
    if m:
        return m.group(1).strip()
    return ""

class EvidenceStore:
    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            die(f"DB not found: {db_path}")
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        if not self._table("segments_fts"):
            die("Missing segments_fts")

        self.has_segments = self._table("segments")
        self.fts_cols = [r[1] for r in self.con.execute("PRAGMA table_info(segments_fts)").fetchall()]

        def pick(cols, cands):
            s=set(cols)
            for c in cands:
                if c in s: return c
            return None

        self.fts_text   = pick(self.fts_cols, ["txt","text","content","segment_text","body"])
        self.fts_corpus = pick(self.fts_cols, ["corpus_id","src"])
        self.fts_sid    = pick(self.fts_cols, ["segment_id","id"])
        if not self.fts_text or not self.fts_sid:
            die(f"Could not detect required columns in segments_fts. cols={self.fts_cols}")

    def _table(self, name: str) -> bool:
        return self.con.execute("select 1 from sqlite_master where type in ('table','view') and name=?", (name,)).fetchone() is not None

    def close(self):
        try: self.con.close()
        except: pass

    def search(self, q: str, corpora: Optional[List[str]], limit: int) -> List[Dict[str,str]]:
        q = (q or "").strip()
        if not q: return []

        where_c = ""
        params: List[Any] = [q]
        if corpora and self.fts_corpus:
            where_c = " AND " + self.fts_corpus + " IN (" + ",".join(["?"]*len(corpora)) + ")"
            params += corpora
        params.append(limit)

        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text"
        if self.fts_corpus:
            cols = f"{cols}, {self.fts_corpus} as corpus_id"

        sql = f"""
          select {cols}
          from segments_fts
          where segments_fts match ?
          {where_c}
          order by bm25(segments_fts)
          limit ?
        """
        rows = self.con.execute(sql, params).fetchall()
        out=[]
        for r in rows:
            out.append({
                "segment_id": str(r["segment_id"]),
                "corpus_id": str(r["corpus_id"]) if "corpus_id" in r.keys() else "",
                "excerpt": norm_ws(str(r["text"]))
            })
        return out

    def audit(self, segment_id: str) -> Optional[Dict[str,str]]:
        # use segments_fts only (fast and enough)
        cols = f"{self.fts_sid} as segment_id, {self.fts_text} as text"
        if self.fts_corpus:
            cols = f"{cols}, {self.fts_corpus} as corpus_id"
        row = self.con.execute(f"select {cols} from segments_fts where {self.fts_sid}=? limit 1", (segment_id,)).fetchone()
        if row:
            return {"segment_id": str(row["segment_id"]),
                    "corpus_id": str(row["corpus_id"]) if "corpus_id" in row.keys() else "",
                    "text": norm_ws(str(row["text"]))}
        return None

def filter_out_pleiades(items: List[Dict[str,str]]) -> List[Dict[str,str]]:
    return [it for it in items if it.get("corpus_id") != PLEIADES_CORPUS_ID]

def rerank_by_terms(items: List[Dict[str,str]], terms: List[str]) -> List[Dict[str,str]]:
    if not terms:
        return items
    ts=[t.lower() for t in terms if t and len(t) >= 3]
    scored=[]
    for idx,it in enumerate(items):
        s=(it.get("excerpt") or "").lower()
        score=sum(1 for t in ts if t in s)
        scored.append((score, idx, it))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [it for _,_,it in scored]

def multi_retrieve(
    store: EvidenceStore,
    query_texts: List[str],
    corpora: List[str],
    cooldown: set,
    scope: str,
    loc_label: str,
    allow_pleiades_narrative: bool,
    rerank_terms: List[str],
) -> Tuple[str, List[Dict[str,str]]]:
    all_items: List[Dict[str,str]] = []
    qs: List[str] = []

    for qt in query_texts:
        q = fts_query(qt)
        qs.append(q)
        hits = store.search(q, corpora if corpora else None, limit=MAX_EVIDENCE*18)
        if not allow_pleiades_narrative:
            hits = filter_out_pleiades(hits)
        all_items.extend(hits)

    # de-dupe
    seen=set()
    uniq=[]
    for it in all_items:
        sid=it["segment_id"]
        if sid in seen:
            continue
        seen.add(sid)
        it["excerpt"] = clamp(it["excerpt"], PROMPT_EXCERPT_CHARS)
        uniq.append(it)

    uniq = rerank_by_terms(uniq, rerank_terms)

    # cooldown + cap
    raw=[]
    for it in uniq:
        sid=it["segment_id"]
        if sid in cooldown:
            continue
        raw.append(it)
        if len(raw) >= MAX_EVIDENCE:
            break

    if scope == "local":
        toks = loc_tokens(loc_label)
        filtered = [it for it in raw if is_loc_relevant(it.get("excerpt",""), toks)]
        if len(filtered) < MIN_PACKET_LOCAL:
            # expand by adding more from raw (even if weakly local) to avoid packet collapse
            for it in raw:
                if it in filtered:
                    continue
                filtered.append(it)
                if len(filtered) >= MIN_PACKET_LOCAL:
                    break
        return " | ".join(qs[:3]) + (" | ..." if len(qs) > 3 else ""), filtered

    return " | ".join(qs[:3]) + (" | ..." if len(qs) > 3 else ""), raw

def system_prompt() -> str:
    return (
      "You are producing a historically grounded exploration experience.\n"
      "Hard rules:\n"
      "1) Every factual claim must be backed by evidence_ids from the packet.\n"
      "2) Output JSON only. No extra keys beyond schema.\n"
      "3) No long verbatim quotes; paraphrase.\n"
      "4) Do NOT use repeated atmospheric cliché openings.\n"
      "5) Answer the user's prompt directly; keep focus.\n"
      "6) If evidence is thin for the requested focus, say so plainly.\n"
    )

PLAN_SCHEMA = {"delta":[{"change":"string","evidence_ids":["segment_id"]}],
               "observations":[{"text":"string","evidence_ids":["segment_id"]}]}
RENDER_SCHEMA = {"narration":"string",
                 "observations":[{"text":"string","evidence_ids":["segment_id"]}],
                 "delta":[{"change":"string","evidence_ids":["segment_id"]}],
                 "claims":[{"claim":"string","evidence_ids":["segment_id"]}]}

def validate_plan(obj: Any, packet_ids: set) -> List[str]:
    e=[]
    allowed={"delta","observations"}
    if not isinstance(obj, dict): return ["plan not object"]
    if set(obj.keys()) - allowed:
        e.append("extra keys not allowed")
    for k in ["delta","observations"]:
        if k not in obj:
            e.append(f"missing {k}")
    if not isinstance(obj.get("delta"), list) or not obj["delta"]:
        e.append("delta must be non-empty list")
    if not isinstance(obj.get("observations"), list) or len(obj["observations"]) < 2:
        e.append("observations must be list len>=2")
    for arr_name in ["delta","observations"]:
        arr=obj.get(arr_name, [])
        if isinstance(arr, list):
            for i,it in enumerate(arr):
                if not isinstance(it, dict):
                    e.append(f"{arr_name}[{i}] not object"); continue
                ev=it.get("evidence_ids")
                if not isinstance(ev, list) or not ev:
                    e.append(f"{arr_name}[{i}].evidence_ids empty"); continue
                bad=[x for x in ev if x not in packet_ids]
                if bad:
                    e.append(f"{arr_name}[{i}] unknown evidence_ids {bad[:4]}")
    return e

def validate_render(obj: Any, packet_ids: set) -> List[str]:
    e=[]
    req={"narration","observations","delta","claims"}
    if not isinstance(obj, dict): return ["render not object"]
    if set(obj.keys()) - req:
        e.append("extra keys not allowed")
    for k in req:
        if k not in obj:
            e.append(f"missing {k}")
    if not isinstance(obj.get("narration"), str) or not obj["narration"].strip():
        e.append("narration empty")
    claims=obj.get("claims")
    if not isinstance(claims, list) or len(claims) < 3:
        e.append("claims must be list len>=3")
    else:
        for i,it in enumerate(claims):
            if not isinstance(it, dict):
                e.append(f"claims[{i}] not object"); continue
            ev=it.get("evidence_ids")
            if not isinstance(ev, list) or not ev:
                e.append(f"claims[{i}].evidence_ids empty"); continue
            bad=[x for x in ev if x not in packet_ids]
            if bad:
                e.append(f"claims[{i}] unknown evidence_ids {bad[:4]}")
    return e

def plan_prompt(state: Dict[str,Any], user_text: str, packet: List[Dict[str,str]]) -> str:
    return json.dumps({
      "task":"Create a TurnPlan for a grounded exploration response.",
      "state": state,
      "user_input": user_text,
      "evidence_packet": packet,
      "required_keys_exactly":["delta","observations"],
      "schema": PLAN_SCHEMA,
      "rules":[
        "JSON only.",
        "Every delta and observation must cite evidence_ids from the packet.",
        "Observations must match the user_input focus (place/topic/time)."
      ]
    }, ensure_ascii=False, separators=(",", ":"))

def render_prompt(state: Dict[str,Any], plan: Dict[str,Any], packet: List[Dict[str,str]]) -> str:
    return json.dumps({
      "task":"Render the response using only the plan and evidence packet.",
      "state": state,
      "plan": plan,
      "evidence_packet": packet,
      "required_keys_exactly":["narration","observations","delta","claims"],
      "schema": RENDER_SCHEMA,
      "rules":[
        "JSON only.",
        "Every factual claim must appear in claims[] with evidence_ids from the packet.",
        "No cliché openings; start concrete and specific.",
        "Keep the narration anchored to the user's prompt."
      ]
    }, ensure_ascii=False, separators=(",", ":"))

def repair_prompt(kind: str, bad: str, errors: List[str], required_keys: List[str], schema: Any) -> str:
    return json.dumps({
      "task": f"Repair invalid {kind} output into valid JSON.",
      "validation_errors": errors,
      "required_keys_exactly": required_keys,
      "schema": schema,
      "invalid_output": clamp(bad, 2000),
      "rules":["Return JSON only.","Include all required keys.","Remove extra keys.","Fix commas/quotes/brackets."]
    }, ensure_ascii=False, separators=(",", ":"))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--corpus", action="append", default=[], help="Restrict narrative retrieval to these corpora (optional)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1900)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    store = EvidenceStore(args.db)
    cooldown = collections.deque(maxlen=COOLDOWN_SIZE)
    cooldown_set = set()

    state = {"loc_label":"Rome","time_of_day":"morning","date_label":"Roman antiquity (mixed sources)","turn":0,
             "scope":"local","pleiades_narrative":"off"}

    last_packet: List[Dict[str,str]] = []

    print("[OK] Vivarium Rome Experience v0.6 (multi-retrieve + auto-goto)")
    print(f"[OK] db={args.db}")
    print("Commands:")
    print("  :goto <place> | :time <...> | :scope <local|global> | :pleiades <on|off>")
    print("  :place <name> (Pleiades lookup) | :find <term> | :audit <segment_id> | :llmtest | :state | :quit")
    print()

    try:
        while True:
            user_in = input("Prompt> ").strip()
            if not user_in:
                continue

            if user_in.startswith(":"):
                parts = user_in[1:].split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("q","quit","exit"):
                    break
                if cmd == "state":
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    continue
                if cmd == "time":
                    if not arg: print("[WARN] :time needs a value"); continue
                    state["time_of_day"] = arg
                    print("[OK] time_of_day =", state["time_of_day"])
                    continue
                if cmd == "goto":
                    if not arg: print("[WARN] :goto needs a place"); continue
                    state["loc_label"] = arg
                    print("[OK] loc_label =", state["loc_label"])
                    continue
                if cmd == "scope":
                    if arg.lower() not in ("local","global"):
                        print("[WARN] :scope must be local|global"); continue
                    state["scope"] = arg.lower()
                    print("[OK] scope =", state["scope"])
                    continue
                if cmd == "pleiades":
                    if arg.lower() not in ("on","off"):
                        print("[WARN] :pleiades must be on|off"); continue
                    state["pleiades_narrative"] = arg.lower()
                    print("[OK] pleiades_narrative =", state["pleiades_narrative"])
                    continue
                if cmd == "llmtest":
                    msgs = [{"role":"user","content":"Return JSON: {\"ok\":true}"}]
                    out = openai_chat(args.base_url, args.model, msgs, 0.0, 30, args.api_key)
                    print("[LLMTEST] raw:", clamp(out, 220))
                    continue
                if cmd == "audit":
                    if not arg: print("[WARN] :audit needs a segment_id"); continue
                    row = store.audit(arg)
                    if not row: print("[WARN] segment not found"); continue
                    print(f"[AUDIT] {row.get('corpus_id','')}:{row['segment_id']}")
                    print(textwrap.fill(clamp(row["text"], 1600), width=90))
                    continue
                if cmd == "place":
                    if not arg: print("[WARN] :place needs a name"); continue
                    q = fts_query(arg)
                    hits = store.search(q, [PLEIADES_CORPUS_ID], limit=6)
                    print(f"[PLACE] hits={len(hits)} (Pleiades)")
                    for h in hits:
                        print(f"- {h.get('corpus_id','')}:{h['segment_id']} :: {clamp(h['excerpt'], 200)}")
                    continue
                if cmd == "find":
                    if not arg: print("[WARN] :find needs a term"); continue
                    q = fts_query(arg)
                    hits = store.search(q, args.corpus if args.corpus else None, limit=10)
                    if state["pleiades_narrative"] != "on":
                        hits = filter_out_pleiades(hits)
                    print(f"[FIND] q={q!r} hits={len(hits)}")
                    for h in hits[:10]:
                        print(f"- {h.get('corpus_id','')}:{h['segment_id']} :: {clamp(h['excerpt'], 200)}")
                    continue

                print("[WARN] unknown command")
                continue

            if len(user_in) < 3:
                print("[WARN] Input too short.")
                continue

            # auto-goto if prompt is "take me to X"
            m = AUTO_GOTO_RE.match(user_in)
            if m:
                target = m.group(3).strip().strip(".")
                if target and len(target) >= 3:
                    hits = store.search(fts_query(target), [PLEIADES_CORPUS_ID], limit=1)
                    if hits:
                        title = parse_pleiades_title(hits[0]["excerpt"]) or target
                        state["loc_label"] = title
                        if args.debug:
                            print(f"[DEBUG] auto-goto -> {state['loc_label']!r} (from {target!r})")

            allow_pl = (state["pleiades_narrative"] == "on")
            bias = time_bias_terms(user_in)
            is_tour = bool(GENERIC_TOUR_RE.match(user_in.strip()))

            # Build multiple query texts to union results
            if is_tour and "rome" in loc_tokens(state["loc_label"]):
                query_texts = [
                    f"{state['loc_label']} {' '.join(ROME_CORE_ANCHORS)} {' '.join(bias)}",
                    f"forum palatine capitoline subura tiber curia rostra comitium {' '.join(bias)}",
                    f"{state['loc_label']} monuments streets basilica temple macellum tabernae {' '.join(bias)}",
                ]
                rerank_terms = ROME_CORE_ANCHORS + bias
            else:
                query_texts = [
                    f"{state['loc_label']} {user_in} {' '.join(bias)}",
                    f"{state['loc_label']} {' '.join(ROME_CORE_ANCHORS[:8])} {user_in}",
                    f"{user_in} {' '.join(bias)}",
                ]
                rerank_terms = expand_synonyms(fts_tokens(user_in)) + bias + loc_tokens(state["loc_label"])

            q_dbg, packet = multi_retrieve(
                store=store,
                query_texts=query_texts,
                corpora=args.corpus,
                cooldown=cooldown_set,
                scope=state["scope"],
                loc_label=state["loc_label"],
                allow_pleiades_narrative=allow_pl,
                rerank_terms=rerank_terms
            )

            if args.debug:
                print(f"[DEBUG] scope={state['scope']} pleiades_narrative={state['pleiades_narrative']} loc={state['loc_label']!r} is_tour={is_tour} q={q_dbg!r} packet_n={len(packet)}")

            if not packet:
                print("[WARN] No evidence returned. Try rephrasing or :find <term>.")
                continue

            last_packet = packet
            packet_ids = {e["segment_id"] for e in packet}

            # plan
            plan_obj=None; plan_last=""; plan_err=[]
            for attempt in range(1, TURNPLAN_RETRIES+1):
                msgs=[{"role":"system","content":system_prompt()},
                      {"role":"user","content":plan_prompt(state, user_in, packet)}]
                if attempt>1:
                    msgs.append({"role":"user","content":repair_prompt("TurnPlan", plan_last, plan_err, ["delta","observations"], PLAN_SCHEMA)})
                out=openai_chat(args.base_url, args.model, msgs, args.temperature, args.max_tokens, args.api_key)
                plan_last=out
                obj,err=try_json(out)
                if err:
                    plan_err=[f"JSON parse error: {err}"]; continue
                plan_err=validate_plan(obj, packet_ids)
                if plan_err: continue
                plan_obj=obj; break
            if plan_obj is None:
                print("[FAIL] plan validation failed:", plan_err[:10])
                print(clamp(plan_last, 900))
                continue

            # render
            render_obj=None; render_last=""; render_err=[]
            for attempt in range(1, RENDER_RETRIES+1):
                msgs=[{"role":"system","content":system_prompt()},
                      {"role":"user","content":render_prompt(state, plan_obj, packet)}]
                if attempt>1:
                    msgs.append({"role":"user","content":repair_prompt("Render", render_last, render_err,
                                                                     ["narration","observations","delta","claims"], RENDER_SCHEMA)})
                out=openai_chat(args.base_url, args.model, msgs, args.temperature, args.max_tokens, args.api_key)
                render_last=out
                obj,err=try_json(out)
                if err:
                    render_err=[f"JSON parse error: {err}"]; continue
                render_err=validate_render(obj, packet_ids)
                if render_err: continue
                render_obj=obj; break
            if render_obj is None:
                print("[FAIL] render validation failed:", render_err[:10])
                print(clamp(render_last, 900))
                continue

            state["turn"] += 1

            # cooldown update
            for e in packet:
                sid=e["segment_id"]
                if sid not in cooldown_set:
                    cooldown.append(sid)
            cooldown_set=set(cooldown)

            narr = de_cliche_opening(render_obj["narration"], state["loc_label"], state["time_of_day"])

            print("\n" + "="*88)
            print(textwrap.fill(narr.strip(), width=90))

            print("\n[Observations]")
            for o in render_obj["observations"]:
                print(f"- {o['text']} (source {','.join(o['evidence_ids'])})")

            print("\n[Delta]")
            for d in render_obj["delta"]:
                print(f"- {d['change']} (source {','.join(d['evidence_ids'])})")

            print("\n[Claims (auditable)]")
            for c in render_obj["claims"]:
                print(f"- {c['claim']} (source {','.join(c['evidence_ids'])})")

            print("="*88 + "\n")

    finally:
        store.close()

if __name__ == "__main__":
    main()