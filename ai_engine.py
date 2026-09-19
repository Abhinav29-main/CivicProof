"""
CivicProof AI Engine
====================
Self-contained computer-vision + NLP pipeline (no external APIs or model downloads)
for classification, priority, duplicates, and ProofWatch — PLUS an optional real
vision-model relevance/authenticity check (see "Vision-model gate" below), which
is the only part of this file that calls out to an external API.

  1. Issue classification     — visual feature space (color, texture, edge, luminance,
                                region geometry) fused with keyword-NLP from the report text.
  2. Priority scoring         — severity base + urgency-language multiplier + community
                                corroboration + aging, with full human-readable reasons.
  3. Duplicate detection      — perceptual hashes (aHash/dHash hamming), haversine
                                proximity, category agreement and token-Jaccard text sim.
  4. ProofWatch verification  — block-SSIM + histogram correlation (scene-match check),
                                after-photo issue re-classification, change metrics and a
                                rendered diff-heatmap artifact.

Thresholds are intentionally exposed at the top for hackathon demo tuning.
"""
import base64
import json
import mimetypes
import os
import re
import math
import time
import colorsys
import urllib.error
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = 40_000_000

# ----------------------------------------------------------------------------
# Categories
# ----------------------------------------------------------------------------
CATEGORIES = [
    "pothole", "garbage", "streetlight", "water_leak", "road_crack",
    "sidewalk_damage", "drainage", "sewage", "traffic_signal",
    "traffic_sign", "road_marking", "electrical_hazard", "fallen_tree",
    "construction_hazard", "building_damage", "bridge_damage",
    "public_property_damage", "other"
]

CATEGORY_META = {
    "pothole": {"label": "Pothole / Road Damage", "severity": 74.0, "icon": "road"},
    "garbage": {"label": "Garbage & Sanitation", "severity": 56.0, "icon": "trash"},
    "streetlight": {"label": "Streetlight Fault", "severity": 63.0, "icon": "lamp"},
    "water_leak": {"label": "Water Leak / Flooding", "severity": 86.0, "icon": "drop"},
    "road_crack": {"label": "Road Crack / Cave-in", "severity": 70.0, "icon": "crack"},
    "sidewalk_damage": {"label": "Sidewalk / Footpath Damage", "severity": 62.0, "icon": "walk"},
    "drainage": {"label": "Drainage Problem", "severity": 72.0, "icon": "drain"},
    "sewage": {"label": "Sewage / Sanitation Problem", "severity": 82.0, "icon": "drop"},
    "traffic_signal": {"label": "Traffic Signal Fault", "severity": 72.0, "icon": "signal"},
    "traffic_sign": {"label": "Traffic Sign Damage", "severity": 58.0, "icon": "sign"},
    "road_marking": {"label": "Road Marking Problem", "severity": 48.0, "icon": "road"},
    "electrical_hazard": {"label": "Electrical / Exposed Wire", "severity": 90.0, "icon": "bolt"},
    "fallen_tree": {"label": "Fallen Tree / Obstruction", "severity": 76.0, "icon": "tree"},
    "construction_hazard": {"label": "Construction Hazard", "severity": 78.0, "icon": "construction"},
    "building_damage": {"label": "Building / Structure Damage", "severity": 82.0, "icon": "building"},
    "bridge_damage": {"label": "Bridge / Culvert Damage", "severity": 88.0, "icon": "bridge"},
    "public_property_damage": {"label": "Public Property Damage", "severity": 64.0, "icon": "property"},
    "other": {"label": "Other Civic Issue", "severity": 40.0, "icon": "flag"},
}

# Relevance gate — sanity-checks a photo against the citizen's own words
# ("what happened" / "describe it briefly") before it's accepted at all.
# Thresholds intentionally exposed here for demo tuning, same as PW below.
RELEVANCE = {
    "min_text_signal": 0.34,   # description must clearly name an issue type before we judge it
    "min_vis_signal":  0.22,   # photo must show *some* visual signature of *some* issue
    "min_overlap":     0.16,   # photo's affinity for the *described* category, below = mismatch
    "gap_trigger":     0.40,   # image confidently reads as *something* (this strongly, or more)...
    "min_gap":         0.28,   # ...and beats the described category's own score by this much = mismatch
}

# Text-priority rule — when the citizen's own words clearly name ONE issue type,
# that wins over pixel statistics (which cannot really "see" a pothole and often
# mistake a rain-filled one for a water leak). The image then only adjusts confidence.
TEXT_LEAD = {
    "min_score":       0.30,   # best category's text score must be at least this...
    "min_lead":        0.15,   # ...and beat the runner-up category by this much
    "strong_score":    0.55,   # at/above this, text wins even if the pixels disagree
    "contradict_vis":  0.45,   # pixels "strongly" prefer another category at/above this...
    "contradict_own":  0.20,   # ...while showing little of the described one (below this)
}

# Categories that routinely look alike in a photo (rain-filled pothole vs. flooding,
# cracked road vs. pothole, blocked drain vs. sewage...). A pixel-level "mismatch"
# between members of the same family is NOT evidence of a wrong photo.
VISUAL_FAMILIES = [
    {"pothole", "road_crack", "water_leak", "drainage", "sewage", "sidewalk_damage"},
    {"garbage", "drainage", "sewage"},
    {"streetlight", "electrical_hazard", "traffic_signal"},
]


def _compatible(a, b):
    return a == b or any(a in fam and b in fam for fam in VISUAL_FAMILIES)


# ---------------------------------------------------------------------------
# Keyword NLP
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "pothole": ["pothole", "hole", "crater", "dug up", "dug-up", "cave in", "cave-in", "bumpy",
                "damaged road", "broken road", "uneven road", "patch", "asphalt", "road repair",
                "manhole", "missing cover", "footpath broken", "road pit", "pit", "hole in road",
                "hole in the road", "big hole", "deep hole"],
    "garbage": ["garbage", "trash", "waste", "dump", "dumping", "litter", "rubbish", "bin",
                "dustbin", "debris", "unclean", "dirty", "sanitation", "plastic"],
    "streetlight": ["streetlight", "street light", "street-light", "lamp", "light pole", "dark street",
                    "not working", "fused", "flicker", "flickering", "bulb", "high mast"],
    "water_leak": ["leak", "leakage", "water leak", "pipe leak", "pipeline", "burst pipe", "burst",
                   "flood", "flooding", "puddle", "stagnant", "gushing", "waterlogging"],
    "road_crack": ["crack", "cracked", "fissure", "fracture", "split", "caving", "sinking",
                   "collapsed", "collapse", "subsidence", "erosion", "wearing"],
    "sidewalk_damage": ["sidewalk", "footpath", "pavement", "pedestrian path", "kerb", "curb",
                        "broken tiles", "broken pavement", "walkway"],
    "drainage": ["drain", "drainage", "storm drain", "blocked drain", "clogged drain",
                 "drain cover", "waterlogging", "drain overflow"],
    "sewage": ["sewage", "sewer", "sewage leak", "sewage overflow", "contamination", "wastewater"],
    "traffic_signal": ["traffic signal", "traffic light", "signal light", "signal not working",
                       "signal broken", "junction signal"],
    "traffic_sign": ["traffic sign", "road sign", "signboard", "sign post", "missing sign",
                     "broken sign", "bent sign"],
    "road_marking": ["road marking", "lane marking", "lane line", "zebra crossing", "crosswalk",
                     "stop line", "faded marking"],
    "electrical_hazard": ["exposed wire", "live wire", "electric wire", "electrical pole",
                          "electric pole", "sparking", "electrocution", "transformer",
                          "fallen cable", "hanging wire"],
    "fallen_tree": ["fallen tree", "tree fallen", "tree branch", "uprooted tree", "tree blocking",
                    "branch blocking", "obstruction"],
    "construction_hazard": ["construction", "construction debris", "unsafe construction", "barricade",
                            "excavation", "open pit", "building work", "road work", "worksite"],
    "building_damage": ["building damage", "wall crack", "damaged building", "collapsed wall",
                        "roof damage", "structural damage", "unsafe building"],
    "bridge_damage": ["bridge damage", "bridge crack", "culvert", "flyover", "overpass",
                      "railing damaged", "bridge railing"],
    "public_property_damage": ["bench", "bus shelter", "bus stop", "public toilet", "park equipment",
                               "playground", "fence", "railing", "public property", "vandalized"],
}

URGENCY_KEYWORDS = {
    1.35: ["injured", "accident", "death", "died", "emergency", "ambulance",
           "electrocut", "exposed wire", "live wire", "fire", "collapsed"],
    1.22: ["school", "hospital", "children", "child", "elderly", "senior citizen",
           "danger", "dangerous", "unsafe", "bus stop", "main road", "highway",
           "market", "crowd", "mosquito", "dengue", "disease", "rain", "monsoon"],
    1.12: ["many days", "weeks", "months", "repeated", "again", "ignored",
           "no action", "urgent", "immediately", "asap", "please", "heavy traffic"],
}

# Terms that on their own almost fully determine the category (weight 2.5 instead of 1).
HEADWORDS = {
    "pothole", "garbage", "trash", "streetlight", "water leak", "pipe leak", "burst pipe",
    "sewage", "sewer", "drainage", "storm drain", "traffic signal", "traffic light",
    "traffic sign", "road sign", "fallen tree", "exposed wire", "live wire", "sidewalk", "footpath",
}

# "pot hole" / "pot-hole" / "street light" / "water logging" ... -> one canonical token,
# applied to BOTH the citizen's text and the keyword lists so they always line up.
_COMPOUNDS = [
    (re.compile(r"\bpot[\s\-]*holes?\b"), "pothole"),
    (re.compile(r"\bpat[\s\-]*holes?\b"), "pothole"),
    (re.compile(r"\bstreet[\s\-]*lights?\b"), "streetlight"),
    (re.compile(r"\bwater[\s\-]*logging\b"), "waterlogging"),
    (re.compile(r"\bman[\s\-]*holes?\b"), "manhole"),
    (re.compile(r"\bdust[\s\-]*bins?\b"), "dustbin"),
    (re.compile(r"\bside[\s\-]*walks?\b"), "sidewalk"),
    (re.compile(r"\bfoot[\s\-]*paths?\b"), "footpath"),
]


def _normalize(text):
    t = (text or "").lower().replace("_", " ")
    for pat, rep in _COMPOUNDS:
        t = pat.sub(rep, t)
    return t


def _kw_regex(kw):
    words = _normalize(kw).replace("-", " ").split()
    body = r"\s+".join(re.escape(w) for w in words)
    # whole-word match (so "hole" no longer fires inside "whole", "bin" inside "combining",
    # "rain" inside "train"), tolerating simple plural / verb endings.
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:s|es|ed|d|ing)?(?![a-z0-9])")


_KW_RE = {cat: [(kw, _kw_regex(kw)) for kw in words] for cat, words in CATEGORY_KEYWORDS.items()}
_URG_RE = [(mult, kw, _kw_regex(kw)) for mult, words in URGENCY_KEYWORDS.items() for kw in words]

STOPWORDS = set("""a an the and or of to in on at is are was were be been it this that with for
from by as near next very there their our my your his her its not no so but if then than too
also just we i you he she they them us""".split())


def _tokens(text):
    out, tok = [], ""
    for ch in (text or "").lower():
        if ch.isalnum():
            tok += ch
        else:
            if tok and tok not in STOPWORDS and len(tok) > 1:
                out.append(tok)
            tok = ""
    if tok and tok not in STOPWORDS and len(tok) > 1:
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------
def _inspect(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.resize((256, int(max(120, round(192 * h / max(w, 1))))), Image.LANCZOS)
    arr = np.asarray(img.resize((256, 192), Image.LANCZOS)).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B

    # grid (4x4) luminance stats — for spatial reasoning
    cells = []
    cs_h, cs_w = 192 // 4, 256 // 4
    for r in range(4):
        row = []
        for c in range(4):
            cell = lum[r * cs_h:(r + 1) * cs_h, c * cs_w:(c + 1) * cs_w]
            row.append(float(cell.mean()))
        cells.append(row)
    cells = np.array(cells)

    brightness = float(lum.mean()) / 255.0
    dark_mask = lum < 72
    dark_ratio = float(dark_mask.mean())
    dark_lower = float(dark_mask[96:, :].mean())
    dark_c = dark_mask.reshape(4, 48, 4, 64).mean(axis=(1, 3))
    dark_irregular = float(dark_c.std() * 2 + dark_c.max() - dark_c.mean())

    gy, gx = np.gradient(lum)
    gmag = np.hypot(gx, gy)
    edge_ratio = float((gmag > 38).mean())
    edge_dense = float((gmag > 85).mean())
    edge_lower = float((gmag[96:, :] > 38).mean())

    mx_, mn_ = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx_ > 8, (mx_ - mn_) / np.maximum(mx_, 1.0), 0.0)
    sat_mean, sat_std = float(sat.mean()), float(sat.std())

    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                         0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    blue_ratio = float(((B > R * 1.12) & (B > G * 1.02) & (B > 95)).mean())
    green_ratio = float(((G > R * 1.08) & (G > B * 1.02) & (G > 80)).mean())
    bright_ratio = float((lum > 232).mean())
    cool_ratio = float(((B > R * 1.01) & (B > G * 0.99) & (B > 108)).mean())   # wet/dusk cool tones
    shine_ratio = float((lum > 200).mean())                                   # specular shimmer

    dark_sel = dark_mask[96:].sum()
    edge_in_dark = float((gmag[96:][dark_mask[96:]] > 38).mean()) if dark_sel > 50 else 0.0

    # hue entropy on sufficiently saturated pixels
    sat_mask = sat > 0.16
    if sat_mask.sum() > 40:
        rr, gg, bb = R[sat_mask] / 255.0, G[sat_mask] / 255.0, B[sat_mask] / 255.0
        mxx = np.maximum.reduce([rr, gg, bb]); mnn = np.minimum.reduce([rr, gg, bb])
        d = np.maximum(mxx - mnn, 1e-6)
        hue = np.where(mxx == rr, ((gg - bb) / d) % 6,
             np.where(mxx == gg, (bb - rr) / d + 2, (rr - gg) / d + 4)) / 6.0
        hist, _ = np.histogram(hue, bins=12, range=(0, 1))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        hue_entropy = float(-(p * np.log2(p)).sum() / 3.585)  # /log2(12)
    else:
        hue_entropy = 0.0

    return {
        "brightness": round(brightness, 4),
        "dark_ratio": round(dark_ratio, 4),
        "dark_lower": round(dark_lower, 4),
        "dark_irregular": round(dark_irregular, 4),
        "edge_ratio": round(edge_ratio, 4),
        "edge_dense": round(edge_dense, 4),
        "edge_lower": round(edge_lower, 4),
        "sat_mean": round(sat_mean, 4),
        "sat_std": round(sat_std, 4),
        "colorfulness": round(colorfulness, 2),
        "blue_ratio": round(blue_ratio, 4),
        "green_ratio": round(green_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "cool_ratio": round(cool_ratio, 4),
        "shine_ratio": round(shine_ratio, 4),
        "edge_in_dark": round(edge_in_dark, 4),
        "hue_entropy": round(hue_entropy, 4),
        "grid_min": round(float(cells.min()) / 255.0, 4),
        "grid_max": round(float(cells.max()) / 255.0, 4),
        "grid_range": round((float(cells.max()) - float(cells.min())) / 255.0, 4),
        "upper_bright": round(float(cells[0].mean()) / 255.0, 4),
        "lower_dark": round(float(cells[3].mean()) / 255.0, 4),
    }


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def image_scores(f):
    """Produce 0..1 affinity per category purely from visual features."""
    b = f["brightness"]
    cool = f.get("cool_ratio", 0.0)
    shine = f.get("shine_ratio", 0.0)
    dk_low = _clamp((f["dark_lower"] - 0.05) / 0.22)
    irreg = _clamp(f["dark_irregular"] / 0.5)
    edge = _clamp(f["edge_ratio"] / 0.30)
    bright_ok = _clamp((b - 0.30) / 0.45)
    color_n = _clamp(f["colorfulness"] / 70)

    pothole = (0.30 * dk_low + 0.24 * irreg + 0.22 * edge + 0.12 * bright_ok
               + 0.12 * (1 - color_n)) \
        * (1 - 0.6 * _clamp((cool - 0.18) / 0.25)) \
        * (1 - 0.5 * _clamp((f["colorfulness"] - 30) / 45)) \
        * (1 - 0.7 * _clamp((f.get("green_ratio", 0) - 0.06) / 0.15))

    garbage = (0.34 * _clamp(f["colorfulness"] / 55)
               + 0.24 * _clamp(f["hue_entropy"] / 0.5)
               + 0.20 * _clamp(f["sat_std"] / 0.20)
               + 0.14 * _clamp((f["edge_ratio"] - 0.10) / 0.25)
               + 0.08 * bright_ok) \
        * (1 - _clamp((cool - 0.15) / 0.25)) \
        * (1 - 0.8 * _clamp((f.get("green_ratio", 0) - 0.10) / 0.18))

    streetlight = (0.46 * _clamp((0.42 - b) / 0.30)
                   + 0.20 * _clamp(f["grid_range"] / 0.5)
                   + 0.16 * _clamp((0.12 - f["edge_ratio"]) / 0.12)
                   + 0.18 * _clamp((f["grid_max"] - 0.40) / 0.40))

    water = (0.42 * _clamp(cool / 0.28)
             + 0.24 * _clamp(shine / 0.05)
             + 0.18 * _clamp((0.22 - f["edge_ratio"]) / 0.22)
             + 0.16 * _clamp((b - 0.32) / 0.40)) \
        * (0.30 + 0.70 * _clamp(cool / 0.12))   # require *some* actual blue/cool tint —
                                                 # smoothness+brightness alone (e.g. a warm
                                                 # sunset poster) can no longer fake a wet-look score

    crack = (0.36 * _clamp(f["edge_dense"] / 0.14)
             + 0.24 * edge
             + 0.20 * _clamp((0.28 - f["colorfulness"]) / 28.0)
             + 0.20 * _clamp((0.18 - f["dark_ratio"]) / 0.18)) \
        * (1 - 0.6 * _clamp((f.get("green_ratio", 0) - 0.06) / 0.15))

    return {
        "pothole": round(pothole, 4),
        "garbage": round(garbage, 4),
        "streetlight": round(streetlight, 4),
        "water_leak": round(water, 4),
        "road_crack": round(crack, 4),
    }


def text_scores(title, description):
    text = _normalize(f"{title or ''} {description or ''}")
    toks = set(_tokens(text))
    raw, matched = {}, {}
    for cat, pairs in _KW_RE.items():
        s, hits = 0.0, []
        for kw, rx in pairs:
            if rx.search(text):
                nk = _normalize(kw)
                s += 2.5 if nk in HEADWORDS else (2.0 if " " in nk else 1.0)
                hits.append(kw)
        raw[cat] = min(1.0, s / 3.0)
        matched[cat] = hits
    urgency_mult, urgency_hits = 1.0, []
    for mult, kw, rx in _URG_RE:
        if rx.search(text):
            urgency_mult = max(urgency_mult, mult)
            urgency_hits.append(kw)
    return {
        "scores": {k: round(v, 4) for k, v in raw.items()},
        "matched": matched,
        "urgency_mult": urgency_mult,
        "urgency_hits": urgency_hits,
        "tokens": sorted(toks),
    }


def _text_leader(txt):
    """(category, score, lead-over-runner-up) for the citizen's own words, or (None, 0, 0)."""
    ranked = sorted(txt["scores"].items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None, 0.0, 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], ranked[0][1], ranked[0][1] - second


def _text_is_decisive(txt, vis):
    """Does the citizen's text clearly name one issue type — and is that safe to trust
    over what the pixel statistics think?"""
    tb, score, lead = _text_leader(txt)
    if not tb or score < TEXT_LEAD["min_score"] or lead < TEXT_LEAD["min_lead"]:
        return None
    if score >= TEXT_LEAD["strong_score"]:
        return tb
    # weak text (one generic word, e.g. just "hole"): only trust it if the photo doesn't
    # strongly point somewhere else
    if vis:
        vtop = max(vis, key=vis.get)
        if (vtop != tb and not _compatible(vtop, tb)
                and vis[vtop] >= TEXT_LEAD["contradict_vis"]
                and vis.get(tb, 0.0) < TEXT_LEAD["contradict_own"]):
            return None
    return tb


def classify_report(image_path=None, title="", description="", vision_result=None):
    """Fuse vision + text into a category decision with calibrated confidence.

    Decision order (why: the pixel statistics cannot truly *see* a pothole and will
    happily call a rain-filled one a water leak, so the citizen's own words come first):
      1. Text names ONE issue type clearly  -> that category; image only tunes confidence.
      2. Otherwise a vision model (if it ran) -> its category.
      3. Otherwise pixel features + weak text fused (offline fallback).
    """
    vis = {}
    feats = {}
    txt = text_scores(title, description)
    if image_path and os.path.exists(image_path):
        feats = _inspect(image_path)
        vis = image_scores(feats)
    has_text = bool((title or "").strip() or (description or "").strip())

    vision_category = None
    vision_confidence = 0.0
    vision_reason = ""
    if isinstance(vision_result, dict) and vision_result.get("ok"):
        candidate = str(vision_result.get("category") or "").strip().lower()
        if candidate in CATEGORY_META:
            vision_category = candidate
            vision_confidence = float(vision_result.get("category_confidence",
                                                        vision_result.get("confidence", 0)))
            vision_reason = str(vision_result.get("reason") or "").strip()

    cats = [c for c in CATEGORIES if c != "other"]
    text_cat = _text_is_decisive(txt, vis)          # None unless the words are clear + trustworthy
    text_score = txt["scores"].get(text_cat, 0.0) if text_cat else 0.0

    def _pack(category, conf, fused, explanation, agreement, decided_by, extra=None):
        out = {
            "category": category,
            "category_label": CATEGORY_META[category]["label"],
            "confidence": round(_clamp(conf / 100.0, 0.30, 0.985) * 100, 1),
            "fused_scores": {k: round(v, 4) for k, v in fused.items()},
            "visual_scores": vis,
            "text_scores": txt["scores"],
            "text_keywords": txt["matched"].get(category, []),
            "urgency_hits": txt["urgency_hits"],
            "urgency_mult": txt["urgency_mult"],
            "features": feats,
            "explanation": explanation,
            "agreement": agreement,
            "decided_by": decided_by,      # "text" | "vision_model" | "text+vision_model" | "image_features"
        }
        if extra:
            out.update(extra)
        return out

    # ---- 1) the citizen's words clearly name one category ----------------------------
    if text_cat:
        img_top = max(vis, key=vis.get) if vis else None
        if vision_category:
            img_cat, img_support = vision_category, (vision_category == text_cat or _compatible(vision_category, text_cat))
            img_strong = vision_confidence >= 60
        else:
            img_cat = img_top
            img_support = bool(vis) and vis.get(text_cat, 0.0) >= 0.25
            img_strong = bool(vis) and img_top is not None and vis[img_top] >= TEXT_LEAD["contradict_vis"]
        conflict = bool(img_cat) and img_cat != text_cat and not _compatible(img_cat, text_cat) and img_strong
        conf = 58 + 30 * text_score
        if img_support:
            conf += 7
        if conflict:
            conf -= 10
        fused = {c: 0.0 for c in cats}
        fused[text_cat] = round(_clamp(conf / 100.0), 4)
        explanation = [f"Your description clearly names '{CATEGORY_META[text_cat]['label']}' "
                       f"({', '.join(txt['matched'][text_cat][:4])})."]
        if img_support and img_cat and img_cat != text_cat:
            explanation.append(f"The photo can look like '{CATEGORY_META[img_cat]['label']}' "
                               f"(e.g. rainwater in a pothole), which is consistent with your report.")
        elif img_support:
            explanation.append("The photo is consistent with that.")
        if conflict:
            explanation.append(f"The photo's pattern resembles '{CATEGORY_META[img_cat]['label']}' — "
                               f"kept your description; flagged for a quick human check.")
        if vision_reason:
            explanation.append(vision_reason)
        return _pack(text_cat, conf, fused, explanation,
                     agreement=bool(img_support), decided_by="text+vision_model" if vision_category else "text",
                     extra={"conflict": bool(conflict), "vision_used": bool(vision_category),
                            **({"vision_category": vision_category} if vision_category else {})})

    # ---- 2) vision model decides (text was vague / silent) ----------------------------
    if vision_category:
        fused = {c: 0.0 for c in cats}
        fused[vision_category] = max(0.50, min(1.0, vision_confidence / 100.0))
        if txt["scores"].get(vision_category, 0) > 0:
            fused[vision_category] = min(1.0, fused[vision_category] + 0.10)
        explanation = [vision_reason or f"Vision model identified '{CATEGORY_META[vision_category]['label']}' from the image."]
        if has_text and txt["matched"].get(vision_category):
            explanation.append(f"Language matches: {', '.join(txt['matched'][vision_category][:4])}")
        return _pack(vision_category, max(50.0, min(99.0, vision_confidence)), fused, explanation,
                     agreement=txt["scores"].get(vision_category, 0) > 0.25, decided_by="vision_model",
                     extra={"vision_category": vision_category, "vision_used": True, "conflict": False})

    # ---- 3) offline fallback: pixel features + weak text ------------------------------
    if vis and has_text:
        wv, wt = 0.55, 0.45
    elif vis:
        wv, wt = 1.0, 0.0
    else:
        wv, wt = 0.0, 1.0

    fused = {}
    for c in cats:
        fused[c] = wv * vis.get(c, 0.0) + wt * txt["scores"].get(c, 0.0)
    best = max(fused, key=fused.get)
    best_score = fused[best]
    runner = sorted(((v, k) for k, v in fused.items() if k != best), reverse=True)[0]

    # confidence: head start for fusing two agreeing signals
    agreement = txt["scores"].get(best, 0) > 0.25 and vis.get(best, 0) > 0.25
    conf = 42 + best_score * 52
    if agreement:
        conf += 7
    if runner[0] > 0:
        conf -= _clamp((runner[0] - best_score * 0.75), 0, 1) * 26
    no_text_support = not txt["matched"].get(best)
    if best_score < 0.20 or (not agreement and no_text_support and conf < 60):
        # weak, unsupported signal — route to manual triage instead of guessing
        best = "other"
        conf = max(38.0, min(58.0, conf))

    explanation = []
    if vis:
        explanation.append(f"Visual affinity {vis.get(best, 0):.2f} for '{best}' "
                           f"(dark lower region={feats.get('dark_lower', 0):.2f}, "
                           f"edge density={feats.get('edge_ratio', 0):.2f}, "
                           f"color chaos={feats.get('colorfulness', 0):.0f})")
    if has_text:
        kw = txt["matched"].get(best, [])
        if kw:
            explanation.append(f"Language matches: {', '.join(kw[:4])}")
    return _pack(best, conf, fused, explanation, agreement=agreement,
                 decided_by="image_features" if not has_text or not txt["matched"].get(best) else "text+image_features",
                 extra={"conflict": False})


# ---------------------------------------------------------------------------
# Relevance gate — does the photo actually match what the citizen typed?
# ---------------------------------------------------------------------------
def check_relevance(image_path, title, description):
    """Sanity-check an uploaded photo against the citizen's 'what happened' (title)
    and 'describe it briefly' (description) text.

    This reuses the same visual-feature and keyword-NLP scoring used for
    classification, but compares the two signals against each other instead of
    fusing them. It is deliberately conservative — a hand-built CV/NLP pipeline
    can't truly "see" a photo, so it only rejects a photo when:
      (a) the text clearly and confidently names a specific issue type, AND
      (b) the photo shows little/no visual signature of that type (or of
          anything at all).
    A vague description, or a description with no recognizable issue keywords,
    is always accepted — there's nothing concrete to contradict.

    Returns: {relevant, confidence, reason, image_category, text_category}
    """
    result = {"relevant": True, "confidence": 60.0, "reason": "",
              "image_category": None, "text_category": None}

    if not image_path or not os.path.exists(image_path):
        result.update(relevant=False, confidence=0.0,
                       reason="No photo was received with this report.")
        return result

    txt = text_scores(title, description)
    txt_best = max(txt["scores"], key=txt["scores"].get)
    txt_best_score = txt["scores"][txt_best]
    text_hits = txt["matched"].get(txt_best, [])

    if not text_hits or txt_best_score < RELEVANCE["min_text_signal"]:
        result["reason"] = "Description too general to verify against the photo — accepted."
        return result
    result["text_category"] = txt_best

    feats = _inspect(image_path)
    vis = image_scores(feats)
    vis_best = max(vis, key=vis.get)
    vis_best_score = vis[vis_best]
    result["image_category"] = vis_best

    if vis_best_score < RELEVANCE["min_vis_signal"]:
        result.update(
            relevant=False,
            confidence=round(_clamp(1 - vis_best_score) * 100, 1),
            reason=(f"The photo doesn't show a clear sign of "
                    f"'{CATEGORY_META[txt_best]['label']}' — it may be the wrong image, "
                    f"too dark or blurry, or unrelated to a civic issue."),
        )
        return result

    overlap = vis.get(txt_best, 0.0)
    gap = vis_best_score - overlap
    mismatch = (vis_best != txt_best) and not _compatible(vis_best, txt_best) and (
        overlap < RELEVANCE["min_overlap"] or                                   # near-zero evidence
        (vis_best_score >= RELEVANCE["gap_trigger"] and gap >= RELEVANCE["min_gap"])  # confidently something else
    )
    if mismatch:
        result.update(
            relevant=False,
            confidence=round(_clamp(max(gap, 1 - overlap)) * 100, 1),
            reason=(f"You described '{CATEGORY_META[txt_best]['label']}' "
                    f"({', '.join(text_hits[:3])}), but the photo looks more like "
                    f"'{CATEGORY_META[vis_best]['label']}'. The image doesn't appear to "
                    f"match what you reported."),
        )
        return result

    result["reason"] = f"Photo is consistent with '{CATEGORY_META[txt_best]['label']}'."
    result["confidence"] = round(_clamp(0.5 + 0.5 * min(vis_best_score, txt_best_score)) * 100, 1)
    return result


# ---------------------------------------------------------------------------
# Photo authenticity — real camera photo vs. screenshot / graphic / collage
# ---------------------------------------------------------------------------
# WHAT THIS CHECK CAN AND CANNOT DO — read before trusting it.
# Three offline rules, all pure pixel statistics:
#   1. screenshots, posters, logos, memes  -> huge areas of perfectly constant colour
#   2. comic pages / multi-panel collages  -> thin black panel borders + an outer frame
#   3. drawings, anime/cartoons, digital paintings, 3-D renders and most AI-generated
#      images -> a *combination* of synthetic-image cues (rule 3, SYNTH below)
# Rule 3 is new. No single statistic separates artwork from photographs, so it scores several
# weak, independent cues and rejects only when enough of them agree (see _synthetic_evidence).
# Measured on real photographs (including JPEG-recompressed and downscaled copies) the score
# stays at or below 3; on drawings, anime art and screenshots it lands at 6-9. That margin is
# what makes the rule safe to act on, but it is still a heuristic: a photorealistic AI image
# with synthetic grain can pass, and a heavily denoised, almost-featureless real photo can be
# refused. The check that really looks at the picture is the vision-model gate below
# (needs ANTHROPIC_API_KEY) — rule 3 is what runs when that is unavailable.
# Rules 1 and 2 stay deliberately strict so real sky-heavy or night-time civic photos are not
# rejected by mistake (the old "flat patch" rule rejected real photos with a clear sky, a
# blurred background or a black night sky).
AUTH = {
    "max_side":        1024,   # working resolution for colour statistics
    "line_side":       2048,   # working resolution for thin panel-border lines
    "const_tol":       1,      # max per-channel step to still count as "the same colour"
    "min_lum_lit":     20,     # ignore near-black (night sky / shadows) — that is not "graphic" evidence
    "graphic_const":   0.72,   # >= this fraction perfectly-flat lit pixels -> screenshot/graphic/poster
    "rule_dark":       45,     # luminance below this counts as a "black line"
    "rule_min_run":    0.30,   # a line must run at least this fraction of the frame
    "rule_thick_max":  0.03,   # ...and be thinner than this fraction of the frame
    "rule_gap":        6,      # px on each side that must NOT be part of the line
    "rule_side_lum":   70,     # both sides of a panel border must be lit (image, not night)
    "frame_run":       0.60,   # an outer frame line must span this much of the edge
    "patch_size":      8,      # legacy stats, still reported
    "flat_std_thresh": 2.5,
}

# Rule 3 — drawing / anime / render / AI-image evidence. Every entry is a threshold for one
# weak cue; `reject_score` is how much total evidence is needed before the photo is refused.
# Calibrated so real photographs score <= 3 and artwork scores >= 6 (see the note above).
SYNTH = {
    "grain_none":    0.06,   # median high-frequency residual in the flattest lit blocks
    "grain_low":     0.18,   # ...a little grain, but far less than a camera sensor produces
    "white_bg":      0.12,   # fraction of the frame that is flat, pure white (paper/canvas)
    "bimodal_hi":    0.50,   # hard-edge / surface-texture ratio: drawn outlines, no texture
    "bimodal_mid":   0.34,
    "cel_flat":      0.58,   # fraction of near-flat colour fill (cel shading)
    "cel_texture":   0.22,   # ...with almost no mid-frequency detail between the fills
    "palette_top8":  0.55,   # 8 quantised colours covering most of a colourful image
    "palette_min_colours": 90,   # (guards against grey-scale photos, which also have few)
    "outline":       0.018,  # fraction of dark, thin, ink-like outline pixels
    "small_side":    320,    # below this (px, long side) the statistics get the benefit of doubt
    "reject_score":  4,      # >= this much combined evidence -> not a camera photo
}


def _thin_lines(lum, dark_thr, min_run, thick_max_frac, gap, side_lum, frame_run):
    """Count thin, long dark lines across `lum` (H x W).
    Returns (interior_lines, frame_lines): interior = a black rule with lit pixels on both
    sides; frame = a black band hugging the image boundary."""
    H, W = lum.shape
    dark = lum < dark_thr

    def runs(mask):
        n = mask.shape[0]
        pad = np.zeros((n, 1), np.int8)
        d = np.diff(np.concatenate([pad, mask.astype(np.int8), pad], axis=1), axis=1)
        out = np.zeros(n, np.float32)
        for i in range(n):
            st = np.flatnonzero(d[i] == 1)
            if len(st):
                out[i] = (np.flatnonzero(d[i] == -1) - st).max()
        return out / mask.shape[1]

    interior = frame = 0
    for run_frac, cover, mean_lum in ((runs(dark), dark.mean(axis=1), lum.mean(axis=1)),
                                      (runs(dark.T), dark.mean(axis=0), lum.mean(axis=0))):
        n = len(run_frac)
        thick_max = max(3, int(round(thick_max_frac * n)))
        cand = run_frac >= min_run
        i = 0
        while i < n:
            if not cand[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and cand[j + 1]:
                j += 1
            thick = j - i + 1
            if thick <= thick_max:
                lo, hi = i - gap, j + gap
                touches_lo, touches_hi = i <= 1, j >= n - 2
                lo_ok = lo >= 0 and run_frac[lo] < min_run * 0.5 and mean_lum[lo] >= side_lum
                hi_ok = hi < n and run_frac[hi] < min_run * 0.5 and mean_lum[hi] >= side_lum
                if lo_ok and hi_ok:
                    interior += 1
                elif (touches_lo and hi_ok and cover[i:j + 1].max() >= frame_run) or \
                     (touches_hi and lo_ok and cover[i:j + 1].max() >= frame_run):
                    frame += 1
            i = j + 1
    return interior, frame


def _synthetic_features(img):
    """Statistics that separate drawn/rendered images from camera photographs.

    `img` is an RGB PIL image already downscaled to AUTH["max_side"].
    Every feature is deliberately cheap and independent of the others:

      grain         median high-frequency residual in the flattest *lit* blocks. A camera
                    sensor always leaves some grain, even after JPEG; flat digital fills
                    leave literally none.
      white_bg      fraction of the frame that is pure white AND perfectly flat — the blank
                    canvas/paper background typical of illustrations and stickers.
      bimodality    (hard edges) / (mid-frequency surface texture). Drawings are built from
                    crisp outlines separating untextured fills, so this ratio runs high;
                    photographs are full of mid-frequency texture, so it runs low.
      flat_fill     fraction of near-flat colour (cel shading).
      texture       fraction of mid-frequency detail.
      palette_top8  how much of the image the 8 most common quantised colours cover.
      colours       how many quantised colours are used at all.
      outline       dark, thin, ink-like strokes sitting against bright fills.
    """
    rgb = np.asarray(img).astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    dx = np.abs(np.diff(rgb, axis=1)).max(axis=2)[:-1, :]
    dy = np.abs(np.diff(rgb, axis=0)).max(axis=2)[:, :-1]
    grad = np.maximum(dx, dy)
    lit = lum[:-1, :-1] >= AUTH["min_lum_lit"]

    f = {}
    f["flat_fill"] = round(float(((grad <= 3) & lit).mean()), 3)
    f["texture"] = round(float(((grad > 6) & (grad < 25)).mean()), 3)
    hard = float((grad >= 50).mean())
    f["bimodality"] = round(hard / (f["texture"] + 1e-6), 3)

    gray = np.asarray(img.convert("L")).astype(np.float32)
    med = np.asarray(img.convert("L").filter(ImageFilter.MedianFilter(3))).astype(np.float32)
    residual = np.abs(gray - med)
    bs, (H, W) = 16, gray.shape
    Hc, Wc = H // bs * bs, W // bs * bs
    f["grain"] = None
    if Hc >= bs and Wc >= bs:
        gb = gray[:Hc, :Wc].reshape(Hc // bs, bs, Wc // bs, bs)
        rb = residual[:Hc, :Wc].reshape(Hc // bs, bs, Wc // bs, bs)
        block_std = gb.std(axis=(1, 3)).ravel()
        block_res = rb.mean(axis=(1, 3)).ravel()
        block_lum = gb.mean(axis=(1, 3)).ravel()
        ok = block_lum >= AUTH["min_lum_lit"]          # ignore night/shadow blocks
        if int(ok.sum()) >= 8:
            order = np.argsort(block_std[ok])
            k = max(4, int(0.25 * int(ok.sum())))      # the quarter flattest lit blocks
            f["grain"] = round(float(np.median(block_res[ok][order[:k]])), 3)

    white = rgb.min(axis=2) >= 250
    f["white_bg"] = round(float((white[:-1, :-1] & (grad <= 2)).mean()), 3)

    q = (rgb // 8).astype(np.int32)                    # 5 bits per channel
    idx = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    counts = np.sort(np.bincount(idx.ravel()))[::-1]
    total = idx.size
    f["palette_top8"] = round(float(counts[:8].sum() / total), 3)
    f["colours"] = int((counts > total * 1e-4).sum())

    dilated = np.asarray(img.convert("L").filter(ImageFilter.MaxFilter(5))).astype(np.float32)
    f["outline"] = round(float(((gray < 90) & (dilated > 185)).mean()), 4)
    return f


def _synthetic_evidence(f):
    """Combine the weak cues in `f` into (score, human-readable signals).

    Nothing here rejects on its own — SYNTH["reject_score"] needs several cues to agree.
    """
    score, signals = 0.0, []
    grain = f.get("grain")
    grain_clean = grain is not None and grain <= SYNTH["grain_low"]
    if grain is not None:
        if grain <= SYNTH["grain_none"]:
            score += 2
            signals.append("no camera grain anywhere in the frame")
        elif grain <= SYNTH["grain_low"]:
            score += 1
            signals.append("almost no camera grain")
    if f["white_bg"] >= SYNTH["white_bg"] and grain_clean:
        score += 2
        signals.append(f"{f['white_bg'] * 100:.0f}% of the frame is flat, pure-white background")
    if f["bimodality"] >= SYNTH["bimodal_hi"]:
        score += 2
        signals.append("hard outlines with no surface texture between them")
    elif f["bimodality"] >= SYNTH["bimodal_mid"]:
        score += 1
        signals.append("edges far sharper than any surface texture")
    if f["flat_fill"] >= SYNTH["cel_flat"] and f["texture"] <= SYNTH["cel_texture"]:
        score += 1
        signals.append("large flat colour fills (cel-shaded look)")
    if f["palette_top8"] >= SYNTH["palette_top8"] and f["colours"] >= SYNTH["palette_min_colours"]:
        score += 1
        signals.append("a handful of colours cover most of a colourful image")
    if f["outline"] >= SYNTH["outline"] and f["bimodality"] >= SYNTH["bimodal_mid"]:
        score += 1
        signals.append("ink-like dark outlines drawn against bright fills")
    return score, signals


# ---------------------------------------------------------------------------
# Provenance — what the FILE says about where it came from
# ---------------------------------------------------------------------------
# Rules 1-3 below look at pixels. They cannot catch a photorealistic render or a modern
# text-to-image generation: measured on a game screenshot and on a Gemini-generated flood
# photo, both score 0 on every pixel cue, because there is nothing cartoon-like about them.
# That is not a tuning problem — a diffusion model's output is *designed* to have photo
# statistics, and telling them apart by hand-written statistics is an open research problem.
#
# What does work is provenance: where the file says it came from.
#   * AI images increasingly ship a signed C2PA manifest saying so. Google's Gemini/Imagen
#     output carries "trainedAlgorithmicMedia" + a SynthID note; Adobe, OpenAI and others
#     write similar manifests. That is a definitive, self-declared answer.
#   * Photos taken on a phone carry EXIF: Make, Model, DateTimeOriginal, exposure, ISO, f-stop.
#   * Game captures, renders, generated images, screenshots and web downloads carry none of
#     that, and often have a frame size no camera produces (an ultrawide 2.35:1 crop, or a
#     64-pixel-grid size like 1408x768 that diffusion models emit).
# So CivicProof weighs provenance as evidence. A file that *declares* itself AI-generated is
# refused outright. A file with no camera metadata is suspicious but not condemned on its own —
# a genuine photo forwarded through WhatsApp, saved from a web page or re-encoded by an editor
# loses its EXIF too, and refusing those would reject most of what citizens actually upload.
# It takes a corroborating signal (a frame shape no camera makes, a generator's 64-pixel grid
# size, or the pixel cues in rule 3) before the photo is refused.
# Set CIVICPROOF_REQUIRE_CAMERA_METADATA=1 for a strict deployment that mandates in-app camera
# capture; then anything without camera metadata is refused.
PROVENANCE = {
    "scan_bytes":  768 * 1024,   # how much of the head/tail to scan for metadata markers
    "require_camera_env": "CIVICPROOF_REQUIRE_CAMERA_METADATA",
    # aspect ratios cameras and phones actually produce (plus their portrait forms)
    "camera_aspects": [1.0, 1.25, 4 / 3, 1.5, 1.6, 16 / 9, 1.85, 2.0],
    "aspect_tol": 0.02,
    "ultrawide": 2.2,          # >= this is a cinematic/game capture shape, not a camera frame
}

# Lower-cased byte markers that mean "this file declares itself synthetic".
AI_PROVENANCE_MARKERS = [
    (b"trainedalgorithmicmedia", "a C2PA manifest marking it as AI-generated"),
    (b"compositewithtrainedalgorithmicmedia", "a C2PA manifest marking it as AI-composited"),
    (b"created by google generative ai", "Google Generative AI"),
    (b"synthid", "a SynthID AI watermark"),
    (b"openai.com", "OpenAI"),
    (b"dall-e", "DALL-E"),
    (b"midjourney", "Midjourney"),
    (b"stable diffusion", "Stable Diffusion"),
    (b"stablediffusion", "Stable Diffusion"),
    (b"automatic1111", "a Stable Diffusion WebUI"),
    (b"comfyui", "ComfyUI"),
    (b"invokeai", "InvokeAI"),
    (b"novelai", "NovelAI"),
    (b"firefly", "Adobe Firefly"),
    (b"imagen", "Google Imagen"),
    (b"leonardo.ai", "Leonardo.Ai"),
    (b"black-forest-labs", "FLUX"),
    (b"flux.1", "FLUX"),
    (b"grok-imagine", "Grok"),
    (b"parameters\x00negative prompt", "a diffusion prompt block"),
]

# Markers left by screen-capture and screen-recording tools.
SCREENSHOT_MARKERS = [
    (b"screenshot", "a screenshot tool"),
    (b"greenshot", "Greenshot"),
    (b"snipaste", "Snipaste"),
    (b"lightshot", "Lightshot"),
    (b"sharex", "ShareX"),
    (b"snagit", "Snagit"),
    (b"nvidia geforce", "NVIDIA GeForce Experience (game capture)"),
    (b"steam screenshot", "Steam (game capture)"),
    (b"xbox game bar", "Xbox Game Bar (game capture)"),
]

_EXPOSURE_TAGS = {33434: "ExposureTime", 33437: "FNumber", 34855: "ISO",
                  37386: "FocalLength", 36867: "DateTimeOriginal", 37385: "Flash"}


def _file_markers(path):
    """Head + tail of the file, lower-cased, for cheap metadata marker matching."""
    n = PROVENANCE["scan_bytes"]
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(n)
            if size > 2 * n:
                fh.seek(-n, os.SEEK_END)
                tail = fh.read(n)
            else:
                tail = b""
        return (head + tail).lower()
    except Exception:
        return b""


def inspect_provenance(image_path):
    """What the file itself claims about its origin.

    Returns {ai_declared, ai_generator, screenshot_tool, camera_evidence, camera_fields,
             width, height, aspect, aspect_is_camera, diffusion_grid, signals}.
    `camera_evidence` counts how many independent camera-only facts are present
    (make/model, capture timestamp, exposure settings, GPS).
    """
    p = {"ai_declared": False, "ai_generator": "", "screenshot_tool": "",
         "camera_evidence": 0, "camera_fields": [], "width": None, "height": None,
         "aspect": None, "aspect_is_camera": True, "diffusion_grid": False, "signals": []}
    if not image_path or not os.path.exists(image_path):
        return p

    blob = _file_markers(image_path)
    for marker, label in AI_PROVENANCE_MARKERS:
        if marker in blob:
            p["ai_declared"] = True
            p["ai_generator"] = label
            p["signals"].append(f"the file carries {label}")
            break
    if not p["ai_declared"]:
        for marker, label in SCREENSHOT_MARKERS:
            if marker in blob:
                p["screenshot_tool"] = label
                p["signals"].append(f"the file was written by {label}")
                break

    try:
        img = Image.open(image_path)
        p["width"], p["height"] = img.size
        exif = img.getexif() or {}
        if exif.get(271) or exif.get(272):                     # Make / Model
            p["camera_evidence"] += 1
            p["camera_fields"].append("camera make/model")
        if exif.get(36867) or exif.get(306):                   # DateTimeOriginal / DateTime
            p["camera_evidence"] += 1
            p["camera_fields"].append("capture timestamp")
        try:
            sub = exif.get_ifd(0x8769) or {}                   # EXIF sub-IFD
        except Exception:
            sub = {}
        shot = [name for tag, name in _EXPOSURE_TAGS.items() if exif.get(tag) or sub.get(tag)]
        if shot:
            p["camera_evidence"] += 1
            p["camera_fields"].append("exposure settings (" + ", ".join(sorted(shot)[:3]) + ")")
        try:
            if exif.get_ifd(0x8825):                           # GPS IFD
                p["camera_evidence"] += 1
                p["camera_fields"].append("GPS tags")
        except Exception:
            pass

        w, h = img.size
        ar = max(w, h) / max(1, min(w, h))
        p["aspect"] = round(ar, 4)
        p["aspect_is_camera"] = any(abs(ar - a) <= PROVENANCE["aspect_tol"]
                                    for a in PROVENANCE["camera_aspects"])
        # A cropped photo can have any ratio, so an odd aspect on its own means nothing.
        # Only a cinematic/ultrawide frame is real evidence: that is a game or video capture,
        # not something a phone camera hands you.
        if ar >= PROVENANCE["ultrawide"]:
            p["signals"].append(f"a {w}x{h} ultrawide frame ({ar:.2f}:1) — a game or video "
                                f"capture shape, not a shape a camera produces")
        # diffusion models emit sizes on a 64-pixel grid; cameras essentially never do
        p["diffusion_grid"] = (w % 64 == 0 and h % 64 == 0 and max(w, h) <= 2048
                               and not p["aspect_is_camera"])
        if p["diffusion_grid"]:
            p["signals"].append(f"a {w}x{h} frame sitting exactly on the 64-pixel grid "
                                f"image generators use")
    except Exception:
        pass

    if p["camera_evidence"]:
        p["signals"].append("camera metadata: " + ", ".join(p["camera_fields"]))
    return p


def require_camera_metadata():
    """Strict provenance mode — OFF by default.

    When on, any file with no camera metadata is refused. That catches every render, game
    capture and generated image outright, but it also refuses perfectly genuine photos that
    lost their EXIF on the way in: anything forwarded through WhatsApp or Telegram, saved
    from a web page, re-encoded by an editor, or re-saved as PNG. Most photos citizens
    actually upload have been through at least one of those, so strict mode is only right
    for a deployment that mandates in-app camera capture.

    Off (the default), missing metadata is still counted as evidence — it just needs a
    corroborating signal before the photo is refused (see rule 4).
    """
    return os.environ.get(PROVENANCE["require_camera_env"], "0") == "1"


def assess_photo_authenticity(image_path):
    """Soft check for whether an image is plausibly ONE camera photo of a physical place.

    Rejects, in order of how certain the evidence is:
      1. screenshot / poster / graphic  — most of the frame is perfectly constant colour;
      2. collage / comic layout         — thin black panel borders plus an outer frame;
      3. drawing / anime / render / AI image — several synthetic-image cues agreeing
                                          (see SYNTH and _synthetic_evidence above).
    Camera metadata (EXIF Make/Model) does not short-circuit rule 3 — it only raises the
    amount of evidence rule 3 needs, since metadata is trivial to copy onto any file.

    Returns: {looks_like_real_photo, confidence, reason, kind, flat_ratio,
              median_block_std, signals, synthetic}
    """
    result = {"looks_like_real_photo": True, "confidence": 55.0, "reason": "", "kind": "photo",
              "flat_ratio": None, "median_block_std": None, "signals": [], "synthetic": None,
              "provenance": None}
    if not image_path or not os.path.exists(image_path):
        result.update(looks_like_real_photo=False, confidence=0.0, kind="missing",
                      reason="No photo was received.")
        return result

    # ---- rule 0: the file declares its own origin --------------------------------------
    prov = inspect_provenance(image_path)
    result["provenance"] = prov
    if prov["ai_declared"]:
        result["signals"].append(prov["signals"][0])
        result.update(
            looks_like_real_photo=False, kind="ai_generated", confidence=99.0,
            reason=(f"This image is signed as AI-generated — it carries {prov['ai_generator']}. "
                    f"CivicProof only accepts a real photo of the problem, taken on site."))
        return result
    if prov["screenshot_tool"]:
        result["signals"].append(prov["signals"][0])
        result.update(
            looks_like_real_photo=False, kind="screenshot", confidence=95.0,
            reason=(f"This is a screen capture, not a photo — the file was written by "
                    f"{prov['screenshot_tool']}. Please upload a photo of the issue itself."))
        return result
    try:
        img = Image.open(image_path)
        has_camera = prov["camera_evidence"] > 0
        img = img.convert("RGB")
        w, h = img.size

        # legacy patch texture stats — kept for the API payload / debugging, not used to reject
        s = min(1.0, 640 / max(w, h))
        small = img if s >= 1.0 else img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        gray = np.asarray(small.convert("L")).astype(np.float32)
        H, W = gray.shape
        bs = AUTH["patch_size"]
        stds = np.array([gray[r:r + bs, c:c + bs].std()
                         for r in range(0, H - bs + 1, bs) for c in range(0, W - bs + 1, bs)])
        if len(stds):
            result["flat_ratio"] = round(float((stds < AUTH["flat_std_thresh"]).mean()), 4)
            result["median_block_std"] = round(float(np.median(stds)), 2)

        if has_camera:
            result["signals"].append("camera EXIF present")

        # ---- rule 1: screenshot / graphic — perfectly constant lit colour everywhere ----------
        s = min(1.0, AUTH["max_side"] / max(w, h))
        mid = img if s >= 1.0 else img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        rgb = np.asarray(mid).astype(np.float32)
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        dx = np.abs(np.diff(rgb, axis=1)).max(axis=2)[:-1, :]
        dy = np.abs(np.diff(rgb, axis=0)).max(axis=2)[:, :-1]
        const_lit = float(((dx <= AUTH["const_tol"]) & (dy <= AUTH["const_tol"])
                           & (lum[:-1, :-1] >= AUTH["min_lum_lit"])).mean())
        result["const_lit"] = round(const_lit, 4)

        # ---- rule 2: collage / comic layout — black panel borders + outer frame ---------------
        s2 = min(1.0, AUTH["line_side"] / max(w, h))
        big = img if s2 >= 1.0 else img.resize((max(1, int(w * s2)), max(1, int(h * s2))), Image.LANCZOS)
        L = np.asarray(big.convert("L")).astype(np.float32)
        f = max(1, int(math.ceil(max(L.shape) / float(AUTH["max_side"]))))
        if f > 1:                                     # min-pool so 1-2 px black rules survive
            H2, W2 = L.shape[0] // f * f, L.shape[1] // f * f
            L = L[:H2, :W2].reshape(H2 // f, f, W2 // f, f).min(axis=(1, 3))
        interior, frame = _thin_lines(L, AUTH["rule_dark"], AUTH["rule_min_run"], AUTH["rule_thick_max"],
                                      AUTH["rule_gap"], AUTH["rule_side_lum"], AUTH["frame_run"])
        result["panel_lines"] = {"interior": interior, "frame": frame}

        # ---- rule 3: drawing / anime / 3-D render / AI-generated image -----------------------
        feats = _synthetic_features(mid)
        synth_score, synth_signals = _synthetic_evidence(feats)
        # EXIF is easy to copy onto any file, so it only makes rule 3 harder to trip, never
        # impossible. Thumbnails get the same benefit of the doubt: resampling destroys grain
        # and texture, so a small image's statistics look synthetic even when it is a photo.
        needed = (SYNTH["reject_score"]
                  + (2 if has_camera else 0)
                  + (2 if max(mid.size) < SYNTH["small_side"] else 0))
        result["synthetic"] = dict(feats, score=synth_score, threshold=needed,
                                   signals=synth_signals)

        if const_lit >= AUTH["graphic_const"]:
            result["signals"].append(f"{const_lit * 100:.0f}% of the frame is perfectly flat colour")
            result.update(
                looks_like_real_photo=False, kind="graphic",
                confidence=round(_clamp(0.55 + const_lit / 2) * 100, 1),
                reason=("This looks like a screenshot, poster or flat graphic rather than a camera photo "
                        "of a real place — most of the image is large areas of one perfectly uniform colour, "
                        "with none of the fine surface texture a real photo has."))
        elif (max(mid.size) >= SYNTH["small_side"]     # thumbnails cannot show real panel layout
              and (interior >= 2 or (interior >= 1 and frame >= 1))):
            result["signals"].append(f"{interior} black panel border(s) + {frame} outer frame line(s)")
            result.update(
                looks_like_real_photo=False, kind="collage", confidence=85.0,
                reason=("This looks like a collage, comic page or illustration board (several panels separated "
                        "by black borders), not a single camera photo of the problem. Please upload one "
                        "clear photo of the issue itself."))
        elif synth_score >= needed:
            result["signals"].extend(synth_signals)
            result.update(
                looks_like_real_photo=False, kind="illustration",
                confidence=round(_clamp(0.55 + 0.05 * (synth_score - needed) + 0.1) * 100, 1),
                reason=("This looks like a drawing, cartoon, render or AI-generated picture rather than a "
                        "photo taken with a camera — " + ("; ".join(synth_signals[:3]) or "no camera-like detail") +
                        ". CivicProof only accepts a real photo of the problem itself."))
        else:
            # ---- rule 4: no positive evidence a camera took this --------------------------
            # Photorealistic renders, game captures and modern generated images all land here:
            # they pass every pixel test, so the only thing left to ask is whether the file
            # carries the fingerprints a camera leaves.
            extra = [s for s in prov["signals"] if "ultrawide" in s or "64-pixel grid" in s]
            prov_score = (0 if has_camera else 2) + 2 * len(extra)
            result["provenance"]["score"] = prov_score
            # Strict mode refuses anything with no camera provenance at all. By default the
            # missing metadata is only evidence (+2), and it takes a corroborating signal —
            # a frame shape no camera makes, or a generator's 64-pixel grid size — to refuse.
            if (require_camera_metadata() and not has_camera) or prov_score >= 4:
                result["signals"].extend((["no camera metadata of any kind"] if not has_camera
                                          else []) + extra)
                result.update(
                    looks_like_real_photo=False, kind="unverified_source",
                    confidence=min(95.0, 60.0 + 10.0 * prov_score),
                    reason=("This file carries no evidence that a camera took it — no make or model, "
                            "no capture time, no exposure settings"
                            + ((", and it has " + extra[0]) if extra else "") +
                            ". Game captures, renders and AI-generated pictures look exactly like this. "
                            "Please take a fresh photo of the issue with your phone camera and upload "
                            "that file directly, without editing or re-saving it."))
            else:
                result["reason"] = ("No sign of a screenshot, graphic, collage or drawing, and the file "
                                    "carries camera metadata — this behaves like a camera photo."
                                    if has_camera else
                                    "No sign of a screenshot, graphic, collage or drawing.")
                result["confidence"] = round(_clamp(0.55 + 0.06 * (needed - synth_score)) * 100, 1)
    except Exception as exc:
        # An image we cannot even decode is not usable evidence — fail closed rather than
        # waving it through.
        result.update(looks_like_real_photo=False, kind="unreadable", confidence=40.0,
                      reason=f"This file could not be read as a photo ({exc}). Please upload a "
                             f"normal JPEG or PNG photo of the issue.")
    return result


# ---------------------------------------------------------------------------
# Vision-model gate — a real look at the photo (the only external API call
# in this file; everything else above is self-contained pixel statistics).
# ---------------------------------------------------------------------------
VISION = {
    "api_key_env":  "ANTHROPIC_API_KEY",     # must be set in the environment to activate
    "disable_env":  "CIVICPROOF_DISABLE_VISION_CHECK",  # set to "1" to force heuristics-only
    "model":        os.environ.get("CIVICPROOF_VISION_MODEL", "claude-haiku-4-5-20251001"),
    "api_url":      "https://api.anthropic.com/v1/messages",
    "api_version":  "2023-06-01",
    "max_tokens":   500,
    "timeout_s":    60,    # a thorough look is allowed to take up to a minute
    "max_bytes":    5 * 1024 * 1024,
}


def _load_image_b64(image_path):
    """Return (base64_str, media_type), downscaling/re-encoding as JPEG if the
    original file is too large for a fast request."""
    with open(image_path, "rb") as fh:
        raw = fh.read()
    if len(raw) <= VISION["max_bytes"]:
        media_type, _ = mimetypes.guess_type(image_path)
        if media_type in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            return base64.b64encode(raw).decode("ascii"), media_type
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def ai_vision_check(image_path, title, description):
    """Ask a real vision-capable Claude model to actually look at the photo and
    judge (a) whether it matches the citizen's title/description and (b) whether
    it looks like a genuine camera photo of a physical scene at all (vs. a
    screenshot, drawing, meme, stock/AI-generated image, etc).

    This is genuine image understanding, unlike the pixel-statistics heuristics
    elsewhere in this file — but it still can't offer a certified, forensic
    "AI-generated: yes/no" verdict; it's a model's best visual judgement, stated
    to the citizen in plain language.

    Returns None/omitted-vision dict when the check can't run (no API key,
    disabled, network/parse failure) so the caller falls back to heuristics.
    Shape on success: {ok: True, relevant, looks_like_real_photo, confidence,
    category_confidence, category, reason}. Shape on failure: {ok: False, error: "..."}.
    """
    if os.environ.get(VISION["disable_env"]) == "1":
        return None
    api_key = os.environ.get(VISION["api_key_env"])
    if not api_key:
        return None
    if not image_path or not os.path.exists(image_path):
        return {"ok": False, "error": "no image file on disk"}

    try:
        b64, media_type = _load_image_b64(image_path)
    except Exception as exc:
        return {"ok": False, "error": f"could not read/encode image: {exc}"}

    prompt = (
        "You are screening a citizen's civic-issue report for a city government app. "
        "The citizen typed a title (answering \"what happened?\") and a short free-text "
        "description, and attached exactly one photo as evidence. Look carefully at the "
        "photo itself — what it actually shows — and judge it against the text.\n\n"
        f"Title (\"what happened?\"): {title!r}\n"
        f"Description (\"describe it briefly\"): {description!r}\n\n"
        "Decide three things:\n"
        "1. relevant — does the photo plausibly show the issue described in the title/"
        "description (same general kind of problem, e.g. flooding/water, garbage, a broken "
        "streetlight, a pothole/road damage, a crack — or, if the text is too vague to pin "
        "down a specific issue, whether the photo at least shows *some* plausible real-world "
        "civic/public-infrastructure problem)? If the photo is clearly unrelated (random "
        "object, meme, artwork, a person's face, an unrelated scene, etc.), mark this false.\n"
        "2. looks_like_real_photo — is this an actual camera photograph of a real physical "
        "place? Mark it FALSE for anything that is not: a drawing, sketch, painting, comic "
        "or manga panel, anime/cartoon art, clip art, logo, poster, meme, screenshot of a "
        "screen, 3-D render or game capture, or a picture that looks generated or heavily "
        "edited by AI. Judge the picture itself, not how well it matches the text, and if "
        "you are in real doubt about whether a camera took it, answer false.\n"
        "3. category — the single most specific civic issue the photo shows. If the citizen's "
        "text names a specific issue type and the photo is consistent with it, use THAT category. "
        "Look at what the water/object is doing, not just its colour: a pothole or damaged road "
        "surface that is filled with rainwater is \"pothole\" (not \"water_leak\"); use "
        "\"water_leak\" only for a leaking/burst pipe or flooding where the water itself is the "
        "problem, and \"drainage\" for blocked or overflowing drains.\n\n"
        "Respond with ONLY a single JSON object and nothing else — no markdown fences, no "
        "preamble — in exactly this shape:\n"
        '{"relevant": true or false, "looks_like_real_photo": true or false, '
        '"confidence": <integer 0-100, your overall confidence in relevant + looks_like_real_photo>, '
        '"category_confidence": <integer 0-100, your confidence in the category alone>, '
        '"category": one of "pothole", "garbage", "streetlight", "water_leak", "road_crack", '
        '"sidewalk_damage", "drainage", "sewage", "traffic_signal", "traffic_sign", '
        '"road_marking", "electrical_hazard", "fallen_tree", "construction_hazard", '
        '"building_damage", "bridge_damage", "public_property_damage", "other", "not_applicable", '
        '"reason": "1-2 short plain sentences explaining your judgement, written directly to '
        'the citizen who uploaded it"}'
    )

    body = json.dumps({
        "model": VISION["model"],
        "max_tokens": VISION["max_tokens"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode("utf-8")

    req = urllib.request.Request(VISION["api_url"], data=body, method="POST", headers={
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": VISION["api_version"],
    })
    try:
        with urllib.request.urlopen(req, timeout=VISION["timeout_s"]) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300]
        return {"ok": False, "error": f"API HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"ok": False, "error": f"API request failed: {exc}"}

    try:
        text = "".join(block.get("text", "") for block in payload.get("content", [])
                       if block.get("type") == "text").strip()
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
        parsed = json.loads(text)
        return {
            "ok": True,
            "relevant": bool(parsed.get("relevant", True)),
            "looks_like_real_photo": bool(parsed.get("looks_like_real_photo", True)),
            "confidence": float(parsed.get("confidence", 50)),
            "category_confidence": float(parsed.get("category_confidence", parsed.get("confidence", 50))),
            "category": str(parsed.get("category") or "other").strip().lower(),
            "reason": str(parsed.get("reason") or "").strip() or "No explanation returned.",
        }
    except Exception as exc:
        return {"ok": False, "error": f"could not parse model response: {exc} — raw: {str(payload)[:300]}"}


def gate_photo(image_path, title, description):
    """Single entry point app.py calls before accepting a report photo.

    Tries a real vision-model judgement first (accurate, but needs
    ANTHROPIC_API_KEY and network — can take up to ~60s, which is expected for
    a thorough look rather than a shallow one). If that's unavailable or fails
    for any reason, falls back to the local pixel-statistics heuristics so the
    app still works offline / without a key, just less reliably.
    """
    def reject(code, message, source, relevance, authenticity, vision=None):
        out = {"accepted": False, "reason_code": code,
               "message": message or "This photo cannot be used for a report.",
               "source": source, "relevance": relevance, "authenticity": authenticity,
               # the caller must discard the upload and send the citizen back to step 1
               "restart": True, "restart_step": 1}
        if vision:
            out["vision"] = vision
        return out

    # The pixel checks are cheap and run in every mode: they are the only thing standing
    # between the app and a drawing when there is no API key, and they are a useful second
    # opinion when there is one (a vision model occasionally calls a clean render a photo).
    auth = assess_photo_authenticity(image_path)
    rel = check_relevance(image_path, title, description)   # always computed: keeps the payload shape stable
    # a specific code so the UI can word the refusal precisely
    auth_code = {"ai_generated": "ai_generated", "screenshot": "screenshot",
                 "collage": "collage", "graphic": "screenshot",
                 "illustration": "not_real_photo", "unverified_source": "unverified_source",
                 "unreadable": "unreadable", "missing": "no_photo"}.get(auth.get("kind"),
                                                                        "not_real_photo")

    vision = ai_vision_check(image_path, title, description)
    if vision and vision.get("ok"):
        relevance = {"relevant": vision["relevant"], "confidence": vision["confidence"],
                     "reason": vision["reason"], "image_category": vision["category"],
                     "text_category": None}
        authenticity = dict(auth)
        authenticity.update(looks_like_real_photo=bool(vision["looks_like_real_photo"]) and
                                                  auth["looks_like_real_photo"],
                            vision_reason=vision["reason"], vision_confidence=vision["confidence"])
        # "not a photo at all" is reported before "photo of the wrong thing": telling someone
        # their cartoon "looks more like garbage" would be confusing.
        if not vision["looks_like_real_photo"]:
            return reject("not_real_photo", vision["reason"], "vision_model",
                          relevance, authenticity, vision)
        if not auth["looks_like_real_photo"]:
            # The model said photo, the pixels disagree — trust the stricter verdict.
            return reject(auth_code, auth["reason"], "heuristic_override",
                          relevance, authenticity, vision)
        if not vision["relevant"]:
            return reject("irrelevant", vision["reason"], "vision_model",
                          relevance, authenticity, vision)
        return {"accepted": True, "reason_code": None, "message": "", "restart": False,
                "source": "vision_model", "vision": vision,
                "relevance": relevance, "authenticity": authenticity}

    # Fallback: no API key configured, or the vision call failed — heuristics only.
    if vision and vision.get("error"):
        print(f"[ai_engine] vision gate unavailable, falling back to heuristics: {vision['error']}")
    if not auth["looks_like_real_photo"]:
        return reject(auth_code, auth["reason"], "heuristic", rel, auth)
    if not rel["relevant"]:
        return reject("irrelevant", rel["reason"], "heuristic", rel, auth)
    return {"accepted": True, "reason_code": None, "message": "", "restart": False,
            "source": "heuristic", "relevance": rel, "authenticity": auth}


def gate_status():
    """Which photo gate is active right now (for logs / a UI badge)."""
    if os.environ.get(VISION["disable_env"]) == "1":
        return {"vision_enabled": False, "mode": "heuristics only (vision check disabled by env)"}
    if os.environ.get(VISION["api_key_env"]):
        return {"vision_enabled": True, "mode": f"vision model ({VISION['model']}), heuristics as fallback"}
    return {"vision_enabled": False,
            "mode": "heuristics only — set ANTHROPIC_API_KEY to enable the vision-model check"}


print(f"[ai_engine] photo gate: {gate_status()['mode']}")
print("[ai_engine] provenance: " + ("STRICT — files without camera metadata are refused"
                                    if require_camera_metadata() else
                                    "missing camera metadata counts as evidence, not an automatic "
                                    "refusal (set CIVICPROOF_REQUIRE_CAMERA_METADATA=1 for strict)"))
if not gate_status()["vision_enabled"]:
    print("[ai_engine] NOTE: offline mode rejects screenshots, graphics, collages and (by combined "
          "pixel evidence) drawings, cartoons and most renders/AI images. It still cannot tell a "
          "real photo of the *wrong thing* from the right one — that needs the vision-model check.")


# ---------------------------------------------------------------------------
# Priority engine
# ---------------------------------------------------------------------------
def priority_label(score):
    if score >= 80:
        return "Critical"
    if score >= 62:
        return "High"
    if score >= 38:
        return "Medium"
    return "Low"


def compute_priority(category, upvotes=0, merged_duplicates=0, urgency_mult=1.0,
                     urgency_hits=None, created_ts=None):
    base = CATEGORY_META.get(category, CATEGORY_META["other"])["severity"]
    score = base * (urgency_mult or 1.0)
    reasons = [{
        "text": f"Base severity {base:.0f} — {CATEGORY_META.get(category, {})['label'] if category in CATEGORY_META else 'uncategorised issue'}",
        "delta": round(base * 0.55, 1),
    }]
    if (urgency_mult or 1.0) > 1.0:
        reasons.append({
            "text": f"Urgent language detected ({', '.join((urgency_hits or [])[:3])}) ×{urgency_mult:.2f}",
            "delta": round(base * (urgency_mult - 1.0), 1),
        })
    if upvotes:
        boost = min(upvotes, 30) * 0.5
        score += boost
        reasons.append({"text": f"{upvotes} citizen confirmation{'s' if upvotes != 1 else ''}", "delta": round(boost, 1)})
    if merged_duplicates:
        boost = min(merged_duplicates, 6) * 4.0
        score += boost
        reasons.append({"text": f"{merged_duplicates} duplicate report(s) merged — corroborates issue", "delta": boost})
    if created_ts:
        days = max(0.0, (time.time() - created_ts) / 86400.0)
        age_boost = min(days, 12) * 0.6
        if age_boost >= 1:
            score += age_boost
            reasons.append({"text": f"Unresolved for {int(days)} day(s) — aging penalty", "delta": round(age_boost, 1)})
    score = round(min(100.0, score), 1)
    return {"score": score, "label": priority_label(score), "reasons": reasons}


# ---------------------------------------------------------------------------
# Perceptual hashes & duplicate detection
# ---------------------------------------------------------------------------
def _ahash(path, size=8):
    g = np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return sum(int(b) << i for i, b in enumerate(bits))


def _dhash(path, size=8):
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return sum(int(b) << i for i, b in enumerate(bits))


def photo_hashes(path):
    try:
        return f"{_ahash(path):016x}", f"{_dhash(path):016x}"
    except Exception:
        return None, None


def hamming(h1, h2):
    if not h1 or not h2:
        return 64
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def text_similarity(text1, text2):
    a, b = set(_tokens(text1)), set(_tokens(text2))
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def find_duplicate(new, candidates, auto_threshold=62, maybe_threshold=40):
    """Return {'match': <report dict|None>, 'score': int, 'possible': bool, 'signals': []}"""
    best = None
    for cand in candidates:
        if cand.get("status") in ("resolved", "duplicate") :
            pass  # resolved/dup still count as references, they prove existence
        score, signals = 0.0, []
        dh = hamming(new.get("phash_d"), cand.get("phash_d"))
        ah = hamming(new.get("phash_a"), cand.get("phash_a"))
        if dh <= 6:
            score += 62; signals.append(f"near-identical photo (dHash distance {dh})")
        elif dh <= 12:
            score += 34; signals.append(f"similar photo (dHash distance {dh})")
        if ah <= 8:
            score += 14; signals.append(f"matching layout (aHash distance {ah})")
        d = None
        if new.get("lat") is not None and cand.get("lat") is not None:
            d = haversine_m(new["lat"], new["lng"], cand["lat"], cand["lng"])
            if d <= 30:
                score += 26; signals.append(f"{d:.0f} m away — same spot")
            elif d <= 80:
                score += 16; signals.append(f"{d:.0f} m away")
            elif d <= 160:
                score += 8;  signals.append(f"{d:.0f} m away")
        if new.get("category") and cand.get("category") == new["category"]:
            score += 14; signals.append("same issue category")
        ts = text_similarity(f"{new.get('title','')} {new.get('description','')}",
                             f"{cand.get('title','')} {cand.get('description','')}")
        if ts >= 0.12:
            score += ts * 34
            signals.append(f"description overlap {ts:.2f}")
        if best is None or score > best["score"]:
            best = {"match": cand, "score": round(score, 1), "signals": signals,
                    "distance_m": round(d, 1) if d is not None else None}
    if best is None or best["score"] < maybe_threshold:
        return {"match": None, "score": best["score"] if best else 0, "possible": False, "signals": []}
    return {
        "match": best["match"],
        "score": best["score"],
        "possible": best["score"] < auto_threshold,
        "signals": best["signals"],
        "distance_m": best["distance_m"],
        "auto_threshold": auto_threshold,
    }


# ---------------------------------------------------------------------------
# ProofWatch — before/after verification
# ---------------------------------------------------------------------------
PW = {
    "identical_dhash": 4,       # after photo basically == before photo
    "scene_ssim_min": 0.42,     # below → suspicious scene mismatch
    "scene_corr_min": 0.78,     # histogram correlation fallback for scene match
    "persist_conf": 56.0,       # issue classifier confidence on after-photo
    "min_change_ratio": 0.02,   # at least this fraction of pixels changed materially
}


def _ssim_global(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                 ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))


def _ssim_blocks(a, b, bs=24):
    vals = []
    h, w = a.shape
    for r in range(0, h - bs + 1, bs):
        for c in range(0, w - bs + 1, bs):
            vals.append(_ssim_global(a[r:r + bs, c:c + bs], b[r:r + bs, c:c + bs]))
    return float(np.mean(vals)), float(np.min(vals))


def _hist_corr(a, b, bins=32):
    cs = []
    for ch in range(3):
        ha, _ = np.histogram(a[..., ch], bins=bins, range=(0, 255))
        hb, _ = np.histogram(b[..., ch], bins=bins, range=(0, 255))
        ha = ha.astype(np.float64); hb = hb.astype(np.float64)
        if ha.std() < 1e-6 or hb.std() < 1e-6:
            cs.append(1.0 if np.allclose(ha, hb) else 0.0)
            continue
        cs.append(float(np.corrcoef(ha, hb)[0, 1]))
    return float(np.mean(cs))


def save_diff_heatmap(before_path, after_path, out_path):
    size = (320, 240)
    a = np.asarray(Image.open(before_path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)
    b = np.asarray(Image.open(after_path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)
    d = np.abs(a - b).mean(axis=2)
    d = np.clip(d / 64.0, 0, 1)
    heat = np.zeros((*d.shape, 3), dtype=np.uint8)
    heat[..., 0] = (np.clip(d * 2.2, 0, 1) * 255).astype(np.uint8)
    heat[..., 1] = (np.clip(d * 1.1 - 0.15, 0, 1) * 200).astype(np.uint8)
    heat[..., 2] = (np.clip(0.35 - d, 0, 1) * 90).astype(np.uint8)
    blend = (b * 0.55 + heat * 0.45).astype(np.uint8)
    Image.fromarray(blend).save(out_path, format="PNG")
    return out_path


def issue_signature(f, category):
    """Scalar strength (0..1) of the *reported issue's* visual signature in a photo.
    Used by ProofWatch: a real repair makes this value collapse between before→after."""
    if category == "pothole":
        return _clamp(f.get("edge_in_dark", 0) / 0.42)                     # roughness inside dark patch
    if category == "garbage":
        return 0.6 * _clamp(f["edge_lower"] / 0.30) + 0.4 * _clamp(f["colorfulness"] / 45)
    if category == "water_leak":
        return 0.75 * _clamp(f.get("cool_ratio", 0) / 0.32) + 0.25 * _clamp(f.get("shine_ratio", 0) / 0.10)
    if category == "streetlight":                                          # darkness / lamp-not-working
        return (0.50 * _clamp((0.40 - f["brightness"]) / 0.30)
                + 0.50 * (1 - _clamp((f.get("grid_max", 0) - 0.34) / 0.20)))
    if category == "road_crack":
        return 0.6 * _clamp(f.get("edge_dense", 0) / 0.12) + 0.4 * _clamp(f["edge_ratio"] / 0.30)
    return 0.5 * _clamp(f["edge_lower"] / 0.30) + 0.5 * _clamp(f["dark_lower"] / 0.40)


def proofwatch_verify(before_path, after_path, category, diff_out=None):
    """Judge an authority 'after' photo against the citizen's original report.

    Steps: (1) identity check, (2) scene-match (SSIM + colour), (3) issue-signature
    delta (category-aware), (4) classifier context. Same scene + signature collapse
    → verified; same scene + signature persists → reopened; otherwise flagged."""
    size = (192, 144)
    A_rgb = np.asarray(Image.open(before_path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)
    B_rgb = np.asarray(Image.open(after_path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)
    A = 0.299 * A_rgb[..., 0] + 0.587 * A_rgb[..., 1] + 0.114 * A_rgb[..., 2]
    Bg = 0.299 * B_rgb[..., 0] + 0.587 * B_rgb[..., 1] + 0.114 * B_rgb[..., 2]

    _, d_b = photo_hashes(before_path)
    _, d_a = photo_hashes(after_path)
    d_dist = hamming(d_b, d_a)

    ssim_g = _ssim_global(A, Bg)
    ssim_b, ssim_min = _ssim_blocks(A, Bg)
    corr = _hist_corr(A_rgb, B_rgb)
    change_ratio = float((np.abs(A - Bg) > 28).mean())

    fb = _inspect(before_path)
    fa = _inspect(after_path)
    sig_before = issue_signature(fb, category)
    sig_after = issue_signature(fa, category)
    sig_drop = (sig_before - sig_after) / max(sig_before, 0.12)

    cls_after = classify_report(image_path=after_path)
    cls_persist = (cls_after["category"] == category and cls_after["confidence"] >= 48)
    issue_conf_after = cls_after["confidence"] if cls_after["category"] == category else 0.0

    notes = []
    if d_dist <= PW["identical_dhash"]:
        verdict = "reopened"
        reason = ("The uploaded completion photo is essentially identical to the citizen's original "
                  "photo — no physical change detected. The repair is treated as not done and the "
                  "ticket has been automatically reopened.")
        notes.append(f"Perceptual hash distance {d_dist} ≤ {PW['identical_dhash']} (copied/duplicate photo).")
    else:
        scene_ok = (ssim_b >= PW["scene_ssim_min"]) or (corr >= PW["scene_corr_min"])
        if not scene_ok:
            verdict = "flagged"
            reason = ("The completion photo does not appear to show the same scene as the original "
                      "report (low structural and colour correlation). Flagged for field re-inspection "
                      "to rule out a mismatched or staged upload.")
            notes.append(f"Block SSIM {ssim_b:.2f} < {PW['scene_ssim_min']} and colour correlation "
                         f"{corr:.2f} < {PW['scene_corr_min']} — different scene.")
        elif sig_drop <= 0.12 and cls_persist:
            verdict = "reopened"
            reason = (f"Same scene confirmed, but the '{CATEGORY_META[category]['label']}' signature "
                      f"is still present (strength {sig_before:.2f} → {sig_after:.2f}) and the classifier "
                      f"still detects the issue at {issue_conf_after:.0f}%. The ticket has been "
                      f"automatically reopened for rework.")
        elif sig_drop <= 0.12 and change_ratio < 0.08:
            verdict = "reopened"
            reason = (f"Same scene confirmed, but almost nothing changed "
                      f"({change_ratio * 100:.0f}% of frame) and the issue signature persists "
                      f"({sig_before:.2f} → {sig_after:.2f}). Ticket automatically reopened.")
        elif sig_drop >= 0.28 and change_ratio >= PW["min_change_ratio"]:
            verdict = "verified"
            reason = (f"Same scene confirmed (SSIM {ssim_b:.2f}, colour correlation {corr:.2f}). The "
                      f"'{CATEGORY_META[category]['label']}' signature collapsed from {sig_before:.2f} "
                      f"to {sig_after:.2f} (−{sig_drop * 100:.0f}%) with {change_ratio * 100:.0f}% of "
                      f"the frame materially altered. Repair verified by ProofWatch.")
        else:
            verdict = "flagged"
            reason = (f"Same scene confirmed, but the evidence is ambiguous: issue signature moved "
                      f"{sig_before:.2f} → {sig_after:.2f} (−{max(sig_drop, 0) * 100:.0f}%) and "
                      f"{change_ratio * 100:.0f}% frame change. Sent for human re-inspection.")

    diff_path = None
    if diff_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(diff_out)), exist_ok=True)
            diff_path = save_diff_heatmap(before_path, after_path, diff_out)
        except Exception:
            diff_path = None

    return {
        "verdict": verdict,                       # verified | reopened | flagged
        "reason": reason,
        "notes": notes,
        "metrics": {
            "ssim_global": round(ssim_g, 4),
            "ssim_block": round(ssim_b, 4),
            "ssim_block_min": round(ssim_min, 4),
            "histogram_correlation": round(corr, 4),
            "change_ratio": round(change_ratio, 4),
            "photo_hash_distance": d_dist,
            "issue_signature_before": round(sig_before, 4),
            "issue_signature_after": round(sig_after, 4),
            "issue_signature_drop": round(sig_drop, 4),
            "after_issue_confidence": round(issue_conf_after, 1),
            "after_top_category": cls_after["category"],
            "after_top_confidence": cls_after["confidence"],
        },
        "thresholds": dict(PW),
        "diff_image": diff_path,
        "checked_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Convenience: one-call pipeline for a fresh citizen report
# ---------------------------------------------------------------------------
def analyze_new_report(image_path, title, description, lat, lng, existing_reports, vision_result=None):
    cls = classify_report(image_path=image_path, title=title, description=description, vision_result=vision_result)
    ph_a, ph_d = (None, None)
    if image_path:
        ph_a, ph_d = photo_hashes(image_path)
    dup = find_duplicate(
        {"phash_a": ph_a, "phash_d": ph_d, "lat": lat, "lng": lng,
         "category": cls["category"], "title": title, "description": description},
        existing_reports,
    )
    pri = compute_priority(cls["category"], urgency_mult=cls["urgency_mult"],
                           urgency_hits=cls["urgency_hits"])
    return {"classification": cls, "hashes": {"a": ph_a, "d": ph_d},
            "duplicate": dup, "priority": pri}