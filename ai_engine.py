"""
CivicProof AI Engine
====================
Self-contained computer-vision + NLP pipeline (no external APIs or model downloads):

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
import os
import math
import time
import colorsys
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = 40_000_000

# ----------------------------------------------------------------------------
# Categories
# ----------------------------------------------------------------------------
CATEGORIES = ["pothole", "garbage", "streetlight", "water_leak", "road_crack", "other"]

CATEGORY_META = {
    "pothole":     {"label": "Pothole / Road Damage", "severity": 74.0, "icon": "road"},
    "garbage":     {"label": "Garbage & Sanitation",  "severity": 56.0, "icon": "trash"},
    "streetlight": {"label": "Streetlight Fault",     "severity": 63.0, "icon": "lamp"},
    "water_leak":  {"label": "Water Leak / Flooding", "severity": 86.0, "icon": "drop"},
    "road_crack":  {"label": "Road Crack / Cave-in",  "severity": 60.0, "icon": "crack"},
    "other":       {"label": "Other Civic Issue",     "severity": 40.0, "icon": "flag"},
}

# ---------------------------------------------------------------------------
# Keyword NLP
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "pothole": ["pothole", "hole", "crater", "dug up", "dug-up", "cave in", "cave-in",
                "bumpy", "damage", "damaged road", "broken road", "uneven road",
                "patch", "asphalt", "tar road", "road repair", "wheel", "two wheeler",
                "manhole", "open drain", "missing cover", "cover stolen", "footpath broken"],
    "garbage": ["garbage", "trash", "waste", "dump", "dumping", "litter", "rubbish",
                "smell", "stench", "odour", "odor", "bin", "dustbin", "debris",
                "stray", "overflow", "unclean", "dirty", "sweep", "sanitation", "plastic"],
    "streetlight": ["streetlight", "street light", "street-light", "lamp", "light",
                    "dark", "pole", "not working", "fused", "flicker", "flickering",
                    "bulb", "glow", "night", "electricity", "tube light", "high mast"],
    "water_leak": ["leak", "leakage", "water", "pipe", "pipeline", "burst", "sewage",
                   "drain", "drainage", "overflow", "flood", "flooding", "puddle",
                   "stagnant", "stagnation", "gushing", "sewer", "manhole", "contamination"],
    "road_crack": ["crack", "cracked", "fissure", "fracture", "split", "gap", "caving",
                   "sinking", "collapsed", "collapse", "widening", "split open",
                   "subsidence", "erosion", "wearing"],
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
             + 0.16 * _clamp((b - 0.32) / 0.40))

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
    text = f"{title or ''} {description or ''}".lower()
    toks = set(_tokens(text))
    raw, matched = {}, {}
    for cat, words in CATEGORY_KEYWORDS.items():
        s, hits = 0.0, []
        for kw in words:
            if kw in text:
                s += 2.0 if " " in kw else 1.0
                hits.append(kw)
        raw[cat] = min(1.0, s / 3.0)
        matched[cat] = hits
    urgency_mult, urgency_hits = 1.0, []
    for mult, words in URGENCY_KEYWORDS.items():
        for kw in words:
            if kw in text:
                urgency_mult = max(urgency_mult, mult)
                urgency_hits.append(kw)
    return {
        "scores": {k: round(v, 4) for k, v in raw.items()},
        "matched": matched,
        "urgency_mult": urgency_mult,
        "urgency_hits": urgency_hits,
        "tokens": sorted(toks),
    }


def classify_report(image_path=None, title="", description=""):
    """Fuse vision + text into a category decision with calibrated confidence."""
    vis = {}
    feats = {}
    txt = text_scores(title, description)
    if image_path and os.path.exists(image_path):
        feats = _inspect(image_path)
        vis = image_scores(feats)
    has_text = bool((title or "").strip() or (description or "").strip())

    cats = [c for c in CATEGORIES if c != "other"]
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
    conf = round(_clamp(conf / 100.0, 0.30, 0.985) * 100, 1)

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
    return {
        "category": best,
        "category_label": CATEGORY_META[best]["label"],
        "confidence": conf,
        "fused_scores": {k: round(v, 4) for k, v in fused.items()},
        "visual_scores": vis,
        "text_scores": txt["scores"],
        "text_keywords": txt["matched"].get(best, []),
        "urgency_hits": txt["urgency_hits"],
        "urgency_mult": txt["urgency_mult"],
        "features": feats,
        "explanation": explanation,
        "agreement": agreement,
    }


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
def analyze_new_report(image_path, title, description, lat, lng, existing_reports):
    cls = classify_report(image_path=image_path, title=title, description=description)
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
