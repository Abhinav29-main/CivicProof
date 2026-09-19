"""
CivicProof API server — Flask
--------------------------------
Citizen reporting, AI triage, duplicate merging and ProofWatch verification.
Run:  python app.py   (serves UI + API on :8000)
"""
import csv
import io
import json
import os
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory, abort

import ai_engine as ai
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
DIFF_DIR = os.path.join(UPLOAD_DIR, "diff")
os.makedirs(DIFF_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

db.init_db()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def save_upload(file_storage, sub=""):
    if not file_storage or not file_storage.filename:
        return None
    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        ext = ".jpg"
    name = f"{uuid.uuid4().hex[:14]}{ext}"
    folder = os.path.join(UPLOAD_DIR, sub) if sub else UPLOAD_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    file_storage.save(path)
    return path


def public_url(path):
    """Map any stored path to a /static URL — tolerant of stale absolute paths
    baked into a seeded DB from a different machine (e.g. /home/user/... or D:\\...)."""
    if not path:
        return ""
    p = str(path).replace("\\", "/")
    if p.startswith("/static/"):
        return p
    if "/uploads/" in p:
        return "/static/uploads/" + p.split("/uploads/", 1)[1]
    try:
        rel = os.path.relpath(path, STATIC_DIR)
    except (ValueError, OSError):
        return ""
    return "/static/" + rel.replace(os.sep, "/")


def resolve_stored(path):
    """Return an existing on-disk path for a stored DB path, even if the stored
    value is an absolute path from another machine (seeded data)."""
    if not path:
        return ""
    if os.path.exists(path):
        return path
    p = str(path).replace("\\", "/")
    if "/uploads/" in p:
        cand = os.path.join(UPLOAD_DIR, p.split("/uploads/", 1)[1])
        if os.path.exists(cand):
            return cand
    return ""


def decorate(rep, with_children=False):
    rep["image_url"] = public_url(rep.pop("image_path", "")) if rep.get("image_path") else ""
    rep["after_image_url"] = public_url(rep.pop("after_image_path", "")) if rep.get("after_image_path") else ""
    pw = rep.get("pw") or {}
    if isinstance(pw, dict) and pw.get("diff_image"):
        di = pw["diff_image"]
        pw["diff_image"] = di if di.startswith("/static/") else public_url(di)
        rep["pw"] = pw
    rep["meta"] = ai.CATEGORY_META.get(rep.get("category"), ai.CATEGORY_META["other"])
    rep["age_days"] = round(max(0.0, (time.time() - rep["created_at"]) / 86400.0), 1)
    if with_children:
        kids = db.list_reports("WHERE parent_id=?", (rep["id"],))
        rep["duplicates"] = [decorate(k) for k in kids]
    return rep


def photo_candidates(exclude_id=None):
    rows = db.list_reports("WHERE status != 'duplicate'")
    return rows


def refresh_priorities():
    """Recompute priority with aging for every live report (cheap, keeps demo fresh)."""
    for r in db.list_reports("WHERE status NOT IN ('resolved','duplicate')"):
        aij = r.get("ai") or {}
        cls_mult = (aij.get("classification") or {}).get("urgency_mult", 1.0)
        cls_hits = (aij.get("classification") or {}).get("urgency_hits", [])
        kids = db.list_reports("WHERE parent_id=?", (r["id"],))
        pri = ai.compute_priority(r["category"], upvotes=r["upvotes"],
                                  merged_duplicates=len(kids),
                                  urgency_mult=cls_mult, urgency_hits=cls_hits,
                                  created_ts=r["created_at"])
        if abs(pri["score"] - (r["priority_score"] or 0)) >= 0.4 or pri["label"] != r["priority_label"]:
            db.update_report(r["id"], {"priority_score": pri["score"], "priority_label": pri["label"]})


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/uploads/<path:fname>")
def uploads(fname):
    return send_from_directory(UPLOAD_DIR, fname)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@app.route("/api/reports", methods=["GET"])
def api_list():
    refresh_priorities()
    status = request.args.get("status")
    category = request.args.get("category")
    q = (request.args.get("q") or "").strip().lower()
    sort = request.args.get("sort", "priority")
    clauses, params = ["1=1"], []
    if status:
        clauses.append("status=?"); params.append(status)
    if category:
        clauses.append("category=?"); params.append(category)
    where = "WHERE " + " AND ".join(clauses)
    order = {"priority": "priority_score DESC", "recent": "created_at DESC",
             "upvotes": "upvotes DESC"}.get(sort, "priority_score DESC")
    reps = [decorate(r) for r in db.list_reports(where, tuple(params), order)]
    if q:
        reps = [r for r in reps if q in (r["title"] + " " + r["description"] + " " +
                                         r["address"] + " " + r["ticket"]).lower()]
    return jsonify({"ok": True, "reports": reps})


@app.route("/api/reports", methods=["POST"])
def api_create():
    photo = request.files.get("photo")
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    reporter = (request.form.get("reporter") or "").strip() or "Anonymous Citizen"
    address = (request.form.get("address") or "").strip()
    try:
        lat = float(request.form.get("lat")) if request.form.get("lat") else None
        lng = float(request.form.get("lng")) if request.form.get("lng") else None
    except ValueError:
        return err("Invalid coordinates")
    if not title:
        return err("A short title is required")
    if not photo:
        return err("A photo of the issue is required")

    img_path = save_upload(photo)
    if not img_path:
        return err("Could not store the photo")

    gate = ai.gate_photo(img_path, title, description)
    if not gate["accepted"]:
        try:
            os.remove(img_path)
        except OSError:
            pass
        return jsonify({
            "ok": False,
            "error": "image_rejected",
            "reason_code": gate["reason_code"],
            "message": gate["message"] + " Please start again with a real photo of the issue.",
            "restart": True,          # the wizard must reset to step 1 — nothing was saved
            "restart_step": 1,
            "source": gate.get("source"),
            "relevance": gate["relevance"],
            "authenticity": gate["authenticity"],
        }), 422

    cands = photo_candidates()
    analysis = ai.analyze_new_report(img_path, title, description, lat, lng, cands, vision_result=gate.get("vision"))
    cls, dup, pri = analysis["classification"], analysis["duplicate"], analysis["priority"]

    is_dup = bool(dup.get("match")) and not dup.get("possible") and dup["match"].get("status") != "resolved"
    fields = {
        "title": title, "description": description, "reporter": reporter,
        "address": address, "lat": lat, "lng": lng,
        "category": cls["category"], "category_label": cls["category_label"],
        "ai_confidence": cls["confidence"],
        "priority_score": pri["score"], "priority_label": pri["label"],
        "phash_a": analysis["hashes"]["a"], "phash_d": analysis["hashes"]["d"],
        "ai_json": json.dumps({"classification": cls, "duplicate": dup, "priority": pri}),
        "image_path": img_path,
        "status": "duplicate" if is_dup else "open",
        "parent_id": dup["match"]["id"] if is_dup else None,
    }
    rid, ticket = db.insert_report(fields)

    if is_dup:
        parent = dup["match"]
        db.log_event(parent["id"], "duplicate_merged",
                     f"{ticket} merged as duplicate — {', '.join(dup['signals'][:2])}",
                     ticket=parent["ticket"])
        # merging corroborates the parent: bump upvotes + priority
        kids = db.list_reports("WHERE parent_id=?", (parent["id"],))
        new_pri = ai.compute_priority(parent["category"],
                                      upvotes=parent["upvotes"] + 1,
                                      merged_duplicates=len(kids),
                                      urgency_mult=cls["urgency_mult"],
                                      urgency_hits=cls["urgency_hits"],
                                      created_ts=parent["created_at"])
        db.update_report(parent["id"], {"upvotes": parent["upvotes"] + 1,
                                        "priority_score": new_pri["score"],
                                        "priority_label": new_pri["label"]})
        status_note = "duplicate_linked"
    else:
        status_note = "created"

    db.log_event(rid, "created",
                 f"AI classified as {cls['category_label']} at {cls['confidence']}% "
                 f"confidence; priority {pri['label']} ({pri['score']})",
                 actor=reporter, ticket=ticket)
    rep = decorate(db.get_report(rid), with_children=True)
    return jsonify({"ok": True, "report": rep, "ai": {"classification": cls,
                    "duplicate": dup, "priority": pri}, "outcome": status_note})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Stateless pre-submission analysis — powers the report wizard's AI preview."""
    photo = request.files.get("photo")
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    try:
        lat = float(request.form.get("lat")) if request.form.get("lat") else None
        lng = float(request.form.get("lng")) if request.form.get("lng") else None
    except ValueError:
        lat = lng = None
    img_path = None
    if photo and photo.filename:
        img_path = save_upload(photo, sub="staging")

    # No photo -> nothing to triage. Previously this fell through to `gate.get(...)` below and
    # raised NameError; the preview must never run without a gated image.
    if not img_path:
        return err("A photo of the issue is required before the AI can analyse it")

    gate = ai.gate_photo(img_path, title, description)
    if not gate["accepted"]:
        try:
            os.remove(img_path)
        except OSError:
            pass
        return jsonify({
            "ok": False,
            "error": "image_rejected",
            "reason_code": gate["reason_code"],
            "message": gate["message"] + " Please start again with a real photo of the issue.",
            "restart": True,          # the wizard must reset to step 1 — the upload was discarded
            "restart_step": 1,
            "source": gate.get("source"),
            "relevance": gate["relevance"],
            "authenticity": gate["authenticity"],
        }), 422

    analysis = ai.analyze_new_report(img_path, title, description, lat, lng, photo_candidates(), vision_result=gate.get("vision"))
    out = {"classification": analysis["classification"],
           "priority": analysis["priority"],
           "duplicate": {k: v for k, v in analysis["duplicate"].items() if k != "match"}}
    m = analysis["duplicate"].get("match")
    if m:
        out["duplicate"]["match"] = {"id": m["id"], "ticket": m["ticket"],
                                     "title": m["title"], "status": m["status"],
                                     "category": m["category"]}
    return jsonify({"ok": True, "ai": out, "staged_photo": public_url(img_path) if img_path else ""})


@app.route("/api/reports/<int:rid>", methods=["GET"])
def api_get(rid):
    rep = db.get_report(rid)
    if not rep:
        return err("Not found", 404)
    refresh_priorities()
    rep = db.get_report(rid)
    out = decorate(rep, with_children=True)
    if rep.get("parent_id"):
        parent = db.get_report(rep["parent_id"])
        out["parent"] = decorate(parent) if parent else None
    out["activity"] = db.get_activity(60, report_id=rid)
    return jsonify({"ok": True, "report": out})


@app.route("/api/reports/<int:rid>/upvote", methods=["POST"])
def api_upvote(rid):
    rep = db.get_report(rid)
    if not rep:
        return err("Not found", 404)
    target = rep
    if rep.get("parent_id"):
        target = db.get_report(rep["parent_id"]) or rep
    kids = db.list_reports("WHERE parent_id=?", (target["id"],))
    mult = (target.get("ai") or {}).get("classification", {}).get("urgency_mult", 1.0)
    hits = (target.get("ai") or {}).get("classification", {}).get("urgency_hits", [])
    pri = ai.compute_priority(target["category"], upvotes=target["upvotes"] + 1,
                              merged_duplicates=len(kids), urgency_mult=mult,
                              urgency_hits=hits, created_ts=target["created_at"])
    db.update_report(target["id"], {"upvotes": target["upvotes"] + 1,
                                    "priority_score": pri["score"],
                                    "priority_label": pri["label"]})
    db.log_event(target["id"], "upvoted",
                 f"+1 citizen confirmation (total {target['upvotes'] + 1}); priority → {pri['label']} ({pri['score']})",
                 actor="citizen", ticket=target["ticket"])
    return jsonify({"ok": True, "upvotes": target["upvotes"] + 1, "priority": pri})


@app.route("/api/reports/<int:rid>/status", methods=["POST"])
def api_status(rid):
    rep = db.get_report(rid)
    if not rep:
        return err("Not found", 404)
    data = request.get_json(force=True) or {}
    status = data.get("status")
    if status not in ("open", "in_progress", "resolved", "reopened"):
        return err("Invalid status")
    updates = {"status": status, "flagged": 0}
    if data.get("assigned_to"):
        updates["assigned_to"] = data["assigned_to"].strip()
    db.update_report(rid, updates)
    db.log_event(rid, "status_changed",
                 f"{rep['status']} → {status}" + (f"; assigned to {updates.get('assigned_to')}" if updates.get("assigned_to") else ""),
                 actor="authority", ticket=rep["ticket"])
    return jsonify({"ok": True, "report": decorate(db.get_report(rid), with_children=True)})


@app.route("/api/reports/<int:rid>/reopen", methods=["POST"])
def api_reopen(rid):
    rep = db.get_report(rid)
    if not rep:
        return err("Not found", 404)
    db.update_report(rid, {"status": "reopened", "resolved_at": None})
    db.log_event(rid, "reopened", "Reopened — issue reported as still present.",
                 actor="citizen", ticket=rep["ticket"])
    return jsonify({"ok": True, "report": decorate(db.get_report(rid), with_children=True)})


@app.route("/api/reports/<int:rid>/resolve", methods=["POST"])
def api_resolve(rid):
    rep = db.get_report(rid)
    if not rep:
        return err("Not found", 404)
    after = request.files.get("after_photo")
    notes_in = (request.form.get("notes") or "").strip()
    before_path = resolve_stored(rep.get("image_path"))
    if not before_path:
        return err("Original photo missing")
    if not after:
        return err("An 'after' photo is required for ProofWatch verification")
    after_path = save_upload(after)
    diff_path = os.path.join(DIFF_DIR, f"diff-{rep['ticket']}-{uuid.uuid4().hex[:6]}.png")

    pw = ai.proofwatch_verify(before_path, after_path, rep["category"], diff_out=diff_path)

    updates = {"after_image_path": after_path, "pw_json": json.dumps(pw)}
    if pw["verdict"] == "verified":
        updates.update({"status": "resolved", "flagged": 0, "resolved_at": time.time()})
        db.log_event(rid, "proofwatch_verified",
                     f"ProofWatch VERIFIED the repair — {pw['reason']}",
                     actor="proofwatch", ticket=rep["ticket"])
    elif pw["verdict"] == "reopened":
        updates.update({"status": "reopened", "flagged": 0, "resolved_at": None})
        db.log_event(rid, "proofwatch_reopened",
                     f"ProofWatch REJECTED the completion photo — ticket auto-reopened. {pw['reason']}",
                     actor="proofwatch", ticket=rep["ticket"])
    else:
        updates.update({"flagged": 1, "resolved_at": None})
        db.log_event(rid, "proofwatch_flagged",
                     f"ProofWatch FLAGGED the completion photo for field re-inspection. {pw['reason']}",
                     actor="proofwatch", ticket=rep["ticket"])
    db.update_report(rid, updates)
    out = decorate(db.get_report(rid), with_children=True)
    return jsonify({"ok": True, "report": out, "proofwatch": pw})


@app.route("/api/reports.csv")
def api_csv():
    refresh_priorities()
    reps = db.list_reports()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ticket", "title", "category", "status", "priority_score", "priority_label",
                "ai_confidence", "upvotes", "lat", "lng", "address", "reporter",
                "created_at", "resolved_at", "proofwatch_verdict"])
    for r in reps:
        w.writerow([r["ticket"], r["title"], r["category"], r["status"], r["priority_score"],
                    r["priority_label"], r["ai_confidence"], r["upvotes"], r["lat"], r["lng"],
                    r["address"], r["reporter"],
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"])),
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(r["resolved_at"])) if r["resolved_at"] else "",
                    (r.get("pw") or {}).get("verdict", "")])
    return (buf.getvalue(), 200, {"Content-Type": "text/csv",
                                  "Content-Disposition": "attachment; filename=civicproof_reports.csv"})


# ---------------------------------------------------------------------------
# stats & activity
# ---------------------------------------------------------------------------
@app.route("/api/stats")
def api_stats():
    refresh_priorities()
    reps = db.list_reports()
    live = [r for r in reps if r["status"] != "duplicate"]
    by_status, by_category, by_priority = {}, {}, {}
    for r in live:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        by_priority[r["priority_label"]] = by_priority.get(r["priority_label"], 0) + 1

    pw_attempts = [r for r in live if (r.get("pw") or {}).get("verdict")]
    pw_verified = sum(1 for r in pw_attempts if r["pw"]["verdict"] == "verified")
    pw_reopened = sum(1 for r in pw_attempts if r["pw"]["verdict"] == "reopened")
    pw_flagged = sum(1 for r in pw_attempts if r["pw"]["verdict"] == "flagged")

    resolved = [r for r in live if r["status"] == "resolved" and r.get("resolved_at")]
    avg_res_h = (sum((r["resolved_at"] - r["created_at"]) for r in resolved) / len(resolved) / 3600.0
                 if resolved else None)

    # 7-day intake series
    now = time.time()
    series = []
    for i in range(6, -1, -1):
        day0 = now - (i + 1) * 86400
        day1 = now - i * 86400
        series.append({
            "label": time.strftime("%a", time.localtime(day1)),
            "count": sum(1 for r in reps if day0 <= r["created_at"] < day1),
        })

    hotspots = sorted(
        [{"lat": r["lat"], "lng": r["lng"], "category": r["category"],
          "status": r["status"], "priority": r["priority_label"],
          "priority_score": r["priority_score"], "id": r["id"], "ticket": r["ticket"],
          "title": r["title"]}
         for r in live if r.get("lat") is not None],
        key=lambda x: -x["priority_score"])

    return jsonify({
        "ok": True,
        "totals": {
            "reports": len(live),
            "open": by_status.get("open", 0),
            "in_progress": by_status.get("in_progress", 0),
            "reopened": by_status.get("reopened", 0),
            "resolved": by_status.get("resolved", 0),
            "duplicates": sum(1 for r in reps if r["status"] == "duplicate"),
            "flagged": sum(1 for r in live if r.get("flagged")),
            "upvotes": sum(r["upvotes"] for r in reps),
            "avg_resolution_hours": round(avg_res_h, 1) if avg_res_h else None,
        },
        "by_status": by_status,
        "by_category": by_category,
        "by_priority": by_priority,
        "proofwatch": {"attempts": len(pw_attempts), "verified": pw_verified,
                       "reopened": pw_reopened, "flagged": pw_flagged,
                       "accuracy": round(pw_verified / len(pw_attempts) * 100, 1) if pw_attempts else None},
        "series7": series,
        "hotspots": hotspots,
        "activity": db.get_activity(14),
    })


@app.route("/api/activity")
def api_activity():
    return jsonify({"ok": True, "activity": db.get_activity(50)})


@app.errorhandler(413)
def too_large(e):
    return err("File too large (20 MB max)", 413)


if __name__ == "__main__":
    # Fresh clone with an empty database? Build the demo dataset automatically
    # (runs the real API + AI pipeline against the bundled seed photos).
    if not db.list_reports():
        try:
            import seed
            print("Empty database detected — seeding demo data (one-time)…")
            seed.main()
        except Exception as exc:  # demo still runs with empty data
            print(f"[warn] auto-seed skipped: {exc}")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)