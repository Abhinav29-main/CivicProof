"""
CivicProof demo seeder.
Drives the REAL API (Flask test client) so the seeded database is exactly what
live users would produce — AI classification, duplicate merging, ProofWatch
verdicts and diff-heatmaps are all genuinely computed against the seed photos.
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import app as app_module   # noqa: E402
import db                  # noqa: E402
import ai_engine as ai     # noqa: E402

SEED = os.path.join(BASE, "static", "uploads", "seed")
DAY = 86400.0
NOW = time.time()

client = app_module.app.test_client()


def wipe():
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init_db()
    # clear generated uploads (keep the seed assets themselves)
    up = os.path.join(BASE, "static", "uploads")
    for name in os.listdir(up):
        p = os.path.join(up, name)
        if os.path.isfile(p):
            os.remove(p)
    for sub in ("diff", "staging"):
        dd = os.path.join(up, sub)
        os.makedirs(dd, exist_ok=True)
        for name in os.listdir(dd):
            os.remove(os.path.join(dd, name))


def report(img, title, desc, reporter, lat, lng, address, days_ago, upvotes):
    with open(os.path.join(SEED, img), "rb") as f:
        resp = client.post("/api/reports", data={
            "photo": (f, img), "title": title, "description": desc,
            "reporter": reporter, "lat": str(lat), "lng": str(lng),
            "address": address,
        }, content_type="multipart/form-data")
    data = resp.get_json()
    if not data.get("ok"):
        raise RuntimeError(f"report failed: {data}")
    rid = data["report"]["id"]
    db.update_report(rid, {"created_at": NOW - days_ago * DAY, "upvotes": upvotes})
    return rid, data


def resolve(rid, after_img, when_days_ago):
    with open(os.path.join(SEED, after_img), "rb") as f:
        resp = client.post(f"/api/reports/{rid}/resolve",
                           data={"after_photo": (f, after_img)},
                           content_type="multipart/form-data")
    data = resp.get_json()
    rep = db.get_report(rid)
    if rep.get("resolved_at"):
        db.update_report(rid, {"resolved_at": NOW - when_days_ago * DAY})
    return data


def set_status(rid, status, assigned=""):
    client.post(f"/api/reports/{rid}/status",
                json={"status": status, "assigned_to": assigned})


def backdate_activity(rid, created, resolved=None):
    with db.connect() as conn:
        conn.execute("UPDATE activity SET created_at=? WHERE report_id=? AND event='created'",
                     (created, rid))
        conn.execute("UPDATE activity SET created_at=? WHERE report_id=? AND event='duplicate_merged'",
                     (created + 0.2 * DAY, rid))
        if resolved:
            conn.execute(
                "UPDATE activity SET created_at=? WHERE report_id=? AND event LIKE 'proofwatch%'",
                (resolved, rid))


def recalc_ai_fields():
    """Recompute stored priority (with reasons) after upvote/backdate adjustments."""
    for r in db.list_reports():
        kids = [k for k in db.list_reports("WHERE parent_id=?", (r["id"],))]
        aij = r.get("ai") or {}
        cls = aij.get("classification") or {}
        pri = ai.compute_priority(
            r["category"], upvotes=r["upvotes"], merged_duplicates=len(kids),
            urgency_mult=cls.get("urgency_mult", 1.0), urgency_hits=cls.get("urgency_hits", []),
            created_ts=r["created_at"])
        aij["priority"] = pri
        db.update_report(r["id"], {"priority_score": pri["score"],
                                   "priority_label": pri["label"],
                                   "ai_json": json.dumps(aij)})


def main():
    wipe()
    print("Seeding CivicProof demo data (real AI pipeline) ...\n")

    created = {}

    rid, d = report("pothole_before.jpg",
                    "Deep pothole near Guindy Metro gate — two-wheelers skidding",
                    "A crater-sized pothole opened up right at the metro gate on Anna Salai. It fills with water and is invisible at night. Two riders fell this week.",
                    "Priya S.", 13.0067, 80.2206, "Anna Salai, near Guindy Metro, Chennai", 6, 23)
    set_status(rid, "in_progress", "CC Roads Division — Team B")
    res = resolve(rid, "pothole_after.jpg", 2)
    print(f"[{d['report']['ticket']}] pothole / Guindy        ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)  proofwatch={res['proofwatch']['verdict']}")
    created[rid] = ("pothole", NOW - 6 * DAY, NOW - 2 * DAY)

    rid2, d = report("garbage_before.jpg",
                     "Garbage pile overflowing for a week — unbearable stench",
                     "Conservancy bins near Pondy Bazaar entrance have been overflowing for days. Stray dogs scatter waste across the pavement every morning. Mosquito menace rising.",
                     "Arun K.", 13.0418, 80.2341, "Pondy Bazaar, T. Nagar, Chennai", 8, 11)
    set_status(rid2, "in_progress", "Urbaser Sumeet — Zone 9 Crew")
    res = resolve(rid2, "garbage_after.jpg", 3)
    print(f"[{d['report']['ticket']}] garbage / T. Nagar       ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)  proofwatch={res['proofwatch']['verdict']}")
    created[rid2] = ("garbage", NOW - 8 * DAY, NOW - 3 * DAY)

    rid3, d = report("streetlight_before.jpg",
                     "Street light pole broken, lamp fused — road pitch dark",
                     "The lamp on LB Road near the bus stop is hanging by its wires and has not worked in weeks. The whole stretch is dark and unsafe for women returning late.",
                     "Meera V.", 13.0012, 80.2565, "LB Road, Adyar, Chennai", 4, 7)
    set_status(rid3, "in_progress", "TANGEDCO Ward 174 — EB Crew 4")
    print(f"[{d['report']['ticket']}] streetlight / Adyar      ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)  status=in_progress")
    created[rid3] = ("streetlight", NOW - 4 * DAY, None)

    rid4, d = report("water_before.jpg",
                     "Water pipe burst — gushing and flooding near bus stop",
                     "Main line burst on 100 Feet Road. Water is gushing like a fountain, flooding the bus stop and the school gate. Children are slipping. Drinking water being wasted for 2 days.",
                     "Dinesh R.", 12.9750, 80.2206, "100 Feet Road, Velachery, Chennai", 3, 19)
    print(f"[{d['report']['ticket']}] water leak / Velachery   ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)  priority={d['ai']['priority']['label']} "
          f"({d['ai']['priority']['score']})")
    created[rid4] = ("water_leak", NOW - 3 * DAY, None)

    rid5, d = report("water_before.jpg",
                     "Water gushing out near Velachery bus stop",
                     "Same pipeline burst — water everywhere, traffic crawling. This is right at the bus stand.",
                     "Kavitha M.", 12.9752, 80.2209, "Velachery Bus Stand, Chennai", 2.6, 0)
    print(f"[{d['report']['ticket']}] duplicate of #{rid4}        outcome={d['outcome']} "
          f"(dup score={d['ai']['duplicate']['score']})")
    created[rid5] = ("duplicate", NOW - 2.6 * DAY, None)

    # R6 — contractor uploaded the SAME photo as completion proof → ProofWatch must reject
    rid6, d = report("garbage_before.jpg",
                     "Uncleared garbage mound beside Triplicane market",
                     "Huge mound of rotting waste right where shoppers walk. Three days, no pickup. Dengue risk.",
                     "Fathima B.", 13.0499, 80.2824, "Triplicane High Rd, near Marina, Chennai", 5, 9)
    set_status(rid6, "in_progress", "Zone 5 — Night SWM Crew")
    res = resolve(rid6, "garbage_before.jpg", 1)   # fraudulent closure with identical photo
    print(f"[{d['report']['ticket']}] garbage / Triplicane     proofwatch={res['proofwatch']['verdict']} "
          f"(identical-photo fraud rejected)  status={db.get_report(rid6)['status']}")
    created[rid6] = ("garbage", NOW - 5 * DAY, None)

    rid7, d = report("water_after.jpg",
                     "Long crack widening along Chaitanya Road",
                     "A long surface crack has appeared along the roadside stretch after the rains. It widens every week — worried it collapses during the monsoon.",
                     "Suresh P.", 13.0604, 80.2493, "Chaitanya Road, Nungambakkam, Chennai", 1.2, 3)
    print(f"[{d['report']['ticket']}] road crack / Nungambakkam ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)")
    created[rid7] = ("road_crack", NOW - 1.2 * DAY, None)

    rid8, d = report("tree_before.jpg",
                     "Fallen tree branch blocking footpath after storm",
                     "Big branch came down last night on 3rd Avenue and the footpath is fully blocked. Elderly people are forced onto the road.",
                     "Lakshmi N.", 13.0095, 80.2662, "3rd Avenue, Besant Nagar, Chennai", 0.5, 5)
    print(f"[{d['report']['ticket']}] fallen tree / Besant Nagar ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)")
    created[rid8] = ("other", NOW - 0.5 * DAY, None)

    rid9, d = report("manhole_before.jpg",
                     "Open manhole with missing cover — children playing nearby",
                     "Manhole cover stolen/broken near the corporation school gate. Completely open pit. Extremely dangerous for children and cyclists. Needs urgent barricading and replacement.",
                     "Joseph A.", 13.0330, 80.2460, "Corporation School Rd, Teynampet, Chennai", 1.8, 14)
    print(f"[{d['report']['ticket']}] manhole / Teynampet      ai={d['ai']['classification']['category']} "
          f"({d['ai']['classification']['confidence']}%)  priority={d['ai']['priority']['label']}")
    created[rid9] = ("pothole", NOW - 1.8 * DAY, None)

    for rid, (_, c, r) in created.items():
        backdate_activity(rid, c, r)

    recalc_ai_fields()
    print("\nSeed complete.\n")


if __name__ == "__main__":
    main()
