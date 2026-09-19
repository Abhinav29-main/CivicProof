# CivicProof 🛡️ — Report it. Prove it. Resolve it.

**AI-powered civic issue reporting with machine-verified repairs.** Citizens snap a photo; the AI
classifies the issue, scores its urgency and silently merges duplicates. When a crew claims the work
is done, **ProofWatch** compares the citizen's original photo with the crew's completion photo and
automatically **reopens tickets for fake or unfinished repairs**.

Built as a fully self-contained hackathon demo for **Chennai** — no API keys, no model downloads,
works completely offline.

---

## ✨ Highlights

| Capability | How it works (all real, all local) |
|---|---|
| **AI issue classification** | Visual feature space (edge density, dark-patch geometry, colour chaos, hue entropy, cool-tone wetness, region luminance) **fused** with keyword NLP over the title/description → category + calibrated confidence. Explains itself on every ticket. |
| **Priority engine** | `severity × urgency-language × corroboration × aging` with per-component **reasons** shown in the UI ("+11.5 — 23 citizen confirmations"). |
| **Duplicate detection & merging** | Perceptual hashes (aHash/dHash hamming distance) + haversine proximity + category agreement + token-Jaccard text overlap → auto-merge score. Duplicates link to the parent ticket and *boost its priority*. |
| **ProofWatch verification** | Identity check (copied-photo fraud), scene match (block-SSIM + colour histogram correlation), then a **category-aware issue-signature** (e.g. roughness-inside-dark-patch for potholes, cool-tone ratio for water) measured before→after. Signature collapse ⇒ *verified*; signature persists ⇒ *reopened*; wrong scene ⇒ *flagged for field check*. Renders a **change heatmap** for every verdict. |
| **Authority console** | Live KPI row, triage queue sorted by AI priority, ProofWatch review feed with before→after thumbs, city map, analytics (donut/bars/7-day intake), full event log, CSV export. |
| **Citizen UX** | 4-step report wizard with a simulated AI-scan sequence, paste/drag-drop photo upload, click-to-pin procedural city map, geolocation, before/after comparison slider, upvotes ("confirm"), ticket timelines. |

## 🗂️ Project layout

```
civicproof/
├── app.py                # Flask API + static hosting (port 8000)
├── ai_engine.py          # classification, priority, dedupe, ProofWatch (PIL + numpy only)
├── db.py                 # SQLite schema & helpers
├── seed.py               # demo data — drives the REAL API end-to-end
├── civicproof.db         # SQLite (created on first run / reseeding)
├── requirements.txt
└── static/
    ├── index.html        # SPA shell + inline SVG icon set (zero CDN deps)
    ├── styles.css        # full dark design system
    ├── js/app.js         # core: API, router, procedural city map, charts, toasts
    ├── js/pages.js       # citizen pages: home, explore, report wizard, issue detail
    ├── js/admin.js       # authority console
    └── uploads/          # citizen photos + seed images + ProofWatch diff heatmaps
```

## 🚀 Run it

```bash
pip install -r requirements.txt   # flask, pillow, numpy
python seed.py                    # build the demo database (stop the server first)
python app.py                     # → http://localhost:8000
```

> Note: `seed.py` wipes and rebuilds the demo data. Stop the app server before reseeding.

## 🎬 5-minute demo script (for judges)

1. **Home** — live pulsing city map, real stats, pipeline explainer.
2. **Report an Issue** — upload *any* photo of a civic problem. Watch the AI scan animation →
   category + confidence arc + priority meter + duplicate sweep. Pin the location on the map, submit,
   get a ticket (`CP-26-XXXX`).
3. **Duplicate magic** — submit the *same* photo again (even with different words/location nearby):
   it's auto-merged into the parent ticket and visibly **boosts its priority**.
4. **Explore** — filter/sort, upvote an issue (priority recomputes live), open the **Velachery water
   leak** ticket to see a merged duplicate (CP-26-0004 ← 0005).
5. **ProofWatch, verified** — open **CP-26-0001 (Guindy pothole)**: drag the before/after slider,
   read the verdict, SSIM/colour/signature metrics, and the change heatmap.
6. **ProofWatch, fraud caught** — open **CP-26-0006 (Triplicane garbage)**: the crew uploaded the
   same photo as "proof" → ProofWatch detected identity and **auto-reopened** the ticket.
7. **Do it live** — in **Authority Console**, open the Adyar streetlight (CP-26-0003), click
   *Complete work*, upload `static/uploads/seed/streetlight_after.jpg` → watch it verify (or upload a
   mismatched photo and watch it get **flagged for field check**).
8. **Analytics & export** — category donut, priority mix, 7-day intake, ProofWatch outcome stats,
   one-click CSV export.

## 🔌 API (all JSON unless noted)

```
POST /api/reports                    multipart: photo, title, description?, lat, lng, address?, reporter?
GET  /api/reports?status&category&q&sort(priority|recent|upvotes)
GET  /api/reports/<id>               full ticket incl. AI, ProofWatch, duplicates, activity
POST /api/reports/<id>/upvote        citizen confirmation → recomputes priority
POST /api/reports/<id>/status        {status, assigned_to?}
POST /api/reports/<id>/resolve       multipart: after_photo → ProofWatch verdict
POST /api/reports/<id>/reopen        citizen-driven reopen
POST /api/analyze                    stateless AI preview for the report wizard
GET  /api/stats                      console KPIs, charts, hotspots, activity
GET  /api/activity                   event feed
GET  /api/reports.csv                export
```

## 🧠 Design notes

- **No black boxes**: every AI decision ships with its intermediate signals (fused category scores,
  matched keywords, dedupe signals, SSIM, signature strength) rendered in the UI.
- **Honest AI**: low-confidence classifications route to `Other / manual triage` instead of
  hallucinating a category.
- **ProofWatch signatures** are category-aware: what "fixed" means differs for a pothole
  (surface roughness), a streetlight (scene darkness vs. lamp glow), flooding (cool-tone ratio)
  or garbage (texture chaos).

## ⚖️ License

MIT — hack it, fork it, ship it.
