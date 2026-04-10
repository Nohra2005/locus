"""
Golden Dataset Visualizer — Locus
==================================

Generates a self-contained HTML report from the last evaluation run.

For each query it shows:
  • The full query image (what Gemini sees as the "query")
  • The CLIP crop (bbox applied — what CLIP used to find the search results)
  • The 5 ground-truth relevant products (annotated in the golden dataset)
  • The 5 actual search results in rank order, each labelled with its Gemini
    VSS score and highlighted green if it matches a ground-truth product_id

Usage:
    python mlops/visualize_golden_dataset.py

Output:
    mlops/golden_dataset_report.html   (open in any browser)

Requires:
    pip install Pillow
"""

import base64
import io
import json
import pathlib
import sys
import urllib.request

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[warn] Pillow not installed — CLIP crop preview will be skipped.")
    print("       pip install Pillow")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = pathlib.Path(__file__).parent
GOLDEN   = HERE / "golden_dataset.json"
RESULTS  = HERE / "gemini_judge_results.json"
OUT_HTML = HERE / "golden_dataset_report.html"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vss_color(score):
    if score is None: return "vss-none"
    if score >= 0.80: return "vss-great"
    if score >= 0.65: return "vss-good"
    if score >= 0.50: return "vss-fair"
    return "vss-poor"


def vss_badge(score, size="normal"):
    cls = vss_color(score)
    txt = f"{score:.2f}" if score is not None else "–"
    sz  = "font-size:13px;padding:3px 8px;" if size == "large" else ""
    return f'<span class="badge {cls}" style="{sz}">{txt}</span>'


def make_img_tag(src, alt="", w=130, h=160, extra_style=""):
    """Render an <img> with graceful error fallback.
    referrerpolicy=no-referrer bypasses CDN hotlink protection (e.g. Zalando img01.ztat.net).
    The fallback SVG is base64-encoded so the onerror attribute contains only a clean
    data URI — no raw angle brackets or percent-encoded chars that confuse JS linters.
    """
    if not src:
        return f'<div class="img-placeholder" style="width:{w}px;height:{h}px;">no image</div>'
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>"
        f"<rect width='{w}' height='{h}' fill='#ddd'/>"
        f"<text x='50%' y='50%' text-anchor='middle' dy='.35em' "
        f"font-size='11' fill='#999'>no img</text></svg>"
    )
    fallback = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return (
        f'<img src="{src}" alt="{alt}" referrerpolicy="no-referrer" '
        f'style="width:{w}px;height:{h}px;object-fit:cover;border-radius:6px;{extra_style}" '
        f'onerror="this.onerror=null;this.src=\'{fallback}\'">'
    )


def get_image_bytes(url):
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Locus-Visualizer/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def make_bbox_crop_src(query_img_url, bbox):
    """
    Crop the query image to the YOLO bounding box and return a data: URI.
    This is the exact image region CLIP embedded to perform the search.
    Returns None if PIL is unavailable or bbox is missing.
    """
    if not PIL_OK or not bbox or not query_img_url:
        return None
    try:
        img_bytes = get_image_bytes(query_img_url)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x1, y1, x2, y2 = bbox
        # Clamp to image dimensions
        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"  [warn] Could not make bbox crop: {e}")
        return None


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f0f2f5; color: #222; min-width: 900px; }

.page-header { background: #1a1a2e; color: #eee; padding: 20px 32px; }
.page-header h1 { font-size: 20px; font-weight: 700; }
.page-header p  { font-size: 12px; color: #aaa; margin-top: 6px; line-height: 1.8; }

/* ── Summary table ── */
.summary-wrap { background: #fff; margin: 20px 28px 8px; border-radius: 10px;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow-x: auto; }
.summary-wrap h2 { padding: 14px 18px; font-size: 14px; border-bottom: 1px solid #eee;
                   background: #fafafa; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: #f4f4f4; padding: 7px 10px; text-align: left;
     font-weight: 600; border-bottom: 2px solid #e0e0e0; white-space: nowrap; }
td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: middle;
     white-space: nowrap; }
tr:hover td { background: #fafbff; }
.anchor-link { color: #3366cc; text-decoration: none; }
.anchor-link:hover { text-decoration: underline; }
.hit-yes { color: #155724; font-weight: 700; }
.hit-no  { color: #721c24; }

/* ── Query cards ── */
.queries { padding: 8px 28px 40px; }
.qcard { background: #fff; border-radius: 10px; margin-bottom: 28px;
         box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }

.qcard-header { display: flex; align-items: flex-start; flex-wrap: wrap;
                gap: 8px; padding: 12px 16px; border-bottom: 1px solid #eee;
                background: #fafafa; }
.qcard-header h3 { font-size: 15px; font-weight: 700; flex: 1 1 200px; }
.qcard-header .cat-tag { font-size: 11px; color: #777;
                          background: #eee; padding: 2px 7px; border-radius: 10px;
                          align-self: center; }
.header-metrics { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.qcard-body { display: grid;
              grid-template-columns: 150px 150px 1fr;
              min-height: 200px; }

/* Column: full query image */
.col-query { padding: 14px 10px; border-right: 1px solid #eee;
             display: flex; flex-direction: column; align-items: center; gap: 6px; }
/* Column: CLIP crop */
.col-crop  { padding: 14px 10px; border-right: 1px solid #eee;
             display: flex; flex-direction: column; align-items: center; gap: 6px; }
.col-label { font-size: 10px; text-transform: uppercase; letter-spacing: .05em;
             color: #888; text-align: center; }

/* Column: panels */
.col-panels { padding: 14px 16px; display: flex; flex-direction: column; gap: 16px;
              min-width: 0; overflow-x: auto; }
.panel-title { font-size: 10px; text-transform: uppercase; letter-spacing: .05em;
               color: #888; margin-bottom: 8px; border-bottom: 1px solid #eee;
               padding-bottom: 4px; }

/* Result rows */
.result-row { display: flex; gap: 10px; flex-wrap: wrap; }
.ritem { display: flex; flex-direction: column; align-items: center; gap: 4px;
         position: relative; flex-shrink: 0; }
.ritem .rank-badge { position: absolute; top: 3px; left: 3px; background: rgba(0,0,0,.6);
                     color: #fff; font-size: 9px; font-weight: 700;
                     padding: 1px 5px; border-radius: 4px; z-index: 1; }
.ritem .name-label { font-size: 10px; color: #555; text-align: center;
                     max-width: 120px; overflow: hidden;
                     text-overflow: ellipsis; white-space: nowrap; }
.gt-label { font-size: 9px; font-weight: 700; color: #155724; }

/* Score badges */
.badge { display: inline-block; padding: 2px 7px; border-radius: 12px;
         font-size: 11px; font-weight: 700; white-space: nowrap; }
.vss-great { background: #d4edda; color: #155724; }
.vss-good  { background: #fff3cd; color: #856404; }
.vss-fair  { background: #ffe5b4; color: #7a4700; }
.vss-poor  { background: #f8d7da; color: #721c24; }
.vss-none  { background: #e2e3e5; color: #383d41; }

.meta-row { font-size: 11px; color: #888; display: flex; flex-wrap: wrap; gap: 8px; }
.meta-row b { color: #444; }
.img-placeholder { background: #eee; display: flex; align-items: center;
                   justify-content: center; font-size: 10px; color: #aaa;
                   border-radius: 6px; text-align: center; padding: 4px; }
"""


# ── HTML renderer ─────────────────────────────────────────────────────────────

def build_results_index(results_data):
    return {(r.get("label") or r.get("query_name", "")): r for r in results_data}


def render(golden_data, results_idx):

    # Sort worst-first by VSS@5
    def get_vss(e):
        return results_idx.get(e.get("query_name", ""), {}).get("vss5", 1.0)

    entries = sorted(golden_data, key=get_vss)

    # ── Summary table ─────────────────────────────────────────────────────────
    sum_rows = []
    for e in entries:
        lbl   = e.get("query_name", "?")
        cat   = e.get("query_category_tag", "")
        r     = results_idx.get(lbl, {})
        vss5  = r.get("vss5")
        hit5  = r.get("hit5")
        gt_vss = r.get("gt_vss")
        gt_found = r.get("gt_found", 0)
        tier  = r.get("bbox_tier", "–")
        sl    = r.get("search_label", "–")
        scores = r.get("scores", [])
        anchor = lbl.replace(" ", "_")
        badge  = vss_badge(vss5)
        sbadges = " ".join(vss_badge(s) for s in scores) if scores else "–"
        hit_html = ('<span class="hit-yes">&#10003; yes</span>' if hit5 else
                    '<span class="hit-no">&#10007; no</span>') if hit5 is not None else "–"
        gt_vss_html = "–"
        if gt_vss is not None:
            warn = ' title="Annotation quality warning: GT products score below 0.80"' if gt_vss < 0.80 else ""
            gt_vss_color = "vss-great" if gt_vss >= 0.80 else "vss-poor"
            gt_vss_html = f'<span class="badge {gt_vss_color}"{warn}>{gt_vss:.2f}</span> <span style="font-size:10px;color:#888;">({gt_found}/5 in top-5)</span>'
        sum_rows.append(
            f"<tr><td><a class='anchor-link' href='#{anchor}'>{lbl}</a></td>"
            f"<td>{cat}</td><td>{badge}</td><td>{sbadges}</td>"
            f"<td>{hit_html}</td><td>{gt_vss_html}</td><td>{tier}</td><td>{sl}</td></tr>"
        )

    # ── Per-query cards ───────────────────────────────────────────────────────
    print("Building query cards (generating CLIP crops from bboxes) ...")
    cards = []
    relevant_ids_map = {e.get("query_name", ""): set(e.get("relevant_product_ids", [])) for e in entries}

    for i, e in enumerate(entries, 1):
        lbl       = e.get("query_name", "?")
        cat       = e.get("query_category_tag", "")
        query_url = e.get("query_image_url", "")
        gt_prods  = e.get("relevant_info", [])
        anchor    = lbl.replace(" ", "_")
        relevant_ids = relevant_ids_map.get(lbl, set())

        r         = results_idx.get(lbl, {})
        vss5      = r.get("vss5")
        hit5      = r.get("hit5")
        gt_vss    = r.get("gt_vss")
        gt_found  = r.get("gt_found", 0)
        tier      = r.get("bbox_tier", "–")
        sl        = r.get("search_label", "")
        shoe_style = r.get("shoe_style")
        detected  = r.get("detected_labels", [])
        bbox      = r.get("bbox")
        result_records = r.get("results", [])   # [{rank,url,name,product_id,score}]

        # CLIP crop
        print(f"  [{i:2d}/{len(entries)}] {lbl[:50]} ...", end=" ", flush=True)
        crop_src = make_bbox_crop_src(query_url, bbox)
        print("crop ok" if crop_src else "no crop")

        crop_col = f"""
      <div class="col-crop">
        <div class="col-label">CLIP crop<br><small style='font-size:9px;color:#aaa'>(what search uses)</small></div>
        {make_img_tag(crop_src, "clip crop", w=120, h=150) if crop_src
         else '<div class="img-placeholder" style="width:120px;height:150px;font-size:10px;">'
              + ("full-image<br>mode" if not bbox else "no crop") + "</div>"}
      </div>"""

        # Ground-truth products
        gt_html = ""
        for p in gt_prods:
            nm  = (p.get("name") or "")[:35]
            img = p.get("image_url", "")
            pid = p.get("product_id", "")
            is_match = pid in relevant_ids
            gt_html += f"""
          <div class="ritem">
            {make_img_tag(img, nm, w=120, h=145,
                extra_style="outline:2px solid #6c757d;outline-offset:1px;" if is_match else "")}
            <span class="name-label" title="{p.get('name','')}">{nm}</span>
          </div>"""

        if not gt_html:
            gt_html = '<p style="font-size:12px;color:#999;font-style:italic;">No ground-truth images</p>'

        # Actual search results (from stored result_records)
        results_html = ""
        if result_records:
            for rec in result_records:
                rank    = rec.get("rank", "?")
                img     = rec.get("url", "")
                nm      = (rec.get("name") or "")[:35]
                pid     = rec.get("product_id", "")
                score   = rec.get("score")
                is_gt   = pid in relevant_ids
                border  = "outline:3px solid #28a745;outline-offset:2px;" if is_gt else ""
                gt_label = '<span class="gt-label">&#10003; GT match</span>' if is_gt else ""
                results_html += f"""
          <div class="ritem">
            <span class="rank-badge">#{rank}</span>
            {make_img_tag(img, nm, w=120, h=145, extra_style=border)}
            {vss_badge(score)}
            {gt_label}
            <span class="name-label" title="{rec.get('name','')}">{nm}</span>
          </div>"""
        else:
            results_html = '<p style="font-size:12px;color:#999;font-style:italic;">Re-run the evaluator to populate result images.</p>'

        # Meta row
        meta = [f"tier: <b>{tier}</b>", f"search_label: <b>{sl or '–'}</b>"]
        if shoe_style: meta.append(f"shoe_style: <b>{shoe_style}</b>")
        if detected:   meta.append(f"detected: {', '.join(detected)}")
        if bbox:       meta.append(f"bbox: {bbox}")
        meta_html = " · ".join(f"<span>{m}</span>" for m in meta)

        hit_note = ""
        if hit5 is not None:
            hit_note = ('<span style="color:#155724;font-weight:700;">&#10003; GT in top-5</span>'
                        if hit5 else
                        '<span style="color:#721c24;">&#10007; GT NOT in top-5</span>')

        gt_vss_note = ""
        gt_warning_banner = ""
        if gt_vss is not None:
            gt_vss_cls = "vss-great" if gt_vss >= 0.80 else "vss-poor"
            gt_vss_note = f'<span style="font-size:11px;color:#666;margin-left:4px;">GT-VSS <span class="badge {gt_vss_cls}" style="font-size:11px;">{gt_vss:.2f}</span> <span style="font-size:10px;color:#999;">({gt_found}/5 GT in results)</span></span>'
            if gt_vss < 0.80:
                gt_warning_banner = (
                    '<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:8px 14px;'
                    'font-size:12px;color:#856404;margin:0;">'
                    '<b>Annotation quality warning:</b> GT products score {:.2f} on average against this query image '
                    '(target: &ge;0.80). The golden dataset annotations for this query may be too loose — '
                    'consider replacing the annotated products with ones Gemini rates &ge;0.85 against the query.'
                    '</div>'.format(gt_vss)
                )

        cards.append(f"""
  <div class="qcard" id="{anchor}">
    <div class="qcard-header">
      <h3>{lbl}</h3>
      <span class="cat-tag">{cat}</span>
      <div class="header-metrics">
        <span style="font-size:11px;color:#888;">VSS@5</span> {vss_badge(vss5, size="large")}
        {hit_note}
        {gt_vss_note}
      </div>
    </div>
    {gt_warning_banner}
    <div class="qcard-body">
      <div class="col-query">
        <div class="col-label">Full query image<br><small style='font-size:9px;color:#aaa'>(what Gemini sees)</small></div>
        {make_img_tag(query_url, lbl, w=120, h=150)}
      </div>
      {crop_col}
      <div class="col-panels">
        <div>
          <div class="panel-title">Ground-truth relevant products ({len(gt_prods)} annotated)</div>
          <div class="result-row">{gt_html}</div>
        </div>
        <div>
          <div class="panel-title">Top-5 search results — Gemini scored each against the full query image (green border = matches ground truth)</div>
          <div class="result-row">{results_html}</div>
        </div>
        <div class="meta-row">{meta_html}</div>
      </div>
    </div>
  </div>""")

    n = len(entries)
    all_vss    = [results_idx.get(e.get("query_name",""), {}).get("vss5") for e in entries if results_idx.get(e.get("query_name",""), {}).get("vss5") is not None]
    all_hit    = [results_idx.get(e.get("query_name",""), {}).get("hit5") for e in entries if results_idx.get(e.get("query_name",""), {}).get("hit5") is not None]
    all_gt_vss = [results_idx.get(e.get("query_name",""), {}).get("gt_vss") for e in entries if results_idx.get(e.get("query_name",""), {}).get("gt_vss") is not None]
    avg_vss    = round(sum(all_vss)/len(all_vss), 3) if all_vss else "–"
    avg_hit    = round(sum(all_hit)/len(all_hit), 3) if all_hit else "–"
    avg_gt_vss = round(sum(all_gt_vss)/len(all_gt_vss), 3) if all_gt_vss else "–"
    n_poor_annot = sum(1 for v in all_gt_vss if v < 0.80)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Locus — Golden Dataset Report</title>
  <style>{CSS}</style>
</head>
<body>

<div class="page-header">
  <h1>Locus — Golden Dataset Evaluation Report</h1>
  <p>
    {n} queries &middot; sorted worst&#8594;best VSS@5 &middot;
    Avg VSS@5: <b>{avg_vss}</b> &middot;
    Hit@5 (GT in top-5): <b>{avg_hit}</b> &middot;
    GT-VSS ceiling: <b>{avg_gt_vss}</b>
    {f'&middot; <span style="color:#e09000;font-weight:700;">{n_poor_annot} queries have GT-VSS &lt; 0.80 (annotation quality warning)</span>' if n_poor_annot else ''} &middot;
    Green border = ground-truth match
  </p>
</div>

<div class="summary-wrap">
  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>Query</th><th>Category</th><th>VSS@5</th>
          <th>Scores #1&#8594;#5</th><th>Hit@5</th><th>GT-VSS (ceiling)</th><th>BBox tier</th><th>Search label</th></tr>
    </thead>
    <tbody>{"".join(sum_rows)}</tbody>
  </table>
</div>

<div class="queries">
{"".join(cards)}
</div>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    if not GOLDEN.exists():
        print(f"ERROR: {GOLDEN} not found"); sys.exit(1)

    golden_data = load_json(GOLDEN)
    print(f"  Golden dataset : {len(golden_data)} queries")

    results_idx = {}
    if RESULTS.exists():
        results_data = load_json(RESULTS)
        results_idx  = build_results_index(results_data)
        has_results  = any("results" in r for r in results_data)
        print(f"  Gemini results : {len(results_idx)} entries"
              + (" (includes result images)" if has_results else
                 " - re-run evaluator to populate result images"))
    else:
        print(f"  Warning: {RESULTS} not found — VSS data will be missing")

    html = render(golden_data, results_idx)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone: {OUT_HTML}")


if __name__ == "__main__":
    main()
