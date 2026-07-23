"""
Interactive web tool to select pruning boxes and recolor them red.

Usage:
    python select_red_boxes.py <image_path> [--out output_path] [--port 5000]

Then open http://<server_ip>:<port> in your browser.
Click boxes to toggle red. Click "Save" to export.
"""

import sys
import argparse
import json
import math
import numpy as np
from PIL import Image, ImageDraw


# ── Grid detection ────────────────────────────────────────────────────────────

def find_bar_height(img_arr):
    """
    Find the label bar at the top.
    The bar is a solid dark background (≈30,30,30) with optional yellow text.
    We look for the first row that transitions from dark-bar to actual content.
    Strategy: scan until we find a row that is NOT mostly near (30,30,30).
    """
    H = img_arr.shape[0]
    for y in range(H):
        row = img_arr[y].astype(float)
        # Bar pixels are all near (30,30,30); content rows have larger variance
        if row.std() > 25 or row.mean() > 90:
            return y
    return 28  # fallback


def infer_grid(total_tokens, img_w, img_h):
    """Same logic as visualize_prefixvlm2.py."""
    aspect = img_w / img_h
    grid_h = max(1, round(math.sqrt(total_tokens / aspect)))
    grid_w = max(1, round(total_tokens / grid_h))
    while grid_h * grid_w < total_tokens:
        grid_w += 1
    return grid_h, grid_w


def detect_grid_from_image(content_arr):
    """
    Detect grid by independently scoring row/col border alignments.
    Returns (grid_h, grid_w, cell_h, cell_w).
    """
    cH, cW = content_arr.shape[:2]
    white = (
        (content_arr[:, :, 0] > 200) &
        (content_arr[:, :, 1] > 200) &
        (content_arr[:, :, 2] > 200)
    ).astype(np.uint8)

    row_sum = white.sum(axis=1).astype(float)  # how many white px in each row
    col_sum = white.sum(axis=0).astype(float)  # how many white px in each col

    # Score n_lines by counting white pixels that fall on border rows/cols.
    # A border is at int(i * length / n) for i in 0..n.
    def score_axis(density, length, n):
        s = 0
        for i in range(n + 1):
            pos = int(i * length / n)
            if pos < length:
                s += density[pos]
        return s

    # Search independently for grid_w (cols) and grid_h (rows)
    # Cell size must be between 8px and 40px (source always uses ~10px cells)
    def best_n(density, length):
        lo = max(10, length // 40)
        hi = length // 8
        best_n, best_s = lo, -1
        for n in range(lo, hi + 1):
            s = score_axis(density, length, n)
            if s > best_s:
                best_s, best_n = s, n
        return best_n

    gw = best_n(col_sum, cW)
    gh = best_n(row_sum, cH)

    return gh, gw, cH / gh, cW / gw


def build_boxes(content_arr, grid_h, grid_w):
    """
    Build boxes using the SAME integer cell boundary formula as visualize_prefixvlm2.py.
    """
    cH, cW = content_arr.shape[:2]
    cell_h = cH / grid_h
    cell_w = cW / grid_w

    white = (
        (content_arr[:, :, 0] > 200) &
        (content_arr[:, :, 1] > 200) &
        (content_arr[:, :, 2] > 200)
    )

    boxes = []
    for row in range(grid_h):
        for col in range(grid_w):
            # Exact same formula as the source visualize code
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w) - 1
            y1 = int((row + 1) * cell_h) - 1

            # Check if this cell has a white 1px border
            border = []
            if y0 < cH:
                border.extend(white[y0, x0:x1+1].tolist())
            if y1 < cH:
                border.extend(white[y1, x0:x1+1].tolist())
            if x0 < cW:
                border.extend(white[y0:y1+1, x0].tolist())
            if x1 < cW:
                border.extend(white[y0:y1+1, x1].tolist())

            boxes.append({
                "id": len(boxes), "row": row, "col": col,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "has_box": any(border),
            })
    return boxes


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_image(img_arr, bar_h, boxes, red_ids):
    result = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(result)
    red_set = set(red_ids)
    for b in boxes:
        if b["id"] in red_set and b["has_box"]:
            # Draw over the white border using exact same coordinates
            draw.rectangle(
                [b["x0"], bar_h + b["y0"], b["x1"], bar_h + b["y1"]],
                outline=(255, 0, 0), width=3,
            )
    return result


# ── Web server ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--grid", default=None, help="e.g. 46x63")
    parser.add_argument("--bar", type=int, default=None)
    args = parser.parse_args()

    out_path = args.out or args.image.replace(".png", "_red.png")

    img = Image.open(args.image).convert("RGB")
    img_arr = np.array(img)

    if args.bar is not None:
        bar_h = args.bar
    else:
        bar_h = 28  # fixed: visualize_prefixvlm2.py always uses bar_h=28

    content_arr = img_arr[bar_h:]
    cH, cW = content_arr.shape[:2]
    print(f"Image {img_arr.shape[1]}x{img_arr.shape[0]}, bar_h={bar_h}, content {cW}x{cH}")

    if args.grid:
        gh, gw = map(int, args.grid.split("x"))
    else:
        gh, gw, _, _ = detect_grid_from_image(content_arr)

    print(f"Grid: {gw} cols x {gh} rows  (cell ~{cW/gw:.2f}x{cH/gh:.2f}px)")

    boxes = build_boxes(content_arr, gh, gw)
    n_white = sum(1 for b in boxes if b["has_box"])
    print(f"Detected {n_white} white-bordered boxes out of {len(boxes)} cells.")

    boxes_json = json.dumps([
        {"id": b["id"], "x0": b["x0"], "y0": b["y0"],
         "x1": b["x1"], "y1": b["y1"], "has_box": b["has_box"]}
        for b in boxes
    ])

    try:
        from flask import Flask, request, jsonify, send_file
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Select Red Boxes</title>
<style>
  body {{ margin:0; background:#111; display:flex; flex-direction:column; align-items:center; }}
  #controls {{ margin:10px; display:flex; align-items:center; flex-wrap:wrap; gap:6px; }}
  button {{ padding:8px 20px; font-size:14px; cursor:pointer; border:none; border-radius:4px; }}
  #status {{ color:#aaa; font-family:monospace; font-size:13px; }}
  canvas {{ cursor:crosshair; max-width:98vw; image-rendering:pixelated; display:block; }}
</style>
</head>
<body>
<div id="controls">
  <button onclick="saveImg()" style="background:#c33;color:#fff">Save</button>
  <button onclick="resetAll()" style="background:#555;color:#fff">Reset</button>
  <span id="status">Loading…</span>
</div>
<canvas id="c"></canvas>
<script>
const BAR_H = {bar_h};
const BOXES  = {boxes_json};
const byId   = Object.fromEntries(BOXES.map(b => [b.id, b]));
const redSet = new Set();

const src  = new Image();
src.src    = '/img?' + Date.now();
const cvs  = document.getElementById('c');
const ctx  = cvs.getContext('2d');

src.onload = () => {{
  cvs.width  = src.naturalWidth;
  cvs.height = src.naturalHeight;
  redraw();
  document.getElementById('status').textContent =
    BOXES.filter(b=>b.has_box).length + ' white boxes — click to toggle red';
}};

function redraw() {{
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(src, 0, 0);
  ctx.strokeStyle = 'red';
  ctx.lineWidth   = 3;
  redSet.forEach(id => {{
    const b = byId[id];
    if (!b || !b.has_box) return;
    // Same rect as PIL: draw.rectangle([x0, bar_h+y0, x1, bar_h+y1])
    ctx.strokeRect(b.x0 + 0.5, BAR_H + b.y0 + 0.5, b.x1 - b.x0, b.y1 - b.y0);
  }});
}}

cvs.addEventListener('click', e => {{
  const r  = cvs.getBoundingClientRect();
  const px = (e.clientX - r.left)  * cvs.width  / r.width;
  const py = (e.clientY - r.top)   * cvs.height / r.height - BAR_H;
  if (py < 0) return;
  for (const b of BOXES) {{
    if (!b.has_box) continue;
    if (px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1) {{
      redSet.has(b.id) ? redSet.delete(b.id) : redSet.add(b.id);
      redraw();
      document.getElementById('status').textContent = redSet.size + ' red boxes selected';
      return;
    }}
  }}
}});

function resetAll() {{
  redSet.clear(); redraw();
  document.getElementById('status').textContent = 'Reset.';
}}

function saveImg() {{
  fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{ red: Array.from(redSet) }})
  }}).then(r=>r.json()).then(d => {{
    document.getElementById('status').textContent = d.msg;
  }});
}}
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return HTML

    @app.route("/img")
    def get_img():
        return send_file(args.image, mimetype="image/png")

    @app.route("/save", methods=["POST"])
    def save():
        red_ids = request.get_json().get("red", [])
        final = render_image(img_arr, bar_h, boxes, red_ids)
        final.save(out_path)
        msg = f"Saved {len(red_ids)} red boxes → {out_path}"
        print(msg)
        return jsonify({"msg": msg})

    print(f"\nOpen in browser:  http://<server_ip>:{args.port}")
    print(f"Output will be:   {out_path}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
