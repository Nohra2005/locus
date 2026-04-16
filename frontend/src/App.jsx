import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import StoreDashboardView from "./StoreDashboardView";

// ── Fix Leaflet marker icons broken by Vite ────────────────────────
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:       "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

const API = "";

const STORE_COORDS = {
  "Zara":          [33.88685, 35.51308],
  "Bershka":       [33.93372, 35.58891],
  "Mike Sport":    [33.86769, 35.54560],
  "Louis Vuitton": [33.89383, 35.50182],
  "Virgin":        [33.88685, 35.51308],
};

const T = {
  bg:          "#0d0c0a",
  bgDeep:      "#090807",
  surface:     "#161410",
  surfaceHov:  "#1c1a16",
  border:      "#2a2620",
  borderFaint: "#1a1814",
  text:        "#e8e2d9",
  textMuted:   "#6b6458",
  textFaint:   "#3d3830",
  accent:      "#c9a96e",
  accentBg:    "rgba(201,169,110,0.08)",
  accentRing:  "rgba(201,169,110,0.2)",
  accentDeep:  "#a8895a",
  green:       "#7aab8a",
  yellow:      "#c9a96e",
  red:         "#c97070",
};

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,600&family=DM+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, #root {
    height: 100%;
    width: 100%;
    background: ${T.bg};
    color: ${T.text};
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 2px; }

  .leaflet-container {
    background: ${T.bgDeep} !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .leaflet-popup-content-wrapper {
    background: ${T.surface} !important;
    color: ${T.text} !important;
    border: 1px solid ${T.border} !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
  }
  .leaflet-popup-tip { background: ${T.surface} !important; }
  .leaflet-popup-close-button { color: ${T.textMuted} !important; top: 6px !important; right: 6px !important; }
  .leaflet-control-zoom a {
    background: ${T.surface} !important;
    color: ${T.textMuted} !important;
    border-color: ${T.border} !important;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .fade-up { animation: fadeUp 0.5s cubic-bezier(0.16,1,0.3,1) forwards; }
  .fade-in { animation: fadeIn 0.35s ease forwards; }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  @media (min-width: 640px)  { .results-grid { grid-template-columns: repeat(3, 1fr); } }
  @media (min-width: 1024px) { .results-grid { grid-template-columns: repeat(4, 1fr); } }

  .nav-link {
    font-size: 0.8rem;
    color: ${T.textMuted};
    background: none;
    border: none;
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    transition: color 0.2s, background 0.2s;
  }
  .nav-link:hover { color: ${T.text}; }
  .nav-link.active {
    background: ${T.surface};
    color: ${T.text};
    border: 1px solid ${T.border};
  }

  .upload-zone {
    border: 1.5px dashed ${T.border};
    border-radius: 16px;
    background: ${T.surface};
    transition: border-color 0.25s, background 0.25s;
    cursor: pointer;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: ${T.accent};
    background: ${T.accentBg};
  }

  .btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    background: ${T.surface};
    color: ${T.text};
    border: 1px solid ${T.border};
    border-radius: 24px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s, color 0.2s;
  }
  .btn-primary:hover {
    border-color: ${T.accent};
    color: ${T.accent};
    background: ${T.accentBg};
  }

  .btn-ghost {
    background: none;
    border: none;
    color: ${T.textMuted};
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    padding: 6px 10px;
    border-radius: 6px;
    transition: color 0.2s, background 0.2s;
  }
  .btn-ghost:hover { color: ${T.text}; background: ${T.surface}; }

  .star-btn {
    background: none;
    border: none;
    padding: 2px 1px;
    cursor: pointer;
    font-size: 0.9rem;
    line-height: 1;
    transition: transform 0.1s, color 0.15s;
    color: ${T.textFaint};
  }
  .star-btn:hover:not(:disabled) { transform: scale(1.2); }
  .star-btn:disabled { cursor: default; }
  .star-btn.filled { color: ${T.accent}; }
  .star-btn.dimmed { color: ${T.borderFaint}; }

  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    height: 2px;
    background: ${T.border};
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: ${T.accent};
    cursor: pointer;
  }
`;

function StyleInjector() {
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);
  return null;
}

function MapRecenter({ center }) {
  const map = useMap();
  useEffect(() => { map.setView(center, map.getZoom()); }, [center, map]);
  return null;
}

const makeStoreIcon = (highlighted) => L.divIcon({
  html: `<div style="
    width:22px; height:22px;
    background:${highlighted ? T.accent : "#6b6458"};
    border-radius:50% 50% 50% 0;
    transform:rotate(-45deg);
    border:2px solid ${highlighted ? "#fff" : T.border};
    box-shadow:0 2px 10px rgba(0,0,0,0.5);
    transition:all 0.2s;
  "></div>`,
  className: "",
  iconSize: [22, 22],
  iconAnchor: [11, 22],
  popupAnchor: [0, -24],
});

function haversineKm([lat1, lon1], [lat2, lon2]) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ══════════════════════════════════════════════════════════════════
// NAVBAR
// ══════════════════════════════════════════════════════════════════
function Navbar({ activeTab, onTab, onLogoClick }) {
  return (
    <nav style={{
      position: "sticky", top: 0, zIndex: 200,
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "14px 28px",
      background: `${T.bg}ee`,
      backdropFilter: "blur(12px)",
      borderBottom: `1px solid ${T.borderFaint}`,
    }}>
      <button
        onClick={onLogoClick}
        style={{ display: "flex", alignItems: "center", gap: "10px", background: "none", border: "none", cursor: "pointer" }}
      >
        <div style={{
          width: 32, height: 32,
          background: `linear-gradient(135deg, ${T.accent}, ${T.accentDeep})`,
          borderRadius: "8px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.75rem",
          boxShadow: `0 2px 12px ${T.accentRing}`,
        }}>✦</div>
        <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.25rem", fontWeight: 600, color: T.text, letterSpacing: "0.02em" }}>
          locus
        </span>
        <span style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase", paddingLeft: "2px" }}>
          shopping made easier
        </span>
      </button>

      <div style={{ display: "flex", gap: "4px" }}>
        {["Discover", "Saved", "History", "Store"].map(tab => (
          <button
            key={tab}
            className={`nav-link ${activeTab === tab ? "active" : ""}`}
            onClick={() => onTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>
    </nav>
  );
}

// ══════════════════════════════════════════════════════════════════
// LANDING
// ══════════════════════════════════════════════════════════════════
function LandingView({ onUpload, error }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file) => {
    if (file && file.type.startsWith("image/")) onUpload(file);
  };

  return (
    <div style={{ minHeight: "calc(100dvh - 61px)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 24px" }}>
      <div className="fade-up" style={{ textAlign: "center", marginBottom: "48px" }}>
        <div style={{ fontSize: "0.7rem", color: T.accent, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "20px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
          <span style={{ fontSize: "0.6rem" }}>✦</span>
          AI-powered visual search
        </div>
        <h1 style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "clamp(2.8rem, 8vw, 4.5rem)", fontWeight: 500, lineHeight: 1.1, color: T.text, letterSpacing: "-0.01em" }}>
          Find what you{" "}<em style={{ color: T.accent, fontStyle: "italic" }}>see</em>
        </h1>
        <p style={{ marginTop: "18px", fontSize: "0.88rem", color: T.textMuted, lineHeight: 1.7, maxWidth: "400px", margin: "18px auto 0" }}>
          Upload any photo and we'll match it against thousands of products across top stores.
        </p>
      </div>

      <div
        className={`upload-zone fade-up ${dragging ? "drag-over" : ""}`}
        style={{ animationDelay: "0.1s", width: "100%", maxWidth: "540px", padding: "56px 32px", textAlign: "center" }}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
      >
        <div style={{ width: 56, height: 56, borderRadius: "50%", background: T.surfaceHov, border: `1px solid ${T.border}`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 20px", fontSize: "1.2rem" }}>↑</div>
        <div style={{ fontSize: "0.95rem", fontWeight: 500, color: T.text, marginBottom: "8px" }}>Drop your photo here</div>
        <div style={{ fontSize: "0.75rem", color: T.textMuted, marginBottom: "24px" }}>PNG · JPG · WEBP — we'll find matching products instantly</div>
        <button className="btn-primary" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>
          <span style={{ color: T.accent }}>✦</span>
          Start searching
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg, image/png, image/webp, image/heic, image/*"
          style={{ position: "absolute", width: "1px", height: "1px", opacity: 0, pointerEvents: "none" }}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
              e.target.value = null;
            }
          }}
        />
      </div>

      {error && (
        <div className="fade-up" style={{ marginTop: "20px", padding: "12px 18px", background: "rgba(201,112,112,0.08)", border: `1px solid rgba(201,112,112,0.2)`, borderRadius: "10px", fontSize: "0.75rem", color: T.red, maxWidth: "540px", width: "100%" }}>
          ⚠ {error}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// LOADING
// ══════════════════════════════════════════════════════════════════
function LoadingView({ label }) {
  return (
    <div style={{ minHeight: "calc(100dvh - 61px)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "16px" }}>
      <div style={{ width: 28, height: 28, border: `1.5px solid ${T.border}`, borderTop: `1.5px solid ${T.accent}`, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
      <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.05em" }}>{label}</span>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// CATEGORY LABELS  (mirrors clip_labels.py CANONICAL_LABELS)
// ══════════════════════════════════════════════════════════════════
const CATEGORY_LABELS = {
  top:        "Top",
  sports_bra: "Sports Bra",
  pants:      "Pants",
  leggings:   "Leggings",
  shorts:     "Shorts",
  skirt:      "Skirt",
  dress:      "Dress",
  sweater:    "Sweater",
  jacket:     "Jacket",
  shoes:      "Shoes",
  hat:        "Hat",
  bag:        "Bag",
  jumpsuit:   "Jumpsuit",
};

// ══════════════════════════════════════════════════════════════════
// DRAW VIEW  — user drags a bbox around the item they want to find
// ══════════════════════════════════════════════════════════════════
function DrawView({ imageURL, imageFile, onConfirm, onBack }) {
  const containerRef = useRef(null);
  const imgRef       = useRef(null);

  const [imgNatural,  setImgNatural]  = useState({ w: 1, h: 1 });
  const [imgRendered, setImgRendered] = useState({ w: 0, h: 0, x: 0, y: 0 });
  const [drawing,     setDrawing]     = useState(false);
  const [startPt,     setStartPt]     = useState(null);
  const [box,         setBox]         = useState(null); // rendered coords {x1,y1,x2,y2}
  const [resizing,    setResizing]    = useState(null); // null | "tl" | "tr" | "bl" | "br"
  const [loading,     setLoading]     = useState(false);

  const measureRendered = (el) => {
    const { width: cw, height: ch } = el.getBoundingClientRect();
    const s = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
    setImgRendered({ w: el.naturalWidth * s, h: el.naturalHeight * s, x: (cw - el.naturalWidth * s) / 2, y: (ch - el.naturalHeight * s) / 2 });
  };

  const onImgLoad = (e) => {
    const el = e.target;
    setImgNatural({ w: el.naturalWidth, h: el.naturalHeight });
    measureRendered(el);
  };

  useEffect(() => {
    const onResize = () => { if (imgRef.current?.naturalWidth) measureRendered(imgRef.current); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Returns pointer position clamped to image bounds, relative to container
  const getPoint = (e) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect  = containerRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: Math.max(imgRendered.x, Math.min(imgRendered.x + imgRendered.w, clientX - rect.left)),
      y: Math.max(imgRendered.y, Math.min(imgRendered.y + imgRendered.h, clientY - rect.top)),
    };
  };

  const onPointerDown = (e) => {
    if (loading || imgRendered.w === 0) return;
    // If a box is already drawn, ignore new touches on the container so the
    // user can scroll freely. They must press "Redraw" to start over.
    const boxIsSet = box && (box.x2 - box.x1) >= 10 && (box.y2 - box.y1) >= 10;
    if (boxIsSet) return;
    e.preventDefault();
    const pt = getPoint(e);
    setStartPt(pt);
    setBox(null);
    setDrawing(true);
  };

  const onPointerMove = (e) => {
    // Handle corner-resize drag
    if (resizing) {
      e.preventDefault();
      const pt = getPoint(e);
      setBox(prev => {
        if (!prev) return prev;
        const b = { ...prev };
        if (resizing.includes("l")) b.x1 = Math.min(pt.x, prev.x2 - 20);
        if (resizing.includes("r")) b.x2 = Math.max(pt.x, prev.x1 + 20);
        if (resizing.includes("t")) b.y1 = Math.min(pt.y, prev.y2 - 20);
        if (resizing.includes("b")) b.y2 = Math.max(pt.y, prev.y1 + 20);
        return b;
      });
      return;
    }
    if (!drawing || !startPt) return;
    e.preventDefault();
    const pt = getPoint(e);
    setBox({
      x1: Math.min(startPt.x, pt.x), y1: Math.min(startPt.y, pt.y),
      x2: Math.max(startPt.x, pt.x), y2: Math.max(startPt.y, pt.y),
    });
  };

  const onPointerUp = (e) => {
    if (resizing) {
      setResizing(null);
      return;
    }
    if (!drawing) return;
    e.preventDefault();
    setDrawing(false);
    // Discard boxes smaller than 10×10 rendered px (accidental tap)
    setBox(prev => (prev && (prev.x2 - prev.x1) >= 10 && (prev.y2 - prev.y1) >= 10) ? prev : null);
  };

  // Convert rendered-space box → natural image pixel coords
  const toNatural = (b) => {
    const sx = imgNatural.w / imgRendered.w;
    const sy = imgNatural.h / imgRendered.h;
    return {
      x1: Math.round((b.x1 - imgRendered.x) * sx),
      y1: Math.round((b.y1 - imgRendered.y) * sy),
      x2: Math.round((b.x2 - imgRendered.x) * sx),
      y2: Math.round((b.y2 - imgRendered.y) * sy),
    };
  };

  const handleFindItem = async () => {
    if (!box || loading) return;
    setLoading(true);
    const natural = toNatural(box);

    // Crop client-side for the confirm preview
    let cropBlob = null;
    try {
      const bitmap = await createImageBitmap(imageFile);
      const cw = Math.max(1, natural.x2 - natural.x1);
      const ch = Math.max(1, natural.y2 - natural.y1);
      const canvas = document.createElement("canvas");
      canvas.width = cw; canvas.height = ch;
      canvas.getContext("2d").drawImage(bitmap, natural.x1, natural.y1, cw, ch, 0, 0, cw, ch);
      cropBlob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.9));
    } catch { /* non-critical — confirm view will show placeholder */ }

    // Ask CLIP for the category
    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", natural.x1); form.append("y1", natural.y1);
    form.append("x2", natural.x2); form.append("y2", natural.y2);
    try {
      const res  = await fetch(`${API}/classify-crop`, { method: "POST", body: form });
      const data = res.ok ? await res.json() : {};
      onConfirm({ bbox: natural, cropBlob, category: data.category || null, allScores: data.all_scores || {} });
    } catch {
      // Network error: still proceed, user can pick category manually
      onConfirm({ bbox: natural, cropBlob, category: null, allScores: {} });
    }
  };

  const hasBox = box && (box.x2 - box.x1) >= 10 && (box.y2 - box.y1) >= 10;

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 28px", borderBottom: `1px solid ${T.borderFaint}` }}>
        <button className="btn-ghost" onClick={onBack} style={{ fontSize: "1rem" }}>←</button>
        <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.08em" }}>
          {hasBox ? "LOOKS GOOD — HIT FIND OR REDRAW" : "DRAW A BOX AROUND THE ITEM"}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "24px" }}>
        <div
          ref={containerRef}
          style={{ position: "relative", width: "100%", maxWidth: 640, cursor: loading ? "wait" : (hasBox ? "default" : "crosshair"), touchAction: (hasBox && !resizing) ? "pan-y" : "none", userSelect: "none" }}
          onMouseDown={onPointerDown}
          onMouseMove={onPointerMove}
          onMouseUp={onPointerUp}
          onMouseLeave={onPointerUp}
          onTouchStart={onPointerDown}
          onTouchMove={onPointerMove}
          onTouchEnd={onPointerUp}
        >
          <img
            ref={imgRef}
            src={imageURL}
            onLoad={onImgLoad}
            alt="uploaded"
            draggable={false}
            onContextMenu={(e) => e.preventDefault()}
            style={{ width: "100%", display: "block", borderRadius: 12, pointerEvents: "none", userSelect: "none" }}
          />
          {imgRendered.w > 0 && (
            <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
              {/* Dim everything outside the drawn box */}
              {hasBox && (
                <>
                  <defs>
                    <mask id="crop-mask">
                      <rect x="0" y="0" width="100%" height="100%" fill="white" />
                      <rect x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1} fill="black" />
                    </mask>
                  </defs>
                  <rect x="0" y="0" width="100%" height="100%" fill="rgba(0,0,0,0.45)" mask="url(#crop-mask)" rx="12" />
                </>
              )}
              {/* The box itself */}
              {box && (
                <rect
                  x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1}
                  fill="rgba(201,169,110,0.08)"
                  stroke={T.accent}
                  strokeWidth={2}
                  strokeDasharray={drawing ? "6 3" : "none"}
                  rx={4}
                />
              )}
            </svg>
          )}
          {/* Draggable corner handles — HTML divs so pointer events work on mobile */}
          {hasBox && !drawing && (
            [
              { id: "tl", cx: box.x1, cy: box.y1, cursor: "nwse-resize" },
              { id: "tr", cx: box.x2, cy: box.y1, cursor: "nesw-resize" },
              { id: "bl", cx: box.x1, cy: box.y2, cursor: "nesw-resize" },
              { id: "br", cx: box.x2, cy: box.y2, cursor: "nwse-resize" },
            ].map(({ id, cx, cy, cursor }) => (
              <div
                key={id}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setResizing(id); }}
                onTouchStart={(e) => { e.stopPropagation(); e.preventDefault(); setResizing(id); }}
                style={{
                  position: "absolute",
                  left: cx, top: cy,
                  width: 28, height: 28,
                  transform: "translate(-50%, -50%)",
                  borderRadius: "50%",
                  background: T.accent,
                  cursor,
                  touchAction: "none",
                  zIndex: 10,
                }}
              />
            ))
          )}
        </div>

        <div style={{ width: "100%", maxWidth: 640, marginTop: 16, display: "flex", gap: 10 }}>
          {hasBox && (
            <button className="btn-ghost" style={{ flexShrink: 0 }} onClick={() => { setBox(null); setStartPt(null); setResizing(null); }}>
              Redraw
            </button>
          )}
          <button
            className="btn-primary"
            style={{ flex: 1, opacity: hasBox && !loading ? 1 : 0.4, pointerEvents: hasBox && !loading ? "auto" : "none" }}
            onClick={handleFindItem}
          >
            {loading
              ? <><div style={{ width: 12, height: 12, border: `1.5px solid ${T.border}`, borderTop: `1.5px solid ${T.accent}`, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} /> Identifying…</>
              : <><span style={{ color: T.accent }}>✦</span> Find this item</>
            }
          </button>
        </div>

        <p style={{ marginTop: 12, fontSize: "0.7rem", color: T.textFaint, textAlign: "center" }}>
          Drag to draw a box · works on mobile too
        </p>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// CONFIRM VIEW  — shows crop + CLIP category prediction for approval
// ══════════════════════════════════════════════════════════════════
function ConfirmView({ cropBlob, predictedCategory, allScores, onSearch, onBack }) {
  const cropURL = useMemo(() => {
    if (!cropBlob) return null;
    const url = URL.createObjectURL(cropBlob);
    return url;
  }, [cropBlob]);

  // Revoke object URL when cropBlob changes or component unmounts
  useEffect(() => {
    return () => { if (cropURL) URL.revokeObjectURL(cropURL); };
  }, [cropURL]);

  const isNotFashion = predictedCategory === "not_fashion";
  const [category, setCategory] = useState(
    isNotFashion ? Object.keys(CATEGORY_LABELS)[0] : (predictedCategory || Object.keys(CATEGORY_LABELS)[0])
  );

  const conf      = (allScores && category) ? (allScores[category] ?? 0) : 0;
  const confLabel = conf > 0.6 ? "high confidence" : conf > 0.35 ? "medium confidence" : "low confidence";
  const confColor = conf > 0.6 ? T.green : conf > 0.35 ? T.yellow : T.textMuted;

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 28px", borderBottom: `1px solid ${T.borderFaint}` }}>
        <button className="btn-ghost" onClick={onBack} style={{ fontSize: "1rem" }}>←</button>
        <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.08em" }}>CONFIRM ITEM</span>
      </div>

      {isNotFashion && (
        <div style={{ margin: "24px 24px 0", padding: "18px 20px", background: "rgba(220,60,60,0.08)", border: "1px solid rgba(220,60,60,0.35)", borderRadius: 12, display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#e05a5a" }}>Not a fashion item</div>
          <div style={{ fontSize: "0.75rem", color: T.textMuted, lineHeight: 1.6 }}>
            The selected area doesn't look like a clothing item or accessory (socks, underwear, belts, etc. aren't searchable).<br />
            Go back and draw a box around a clothing item instead.
          </div>
          <button className="btn-ghost" style={{ alignSelf: "flex-start", marginTop: 4 }} onClick={onBack}>← Redraw</button>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "32px 24px", gap: 28, display: isNotFashion ? "none" : "flex" }}>

        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", width: "100%", maxWidth: 480 }}>
          {/* Crop thumbnail */}
          <div style={{ width: 100, height: 130, flexShrink: 0, background: T.surface, border: `1.5px solid ${T.accent}`, borderRadius: 10, overflow: "hidden" }}>
            {cropURL
              ? <img src={cropURL} alt="crop" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.4rem" }}>✦</div>
            }
          </div>

          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Predicted category */}
            <div>
              <div style={{ fontSize: "0.6rem", color: T.textMuted, letterSpacing: "0.1em", marginBottom: 6 }}>DETECTED AS</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.4rem", fontWeight: 600, color: T.text }}>
                  {CATEGORY_LABELS[category] ?? "Unknown"}
                </span>
                {conf > 0 && (
                  <span style={{ fontSize: "0.62rem", color: confColor, border: `1px solid ${confColor}`, borderRadius: 20, padding: "2px 8px" }}>
                    {confLabel}
                  </span>
                )}
              </div>
            </div>

            {/* Correction dropdown */}
            <div>
              <div style={{ fontSize: "0.6rem", color: T.textMuted, letterSpacing: "0.1em", marginBottom: 6 }}>NOT RIGHT? CHANGE IT</div>
              <div style={{ position: "relative" }}>
                <select
                  value={category}
                  onChange={e => setCategory(e.target.value)}
                  style={{
                    width: "100%", background: T.surface, color: T.text,
                    border: `1px solid ${T.border}`, borderRadius: 8,
                    padding: "7px 32px 7px 10px", fontSize: "0.82rem",
                    fontFamily: "'DM Sans', sans-serif", cursor: "pointer",
                    appearance: "none", outline: "none",
                  }}
                >
                  {Object.entries(CATEGORY_LABELS).map(([val, lbl]) => (
                    <option key={val} value={val}>{lbl}</option>
                  ))}
                </select>
                <svg style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }} width="10" height="6" viewBox="0 0 10 6">
                  <path d="M0 0l5 6 5-6z" fill={T.textMuted} />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <button
          className="btn-primary"
          style={{ width: "100%", maxWidth: 480 }}
          onClick={() => onSearch(category)}
        >
          <span style={{ color: T.accent }}>✦</span> Search
        </button>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULT CARD
// ══════════════════════════════════════════════════════════════════
function ResultCard({ result, category, onFeedback, highlighted, judgeScore }) {
  const [vote, setVote]       = useState(null);
  const [hoverStar, setHover] = useState(null);

  const score    = result.score ?? 0;
  const payload  = result.payload ?? {};
  const raw      = payload.image_url ?? "";
  const imageURL = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;

  const displayScore = judgeScore ?? score;
  const isJudged     = judgeScore !== null && judgeScore !== undefined;
  const scoreColor   = displayScore >= 0.80 ? T.green : displayScore >= 0.60 ? T.yellow : T.red;

  const handleVote = async (stars) => {
    if (stars === vote) return; // same rating clicked again — no-op
    setVote(stars);
    setHover(null);
    await onFeedback(
      payload.product_id || payload.item_id || payload.image_url,
      stars,
      payload.item_name || "",
      payload.store     || "",
      category          || "",
      payload.image_url || "",
    );
  };

  const fillUpTo = vote !== null ? vote : (hoverStar ?? 0);

  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${highlighted ? T.accent : T.border}`,
      borderRadius: "12px",
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      transition: "border-color 0.2s",
    }}>
      <div style={{ aspectRatio: "3/4", background: T.bgDeep, overflow: "hidden", position: "relative" }}>
        {imageURL
          ? <img src={imageURL} alt={payload.item_name || "product"} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.5rem" }}>✦</div>
        }
        <span style={{ position: "absolute", top: 8, right: 8, display: "flex", alignItems: "center", gap: 3, background: "rgba(0,0,0,0.55)", borderRadius: 5, padding: "2px 6px" }}>
          <span style={{ fontSize: "0.72rem", fontWeight: 600, color: scoreColor }}>{Math.round(displayScore * 100)}%</span>
          {isJudged
            ? <span style={{ fontSize: "0.52rem", color: T.accent, border: `1px solid ${T.accent}`, borderRadius: 3, padding: "0px 3px", lineHeight: 1.6 }}>AI</span>
            : <span style={{ fontSize: "0.52rem", color: T.textFaint }}>~</span>
          }
        </span>
      </div>

      <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: "8px", flex: 1 }}>
        <div style={{ fontSize: "0.78rem", fontWeight: 500, color: T.text, lineHeight: 1.4 }}>
          {payload.item_name || "Product"}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "0.68rem", color: T.textMuted }}>{payload.store || ""}</span>
        </div>

        <div style={{ display: "flex", gap: "1px", marginTop: "auto", paddingTop: "4px" }}>
          {[1, 2, 3, 4, 5].map(star => (
            <button
              key={star}
              className={`star-btn ${star <= fillUpTo ? "filled" : ""}`}
              onMouseEnter={() => setHover(star)}
              onMouseLeave={() => setHover(null)}
              onClick={() => handleVote(star)}
              title={`Rate ${star} star${star > 1 ? "s" : ""}`}
            >
              ★
            </button>
          ))}
          {vote !== null && (
            <span style={{ fontSize: "0.6rem", color: T.textMuted, marginLeft: "4px", alignSelf: "center" }}>
              {vote >= 4 ? "great" : vote === 3 ? "ok" : "poor"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULTS VIEW
// ══════════════════════════════════════════════════════════════════
function ResultsView({ results, categoryInfo, selectedItem, queryImageURL, judgeScores, radius, setRadius, userLocation, onFeedback, onReset }) {
  const [highlightedStore, setHighlightedStore] = useState(null);
  const userLL   = userLocation || [33.8869, 35.5131];
  const category = categoryInfo?.category || selectedItem?.search_label || "";

  const storeResults = {};
  results.forEach(r => {
    const store = r.payload?.store;
    if (store) {
      if (!storeResults[store]) storeResults[store] = [];
      storeResults[store].push(r);
    }
  });

  const storesInRadius = Object.keys(storeResults).filter(store => {
    const coords = STORE_COORDS[store] ?? userLL;
    return haversineKm(userLL, coords) <= radius;
  });

  const visibleResults = results.filter(r => !r.payload?.store || storesInRadius.includes(r.payload.store));
  const unfilteredResults = highlightedStore
    ? visibleResults.filter(r => r.payload?.store === highlightedStore)
    : visibleResults;

  // Re-sort by AI judge score once available; unjudged items stay at end in CLIP order
  const displayResults = [...unfilteredResults].sort((a, b) => {
    const flatA = a.payload ?? a;
    const flatB = b.payload ?? b;
    const pidA  = flatA.product_id || flatA.item_id || flatA.image_url;
    const pidB  = flatB.product_id || flatB.item_id || flatB.image_url;
    const sA    = judgeScores[pidA];
    const sB    = judgeScores[pidB];
    if (sA !== undefined && sB !== undefined) return sB - sA;
    if (sA !== undefined) return -1;
    if (sB !== undefined) return 1;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)", paddingBottom: "48px" }}>

      {/* ── Sticky header bar ── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 28px",
        borderBottom: `1px solid ${T.borderFaint}`,
        position: "sticky", top: "61px", zIndex: 100,
        background: `${T.bg}f0`, backdropFilter: "blur(10px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <button className="btn-ghost" onClick={onReset} style={{ fontSize: "1rem" }}>←</button>
          <div>
            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1rem", fontWeight: 600, color: T.text, textTransform: "capitalize" }}>
              {selectedItem?.label || category || "Results"}
            </div>
            <div style={{ fontSize: "0.68rem", color: T.textMuted }}>
              {displayResults.length} match{displayResults.length !== 1 ? "es" : ""}
              {storesInRadius.length > 0 && ` · ${storesInRadius.length} store${storesInRadius.length > 1 ? "s" : ""} nearby`}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "0.68rem", color: T.textMuted }}>radius</span>
          <input type="range" min={1} max={50} value={radius} onChange={e => setRadius(Number(e.target.value))} style={{ width: 80 }} />
          <span style={{ fontSize: "0.72rem", color: T.accent, minWidth: 36 }}>{radius} km</span>
        </div>
      </div>

      {/* ── Query image strip ── */}
      {queryImageURL && (
        <div style={{
          display: "flex", alignItems: "center", gap: "16px",
          padding: "10px 28px",
          borderBottom: `1px solid ${T.borderFaint}`,
          background: T.bgDeep,
        }}>
          <span style={{ fontSize: "0.62rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", flexShrink: 0 }}>
            You searched
          </span>
          <img
            src={queryImageURL}
            alt="your search crop"
            style={{
              height: 68,
              width: "auto",
              maxWidth: 100,
              objectFit: "cover",
              borderRadius: 8,
              border: `1.5px solid ${T.accent}`,
              flexShrink: 0,
            }}
          />
          <div style={{ fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.6 }}>
            <div style={{ color: T.text, fontWeight: 500, textTransform: "capitalize", marginBottom: 2 }}>
              {CATEGORY_LABELS[selectedItem?.search_label] || selectedItem?.label || "item"}
            </div>
            {categoryInfo?.confidence != null && (
              <div>{Math.round(categoryInfo.confidence * 100)}% confidence</div>
            )}
          </div>
        </div>
      )}

      <div style={{ padding: "24px 28px" }}>
        {/* Map */}
        <div style={{ borderRadius: 12, overflow: "hidden", border: `1px solid ${T.border}`, marginBottom: 28, height: 220 }}>
          <MapContainer center={userLL} zoom={11} style={{ height: "100%", width: "100%" }} zoomControl={false}>
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="&copy; CartoDB" />
            <MapRecenter center={userLL} />
            {storesInRadius.map(store => {
              const coords = STORE_COORDS[store] ?? userLL;
              return (
                <Marker key={store} position={coords} icon={makeStoreIcon(store === highlightedStore)} eventHandlers={{ click: () => setHighlightedStore(store === highlightedStore ? null : store) }}>
                  <Popup>
                    <div style={{ lineHeight: 1.7 }}>
                      <strong style={{ color: T.accent }}>{store}</strong><br />
                      {storeResults[store]?.length} matching item{storeResults[store]?.length !== 1 ? "s" : ""}
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>
        </div>

        {storesInRadius.length === 0 && (
          <div style={{ fontSize: "0.72rem", color: T.textMuted, textAlign: "center", marginTop: "10px" }}>
            No stores within {radius} km. Try increasing the radius.
          </div>
        )}

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
          <span style={{ fontSize: "0.7rem", color: T.textMuted, letterSpacing: "0.08em" }}>
            MATCHES
            {highlightedStore && (
              <span style={{ color: T.accent, marginLeft: "8px" }}>
                · {highlightedStore}
                <button onClick={() => setHighlightedStore(null)} style={{ background: "none", border: "none", color: T.textMuted, cursor: "pointer", marginLeft: "6px" }}>✕</button>
              </span>
            )}
          </span>
        </div>

        {displayResults.length === 0 ? (
          <div style={{ textAlign: "center", padding: "60px 0", fontSize: "0.8rem", color: T.textMuted, lineHeight: 2 }}>
            No results found.<br />
            <span style={{ fontSize: "0.72rem" }}>The database may be empty — index items first.</span>
          </div>
        ) : (
          <div className="results-grid">
            {displayResults.map((result) => {
              const flat = result.payload ?? result;
              const pid  = flat.product_id || flat.item_id || flat.image_url;
              return (
                <ResultCard
                  key={pid}
                  result={result}
                  category={category}
                  onFeedback={onFeedback}
                  highlighted={flat.store === highlightedStore}
                  judgeScore={judgeScores[pid] ?? null}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// ROOT
// ══════════════════════════════════════════════════════════════════
export default function App() {
  const [view,             setView]             = useState("landing");
  const [activeTab,        setActiveTab]        = useState("Discover");
  const [imageFile,        setImageFile]        = useState(null);
  const [imageURL,         setImageURL]         = useState(null);
  const [drawnBbox,        setDrawnBbox]        = useState(null);  // {x1,y1,x2,y2} natural px
  const [cropBlob,         setCropBlob]         = useState(null);  // Blob from DrawView crop
  const [predictedCategory,setPredictedCategory]= useState(null);
  const [allScores,        setAllScores]        = useState({});
  const [selectedItem,     setSelectedItem]     = useState(null);
  const [results,          setResults]          = useState([]);
  const [categoryInfo,     setCategoryInfo]     = useState(null);
  const [queryImageURL,    setQueryImageURL]    = useState(null);
  const [searchId,         setSearchId]         = useState(null);
  const [judgeScores,      setJudgeScores]      = useState({});   // {product_id: float}
  const [radius,           setRadius]           = useState(5);
  const [userLocation,     setUserLocation]     = useState(null);
  const [error,            setError]            = useState(null);

  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      pos => setUserLocation([pos.coords.latitude, pos.coords.longitude]),
      ()  => setUserLocation([33.8869, 35.5131])
    );
  }, []);

  // Poll for Groq judge scores after results are shown.
  // Judge runs in background on the server (~2.5s per result, top-5 only).
  // We poll every 3s, stop when all 5 are scored or after 8 attempts (~24s).
  useEffect(() => {
    if (view !== "results" || !searchId) return;
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 8) { clearInterval(interval); return; }
      try {
        const res = await fetch(`${API}/judge-scores/${searchId}`);
        if (!res.ok) return;
        const scores = await res.json();
        if (Object.keys(scores).length > 0) setJudgeScores(scores);
        if (Object.keys(scores).length >= 3) clearInterval(interval);
      } catch { /* non-critical */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [view, searchId]);

  const prepareImage = (file) => new Promise((resolve, reject) => {
    const MAX = 800;
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      let { naturalWidth: w, naturalHeight: h } = img;
      if (w > MAX || h > MAX) {
        if (w > h) { h = Math.round(h * MAX / w); w = MAX; }
        else        { w = Math.round(w * MAX / h); h = MAX; }
      }
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      canvas.toBlob(blob => {
        if (!blob) { reject(new Error("Canvas toBlob failed")); return; }
        resolve(new File([blob], "photo.jpg", { type: "image/jpeg" }));
      }, "image/jpeg", 0.88);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image load failed")); };
    img.src = url;
  });

  const handleUpload = useCallback(async (file) => {
    setError(null);
    let prepared;
    try { prepared = await prepareImage(file); } catch { prepared = file; }
    setImageFile(prepared);
    setImageURL(URL.createObjectURL(prepared));
    setView("drawing");
  }, []);

  // Called by DrawView once the user has drawn a box and CLIP has classified it
  const handleDrawConfirm = useCallback(({ bbox, cropBlob: blob, category, allScores: scores }) => {
    setDrawnBbox(bbox);
    setCropBlob(blob);
    setPredictedCategory(category);
    setAllScores(scores || {});
    setView("confirming");
  }, []);

  // Called by ConfirmView once the user has approved / corrected the category
  const handleConfirm = useCallback(async (confirmedCategory) => {
    setView("searching");
    const { x1, y1, x2, y2 } = drawnBbox;

    // Turn the crop blob into an object URL for the results strip
    if (cropBlob) {
      setQueryImageURL(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(cropBlob); });
    }
    setSelectedItem({ search_label: confirmedCategory, label: CATEGORY_LABELS[confirmedCategory] || confirmedCategory, bbox: [x1, y1, x2, y2] });

    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", x1); form.append("y1", y1);
    form.append("x2", x2); form.append("y2", y2);
    form.append("search_label", confirmedCategory);
    try {
      const res = await fetch(`${API}/search`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const reshaped = (data.matches || data.results || []).map(m => ({
        score: m.score,
        payload: {
          store:      m.store      || m.store_name,
          image_url:  m.image_url  || m.image_filename,
          item_name:  m.name,
          product_id: m.product_id,
          item_id:    m.item_id,
        }
      }));
      const seen = new Map();
      for (const r of reshaped) {
        const id = r.payload.product_id || r.payload.item_id || r.payload.image_url;
        if (!seen.has(id) || r.score > seen.get(id).score) seen.set(id, r);
      }
      const deduped = Array.from(seen.values()).sort((a, b) => b.score - a.score);
      setResults(deduped);
      const detCat  = data.detected_category;
      const confMap = data.category_confidence;
      const confVal = (confMap && typeof confMap === "object") ? (confMap[detCat] ?? 0) : 0;
      setCategoryInfo({ category: detCat, confidence: confVal });
      setSearchId(data.search_id || null);
      setJudgeScores({});
      setView("results");
    } catch (e) {
      setError(`Search failed: ${e.message}`);
      setView("confirming");
    }
  }, [imageFile, drawnBbox, cropBlob]);

  const handleFeedback = useCallback(async (productId, rating, name, store, category, imageUrl) => {
    try {
      await fetch(`${API}/feedback`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          result_product_id: productId || "",
          result_image_url:  imageUrl  || "",
          result_name:       name      || "",
          store_name:        store     || "",
          category:          category  || "",
          rating,
        }),
      });
    } catch { /* non-critical */ }
  }, []);

  const reset = useCallback(() => {
    setView("landing");
    setImageFile(null);
    if (imageURL)      URL.revokeObjectURL(imageURL);
    if (queryImageURL) URL.revokeObjectURL(queryImageURL);
    setImageURL(null);
    setQueryImageURL(null);
    setDrawnBbox(null);
    setCropBlob(null);
    setPredictedCategory(null);
    setAllScores({});
    setResults([]);
    setSelectedItem(null);
    setCategoryInfo(null);
    setSearchId(null);
    setJudgeScores({});
    setError(null);
  }, [imageURL, queryImageURL]);

  return (
    <>
      <StyleInjector />
      <Navbar activeTab={activeTab} onTab={setActiveTab} onLogoClick={reset} />

      {activeTab === "Store" ? (
        <StoreDashboardView />
      ) : (
        <>
          {view === "landing"    && <LandingView onUpload={handleUpload} error={error} />}
          {view === "drawing"    && <DrawView imageURL={imageURL} imageFile={imageFile} onConfirm={handleDrawConfirm} onBack={reset} />}
          {view === "confirming" && <ConfirmView cropBlob={cropBlob} predictedCategory={predictedCategory} allScores={allScores} onSearch={handleConfirm} onBack={() => setView("drawing")} />}
          {view === "searching"  && <LoadingView label="Finding matches…" />}
          {view === "results"    && (
            <ResultsView
              results={results}
              categoryInfo={categoryInfo}
              selectedItem={selectedItem}
              queryImageURL={queryImageURL}
              judgeScores={judgeScores}
              radius={radius}
              setRadius={setRadius}
              userLocation={userLocation}
              onFeedback={handleFeedback}
              onReset={reset}
            />
          )}
        </>
      )}
    </>
  );
}