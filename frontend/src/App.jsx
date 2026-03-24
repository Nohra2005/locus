import { useState, useRef, useEffect, useCallback } from "react";
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

  .detection-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 14px 18px;
    background: ${T.surface};
    border: 1px solid ${T.border};
    border-radius: 12px;
    color: ${T.text};
    cursor: pointer;
    text-align: left;
    font-family: 'DM Sans', sans-serif;
    transition: border-color 0.2s, background 0.2s;
  }
  .detection-card:hover {
    border-color: ${T.accent};
    background: ${T.accentBg};
  }

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
// SELECT VIEW
// ══════════════════════════════════════════════════════════════════
function SelectView({ imageURL, detections, onSelect, onBack }) {
  const imgRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [imgNatural, setImgNatural] = useState({ w: 1, h: 1 });
  const [imgRendered, setImgRendered] = useState({ w: 0, h: 0, x: 0, y: 0 });

  const onImgLoad = (e) => {
    const el = e.target;
    setImgNatural({ w: el.naturalWidth, h: el.naturalHeight });
    const { width: cw, height: ch } = el.getBoundingClientRect();
    const scale = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
    setImgRendered({ w: el.naturalWidth * scale, h: el.naturalHeight * scale, x: (cw - el.naturalWidth * scale) / 2, y: (ch - el.naturalHeight * scale) / 2 });
  };

  useEffect(() => {
    const handleResize = () => {
      const el = imgRef.current;
      if (!el || !el.naturalWidth) return;
      const { width: cw, height: ch } = el.getBoundingClientRect();
      const scale = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
      setImgRendered({ w: el.naturalWidth * scale, h: el.naturalHeight * scale, x: (cw - el.naturalWidth * scale) / 2, y: (ch - el.naturalHeight * scale) / 2 });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const scale = (x, y, x2, y2) => {
    const sw = imgRendered.w / imgNatural.w;
    const sh = imgRendered.h / imgNatural.h;
    return { x: imgRendered.x + x * sw, y: imgRendered.y + y * sh, w: (x2 - x) * sw, h: (y2 - y) * sh };
  };

  const COLORS = ["#c9a96e", "#6eb5c9", "#c96e8a", "#6e9bc9", "#8ac96e"];

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 28px", borderBottom: `1px solid ${T.borderFaint}` }}>
        <button className="btn-ghost" onClick={onBack} style={{ fontSize: "1rem" }}>←</button>
        <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.08em" }}>TAP A BOX TO SEARCH THAT ITEM</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "24px", position: "relative" }}>
        <div style={{ position: "relative", width: "100%", maxWidth: 640 }}>
          <img ref={imgRef} src={imageURL} onLoad={onImgLoad} alt="uploaded" style={{ width: "100%", display: "block", borderRadius: 12 }} />
          {imgRendered.w > 0 && (
            <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
              {detections.map((det, i) => {
                const [x1, y1, x2, y2] = det.bbox;
                const b = scale(x1, y1, x2, y2);
                const color = COLORS[i % COLORS.length];
                return (
                  <g key={i}>
                    <rect x={b.x} y={b.y} width={b.w} height={b.h} fill={hovered === i ? `${color}20` : "transparent"} stroke={color} strokeWidth={hovered === i ? 2.5 : 1.5} rx={4} style={{ pointerEvents: "all", cursor: "pointer" }} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)} onClick={() => onSelect(det)} />
                    <rect x={b.x} y={b.y - 22} width={Math.min((det.search_label || "").length * 7 + 16, 120)} height={20} fill={color} rx={3} />
                    <text x={b.x + 8} y={b.y - 8} fill={T.bg} fontSize={10} fontFamily="DM Sans, sans-serif" fontWeight={500}>{(det.search_label || "item").toUpperCase()}</text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        <div style={{ width: "100%", maxWidth: 640, marginTop: 20, display: "flex", flexDirection: "column", gap: 10 }}>
          {detections.map((det, i) => {
            const color = COLORS[i % COLORS.length];
            return (
              <button key={i} className="detection-card" style={{ borderColor: hovered === i ? color : T.border, background: hovered === i ? `${color}10` : T.surface }} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)} onClick={() => onSelect(det)}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <div style={{ width: 10, height: 10, borderRadius: "2px", background: color, flexShrink: 0 }} />
                  <div>
                    <div style={{ fontSize: "0.88rem", fontWeight: 500, textTransform: "capitalize" }}>{det.search_label || det.label || "Item"}</div>
                    {det.score != null && <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: "2px" }}>confidence {Math.round(det.score * 100)}%</div>}
                  </div>
                </div>
                <span style={{ color: T.accent }}>→</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULT CARD
// ══════════════════════════════════════════════════════════════════
function ResultCard({ result, category, onFeedback, highlighted }) {
  const [vote, setVote]       = useState(null);
  const [hoverStar, setHover] = useState(null);

  const score    = result.score ?? 0;
  const payload  = result.payload ?? {};
  const raw      = payload.image_url ?? "";
  const imageURL = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;

  const scoreColor = score >= 0.80 ? T.green : score >= 0.60 ? T.yellow : T.red;

  const handleVote = async (stars) => {
    if (vote !== null) return;
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
      <div style={{ aspectRatio: "3/4", background: T.bgDeep, overflow: "hidden" }}>
        {imageURL
          ? <img src={imageURL} alt={payload.item_name || "product"} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.5rem" }}>✦</div>
        }
      </div>

      <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: "8px", flex: 1 }}>
        <div style={{ fontSize: "0.78rem", fontWeight: 500, color: T.text, lineHeight: 1.4 }}>
          {payload.item_name || "Product"}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "0.68rem", color: T.textMuted }}>{payload.store || ""}</span>
          <span style={{ fontSize: "0.72rem", fontWeight: 600, color: scoreColor }}>{Math.round(score * 100)}%</span>
        </div>

        <div
          style={{ display: "flex", gap: "1px", marginTop: "auto", paddingTop: "4px" }}
          onMouseLeave={() => vote === null && setHover(null)}
        >
          {[1, 2, 3, 4, 5].map(star => (
            <button
              key={star}
              className={`star-btn ${star <= fillUpTo ? "filled" : vote !== null ? "dimmed" : ""}`}
              disabled={vote !== null}
              onMouseEnter={() => vote === null && setHover(star)}
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
function ResultsView({ results, categoryInfo, selectedItem, queryImageURL, radius, setRadius, userLocation, onFeedback, onReset }) {
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
  const displayResults = highlightedStore
    ? visibleResults.filter(r => r.payload?.store === highlightedStore)
    : visibleResults;

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
              {selectedItem?.search_label || selectedItem?.label || "item"}
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
            {displayResults.map((result, i) => (
              <ResultCard
                key={i}
                result={result}
                category={category}
                onFeedback={onFeedback}
                highlighted={result.payload?.store === highlightedStore}
              />
            ))}
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
  const [view,          setView]          = useState("landing");
  const [activeTab,     setActiveTab]     = useState("Discover");
  const [imageFile,     setImageFile]     = useState(null);
  const [imageURL,      setImageURL]      = useState(null);
  const [detections,    setDetections]    = useState([]);
  const [selectedItem,  setSelectedItem]  = useState(null);
  const [results,       setResults]       = useState([]);
  const [categoryInfo,  setCategoryInfo]  = useState(null);
  const [queryImageURL, setQueryImageURL] = useState(null);  // cropped bbox shown in results
  const [radius,        setRadius]        = useState(5);
  const [userLocation,  setUserLocation]  = useState(null);
  const [error,         setError]         = useState(null);

  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      pos => setUserLocation([pos.coords.latitude, pos.coords.longitude]),
      ()  => setUserLocation([33.8869, 35.5131])
    );
  }, []);

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
    setView("detecting");
    setError(null);
    let prepared;
    try { prepared = await prepareImage(file); } catch { prepared = file; }
    setImageFile(prepared);
    setImageURL(URL.createObjectURL(prepared));
    const form = new FormData();
    form.append("file", prepared);
    try {
      const res = await fetch(`${API}/detect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDetections(data.detections || []);
      setView("selecting");
    } catch (e) {
      setError(`Detection failed: ${e.message}. Is the gateway running on port 8000?`);
      setView("landing");
    }
  }, []);

  const handleSelect = useCallback(async (detection) => {
    setSelectedItem(detection);
    setView("searching");
    const [x1, y1, x2, y2] = detection.bbox;

    // ── Crop query image for display in results ───────────────────────────
    try {
      const bitmap = await createImageBitmap(imageFile);
      const cw = Math.max(1, Math.round(x2 - x1));
      const ch = Math.max(1, Math.round(y2 - y1));
      const canvas = document.createElement("canvas");
      canvas.width  = cw;
      canvas.height = ch;
      canvas.getContext("2d").drawImage(bitmap, x1, y1, cw, ch, 0, 0, cw, ch);
      canvas.toBlob(blob => {
        if (blob) {
          setQueryImageURL(prev => {
            if (prev) URL.revokeObjectURL(prev);
            return URL.createObjectURL(blob);
          });
        }
      }, "image/jpeg", 0.9);
    } catch { /* non-critical */ }
    // ─────────────────────────────────────────────────────────────────────

    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", x1); form.append("y1", y1);
    form.append("x2", x2); form.append("y2", y2);
    if (detection.search_label || detection.label) {
      form.append("search_label", detection.search_label || detection.label);
    }
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
      setCategoryInfo({ category: data.detected_category, confidence: data.category_confidence });
      setView("results");
    } catch (e) {
      setError(`Search failed: ${e.message}`);
      setView("selecting");
    }
  }, [imageFile]);

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
    setDetections([]);
    setResults([]);
    setSelectedItem(null);
    setCategoryInfo(null);
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
          {view === "landing"   && <LandingView onUpload={handleUpload} error={error} />}
          {view === "detecting" && <LoadingView label="Analyzing image…" />}
          {view === "selecting" && (
            <SelectView imageURL={imageURL} detections={detections} onSelect={handleSelect} onBack={reset} />
          )}
          {view === "searching" && <LoadingView label="Finding matches…" />}
          {view === "results"   && (
            <ResultsView
              results={results}
              categoryInfo={categoryInfo}
              selectedItem={selectedItem}
              queryImageURL={queryImageURL}
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