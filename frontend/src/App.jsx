import { useState, useRef, useEffect, useCallback } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ── Fix Leaflet marker icons broken by Vite ────────────────────────
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:       "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

// ── Proxy handles routing — no need for absolute URL ──────────────
const API = "";

// ── Demo store GPS coords (ABC Achrafieh area) ─────────────────────
const STORE_COORDS = {
  "Zara":          [33.88690, 35.51310],
  "Bershka":       [33.88720, 35.51350],
  "Mike Sport":    [33.88650, 35.51280],
  "Louis Vuitton": [33.88750, 35.51400],
  "Virgin":        [33.88600, 35.51250],
};

// ── Design tokens — matching your existing locus aesthetic ─────────
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
  accent:      "#c9a96e",   // warm gold
  accentBg:    "rgba(201,169,110,0.08)",
  accentRing:  "rgba(201,169,110,0.2)",
  accentDeep:  "#a8895a",
  green:       "#7aab8a",
  yellow:      "#c9a96e",
  red:         "#c97070",
};

// ── Global CSS ─────────────────────────────────────────────────────
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

  /* Leaflet overrides */
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

  .fb-btn {
    background: none;
    border: 1px solid ${T.border};
    border-radius: 6px;
    color: ${T.textMuted};
    cursor: pointer;
    font-size: 0.75rem;
    padding: 4px 10px;
    transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
  }
  .fb-btn:hover           { border-color: ${T.accent}; color: ${T.accent}; }
  .fb-btn.voted-up        { background: rgba(122,171,138,0.12); border-color: ${T.green}; color: ${T.green}; }
  .fb-btn.voted-down      { background: rgba(201,112,112,0.12); border-color: ${T.red};   color: ${T.red};   }
  .fb-btn:disabled        { cursor: default; }

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

// ── Leaflet map re-centering ───────────────────────────────────────
function MapRecenter({ center }) {
  const map = useMap();
  useEffect(() => { map.setView(center, map.getZoom()); }, [center, map]);
  return null;
}

// ── Custom warm store pin ──────────────────────────────────────────
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

// ── Haversine distance ─────────────────────────────────────────────
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
      {/* Logo */}
      <button
        onClick={onLogoClick}
        style={{
          display: "flex", alignItems: "center", gap: "10px",
          background: "none", border: "none", cursor: "pointer",
        }}
      >
        <div style={{
          width: 32, height: 32,
          background: `linear-gradient(135deg, ${T.accent}, ${T.accentDeep})`,
          borderRadius: "8px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.75rem",
          boxShadow: `0 2px 12px ${T.accentRing}`,
        }}>
          ✦
        </div>
        <span style={{
          fontFamily: "'Cormorant Garamond', serif",
          fontSize: "1.25rem",
          fontWeight: 600,
          color: T.text,
          letterSpacing: "0.02em",
        }}>
          locus
        </span>
        <span style={{
          fontSize: "0.65rem",
          color: T.textMuted,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          paddingLeft: "2px",
        }}>
          shopping made easier
        </span>
      </button>

      {/* Nav links */}
      <div style={{ display: "flex", gap: "4px" }}>
        {["Discover", "Saved", "History"].map(tab => (
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
    <div style={{
      minHeight: "calc(100dvh - 61px)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "40px 24px",
    }}>
      {/* Hero */}
      <div className="fade-up" style={{ textAlign: "center", marginBottom: "48px" }}>
        <div style={{
          fontSize: "0.7rem",
          color: T.accent,
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          marginBottom: "20px",
          display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
        }}>
          <span style={{ fontSize: "0.6rem" }}>✦</span>
          AI-powered visual search
        </div>

        <h1 style={{
          fontFamily: "'Cormorant Garamond', serif",
          fontSize: "clamp(2.8rem, 8vw, 4.5rem)",
          fontWeight: 500,
          lineHeight: 1.1,
          color: T.text,
          letterSpacing: "-0.01em",
        }}>
          Find what you{" "}
          <em style={{ color: T.accent, fontStyle: "italic" }}>see</em>
        </h1>

        <p style={{
          marginTop: "18px",
          fontSize: "0.88rem",
          color: T.textMuted,
          lineHeight: 1.7,
          maxWidth: "400px",
          margin: "18px auto 0",
        }}>
          Upload any photo and we'll match it against thousands of
          products across top stores.
        </p>
      </div>

      {/* Upload zone */}
      <div
        className={`upload-zone fade-up ${dragging ? "drag-over" : ""}`}
        style={{
          animationDelay: "0.1s",
          width: "100%",
          maxWidth: "540px",
          padding: "56px 32px",
          textAlign: "center",
        }}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
      >
        {/* Upload icon */}
        <div style={{
          width: 56, height: 56,
          borderRadius: "50%",
          background: T.surfaceHov,
          border: `1px solid ${T.border}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          margin: "0 auto 20px",
          fontSize: "1.2rem",
        }}>
          ↑
        </div>

        <div style={{ fontSize: "0.95rem", fontWeight: 500, color: T.text, marginBottom: "8px" }}>
          Drop your photo here
        </div>
        <div style={{ fontSize: "0.75rem", color: T.textMuted, marginBottom: "24px" }}>
          PNG · JPG · WEBP — we'll find matching products instantly
        </div>

        <button
          className="btn-primary"
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
        >
          <span style={{ color: T.accent }}>✦</span>
          Start searching
        </button>

        <input
          ref={inputRef}
          type="file"
          // explicitly list formats so mobile OS knows to open the full gallery
          accept="image/jpeg, image/png, image/webp, image/heic, image/*"
          // visually hidden, but NOT display: none
          style={{ 
            position: "absolute", 
            width: "1px", 
            height: "1px", 
            opacity: 0, 
            pointerEvents: "none" 
          }}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
              // Reset the input value so you can upload the same file twice if needed
              e.target.value = null; 
            }
          }}
        />
      </div>

      {error && (
        <div className="fade-up" style={{
          marginTop: "20px",
          padding: "12px 18px",
          background: "rgba(201,112,112,0.08)",
          border: `1px solid rgba(201,112,112,0.2)`,
          borderRadius: "10px",
          fontSize: "0.75rem",
          color: T.red,
          maxWidth: "540px",
          width: "100%",
        }}>
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
    <div style={{
      minHeight: "calc(100dvh - 61px)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: "16px",
    }}>
      <div style={{
        width: 28, height: 28,
        border: `1.5px solid ${T.border}`,
        borderTop: `1.5px solid ${T.accent}`,
        borderRadius: "50%",
        animation: "spin 0.9s linear infinite",
      }} />
      <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.05em" }}>
        {label}
      </span>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// SELECT VIEW  — image + SVG bbox overlay
// ══════════════════════════════════════════════════════════════════
function SelectView({ imageURL, detections, onSelect, onBack }) {
  const imgRef        = useRef(null);
  const [hovered, setHovered]   = useState(null);   // index of hovered box
  const [imgNatural, setImgNatural] = useState({ w: 1, h: 1 }); // original px
  const [imgRendered, setImgRendered] = useState({ w: 0, h: 0, x: 0, y: 0 }); // on-screen

  // Once image loads, record natural size AND calculate dimensions
  const onImgLoad = (e) => {
    const el = e.target;
    setImgNatural({ w: el.naturalWidth, h: el.naturalHeight });
    
    // Measure the image to fit the bounding boxes perfectly
    const { width: cw, height: ch } = el.getBoundingClientRect();
    const scale = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
    setImgRendered({ 
      w: el.naturalWidth * scale, 
      h: el.naturalHeight * scale, 
      x: (cw - (el.naturalWidth * scale)) / 2, 
      y: (ch - (el.naturalHeight * scale)) / 2 
    });
  };

  // Handle window resizing without triggering React Compiler warnings
  useEffect(() => {
    const handleResize = () => {
      const el = imgRef.current;
      if (!el || !el.naturalWidth) return;
      
      const { width: cw, height: ch } = el.getBoundingClientRect();
      const scale = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
      setImgRendered({ 
        w: el.naturalWidth * scale, 
        h: el.naturalHeight * scale, 
        x: (cw - (el.naturalWidth * scale)) / 2, 
        y: (ch - (el.naturalHeight * scale)) / 2 
      });
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []); // Empty array means it binds once cleanly

  // Scale a bbox coord from original px → rendered px
  const scale = (x, y, x2, y2) => {
    const sw = imgRendered.w / imgNatural.w;
    const sh = imgRendered.h / imgNatural.h;
    return {
      x:  imgRendered.x + x  * sw,
      y:  imgRendered.y + y  * sh,
      w:  (x2 - x) * sw,
      h:  (y2 - y) * sh,
    };
  };

  // Box colours cycling (gold, teal, rose, sky, lime)
  const COLORS = ["#c9a96e", "#6eb5c9", "#c96e8a", "#6e9bc9", "#8ac96e"];

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)" }}>
      {/* Sub-header */}
      <div style={{
        display: "flex", alignItems: "center", gap: "12px",
        padding: "16px 28px",
        borderBottom: `1px solid ${T.borderFaint}`,
      }}>
        <button className="btn-ghost" onClick={onBack} style={{ fontSize: "1rem" }}>←</button>
        <span style={{ fontSize: "0.75rem", color: T.textMuted, letterSpacing: "0.08em" }}>
          TAP A BOX TO SEARCH THAT ITEM
        </span>
      </div>

      <div style={{ maxWidth: "720px", margin: "0 auto" }}>
        {/* ── Image + SVG overlay ── */}
        <div style={{
          position: "relative",
          width: "100%",
          background: T.bgDeep,
          lineHeight: 0,          // removes extra gap under img
        }}>
          <img
            ref={imgRef}
            src={imageURL}
            alt="Uploaded"
            onLoad={onImgLoad}
            style={{
              width: "100%",
              maxHeight: "55dvh",
              objectFit: "contain",
              display: "block",
            }}
          />

          {/* SVG sits exactly over the image container */}
          {imgRendered.w > 0 && (
            <svg
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",   // boxes below handle their own events
              }}
            >
              {detections.map((det, i) => {
                const [x1, y1, x2, y2] = det.bbox;
                const b = scale(x1, y1, x2, y2);
                const color = COLORS[i % COLORS.length];
                const isHov = hovered === i;

                return (
                  <g key={i} style={{ pointerEvents: "all", cursor: "pointer" }}
                    onClick={() => onSelect(det)}
                    onMouseEnter={() => setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    {/* Dim overlay on unhovered boxes */}
                    {hovered !== null && !isHov && (
                      <rect
                        x={b.x} y={b.y} width={b.w} height={b.h}
                        fill="rgba(0,0,0,0.35)"
                        rx={3}
                      />
                    )}

                    {/* Fill flash on hover */}
                    <rect
                      x={b.x} y={b.y} width={b.w} height={b.h}
                      fill={isHov ? `${color}18` : "transparent"}
                      rx={3}
                      style={{ transition: "fill 0.15s" }}
                    />

                    {/* Border */}
                    <rect
                      x={b.x} y={b.y} width={b.w} height={b.h}
                      fill="none"
                      stroke={color}
                      strokeWidth={isHov ? 2.5 : 1.5}
                      strokeDasharray={isHov ? "none" : "6 3"}
                      rx={3}
                      style={{ transition: "stroke-width 0.15s" }}
                    />

                    {/* Label pill */}
                    <rect
                      x={b.x} y={b.y - 22}
                      width={Math.min((det.label || det.search_label || "item").length * 7.5 + 16, b.w)}
                      height={20}
                      fill={color}
                      rx={4}
                    />
                    <text
                      x={b.x + 8}
                      y={b.y - 7}
                      fill="#0d0c0a"
                      fontSize="10"
                      fontFamily="DM Sans, sans-serif"
                      fontWeight="600"
                    >
                      {det.label || det.search_label || `item ${i + 1}`}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {/* ── Detection list below image ── */}
        <div style={{ padding: "20px 28px 32px" }}>
          {detections.length === 0 ? (
            <div style={{
              textAlign: "center", padding: "48px 0",
              fontSize: "0.8rem", color: T.textMuted, lineHeight: 2,
            }}>
              No clothing detected.<br />
              <span style={{ fontSize: "0.7rem" }}>Try a clearer photo with the item centered.</span>
            </div>
          ) : (
            <>
              <div style={{ fontSize: "0.7rem", color: T.textMuted, marginBottom: "14px", letterSpacing: "0.05em" }}>
                {detections.length} item{detections.length > 1 ? "s" : ""} detected
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {detections.map((det, i) => {
                  const color = COLORS[i % COLORS.length];
                  return (
                    <button
                      key={i}
                      className="detection-card"
                      onClick={() => onSelect(det)}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      style={{
                        borderColor: hovered === i ? color : T.border,
                        background:  hovered === i ? `${color}10` : T.surface,
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        {/* Color swatch matching the box */}
                        <div style={{
                          width: 10, height: 10,
                          borderRadius: "2px",
                          background: color,
                          flexShrink: 0,
                        }} />
                        <div>
                          <div style={{ fontSize: "0.88rem", fontWeight: 500, textTransform: "capitalize" }}>
                            {det.label || det.search_label || "Item"}
                          </div>
                          {det.score != null && (
                            <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: "2px" }}>
                              confidence {Math.round(det.score * 100)}%
                            </div>
                          )}
                        </div>
                      </div>
                      <span style={{ color: T.accent }}>→</span>
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULT CARD
// ══════════════════════════════════════════════════════════════════
function ResultCard({ result, onFeedback, highlighted }) {
  const [vote, setVote] = useState(null);
  const score   = result.score ?? 0;
  const payload = result.payload ?? {};
  const raw     = payload.image_url ?? "";
  const imageURL = raw
    ? (raw.startsWith("http") ? raw : `${API}${raw}`)
    : null;

  const scoreColor =
    score >= 0.80 ? T.green :
    score >= 0.60 ? T.yellow :
                    T.red;

  const handleVote = async (rating) => {
    if (vote !== null) return;
    setVote(rating);
    if (payload.item_id) await onFeedback(payload.item_id, rating);
  };

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
      {/* Image */}
      <div style={{ aspectRatio: "3/4", background: T.bgDeep, overflow: "hidden" }}>
        {imageURL ? (
          <img
            src={imageURL}
            alt={payload.item_name || "Product"}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform 0.3s" }}
            onMouseEnter={e => e.target.style.transform = "scale(1.04)"}
            onMouseLeave={e => e.target.style.transform = "scale(1)"}
            onError={e => { e.target.style.display = "none"; }}
          />
        ) : (
          <div style={{
            width: "100%", height: "100%",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: T.textFaint, fontSize: "1.5rem",
          }}>✦</div>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: "10px 12px", flex: 1, display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ fontSize: "0.78rem", fontWeight: 500, color: T.text }}>
          {payload.store || "—"}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ fontSize: "0.7rem", color: scoreColor, fontWeight: 500 }}>
            {Math.round(score * 100)}% match
          </span>
        </div>

        {score < 0.60 && (
          <div style={{
            padding: "3px 7px",
            background: "rgba(201,112,112,0.08)",
            border: "1px solid rgba(201,112,112,0.2)",
            borderRadius: "4px",
            fontSize: "0.6rem",
            color: T.red,
          }}>
            ⚠ LOW MATCH
          </div>
        )}

        <div style={{ display: "flex", gap: "6px", marginTop: "auto", paddingTop: "6px" }}>
          <button
            className={`fb-btn ${vote === 1 ? "voted-up" : ""}`}
            onClick={() => handleVote(1)}
            disabled={vote !== null}
          >👍</button>
          <button
            className={`fb-btn ${vote === -1 ? "voted-down" : ""}`}
            onClick={() => handleVote(-1)}
            disabled={vote !== null}
          >👎</button>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULTS VIEW
// ══════════════════════════════════════════════════════════════════
function ResultsView({ results, categoryInfo, selectedItem, imageURL, radius, setRadius, userLocation, onFeedback, onReset }) {
  const [highlightedStore, setHighlightedStore] = useState(null);
  const userLL = userLocation || [33.8869, 35.5131];

  // Group results by store
  const storeResults = {};
  results.forEach(r => {
    const store = r.payload?.store;
    if (store) {
      if (!storeResults[store]) storeResults[store] = [];
      storeResults[store].push(r);
    }
  });

  // Filter stores within radius
  const storesInRadius = Object.keys(storeResults).filter(store => {
    const coords = STORE_COORDS[store] ?? userLL;
    return haversineKm(userLL, coords) <= radius;
  });

  const visibleResults = results.filter(r =>
    !r.payload?.store || storesInRadius.includes(r.payload.store)
  );

  const displayResults = highlightedStore
    ? visibleResults.filter(r => r.payload?.store === highlightedStore)
    : visibleResults;

  return (
    <div className="fade-in" style={{ minHeight: "calc(100dvh - 61px)", paddingBottom: "48px" }}>
      {/* Sub-header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 28px",
        borderBottom: `1px solid ${T.borderFaint}`,
        position: "sticky", top: "61px", zIndex: 100,
        background: `${T.bg}f0`,
        backdropFilter: "blur(10px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <button className="btn-ghost" onClick={onReset} style={{ fontSize: "1rem" }}>←</button>
          <div>
            <div style={{
              fontFamily: "'Cormorant Garamond', serif",
              fontSize: "1rem", fontWeight: 600, color: T.text,
              textTransform: "capitalize",
            }}>
              {selectedItem?.label || categoryInfo?.category || "Results"}
            </div>
            <div style={{ fontSize: "0.68rem", color: T.textMuted }}>
              {displayResults.length} match{displayResults.length !== 1 ? "es" : ""}
              {storesInRadius.length > 0 && ` · ${storesInRadius.length} store${storesInRadius.length > 1 ? "s" : ""}`}
            </div>
          </div>
        </div>

        {imageURL && (
          <img src={imageURL} style={{
            width: 38, height: 38,
            borderRadius: "6px",
            objectFit: "cover",
            border: `1px solid ${T.border}`,
          }} />
        )}
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 28px 0" }}>
        {/* Category confidence warning */}
        {categoryInfo?.confidence != null && categoryInfo.confidence < 0.60 && (
          <div style={{
            marginBottom: "20px",
            padding: "10px 16px",
            background: T.accentBg,
            border: `1px solid ${T.accentRing}`,
            borderRadius: "10px",
            fontSize: "0.72rem",
            color: T.accent,
            display: "flex", alignItems: "center", gap: "10px",
          }}>
            <span>✦</span>
            Low category confidence ({Math.round(categoryInfo.confidence * 100)}%) — results may be mixed
          </div>
        )}

        {/* Map */}
        <div style={{ marginBottom: "28px" }}>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            marginBottom: "12px",
          }}>
            <span style={{ fontSize: "0.7rem", color: T.textMuted, letterSpacing: "0.08em" }}>
              NEARBY STORES
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "0.7rem", color: T.textMuted }}>Radius</span>
              <input
                type="range" min={1} max={15} step={0.5}
                value={radius}
                onChange={e => setRadius(Number(e.target.value))}
                style={{ width: 90 }}
              />
              <span style={{ fontSize: "0.72rem", color: T.accent, minWidth: "36px" }}>
                {radius} km
              </span>
            </div>
          </div>

          <div style={{
            height: "240px",
            borderRadius: "14px",
            overflow: "hidden",
            border: `1px solid ${T.border}`,
          }}>
            <MapContainer center={userLL} zoom={14} style={{ height: "100%", width: "100%" }} zoomControl={true}>
              <MapRecenter center={userLL} />
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              />
              <Circle
                center={userLL}
                radius={radius * 1000}
                pathOptions={{
                  color: T.accent,
                  fillColor: T.accent,
                  fillOpacity: 0.05,
                  weight: 1,
                  dashArray: "5 5",
                }}
              />
              {storesInRadius.map(store => (
                <Marker
                  key={store}
                  position={STORE_COORDS[store] ?? userLL}
                  icon={makeStoreIcon(store === highlightedStore)}
                  eventHandlers={{
                    click: () => setHighlightedStore(store === highlightedStore ? null : store),
                  }}
                >
                  <Popup>
                    <div style={{ lineHeight: 1.7 }}>
                      <strong style={{ color: T.accent }}>{store}</strong><br />
                      {storeResults[store]?.length} matching item{storeResults[store]?.length !== 1 ? "s" : ""}
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </div>

          {storesInRadius.length === 0 && (
            <div style={{ fontSize: "0.72rem", color: T.textMuted, textAlign: "center", marginTop: "10px" }}>
              No stores within {radius} km. Try increasing the radius.
            </div>
          )}
        </div>

        {/* Results grid */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          marginBottom: "14px",
        }}>
          <span style={{ fontSize: "0.7rem", color: T.textMuted, letterSpacing: "0.08em" }}>
            MATCHES
            {highlightedStore && (
              <span style={{ color: T.accent, marginLeft: "8px" }}>
                · {highlightedStore}
                <button
                  onClick={() => setHighlightedStore(null)}
                  style={{ background: "none", border: "none", color: T.textMuted, cursor: "pointer", marginLeft: "6px" }}
                >✕</button>
              </span>
            )}
          </span>
        </div>

        {displayResults.length === 0 ? (
          <div style={{
            textAlign: "center", padding: "60px 0",
            fontSize: "0.8rem", color: T.textMuted, lineHeight: 2,
          }}>
            No results found.<br />
            <span style={{ fontSize: "0.72rem" }}>The database may be empty — index items first.</span>
          </div>
        ) : (
          <div className="results-grid">
            {displayResults.map((result, i) => (
              <ResultCard
                key={i}
                result={result}
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
  const [view,         setView]         = useState("landing");
  const [activeTab,    setActiveTab]    = useState("Discover");
  const [imageFile,    setImageFile]    = useState(null);
  const [imageURL,     setImageURL]     = useState(null);
  const [detections,   setDetections]   = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [results,      setResults]      = useState([]);
  const [categoryInfo, setCategoryInfo] = useState(null);
  const [radius,       setRadius]       = useState(5);
  const [userLocation, setUserLocation] = useState(null);
  const [error,        setError]        = useState(null);

  // Request GPS on mount
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      pos => setUserLocation([pos.coords.latitude, pos.coords.longitude]),
      ()  => setUserLocation([33.8869, 35.5131]) // fallback: ABC Achrafieh
    );
  }, []);

  // ── Image preprocessing ────────────────────────────────────────
  // Fixes two mobile problems at once:
  //   1. EXIF orientation  — phone cameras store pixels rotated; drawing
  //      to canvas makes the browser bake in the correct rotation so the
  //      server always gets an upright image → correct YOLO detections
  //      and correct bbox coordinates.
  //   2. Resize to ≤ 800px — phones take 12–50 MP photos; YOLO only
  //      needs 640px. Smaller upload = 60-80% less transfer + processing.
  const prepareImage = (file) => new Promise((resolve, reject) => {
    const MAX = 800; // px on longest side — enough for YOLO accuracy
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);

      // Compute target size preserving aspect ratio
      let { naturalWidth: w, naturalHeight: h } = img;
      if (w > MAX || h > MAX) {
        if (w > h) { h = Math.round(h * MAX / w); w = MAX; }
        else        { w = Math.round(w * MAX / h); h = MAX; }
      }

      const canvas = document.createElement("canvas");
      canvas.width  = w;
      canvas.height = h;

      // Drawing respects the browser's EXIF correction automatically —
      // the resulting canvas has no rotation tag, pixels are upright.
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);

      canvas.toBlob(blob => {
        if (!blob) { reject(new Error("Canvas toBlob failed")); return; }
        // Wrap blob in a File so FormData sends it with a filename
        resolve(new File([blob], "photo.jpg", { type: "image/jpeg" }));
      }, "image/jpeg", 0.88); // 0.88 quality — sharp enough, smaller payload
    };

    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image load failed")); };
    img.src = url;
  });

  const handleUpload = useCallback(async (file) => {
    setView("detecting");
    setError(null);

    let prepared;
    try {
      prepared = await prepareImage(file);
    } catch {
      // If canvas fails for any reason, fall back to original file
      prepared = file;
    }

    // Show the corrected image (no EXIF rotation weirdness in preview)
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
    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", x1);
    form.append("y1", y1);
    form.append("x2", x2);
    form.append("y2", y2);
    if (detection.search_label || detection.label) {
      form.append("search_label", detection.search_label || detection.label);
    }

    try {
      const res = await fetch(`${API}/search`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
      setCategoryInfo({ category: data.detected_category, confidence: data.category_confidence });
      setView("results");
    } catch (e) {
      setError(`Search failed: ${e.message}`);
      setView("selecting");
    }
  }, [imageFile]);

  const handleFeedback = useCallback(async (itemId, rating) => {
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_id: itemId, rating }),
      });
    } catch { /* non-critical */ }
  }, []);

  const reset = useCallback(() => {
    setView("landing");
    setImageFile(null);
    if (imageURL) URL.revokeObjectURL(imageURL);
    setImageURL(null);
    setDetections([]);
    setResults([]);
    setSelectedItem(null);
    setCategoryInfo(null);
    setError(null);
  }, [imageURL]);

  return (
    <>
      <StyleInjector />
      <Navbar
        activeTab={activeTab}
        onTab={setActiveTab}
        onLogoClick={reset}
      />

      {view === "landing"    && <LandingView  onUpload={handleUpload} error={error} />}
      {view === "detecting"  && <LoadingView  label="Analyzing image…" />}
      {view === "selecting"  && (
        <SelectView
          imageURL={imageURL}
          detections={detections}
          onSelect={handleSelect}
          onBack={reset}
        />
      )}
      {view === "searching"  && <LoadingView  label="Finding matches…" />}
      {view === "results"    && (
        <ResultsView
          results={results}
          categoryInfo={categoryInfo}
          selectedItem={selectedItem}
          imageURL={imageURL}
          radius={radius}
          setRadius={setRadius}
          userLocation={userLocation}
          onFeedback={handleFeedback}
          onReset={reset}
        />
      )}
    </>
  );
}