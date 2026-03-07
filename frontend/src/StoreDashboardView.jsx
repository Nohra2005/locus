import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

// ── Theme (matches App.jsx) ────────────────────────────────────────
const T = {
  bg:          "#080808",
  bgDeep:      "#040404",
  surface:     "#111111",
  surfaceHov:  "#181818",
  border:      "#222222",
  borderFaint: "#141414",
  accent:      "#c9a96e",
  accentDeep:  "#a07840",
  accentBg:    "rgba(201,169,110,0.06)",
  accentRing:  "rgba(201,169,110,0.25)",
  text:        "#f0ede8",
  textMuted:   "#6b6458",
  green:       "#7aab8a",
  yellow:      "#c9a96e",
  red:         "#c97070",
};

// ══════════════════════════════════════════════════════════════════
// STORE DASHBOARD VIEW
// ══════════════════════════════════════════════════════════════════
export default function StoreDashboardView() {
  const [activeTab,  setActiveTab]  = useState("csv");   // "csv" | "scrape"
  const [storeName,  setStoreName]  = useState("");
  const [mallName,   setMallName]   = useState("");
  const [infoSaved,  setInfoSaved]  = useState(false);

  const handleSaveInfo = () => {
    if (storeName.trim() && mallName.trim()) setInfoSaved(true);
  };

  return (
    <div className="fade-in" style={{
      minHeight: "calc(100dvh - 61px)",
      maxWidth: 780,
      margin: "0 auto",
      padding: "40px 24px 64px",
    }}>

      {/* ── Page header ────────────────────────────────────────── */}
      <div className="fade-up" style={{ marginBottom: 36 }}>
        <div style={{
          fontSize: "0.68rem", color: T.accent,
          letterSpacing: "0.18em", textTransform: "uppercase",
          marginBottom: 12,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{ fontSize: "0.6rem" }}>✦</span>
          Store Portal
        </div>
        <h1 style={{
          fontFamily: "'Cormorant Garamond', serif",
          fontSize: "clamp(2rem, 6vw, 3rem)",
          fontWeight: 500,
          color: T.text,
          lineHeight: 1.1,
          margin: 0,
        }}>
          Catalogue Manager
        </h1>
        <p style={{
          marginTop: 10,
          fontSize: "0.82rem",
          color: T.textMuted,
          lineHeight: 1.7,
        }}>
          Index your store's products into Locus so shoppers can find them visually.
        </p>
      </div>

      {/* ── Store info card ─────────────────────────────────────── */}
      <div className="fade-up" style={{
        background: T.surface,
        border: `1px solid ${infoSaved ? T.accent : T.border}`,
        borderRadius: 14,
        padding: "24px 28px",
        marginBottom: 28,
        transition: "border-color 0.3s",
        animationDelay: "0.05s",
      }}>
        <div style={{
          fontSize: "0.68rem", color: infoSaved ? T.accent : T.textMuted,
          letterSpacing: "0.14em", textTransform: "uppercase",
          marginBottom: 18,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          {infoSaved ? "✦ Store confirmed" : "01 — Store identity"}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
          <StoreInput
            label="Store name"
            placeholder="e.g. Zara, Pull & Bear…"
            value={storeName}
            onChange={setStoreName}
            disabled={infoSaved}
          />
          <StoreInput
            label="Mall name"
            placeholder="e.g. ABC Achrafieh…"
            value={mallName}
            onChange={setMallName}
            disabled={infoSaved}
          />
        </div>

        {!infoSaved ? (
          <button
            className="btn-primary"
            onClick={handleSaveInfo}
            disabled={!storeName.trim() || !mallName.trim()}
            style={{ opacity: (!storeName.trim() || !mallName.trim()) ? 0.4 : 1 }}
          >
            <span style={{ color: T.accent }}>✦</span>
            Confirm store
          </button>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: "0.8rem", color: T.text }}>
              <strong style={{ color: T.accent }}>{storeName}</strong>
              <span style={{ color: T.textMuted }}> at </span>
              <strong style={{ color: T.text }}>{mallName}</strong>
            </div>
            <button
              className="btn-ghost"
              onClick={() => setInfoSaved(false)}
              style={{ fontSize: "0.72rem" }}
            >
              Edit
            </button>
          </div>
        )}
      </div>

      {/* ── Upload method tabs ───────────────────────────────────── */}
      {infoSaved && (
        <div className="fade-up" style={{ animationDelay: "0.1s" }}>

          {/* Tab switcher */}
          <div style={{
            display: "flex", gap: 4, marginBottom: 24,
            background: T.surface,
            border: `1px solid ${T.border}`,
            borderRadius: 10,
            padding: 4,
            width: "fit-content",
          }}>
            {[
              { id: "csv",    label: "CSV / Excel",    icon: "📋" },
              { id: "scrape", label: "Scrape Website",  icon: "🌐" },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: "8px 20px",
                  borderRadius: 7,
                  border: "none",
                  cursor: "pointer",
                  fontSize: "0.8rem",
                  fontFamily: "'DM Sans', sans-serif",
                  fontWeight: activeTab === tab.id ? 600 : 400,
                  color:      activeTab === tab.id ? T.text : T.textMuted,
                  background: activeTab === tab.id ? T.surfaceHov : "transparent",
                  transition: "all 0.2s",
                  display: "flex", alignItems: "center", gap: 6,
                }}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          {/* Panels */}
          {activeTab === "csv"    && <CsvUploadPanel    storeName={storeName} mallName={mallName} />}
          {activeTab === "scrape" && <ScrapeWebsitePanel storeName={storeName} mallName={mallName} />}
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
// CSV UPLOAD PANEL
// ══════════════════════════════════════════════════════════════════
function CsvUploadPanel({ storeName, mallName }) {
  const [file,     setFile]     = useState(null);
  const [rows,     setRows]     = useState([]);
  const [headers,  setHeaders]  = useState([]);
  const [status,   setStatus]   = useState("idle"); // idle|indexing|done
  const [progress, setProgress] = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,   setErrors]   = useState([]);
  const inputRef = useRef();

  const handleFile = async (f) => {
    if (!f) return;
    setFile(f);
    setStatus("idle");
    setErrors([]);

    // Parse CSV client-side — no library needed for simple CSVs
    const text = await f.text();
    const lines = text.trim().split("\n").filter(Boolean);
    if (lines.length < 2) return;

    const hdrs = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, "").toLowerCase());
    const data = lines.slice(1).map(line => {
      const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
      return Object.fromEntries(hdrs.map((h, i) => [h, vals[i] ?? ""]));
    });

    setHeaders(hdrs);
    setRows(data);
  };

  const handleIndex = async () => {
    if (!rows.length) return;
    setStatus("indexing");
    setErrors([]);
    const total = rows.length;
    let success = 0, failed = 0;
    const errs = [];

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      setProgress({ done: i, total, success, failed, current: row.name || `Row ${i + 1}` });

      try {
        const resp = await fetch(`${API}/add-bulk`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name:      row.name      || `Product ${i + 1}`,
            store:     storeName,
            mall:      mallName,
            image_url: row.image_url || row.image || "",
            price:     row.price     || "",
            category:  row.category  || "",
          }),
        });
        if (resp.ok) { success++; } else { failed++; errs.push(`Row ${i+1}: HTTP ${resp.status}`); }
      } catch (e) {
        failed++;
        errs.push(`Row ${i+1}: ${e.message}`);
      }
    }

    setProgress({ done: total, total, success, failed, current: "" });
    setErrors(errs);
    setStatus("done");
  };

  const hasRequired = headers.includes("name") && (headers.includes("image_url") || headers.includes("image"));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* Format hint */}
      <InfoBox title="Required CSV columns">
        <code style={{ fontSize: "0.78rem", color: T.accent }}>name</code>
        <span style={{ color: T.textMuted }}> · </span>
        <code style={{ fontSize: "0.78rem", color: T.accent }}>image_url</code>
        <span style={{ color: T.textMuted }}> · </span>
        <code style={{ fontSize: "0.78rem", color: T.textMuted }}>price (optional)</code>
        <span style={{ color: T.textMuted }}> · </span>
        <code style={{ fontSize: "0.78rem", color: T.textMuted }}>category (optional)</code>
        <div style={{ marginTop: 8, fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.7 }}>
          image_url must be a publicly accessible photo URL. AI will auto-detect category if missing.
        </div>
      </InfoBox>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        style={{
          border: `1.5px dashed ${file ? T.accent : T.border}`,
          borderRadius: 12,
          padding: "36px 24px",
          textAlign: "center",
          cursor: "pointer",
          background: file ? T.accentBg : T.surface,
          transition: "all 0.25s",
        }}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
      >
        <div style={{ fontSize: "1.8rem", marginBottom: 10 }}>{file ? "📄" : "⬆"}</div>
        <div style={{ fontSize: "0.85rem", color: T.text, marginBottom: 4 }}>
          {file ? file.name : "Drop CSV or Excel file here"}
        </div>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
          {file ? `${rows.length} products detected` : "or click to browse"}
        </div>
        <input
          ref={inputRef} type="file" accept=".csv,.xlsx,.xls"
          style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])}
        />
      </div>

      {/* Preview table */}
      {rows.length > 0 && !hasRequired && (
        <div style={{
          background: "rgba(201,112,112,0.06)",
          border: `1px solid rgba(201,112,112,0.2)`,
          borderRadius: 10, padding: "12px 16px",
          fontSize: "0.78rem", color: T.red,
        }}>
          ⚠ Missing required columns. Found: {headers.join(", ")}
        </div>
      )}

      {rows.length > 0 && hasRequired && status === "idle" && (
        <>
          <PreviewTable rows={rows.slice(0, 5)} headers={headers} />
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <button className="btn-primary" onClick={handleIndex}>
              <span style={{ color: T.accent }}>✦</span>
              Index {rows.length} products
            </button>
            <span style={{ fontSize: "0.72rem", color: T.textMuted }}>
              ~{Math.ceil(rows.length * 4 / 60)} min estimated
            </span>
          </div>
        </>
      )}

      {/* Progress */}
      {status === "indexing" && (
        <IndexingProgress progress={progress} />
      )}

      {/* Done */}
      {status === "done" && (
        <DoneCard progress={progress} errors={errors} storeName={storeName} />
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
// SCRAPE WEBSITE PANEL
// ══════════════════════════════════════════════════════════════════
function ScrapeWebsitePanel({ storeName, mallName }) {
  const [url,       setUrl]       = useState("");
  const [maxItems,  setMaxItems]  = useState(20);
  const [scraping,  setScraping]  = useState(false);
  const [products,  setProducts]  = useState(null);   // null = not yet scraped
  const [selected,  setSelected]  = useState({});
  const [status,    setStatus]    = useState("idle");
  const [progress,  setProgress]  = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,    setErrors]    = useState([]);
  const [scrapeErr, setScrapeErr] = useState("");

  const handleScrape = async () => {
    if (!url.startsWith("http")) { setScrapeErr("Please enter a valid URL starting with https://"); return; }
    setScrapeErr("");
    setScraping(true);
    setProducts(null);
    setSelected({});
    setStatus("idle");

    try {
      const resp = await fetch(`${API}/scrape`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, max_products: maxItems }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const prods = data.products || [];
      setProducts(prods);
      setSelected(Object.fromEntries(prods.map((_, i) => [i, true])));
    } catch (e) {
      setScrapeErr(`Scrape failed: ${e.message}. Is Docker running?`);
    } finally {
      setScraping(false);
    }
  };

  const toggleAll = (val) =>
    setSelected(Object.fromEntries((products || []).map((_, i) => [i, val])));

  const selectedItems = (products || []).filter((_, i) => selected[i]);

  const handleIndex = async () => {
    if (!selectedItems.length) return;
    setStatus("indexing");
    setErrors([]);
    const total = selectedItems.length;
    let success = 0, failed = 0;
    const errs = [];

    for (let i = 0; i < selectedItems.length; i++) {
      const p = selectedItems[i];
      setProgress({ done: i, total, success, failed, current: p.name || `Product ${i + 1}` });
      try {
        const resp = await fetch(`${API}/add-bulk`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: p.name, store: storeName, mall: mallName,
            image_url: p.image_url, price: p.price || "", category: "",
          }),
        });
        if (resp.ok) { success++; } else { failed++; errs.push(`${p.name}: HTTP ${resp.status}`); }
      } catch (e) {
        failed++; errs.push(`${p.name}: ${e.message}`);
      }
    }

    setProgress({ done: total, total, success, failed, current: "" });
    setErrors(errs);
    setStatus("done");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      <InfoBox title="How scraping works">
        <div style={{ fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.8 }}>
          Paste your store's product listing page URL. Locus will extract product names and images,
          show you a preview, and you confirm which items to add to the catalogue.
          Works best with <span style={{ color: T.accent }}>Shopify</span> and <span style={{ color: T.accent }}>WooCommerce</span> stores.
        </div>
      </InfoBox>

      {/* URL input + controls */}
      <div style={{
        background: T.surface,
        border: `1px solid ${T.border}`,
        borderRadius: 12,
        padding: "24px 24px 20px",
      }}>
        <label style={{ fontSize: "0.68rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase" }}>
          Product listing URL
        </label>
        <div style={{ display: "flex", gap: 10, marginTop: 8, marginBottom: 18 }}>
          <input
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleScrape()}
            placeholder="https://yourstore.com/collections/all"
            style={{
              flex: 1,
              background: T.bgDeep,
              border: `1px solid ${T.border}`,
              borderRadius: 8,
              padding: "10px 14px",
              color: T.text,
              fontSize: "0.82rem",
              fontFamily: "'DM Sans', sans-serif",
              outline: "none",
            }}
          />
          <button
            className="btn-primary"
            onClick={handleScrape}
            disabled={scraping || !url.trim()}
            style={{ whiteSpace: "nowrap", opacity: (scraping || !url.trim()) ? 0.5 : 1 }}
          >
            {scraping ? (
              <>
                <div style={{
                  width: 12, height: 12,
                  border: `1.5px solid ${T.border}`,
                  borderTop: `1.5px solid ${T.accent}`,
                  borderRadius: "50%",
                  animation: "spin 0.9s linear infinite",
                }} />
                Fetching…
              </>
            ) : (
              <><span style={{ color: T.accent }}>✦</span> Scrape</>
            )}
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <label style={{ fontSize: "0.68rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap" }}>
            Max products
          </label>
          <input
            type="range" min={5} max={100} step={5}
            value={maxItems}
            onChange={e => setMaxItems(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: "0.8rem", color: T.accent, minWidth: 28, textAlign: "right" }}>
            {maxItems}
          </span>
        </div>
      </div>

      {scrapeErr && (
        <div style={{
          background: "rgba(201,112,112,0.06)",
          border: `1px solid rgba(201,112,112,0.2)`,
          borderRadius: 10, padding: "12px 16px",
          fontSize: "0.78rem", color: T.red,
        }}>
          ⚠ {scrapeErr}
        </div>
      )}

      {/* Product preview grid */}
      {products !== null && status === "idle" && (
        <>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontSize: "0.68rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              {products.length} products found · {selectedItems.length} selected
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-ghost" onClick={() => toggleAll(true)}  style={{ fontSize: "0.72rem" }}>Select all</button>
              <button className="btn-ghost" onClick={() => toggleAll(false)} style={{ fontSize: "0.72rem" }}>Deselect all</button>
            </div>
          </div>

          {products.length === 0 ? (
            <div style={{
              background: T.surface, border: `1px solid ${T.border}`,
              borderRadius: 12, padding: "48px 24px", textAlign: "center",
            }}>
              <div style={{ fontSize: "1.5rem", marginBottom: 12 }}>🔍</div>
              <div style={{ fontSize: "0.85rem", color: T.textMuted }}>No products found on this page.</div>
              <div style={{ fontSize: "0.72rem", color: T.textMuted, marginTop: 6 }}>
                Try a different URL or use the CSV upload method instead.
              </div>
            </div>
          ) : (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
              gap: 14,
            }}>
              {products.map((p, i) => (
                <ScrapedProductCard
                  key={i}
                  product={p}
                  selected={!!selected[i]}
                  onToggle={() => setSelected(s => ({ ...s, [i]: !s[i] }))}
                />
              ))}
            </div>
          )}

          {selectedItems.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <button className="btn-primary" onClick={handleIndex}>
                <span style={{ color: T.accent }}>✦</span>
                Index {selectedItems.length} selected
              </button>
              <span style={{ fontSize: "0.72rem", color: T.textMuted }}>
                ~{Math.ceil(selectedItems.length * 4 / 60)} min estimated
              </span>
            </div>
          )}
        </>
      )}

      {status === "indexing" && <IndexingProgress progress={progress} />}
      {status === "done"     && <DoneCard progress={progress} errors={errors} storeName={storeName} />}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
// SHARED SUB-COMPONENTS
// ══════════════════════════════════════════════════════════════════

function StoreInput({ label, placeholder, value, onChange, disabled }) {
  return (
    <div>
      <label style={{
        fontSize: "0.65rem", color: T.textMuted,
        letterSpacing: "0.12em", textTransform: "uppercase",
        display: "block", marginBottom: 8,
      }}>
        {label}
      </label>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        style={{
          width: "100%",
          background: disabled ? T.bgDeep : T.bgDeep,
          border: `1px solid ${T.border}`,
          borderRadius: 8,
          padding: "10px 14px",
          color: disabled ? T.textMuted : T.text,
          fontSize: "0.82rem",
          fontFamily: "'DM Sans', sans-serif",
          outline: "none",
          boxSizing: "border-box",
          opacity: disabled ? 0.6 : 1,
        }}
      />
    </div>
  );
}

function InfoBox({ title, children }) {
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderLeft: `3px solid ${T.accent}`,
      borderRadius: "0 10px 10px 0",
      padding: "14px 18px",
    }}>
      <div style={{
        fontSize: "0.65rem", color: T.accent,
        letterSpacing: "0.14em", textTransform: "uppercase",
        marginBottom: 8,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function PreviewTable({ rows, headers }) {
  const showCols = headers.slice(0, 4);
  return (
    <div style={{ overflowX: "auto", borderRadius: 10, border: `1px solid ${T.border}` }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.75rem" }}>
        <thead>
          <tr style={{ background: T.surfaceHov }}>
            {showCols.map(h => (
              <th key={h} style={{
                padding: "10px 14px", textAlign: "left",
                color: T.textMuted, fontWeight: 500,
                letterSpacing: "0.08em", textTransform: "uppercase",
                fontSize: "0.65rem",
                borderBottom: `1px solid ${T.border}`,
              }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${T.borderFaint}` }}>
              {showCols.map(h => (
                <td key={h} style={{
                  padding: "9px 14px", color: T.text,
                  maxWidth: 180, overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {row[h] || "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{
        padding: "8px 14px",
        fontSize: "0.65rem", color: T.textMuted,
        borderTop: `1px solid ${T.borderFaint}`,
      }}>
        Showing first {rows.length} rows
      </div>
    </div>
  );
}

function ScrapedProductCard({ product, selected, onToggle }) {
  return (
    <div
      onClick={onToggle}
      style={{
        background: T.surface,
        border: `1px solid ${selected ? T.accent : T.border}`,
        borderRadius: 10,
        overflow: "hidden",
        cursor: "pointer",
        transition: "border-color 0.2s, transform 0.15s",
        transform: selected ? "none" : "scale(0.98)",
        opacity: selected ? 1 : 0.55,
      }}
    >
      {/* Image */}
      <div style={{
        aspectRatio: "3/4",
        background: T.bgDeep,
        overflow: "hidden",
        position: "relative",
      }}>
        {product.image_url ? (
          <img
            src={product.image_url}
            alt={product.name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={e => { e.target.style.display = "none"; }}
          />
        ) : (
          <div style={{
            width: "100%", height: "100%",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: T.textMuted, fontSize: "1.5rem",
          }}>
            🖼
          </div>
        )}
        {/* Selected checkmark */}
        <div style={{
          position: "absolute", top: 8, right: 8,
          width: 20, height: 20,
          borderRadius: "50%",
          background: selected ? T.accent : "rgba(0,0,0,0.4)",
          border: `1.5px solid ${selected ? T.accent : T.border}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.6rem", color: T.bg,
          transition: "all 0.2s",
        }}>
          {selected ? "✓" : ""}
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: "10px 12px" }}>
        <div style={{
          fontSize: "0.75rem", color: T.text,
          overflow: "hidden", textOverflow: "ellipsis",
          whiteSpace: "nowrap", lineHeight: 1.4,
        }}>
          {product.name || "Unnamed product"}
        </div>
        {product.price && (
          <div style={{ fontSize: "0.68rem", color: T.accent, marginTop: 3 }}>
            {product.price}
          </div>
        )}
      </div>
    </div>
  );
}

function IndexingProgress({ progress }) {
  const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: 12,
      padding: "24px 28px",
    }}>
      <div style={{
        fontSize: "0.68rem", color: T.accent,
        letterSpacing: "0.14em", textTransform: "uppercase",
        marginBottom: 16,
      }}>
        Indexing in progress
      </div>

      {/* Progress bar */}
      <div style={{
        height: 2, background: T.border, borderRadius: 2,
        marginBottom: 12, overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: `linear-gradient(90deg, ${T.accentDeep}, ${T.accent})`,
          borderRadius: 2, transition: "width 0.4s ease",
        }} />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
          {progress.current && (
            <span>Processing: <span style={{ color: T.text }}>{progress.current}</span></span>
          )}
        </div>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
          {progress.done} / {progress.total}
        </div>
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ fontSize: "0.78rem" }}>
          <span style={{ color: T.green }}>✓ {progress.success}</span>
          <span style={{ color: T.textMuted }}> indexed</span>
        </div>
        {progress.failed > 0 && (
          <div style={{ fontSize: "0.78rem" }}>
            <span style={{ color: T.red }}>✕ {progress.failed}</span>
            <span style={{ color: T.textMuted }}> failed</span>
          </div>
        )}
      </div>
    </div>
  );
}

function DoneCard({ progress, errors, storeName }) {
  return (
    <div style={{
      background: progress.success > 0
        ? "rgba(122,171,138,0.05)"
        : "rgba(201,112,112,0.05)",
      border: `1px solid ${progress.success > 0 ? T.green : T.red}`,
      borderRadius: 12,
      padding: "24px 28px",
    }}>
      <div style={{
        fontFamily: "'Cormorant Garamond', serif",
        fontSize: "1.4rem",
        color: progress.success > 0 ? T.green : T.red,
        marginBottom: 6,
      }}>
        {progress.success > 0 ? "Done!" : "Indexing failed"}
      </div>
      <div style={{ fontSize: "0.8rem", color: T.textMuted, marginBottom: 16 }}>
        {progress.success > 0 && (
          <><strong style={{ color: T.text }}>{progress.success} products</strong> from <strong style={{ color: T.accent }}>{storeName}</strong> are now searchable in Locus.</>
        )}
        {progress.failed > 0 && (
          <> · <span style={{ color: T.red }}>{progress.failed} failed</span></>
        )}
      </div>

      {errors.length > 0 && (
        <details style={{ marginTop: 10 }}>
          <summary style={{ fontSize: "0.72rem", color: T.textMuted, cursor: "pointer" }}>
            Show {errors.length} error{errors.length > 1 ? "s" : ""}
          </summary>
          <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 4 }}>
            {errors.map((e, i) => (
              <div key={i} style={{ fontSize: "0.7rem", color: T.red, fontFamily: "monospace" }}>{e}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
