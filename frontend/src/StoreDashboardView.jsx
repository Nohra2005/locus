import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

// ── Theme ──────────────────────────────────────────────────────────────────────
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


// ══════════════════════════════════════════════════════════════════════════════
// SHARED BATCH INDEXING HELPER
// ══════════════════════════════════════════════════════════════════════════════
async function runBatchIndex(items, storeName, mallName, setProgress, setErrors, setStatus) {
  setStatus("indexing");
  setErrors([]);

  const CHUNK_SIZE = 50;
  const total = items.length;
  let success = 0;
  let failed  = 0;
  const errs  = [];

  for (let start = 0; start < total; start += CHUNK_SIZE) {
    const chunk      = items.slice(start, start + CHUNK_SIZE);
    const batchNum   = Math.floor(start / CHUNK_SIZE) + 1;
    const totalBatch = Math.ceil(total / CHUNK_SIZE);

    setProgress({
      done:    start,
      total,
      success,
      failed,
      current: `Batch ${batchNum}/${totalBatch} (items ${start + 1}–${Math.min(start + CHUNK_SIZE, total)})`,
    });

    try {
      const resp = await fetch(`${API}/add-bulk-batch`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: chunk.map(row => ({
            name:      row.name      || "Product",
            store:     storeName,
            mall:      mallName,
            image_url: row.image_url || row.image || "",
            price:     row.price     || "",
            category:  row.category  || "",
          })),
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        success += data.success || 0;
        failed  += (data.failed || []).length;
        (data.failed || []).forEach(f => errs.push(`${f.item}: ${f.error}`));
      } else {
        failed += chunk.length;
        errs.push(`Batch ${batchNum}: HTTP ${resp.status}`);
      }
    } catch (e) {
      failed += chunk.length;
      errs.push(`Batch ${batchNum}: ${e.message}`);
    }
  }

  setProgress({ done: total, total, success, failed, current: "" });
  setErrors(errs);
  setStatus("done");
}


// ══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
export default function StoreDashboardView() {
  const [activeTab, setActiveTab] = useState("csv");
  const [storeName, setStoreName] = useState("");
  const [mallName,  setMallName]  = useState("");
  const [infoSaved, setInfoSaved] = useState(false);

  const handleSaveInfo = () => {
    if (storeName.trim() && mallName.trim()) setInfoSaved(true);
  };

  return (
    <div className="fade-in" style={{
      minHeight: "calc(100dvh - 61px)",
      maxWidth: 820,
      margin: "0 auto",
      padding: "40px 24px 64px",
    }}>

      {/* Header */}
      <div className="fade-up" style={{ marginBottom: 36 }}>
        <div style={{
          fontSize: "0.68rem", color: T.accent,
          letterSpacing: "0.18em", textTransform: "uppercase",
          marginBottom: 12, display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{ fontSize: "0.6rem" }}>✦</span> Store Portal
        </div>
        <h1 style={{
          fontFamily: "'Cormorant Garamond', serif",
          fontSize: "clamp(2rem, 6vw, 3rem)",
          fontWeight: 500, color: T.text, lineHeight: 1.1, margin: 0,
        }}>
          Catalogue Manager
        </h1>
        <p style={{ marginTop: 10, fontSize: "0.82rem", color: T.textMuted, lineHeight: 1.7 }}>
          Index your store's products into Locus so shoppers can find them visually.
        </p>
      </div>

      {/* Store identity card */}
      <div className="fade-up" style={{
        background: T.surface,
        border: `1px solid ${infoSaved ? T.accent : T.border}`,
        borderRadius: 14, padding: "24px 28px", marginBottom: 28,
        transition: "border-color 0.3s", animationDelay: "0.05s",
      }}>
        <div style={{
          fontSize: "0.68rem", color: infoSaved ? T.accent : T.textMuted,
          letterSpacing: "0.14em", textTransform: "uppercase",
          marginBottom: 18, display: "flex", alignItems: "center", gap: 8,
        }}>
          {infoSaved ? "✦ Store confirmed" : "01 — Store identity"}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
          <StoreInput label="Store name" placeholder="e.g. Zara, Pull & Bear…"
            value={storeName} onChange={setStoreName} disabled={infoSaved} />
          <StoreInput label="Mall name" placeholder="e.g. ABC Achrafieh…"
            value={mallName} onChange={setMallName} disabled={infoSaved} />
        </div>

        {!infoSaved ? (
          <button className="btn-primary" onClick={handleSaveInfo}
            disabled={!storeName.trim() || !mallName.trim()}
            style={{ opacity: (!storeName.trim() || !mallName.trim()) ? 0.4 : 1 }}>
            <span style={{ color: T.accent }}>✦</span> Confirm store
          </button>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: "0.8rem", color: T.text }}>
              <strong style={{ color: T.accent }}>{storeName}</strong>
              <span style={{ color: T.textMuted }}> at </span>
              <strong style={{ color: T.text }}>{mallName}</strong>
            </div>
            <button className="btn-ghost" onClick={() => setInfoSaved(false)} style={{ fontSize: "0.72rem" }}>Edit</button>
          </div>
        )}
      </div>

      {/* Tabs */}
      {infoSaved && (
        <div className="fade-up" style={{ animationDelay: "0.1s" }}>
          <div style={{
            display: "flex", gap: 4, marginBottom: 24,
            background: T.surface, border: `1px solid ${T.border}`,
            borderRadius: 10, padding: 4, width: "fit-content",
          }}>
            {[
              { id: "csv",       label: "CSV / Excel",   icon: "📋" },
              { id: "scrape",    label: "Scrape Website", icon: "🌐" },
              { id: "catalogue", label: "My Catalogue",   icon: "🗄️" },
            ].map(tab => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                padding: "8px 20px", borderRadius: 7, border: "none", cursor: "pointer",
                fontSize: "0.8rem", fontFamily: "'DM Sans', sans-serif",
                fontWeight: activeTab === tab.id ? 600 : 400,
                color:      activeTab === tab.id ? T.text : T.textMuted,
                background: activeTab === tab.id ? T.surfaceHov : "transparent",
                transition: "all 0.2s", display: "flex", alignItems: "center", gap: 6,
              }}>
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          {activeTab === "csv"       && <CsvUploadPanel     storeName={storeName} mallName={mallName} />}
          {activeTab === "scrape"    && <ScrapeWebsitePanel storeName={storeName} mallName={mallName} />}
          {activeTab === "catalogue" && <CataloguePanel     storeName={storeName} />}
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// MY CATALOGUE PANEL
// ══════════════════════════════════════════════════════════════════════════════
function CataloguePanel({ storeName }) {
  const [products, setProducts] = useState([]);
  const [total,    setTotal]    = useState(0);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");
  const [offset,   setOffset]   = useState(0);
  const [search,   setSearch]   = useState("");
  const [deleting, setDeleting] = useState({});
  const LIMIT = 24;

const fetchProducts = useCallback(async (off = 0) => {
    setLoading(true); setError("");
    try {
      const resp = await fetch(`${API}/store-catalogue?store_name=${encodeURIComponent(storeName)}&limit=${LIMIT}&offset=${off}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setProducts(data.products || []);
      setTotal(data.total || 0);
      setOffset(off);
    } catch (e) {
      setError(`Failed to load catalogue: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [storeName]);

  useEffect(() => { fetchProducts(0); }, [fetchProducts]);

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Remove "${name}" from the catalogue?`)) return;
    setDeleting(d => ({ ...d, [id]: true }));
    try {
      const resp = await fetch(`${API}/store-catalogue/item/${id}`, { method: "DELETE" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setProducts(ps => ps.filter(p => p.id !== id));
      setTotal(t => t - 1);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    } finally {
      setDeleting(d => ({ ...d, [id]: false }));
    }
  };

  const catColor = {
    shirt: "#6e9ecf", sweater: "#9b8abf", jacket: "#7aab8a", coat: "#7aab8a",
    dress: "#c97070", jumpsuit: "#c9a96e", skirt: "#c9c06e", pants: "#8aabcf",
    shorts: "#8aabcf", shoes: "#c9a96e", bag: "#bf9a7a", glasses: "#aaaaaa",
    hat: "#aaaaaa", watch: "#c9c96e", scarf: "#bf9a7a",
  };

  const filtered = products.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.category_tag || "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div style={{ fontSize: "0.78rem", color: T.textMuted }}>
          <span style={{ color: T.accent, fontWeight: 600 }}>{total}</span> products indexed for{" "}
          <span style={{ color: T.text }}>{storeName}</span>
        </div>
        <button className="btn-ghost" onClick={() => fetchProducts(offset)} disabled={loading} style={{ fontSize: "0.72rem" }}>
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by name or category…"
        style={{ width: "100%", background: T.bgDeep, border: `1px solid ${T.border}`, borderRadius: 8, padding: "10px 14px", color: T.text, fontSize: "0.82rem", fontFamily: "'DM Sans', sans-serif", outline: "none", boxSizing: "border-box" }}
      />

      {error && (
        <div style={{ background: "rgba(201,112,112,0.06)", border: `1px solid rgba(201,112,112,0.2)`, borderRadius: 10, padding: "12px 16px", fontSize: "0.78rem", color: T.red }}>⚠ {error}</div>
      )}

      {loading ? (
        <div style={{ textAlign: "center", padding: "60px 0", color: T.textMuted, fontSize: "0.82rem" }}>Loading products…</div>
      ) : filtered.length === 0 ? (
        <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "60px 24px", textAlign: "center" }}>
          <div style={{ fontSize: "2rem", marginBottom: 14 }}>🗄️</div>
          <div style={{ fontSize: "0.85rem", color: T.textMuted }}>
            {total === 0 ? "No products indexed yet. Use CSV or Scrape to add products." : "No products match your search."}
          </div>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 14 }}>
          {filtered.map(p => (
            <CatalogueCard key={p.id} product={p} catColor={catColor} deleting={!!deleting[p.id]} onDelete={() => handleDelete(p.id, p.name)} />
          ))}
        </div>
      )}

      {total > LIMIT && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginTop: 8 }}>
          <button className="btn-ghost" onClick={() => fetchProducts(Math.max(0, offset - LIMIT))} disabled={offset === 0 || loading} style={{ fontSize: "0.78rem" }}>← Previous</button>
          <span style={{ fontSize: "0.75rem", color: T.textMuted }}>{offset + 1}–{Math.min(offset + LIMIT, total)} of {total}</span>
          <button className="btn-ghost" onClick={() => fetchProducts(offset + LIMIT)} disabled={offset + LIMIT >= total || loading} style={{ fontSize: "0.78rem" }}>Next →</button>
        </div>
      )}
    </div>
  );
}

function CatalogueCard({ product: p, catColor, deleting, onDelete }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} style={{ background: T.surface, border: `1px solid ${hovered ? T.accentDeep : T.border}`, borderRadius: 10, overflow: "hidden", position: "relative", transition: "border-color 0.2s" }}>
      <div style={{ width: "100%", aspectRatio: "1 / 1", background: T.bgDeep, overflow: "hidden" }}>
        {p.image_url ? (
          <img src={p.image_url} alt={p.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textMuted, fontSize: "1.8rem" }}>📦</div>
        )}
      </div>
      <div style={{ padding: "10px 10px 8px" }}>
        <div style={{ fontSize: "0.75rem", color: T.text, fontWeight: 500, lineHeight: 1.35, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", marginBottom: 8 }}>{p.name}</div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
          {p.category_tag && (
            <span style={{ fontSize: "0.6rem", color: catColor[p.category_tag] || T.textMuted, background: `${catColor[p.category_tag] || T.textMuted}18`, border: `1px solid ${catColor[p.category_tag] || T.textMuted}40`, borderRadius: 4, padding: "2px 6px", textTransform: "capitalize", letterSpacing: "0.06em" }}>{p.category_tag}</span>
          )}
          {p.price && <span style={{ fontSize: "0.7rem", color: T.accent, whiteSpace: "nowrap" }}>{p.price}</span>}
        </div>
      </div>
      {hovered && (
        <button onClick={onDelete} disabled={deleting} style={{ position: "absolute", top: 6, right: 6, width: 26, height: 26, background: "rgba(0,0,0,0.75)", border: `1px solid ${T.border}`, borderRadius: "50%", color: T.red, cursor: "pointer", fontSize: "0.7rem", display: "flex", alignItems: "center", justifyContent: "center" }}>
          {deleting ? "…" : "✕"}
        </button>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// CSV UPLOAD PANEL
// ══════════════════════════════════════════════════════════════════════════════
function CsvUploadPanel({ storeName, mallName }) {
  const [file,     setFile]     = useState(null);
  const [rows,     setRows]     = useState([]);
  const [headers,  setHeaders]  = useState([]);
  const [status,   setStatus]   = useState("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,   setErrors]   = useState([]);
  const inputRef = useRef();

  const handleFile = async (f) => {
    if (!f) return;
    setFile(f); setStatus("idle"); setErrors([]);
    const text  = await f.text();
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

  const hasRequired = headers.includes("name") && headers.includes("image_url");
  const handleIndex = () =>
    runBatchIndex(rows, storeName, mallName, setProgress, setErrors, setStatus);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      <InfoBox title="Required columns">
        <div style={{ fontSize: "0.72rem", color: T.textMuted, fontFamily: "'DM Mono', monospace", lineHeight: 1.8 }}>
          <strong style={{ color: T.text }}>name</strong> &nbsp;|&nbsp;
          <strong style={{ color: T.text }}>image_url</strong> &nbsp;|&nbsp;
          price (optional) &nbsp;|&nbsp; category (optional)
        </div>
      </InfoBox>

      {/* ── ACTION BAR — appears at top as soon as a valid file is loaded ── */}
      {rows.length > 0 && hasRequired && status === "idle" && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: T.accentBg, border: `1px solid ${T.accentRing}`,
          borderRadius: 12, padding: "16px 20px",
        }}>
          <div>
            <div style={{ fontSize: "0.82rem", color: T.text, fontWeight: 500 }}>
              <span style={{ color: T.accent }}>{rows.length}</span> products ready
            </div>
            <div style={{ fontSize: "0.7rem", color: T.textMuted, marginTop: 2 }}>
              {file?.name} · ~{Math.max(1, Math.ceil(rows.length * 0.5 / 60))} min estimated
            </div>
          </div>
          <button className="btn-primary" onClick={handleIndex}>
            <span style={{ color: T.accent }}>✦</span> Index {rows.length} products
          </button>
        </div>
      )}

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
        style={{
          border: `2px dashed ${file ? T.accent : T.border}`,
          borderRadius: 12, padding: "40px 24px", textAlign: "center",
          cursor: "pointer", transition: "border-color 0.2s",
          background: file ? T.accentBg : "transparent",
        }}
      >
        <div style={{ fontSize: "1.5rem", marginBottom: 10 }}>{file ? "📄" : "⬆️"}</div>
        <div style={{ fontSize: "0.82rem", color: file ? T.accent : T.textMuted }}>
          {file ? file.name : "Drop CSV / Excel here or click to browse"}
        </div>
        <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls"
          style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
      </div>

      {headers.length > 0 && (
        <div style={{ fontSize: "0.72rem", color: hasRequired ? T.green : T.red }}>
          {hasRequired ? "✓ Required columns found" : "✕ Missing: name, image_url"} — Found: {headers.join(", ")}
        </div>
      )}

      {/* Collapsible preview */}
      {rows.length > 0 && hasRequired && status === "idle" && (
        <details>
          <summary style={{ fontSize: "0.72rem", color: T.textMuted, cursor: "pointer", marginBottom: 10, userSelect: "none" }}>
            Preview first 5 rows
          </summary>
          <PreviewTable rows={rows.slice(0, 5)} headers={headers} />
        </details>
      )}

      {status === "indexing" && <IndexingProgress progress={progress} />}
      {status === "done"     && <DoneCard progress={progress} errors={errors} storeName={storeName} />}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// SCRAPE WEBSITE PANEL
// ══════════════════════════════════════════════════════════════════════════════
function ScrapeWebsitePanel({ storeName, mallName }) {
  const [url,       setUrl]       = useState("");
  const [scraping,  setScraping]  = useState(false);
  const [products,  setProducts]  = useState(null);
  const [selected,  setSelected]  = useState({});
  const [status,    setStatus]    = useState("idle");
  const [progress,  setProgress]  = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,    setErrors]    = useState([]);
  const [scrapeErr, setScrapeErr] = useState("");

  const handleScrape = async () => {
    if (!url.startsWith("http")) { setScrapeErr("Please enter a valid URL starting with https://"); return; }
    setScrapeErr(""); setScraping(true); setProducts(null); setSelected({}); setStatus("idle");
    try {
      const resp = await fetch(`${API}/scrape`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, max_products: 0 }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data  = await resp.json();
      const prods = data.products || [];
      setProducts(prods);
      setSelected(Object.fromEntries(prods.map((_, i) => [i, true])));
    } catch (e) {
      setScrapeErr(`Scrape failed: ${e.message}`);
    } finally {
      setScraping(false);
    }
  };

  const toggleAll     = (val) => setSelected(Object.fromEntries((products || []).map((_, i) => [i, val])));
  const selectedItems = (products || []).filter((_, i) => selected[i]);
  const handleIndex   = () =>
    runBatchIndex(selectedItems, storeName, mallName, setProgress, setErrors, setStatus);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <InfoBox title="How scraping works">
        <div style={{ fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.8 }}>
          Paste your store's product listing URL. Works best with{" "}
          <span style={{ color: T.accent }}>Shopify</span> stores — fetches all products automatically.
        </div>
      </InfoBox>

      <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "20px 24px" }}>
        <label style={{ fontSize: "0.68rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase" }}>Product listing URL</label>
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          <input value={url} onChange={e => setUrl(e.target.value)} onKeyDown={e => e.key === "Enter" && handleScrape()}
            placeholder="https://yourstore.com/collections/all"
            style={{ flex: 1, background: T.bgDeep, border: `1px solid ${T.border}`, borderRadius: 8, padding: "10px 14px", color: T.text, fontSize: "0.82rem", fontFamily: "'DM Sans', sans-serif", outline: "none" }}
          />
          <button className="btn-primary" onClick={handleScrape} disabled={scraping || !url.trim()} style={{ whiteSpace: "nowrap", opacity: (scraping || !url.trim()) ? 0.5 : 1 }}>
            {scraping ? (
              <><div style={{ width: 12, height: 12, border: `1.5px solid ${T.border}`, borderTop: `1.5px solid ${T.accent}`, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} /> Fetching…</>
            ) : (
              <><span style={{ color: T.accent }}>✦</span> Scrape</>
            )}
          </button>
        </div>
      </div>

      {scrapeErr && (
        <div style={{ background: "rgba(201,112,112,0.06)", border: `1px solid rgba(201,112,112,0.2)`, borderRadius: 10, padding: "12px 16px", fontSize: "0.78rem", color: T.red }}>⚠ {scrapeErr}</div>
      )}

      {products !== null && status === "idle" && (
        <>
          {/* ── ACTION BAR — at top before the product grid ── */}
          {selectedItems.length > 0 && (
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              background: T.accentBg, border: `1px solid ${T.accentRing}`,
              borderRadius: 12, padding: "16px 20px",
            }}>
              <div>
                <div style={{ fontSize: "0.82rem", color: T.text, fontWeight: 500 }}>
                  <span style={{ color: T.accent }}>{selectedItems.length}</span> products selected
                </div>
                <div style={{ fontSize: "0.7rem", color: T.textMuted, marginTop: 2 }}>
                  ~{Math.max(1, Math.ceil(selectedItems.length * 0.5 / 60))} min estimated
                </div>
              </div>
              <button className="btn-primary" onClick={handleIndex}>
                <span style={{ color: T.accent }}>✦</span> Index {selectedItems.length} selected
              </button>
            </div>
          )}

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
            <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "48px 24px", textAlign: "center" }}>
              <div style={{ fontSize: "1.5rem", marginBottom: 12 }}>🔍</div>
              <div style={{ fontSize: "0.85rem", color: T.textMuted }}>No products found on this page.</div>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 14 }}>
              {products.map((p, i) => (
                <ScrapedProductCard key={i} product={p} selected={!!selected[i]}
                  onToggle={() => setSelected(s => ({ ...s, [i]: !s[i] }))} />
              ))}
            </div>
          )}
        </>
      )}

      {status === "indexing" && <IndexingProgress progress={progress} />}
      {status === "done"     && <DoneCard progress={progress} errors={errors} storeName={storeName} />}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// SHARED SUB-COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════

function StoreInput({ label, placeholder, value, onChange, disabled }) {
  return (
    <div>
      <label style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>{label}</label>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} disabled={disabled}
        style={{ width: "100%", background: T.bgDeep, border: `1px solid ${T.border}`, borderRadius: 8, padding: "10px 14px", color: disabled ? T.textMuted : T.text, fontSize: "0.82rem", fontFamily: "'DM Sans', sans-serif", outline: "none", boxSizing: "border-box", opacity: disabled ? 0.6 : 1 }}
      />
    </div>
  );
}

function InfoBox({ title, children }) {
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderLeft: `3px solid ${T.accent}`, borderRadius: "0 10px 10px 0", padding: "14px 18px" }}>
      <div style={{ fontSize: "0.65rem", color: T.accent, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 8 }}>{title}</div>
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
              <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: T.textMuted, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase", fontSize: "0.65rem", borderBottom: `1px solid ${T.border}` }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${T.borderFaint}` }}>
              {showCols.map(h => (
                <td key={h} style={{ padding: "9px 14px", color: T.text, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row[h] || "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ padding: "8px 14px", fontSize: "0.65rem", color: T.textMuted, borderTop: `1px solid ${T.borderFaint}` }}>Showing first {rows.length} rows</div>
    </div>
  );
}

function ScrapedProductCard({ product, selected, onToggle }) {
  return (
    <div onClick={onToggle} style={{ background: T.surface, border: `1px solid ${selected ? T.accent : T.border}`, borderRadius: 10, overflow: "hidden", cursor: "pointer", transition: "border-color 0.2s, transform 0.15s", transform: selected ? "none" : "scale(0.98)", opacity: selected ? 1 : 0.65 }}>
      <div style={{ width: "100%", aspectRatio: "1 / 1", background: T.bgDeep, overflow: "hidden" }}>
        {product.image_url ? (
          <img src={product.image_url} alt={product.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} onError={e => { e.target.style.display = "none"; }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textMuted, fontSize: "1.5rem" }}>📦</div>
        )}
      </div>
      <div style={{ padding: "8px 10px" }}>
        <div style={{ fontSize: "0.72rem", color: T.text, lineHeight: 1.35, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{product.name}</div>
        {product.price && <div style={{ fontSize: "0.68rem", color: T.accent, marginTop: 4 }}>{product.price}</div>}
      </div>
    </div>
  );
}

function IndexingProgress({ progress }) {
  const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "24px 28px" }}>
      <div style={{ fontSize: "0.68rem", color: T.accent, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 16 }}>Indexing in progress</div>
      <div style={{ height: 2, background: T.border, borderRadius: 2, marginBottom: 12, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg, ${T.accentDeep}, ${T.accent})`, borderRadius: 2, transition: "width 0.4s ease" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
          {progress.current && <span>Processing: <span style={{ color: T.text }}>{progress.current}</span></span>}
        </div>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>{progress.done} / {progress.total}</div>
      </div>
      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ fontSize: "0.78rem" }}><span style={{ color: T.green }}>✓ {progress.success}</span><span style={{ color: T.textMuted }}> indexed</span></div>
        {progress.failed > 0 && <div style={{ fontSize: "0.78rem" }}><span style={{ color: T.red }}>✕ {progress.failed}</span><span style={{ color: T.textMuted }}> failed</span></div>}
      </div>
    </div>
  );
}

function DoneCard({ progress, errors, storeName }) {
  return (
    <div style={{ background: progress.success > 0 ? "rgba(122,171,138,0.05)" : "rgba(201,112,112,0.05)", border: `1px solid ${progress.success > 0 ? T.green : T.red}`, borderRadius: 12, padding: "24px 28px" }}>
      <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.4rem", color: progress.success > 0 ? T.green : T.red, marginBottom: 6 }}>
        {progress.success > 0 ? "Done!" : "Indexing failed"}
      </div>
      <div style={{ fontSize: "0.82rem", color: T.textMuted, marginBottom: 16 }}>
        {progress.success > 0 && <><span style={{ color: T.green }}>{progress.success} products</span> from <strong style={{ color: T.text }}>{storeName}</strong> are now searchable in Locus.</>}
        {progress.failed > 0 && <> <span style={{ color: T.red }}>{progress.failed} failed.</span></>}
      </div>
      {errors.length > 0 && (
        <details>
          <summary style={{ fontSize: "0.72rem", color: T.textMuted, cursor: "pointer", marginBottom: 8 }}>Show {errors.length} errors</summary>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {errors.map((e, i) => <div key={i} style={{ fontSize: "0.68rem", color: T.red, fontFamily: "'DM Mono', monospace" }}>{e}</div>)}
          </div>
        </details>
      )}
    </div>
  );
}