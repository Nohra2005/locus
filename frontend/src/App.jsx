// placeholder data for your grid
const MOCK_DISCOVERIES = [
  { id: 1, name: "Product match 1" },
  { id: 2, name: "Product match 2" },
  { id: 3, name: "Product match 3" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-[#0e0e0e] text-white flex flex-col font-sans selection:bg-[#a68e64]/30">
      
      {/* header section */}
      <header className="w-full flex items-center justify-between px-8 border-b border-[#1a1a1a] h-17.5 shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-9 h-9 bg-[#a68e64] rounded-xl flex items-center justify-center shadow-[0_0_15px_rgba(166,142,100,0.2)]">
            <div className="w-3.5 h-3.5 rounded-full border-2 border-dashed border-[#0e0e0e]"></div>
          </div>
          
          <div className="flex items-baseline gap-2.5">
            <h1 className="text-xl font-bold tracking-tight">locus</h1>
            <span className="text-[9px] text-gray-500 font-bold tracking-[0.2em] uppercase whitespace-nowrap">
              Shopping made easier
            </span>
          </div>
        </div>

        <nav className="flex items-center gap-6">
          <button className="px-4 py-1.5 rounded-full bg-[#1a1a1a] text-[10px] font-bold text-white border border-[#333] tracking-wide">
            Discover
          </button>
          <button className="text-[10px] font-bold text-gray-500 hover:text-white uppercase tracking-wide">Saved</button>
          <button className="text-[10px] font-bold text-gray-500 hover:text-white uppercase tracking-wide">History</button>
        </nav>
      </header>

      {/* main view wrapper - forced to fit screen height */}
      <div className="flex flex-col h-[calc(100vh-70px)]">
        
        {/* hero & upload container - centered in the remaining height */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 max-h-212.5">
          
          {/* hero section - tighter margins */}
          <div className="text-center mb-8">
            <div className="text-[#a68e64] text-[9px] font-bold tracking-[0.3em] mb-3 flex items-center justify-center gap-2 uppercase">
              <span className="text-xs">✦</span> AI-powered visual search
            </div>
            <h2 className="text-5xl md:text-6xl font-serif mb-4 tracking-tight leading-tight">
              Find what you <span className="italic text-[#d2be9b]">see</span>
            </h2>
            <p className="text-gray-500 text-sm md:text-base max-w-lg mx-auto">
              Upload any photo and we'll match it against thousands of products across top stores.
            </p>
          </div>

          {/* upload dropzone - scaled for visibility */}
          <div className="w-full max-w-2xl border border-dashed border-[#222] bg-[#111] rounded-4xl p-12 md:p-16 flex flex-col items-center justify-center transition-all hover:border-[#333] hover:bg-[#141414] cursor-pointer group relative">
            
            <div className="w-14 h-14 rounded-full border border-[#222] flex items-center justify-center mb-6 bg-[#181818]">
              <svg className="w-5 h-5 text-[#a68e64]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>

            <h3 className="text-xl font-bold mb-2 tracking-tight">Drop your photo here</h3>
            <p className="text-gray-500 text-[10px] font-serif italic mb-8">Match products instantly</p>

            <button className="px-8 py-3 rounded-full border border-[#2a2a2a] bg-[#181818] text-[#a68e64] text-[10px] font-bold tracking-[0.15em] uppercase hover:bg-[#222]">
              Select file
            </button>
          </div>
        </div>

        {/* hint for scrolling */}
        <div className="flex flex-col items-center pb-4 opacity-40">
          <span className="text-[9px] uppercase tracking-[0.2em] mb-2">Scroll to explore</span>
          <div className="w-px h-8 bg-linear-to-b from-[#a68e64] to-transparent"></div>
        </div>
      </div>

      {/* discoveries section - starts after the fold */}
      <section className="w-full max-w-5xl mx-auto py-24 px-6">
        <div className="flex justify-between items-end mb-10 border-b border-[#1a1a1a] pb-5">
          <h4 className="text-[10px] font-bold tracking-[0.25em] uppercase text-gray-500">Recent discoveries</h4>
          <button className="text-[10px] font-bold text-[#a68e64] uppercase tracking-widest">View all →</button>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
          {MOCK_DISCOVERIES.map((item) => (
            <div key={item.id} className="aspect-4/5 bg-[#111] rounded-4xl border border-[#1a1a1a] flex items-center justify-center group cursor-pointer hover:border-[#a68e64]/50 transition-all">
              <span className="text-gray-600 font-serif italic text-sm">{item.name}</span>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}