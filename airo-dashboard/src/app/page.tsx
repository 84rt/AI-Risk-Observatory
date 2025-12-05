'use client';

import { useState, useMemo } from 'react';
import { 
  StackedBarChart, 
  GenericHeatmap, 
  COLORS 
} from '@/components/overview-charts';
import { MOCK_DATA, RiskCategory } from '@/data/mockData';

// --- View Definitions ---
const VIEWS = [
  { id: 1, title: 'Risk Trends by Type', description: 'Evolution of risk categories over time.' },
  { id: 2, title: 'Risk Trends by Sector', description: 'Volume of risk mentions across sectors.' },
  { id: 3, title: 'Sector Heatmap', description: 'Intensity of risk reporting by sector and year.' },
  { id: 4, title: 'Confidence Heatmap', description: 'Quality and evidence of risk disclosures.' },
];

export default function Dashboard() {
  const [activeView, setActiveView] = useState<number>(1);

  // --- Data Processing ---
  
  // 1. Risk Trends (Stacked by Risk Type)
  const riskTrendData = useMemo(() => {
    const map = new Map<number, any>();
    // Initialize years
    for (let y = 2021; y <= 2025; y++) map.set(y, { year: y });
    
    MOCK_DATA.forEach(d => {
      if (d.year < 2021 || d.year > 2025) return;
    if (d.dominant_risk) {
        const entry = map.get(d.year);
        entry[d.dominant_risk] = (entry[d.dominant_risk] || 0) + 1;
      }
    });
    return Array.from(map.values()).sort((a, b) => a.year - b.year);
  }, []);

  // 2. Sector Trends (Stacked by Sector)
  const sectorTrendData = useMemo(() => {
    const map = new Map<number, any>();
    for (let y = 2021; y <= 2025; y++) map.set(y, { year: y });
    
    MOCK_DATA.forEach(d => {
      if (d.year < 2021 || d.year > 2025) return;
      if (d.ai_mentioned) { // Count all AI risk mentions
        const entry = map.get(d.year);
        entry[d.sector] = (entry[d.sector] || 0) + 1;
      }
    });
    return Array.from(map.values()).sort((a, b) => a.year - b.year);
  }, []);

  // 3. Sector Heatmap Data (Year x Sector)
  const sectorHeatmapData = useMemo(() => {
    const data: { x: number; y: string; value: number }[] = [];
    const counts = new Map<string, number>();

    MOCK_DATA.forEach(d => {
      if (d.year < 2021 || d.year > 2025) return;
      if (d.ai_mentioned) {
        const key = `${d.year}-${d.sector}`;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    });

    counts.forEach((val, key) => {
      const [year, sector] = key.split('-');
      data.push({ x: parseInt(year), y: sector, value: val });
    });
    return data;
  }, []);

  // 4. Confidence Heatmap Data (Year x Confidence Score)
  const confidenceHeatmapData = useMemo(() => {
    const data: { x: number; y: number; value: number }[] = [];
    const counts = new Map<string, number>();

    MOCK_DATA.forEach(d => {
      if (d.year < 2021 || d.year > 2025) return;
      if (d.ai_mentioned) {
        // Only count mentions, assuming confidence_score is relevant only if AI mentioned
        // Actually mockData logic assigns confidence score if ai_mentioned is true.
        // It defaults to 0 but logic updates it.
        const key = `${d.year}-${d.confidence_score}`;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    });

    counts.forEach((val, key) => {
      const [year, score] = key.split('-');
      data.push({ x: parseInt(year), y: parseInt(score), value: val });
    });
    return data;
  }, []);

  // --- Constants ---
  const riskKeys = Object.keys(COLORS).filter(k => 
    !['Financials', 'Technology', 'Healthcare', 'Industrials', 'Energy', 'Consumer Discretionary', '0', '1', '2', 'default'].includes(k)
  );
  
  const sectorKeys = ['Financials', 'Technology', 'Healthcare', 'Industrials', 'Energy', 'Consumer Discretionary'];
  const years = [2021, 2022, 2023, 2024, 2025];
  const confidenceLevels = [2, 1, 0]; // Display High to Low

  return (
    <div className="flex min-h-screen bg-slate-50 font-sans text-slate-900">
      
      {/* --- Sidebar --- */}
      <aside className="w-80 bg-white border-r border-slate-200 flex-shrink-0 fixed h-full z-10 overflow-y-auto">
        <div className="p-8 border-b border-slate-100">
          <h1 className="text-xl font-bold tracking-tight text-slate-900">
                 AI Risk Observatory
               </h1>
          <p className="text-xs text-slate-500 mt-2 font-light">
            Analysis of Annual Reports (2021-2025)
          </p>
           </div>
           
        <nav className="p-6 space-y-2">
          {VIEWS.map((view) => (
              <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`w-full text-left p-4 rounded-lg transition-all duration-200 group ${
                activeView === view.id 
                  ? 'bg-slate-900 text-white shadow-md ring-1 ring-slate-900' 
                  : 'hover:bg-slate-50 text-slate-600 border border-transparent hover:border-slate-200'
              }`}
            >
              <h3 className={`font-semibold text-sm ${activeView === view.id ? 'text-white' : 'text-slate-900'}`}>
                {view.title}
              </h3>
              <p className={`text-xs mt-1 leading-relaxed ${activeView === view.id ? 'text-slate-300' : 'text-slate-500'}`}>
                {view.description}
              </p>
                </button>
              ))}
        </nav>
      </aside>

      {/* --- Main Content --- */}
      <main className="flex-1 ml-80 p-12 max-w-[1600px]">
        <div className="mb-8">
          <h2 className="text-3xl font-light text-slate-900">
            {VIEWS.find(v => v.id === activeView)?.title}
          </h2>
          <p className="text-slate-500 mt-2 text-lg font-light max-w-2xl">
            {VIEWS.find(v => v.id === activeView)?.description}
          </p>
      </div>

        <div className="mt-8">
          {activeView === 1 && (
            <div className="space-y-6">
              <StackedBarChart 
                data={riskTrendData} 
                xAxisKey="year" 
                stackKeys={riskKeys} 
              />
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-sm text-slate-600">
                <p><strong>Insight:</strong> This chart shows the aggregate volume of risk mentions broken down by category. Look for shifting dominance in risk types over the 5-year period.</p>
              </div>
            </div>
          )}

          {activeView === 2 && (
            <div className="space-y-6">
              <StackedBarChart 
                data={sectorTrendData} 
                xAxisKey="year" 
                stackKeys={sectorKeys} 
              />
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-sm text-slate-600">
                <p><strong>Insight:</strong> This chart tracks total risk mentions by industry sector. It highlights which industries are becoming more vocal about AI risks over time.</p>
          </div>
            </div>
          )}

          {activeView === 3 && (
            <div className="space-y-6">
              <GenericHeatmap 
                data={sectorHeatmapData}
                xLabels={years}
                yLabels={sectorKeys}
              />
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-sm text-slate-600">
                <p><strong>Insight:</strong> Darker cells indicate higher reporting intensity. This matrix helps identify sector-specific surges in risk awareness (e.g., Financials in 2024 vs Technology in 2023).</p>
              </div>
            </div>
          )}

          {activeView === 4 && (
            <div className="space-y-6">
              <GenericHeatmap 
                data={confidenceHeatmapData}
                xLabels={years}
                yLabels={confidenceLevels}
                valueFormatter={(val) => `${val} firms`}
              />
               <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="bg-white p-4 rounded-lg border border-slate-200 text-sm">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-slate-900"></span>
                    <strong className="text-slate-900">Score 2 (High)</strong>
                  </div>
                  <p className="text-slate-500">Concrete evidence of governance, specific system mentions, or advanced risk mitigation strategies.</p>
                </div>
                <div className="bg-white p-4 rounded-lg border border-slate-200 text-sm">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-slate-500"></span>
                    <strong className="text-slate-900">Score 1 (Medium)</strong>
          </div>
                  <p className="text-slate-500">Contextual disclosures, awareness of risks but limited specific action or "boilerplate plus" language.</p>
            </div>
                <div className="bg-white p-4 rounded-lg border border-slate-200 text-sm">
                   <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-slate-300"></span>
                    <strong className="text-slate-900">Score 0 (Low)</strong>
          </div>
                  <p className="text-slate-500">Generic boilerplate, minimal detail, or no evidence of governance despite mentioning AI.</p>
           </div>
             </div>
           </div>
          )}
      </div>
      </main>
    </div>
  );
}
