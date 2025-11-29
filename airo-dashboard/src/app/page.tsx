'use client';

import { useState } from 'react';
import { OverviewCharts, TrendChart, SectorHeatmap, GovernanceScatterPlot, QualityTrendChart } from '@/components/overview-charts';
import { MOCK_DATA, RiskCategory } from '@/data/mockData';

// --- Reusable Components ---

function SectionHeader({ title, subtitle }: { title: string, subtitle: string }) {
  return (
    <div className="mb-6">
      <h2 className="text-2xl font-light text-slate-900 mb-2">{title}</h2>
      <p className="text-slate-500 text-lg font-light leading-relaxed max-w-3xl">
        {subtitle}
      </p>
    </div>
  );
}

function MethodologyCard({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 text-sm">
      <h4 className="font-medium text-slate-900 mb-2 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
        {title}
      </h4>
      <div className="text-slate-600 space-y-2 leading-relaxed">
        {children}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [selectedSector, setSelectedSector] = useState<string | null>(null);
  
  // --- 1. Filter Data ---
  const currentYear = 2024;
  
  // Apply sector filter if selected
  const filteredHistory = selectedSector 
    ? MOCK_DATA.filter(d => d.sector === selectedSector)
    : MOCK_DATA;
    
  const currentData = filteredHistory.filter(d => d.year === currentYear);

  // --- 2. Prepare Chart Data ---
  
  // A. Bar Chart: Risk Distribution
  const riskCounts: Record<string, number> = {};
  currentData.forEach(d => {
    if (d.dominant_risk) {
      riskCounts[d.dominant_risk] = (riskCounts[d.dominant_risk] || 0) + 1;
    }
  });
  const riskChartData = Object.entries(riskCounts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  // B. Stacked Area Chart: Trends
  const trendDataMap: Record<number, { year: number; [key: string]: number }> = {};
  filteredHistory.forEach(d => {
    if (!trendDataMap[d.year]) trendDataMap[d.year] = { year: d.year };
    if (d.dominant_risk) {
      trendDataMap[d.year][d.dominant_risk] = (trendDataMap[d.year][d.dominant_risk] || 0) + 1;
    }
  });
  const trendChartData = Object.values(trendDataMap).sort((a, b) => a.year - b.year);

  // C. Heatmap: Sector vs Risk
  const allCurrentData = MOCK_DATA.filter(d => d.year === currentYear);
  const uniqueSectors = Array.from(new Set(MOCK_DATA.map(d => d.sector))).sort();
  
  const sectorMap: Record<string, Record<string, number>> = {};
  allCurrentData.forEach(d => {
    if (!sectorMap[d.sector]) sectorMap[d.sector] = {} as any;
    if (d.dominant_risk) {
      sectorMap[d.sector][d.dominant_risk] = (sectorMap[d.sector][d.dominant_risk] || 0) + 1;
    }
  });
  const heatmapData = Object.entries(sectorMap).map(([sector, risks]) => ({
    sector,
    risks: risks as Record<RiskCategory, number>
  }));

  // D. Scatter Plot: Risk vs Governance
  const maturityMap = { 'none': 0, 'basic': 1, 'intermediate': 2, 'advanced': 3 };
  const scatterData = currentData.map(d => {
    // Add jitter to Y for swarm plot effect
    // Base value (0, 1, 2, 3) + random offset (-0.25 to +0.25)
    const baseY = maturityMap[d.governance_maturity] || 0;
    const jitterY = (Math.random() * 0.5) - 0.25; 
    const jitterX = (Math.random() - 0.5) * 0.8; // Add horizontal jitter too
    return {
      name: d.firm_name,
      x: d.risk_mentions_count + jitterX,
      y: baseY + jitterY,
      rawY: baseY, // Keep raw for tooltip mapping
      z: d.mitigation_gap_score,
      sector: d.sector
    };
  }).filter(d => d.x > 0); // Only show firms with risk mentions

  // E. Frontier Signal
  const frontierCount = currentData.filter(d => d.frontier_ai_mentioned).length;
  const frontierPct = Math.round((frontierCount / currentData.length) * 100) || 0;
  const prevYearData = filteredHistory.filter(d => d.year === currentYear - 1);
  const prevFrontierCount = prevYearData.filter(d => d.frontier_ai_mentioned).length;
  const prevFrontierPct = Math.round((prevFrontierCount / prevYearData.length) * 100) || 0;
  const frontierDelta = frontierPct - prevFrontierPct;

  // F. Specificity Trends (Time Series)
  const qualityTrendMap: Record<number, { year: number; Concrete: number; Contextual: number; Boilerplate: number }> = {};
  
  filteredHistory.forEach(d => {
    if (!qualityTrendMap[d.year]) {
        qualityTrendMap[d.year] = { year: d.year, Concrete: 0, Contextual: 0, Boilerplate: 0 };
    }
    const level = d.specificity_level || 'boilerplate';
    const key = level.charAt(0).toUpperCase() + level.slice(1) as 'Concrete' | 'Contextual' | 'Boilerplate';
    // @ts-ignore
    if (qualityTrendMap[d.year][key] !== undefined) {
        // @ts-ignore
        qualityTrendMap[d.year][key]++;
    }
  });
  
  const qualityTrendData = Object.values(qualityTrendMap).sort((a, b) => a.year - b.year);

  return (
    <div className="bg-white min-h-screen font-sans pb-24">
      
      {/* --- Header --- */}
      <div className="sticky top-0 z-50 border-b border-gray-200 bg-white/95 backdrop-blur-sm py-6 shadow-sm">
        <div className="w-full max-w-[1920px] mx-auto px-8 md:px-12">
           <div className="flex justify-between items-center mb-4">
             <div>
               <h1 className="text-3xl font-normal tracking-tight text-slate-900">
                 AI Risk Observatory
               </h1>
             </div>
             <div className="text-sm text-slate-500 font-light">
               Analysis of UK Annual Reports (2024)
             </div>
           </div>
           
           {/* Minimal Filter Menu */}
           <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-2">Filter by Sector:</span>
              <button
                onClick={() => setSelectedSector(null)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  !selectedSector 
                    ? 'bg-slate-900 text-white shadow-md' 
                    : 'bg-white border border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                All Sectors
              </button>
              {uniqueSectors.map(sector => (
                <button
                  key={sector}
                  onClick={() => setSelectedSector(sector)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    selectedSector === sector 
                      ? 'bg-slate-900 text-white shadow-md' 
                      : 'bg-white border border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  {sector}
                </button>
              ))}
           </div>
        </div>
      </div>

      <div className="w-full max-w-[1920px] mx-auto px-8 md:px-12 space-y-24 mt-12">

        {/* --- Section 1: The Cowboy Matrix --- */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-8">
            <SectionHeader 
              title="Adoption vs. Governance Maturity" 
              subtitle="Identifying firms that are deploying AI rapidly without commensurate governance frameworks (the 'Cowboys')."
            />
            <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm h-[450px]">
              <GovernanceScatterPlot data={scatterData} />
            </div>
          </div>
          <div className="lg:col-span-4 flex flex-col justify-center space-y-6">
            <MethodologyCard title="How to read this chart">
              <p>Each dot represents a company. The <strong>X-axis</strong> shows adoption intensity (volume of risk mentions), while the <strong>Y-axis</strong> shows governance maturity (0-3 scale).</p>
              <p className="mt-2"><strong>Target Zone:</strong> Top-right (High Adoption, Advanced Governance).</p>
              <p><strong>Danger Zone:</strong> Bottom-right (High Adoption, No Governance).</p>
            </MethodologyCard>
            <MethodologyCard title="Data Source">
              <p>Governance scores are derived by LLM analysis of annual reports, looking for specific keywords like "AI Ethics Committee", "Model Validation", or "Third-party Audit".</p>
            </MethodologyCard>
          </div>
        </section>

        {/* --- Section 2: Risk Evolution --- */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-4 flex flex-col justify-center space-y-6 order-2 lg:order-1">
             <MethodologyCard title="What defines a 'Risk'?">
               <p>We classify risks into 6 Tier-1 categories based on the <strong>AIRO Taxonomy</strong>.</p>
               <ul className="list-disc pl-4 space-y-1 mt-2 text-slate-500">
                 <li><strong>Operational:</strong> Reliability, hallucination.</li>
                 <li><strong>Security:</strong> Cyber, adversarial attacks.</li>
                 <li><strong>Frontier:</strong> Loss of control, systemic.</li>
               </ul>
             </MethodologyCard>
          </div>
          <div className="lg:col-span-8 order-1 lg:order-2">
            <SectionHeader 
              title="Risk Evolution (2020-2024)" 
              subtitle="Tracking how the composition of disclosed AI risks has shifted over the last 5 years."
            />
            <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm h-[400px]">
              <TrendChart data={trendChartData} />
            </div>
          </div>
        </section>

        {/* --- Section 3: Sector Heatmap --- */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-8">
            <SectionHeader 
              title="Sector Risk Heatmap" 
              subtitle="Comparing risk concentrations across industries. Darker colors indicate higher frequency of mentions."
            />
            <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm">
              <SectorHeatmap 
                data={heatmapData} 
              />
            </div>
          </div>
          <div className="lg:col-span-4 flex flex-col justify-center space-y-6">
            <MethodologyCard title="Interpretation">
              <p>Darker cells indicate a higher concentration of risk mentions. This view helps identify which sectors are most exposed to specific risk types.</p>
            </MethodologyCard>
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-5 text-sm">
              <h4 className="font-medium text-blue-900 mb-2">Key Insight</h4>
              <p className="text-blue-800">
                Financial Services firms are currently leading in "Regulatory Compliance" risks, likely due to the impending EU AI Act and FCA guidelines.
              </p>
            </div>
          </div>
        </section>

        {/* --- Section 4: Disclosure Quality --- */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12 pb-12">
           <div className="lg:col-span-4 flex flex-col justify-center space-y-6 order-2 lg:order-1">
             <MethodologyCard title="The 'Fluff' Detector">
               <p>We use an LLM to grade every disclosure on specificity:</p>
               <ul className="list-disc pl-4 space-y-1 mt-2 text-slate-500">
                 <li><strong>Boilerplate:</strong> Generic statements applicable to any firm.</li>
                 <li><strong>Concrete:</strong> Specific systems, quantified impacts, or named technologies.</li>
               </ul>
             </MethodologyCard>
           </div>
           <div className="lg:col-span-8 order-1 lg:order-2">
             <SectionHeader 
               title="Disclosure Quality Evolution" 
               subtitle="Tracking the substance of AI disclosures over time. Are firms becoming more specific or staying generic?"
             />
             <div className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm h-[400px]">
               <QualityTrendChart data={qualityTrendData} />
             </div>
           </div>
        </section>

      </div>
    </div>
  );
}
