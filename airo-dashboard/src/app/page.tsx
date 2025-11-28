'use client';

import { OverviewCharts, TrendChart, SectorHeatmap } from '@/components/overview-charts';
import { MOCK_DATA, RiskCategory } from '@/data/mockData';

export default function Dashboard() {
  // --- 1. Filter Data (Latest Year) ---
  const currentYear = 2024;
  const currentData = MOCK_DATA.filter(d => d.year === currentYear);

  // --- 2. Prepare Chart Data ---
  
  // A. Bar Chart: Risk Distribution
  const riskCounts: Record<string, number> = {};
  currentData.forEach(d => {
    if (d.dominant_risk) {
      riskCounts[d.dominant_risk] = (riskCounts[d.dominant_risk] || 0) + 1;
    }
  });
  const riskChartData = Object.entries(riskCounts).map(([name, value]) => ({ name, value }));

  // B. Line Chart: Trends (2020-2024)
  const trendDataMap: Record<number, { year: number; [key: string]: number }> = {};
  MOCK_DATA.forEach(d => {
    if (!trendDataMap[d.year]) trendDataMap[d.year] = { year: d.year };
    if (d.dominant_risk) {
      trendDataMap[d.year][d.dominant_risk] = (trendDataMap[d.year][d.dominant_risk] || 0) + 1;
    }
  });
  const trendChartData = Object.values(trendDataMap).sort((a, b) => a.year - b.year);

  // C. Heatmap: Sector vs Risk
  const sectorMap: Record<string, Record<string, number>> = {};
  currentData.forEach(d => {
    if (!sectorMap[d.sector]) sectorMap[d.sector] = {} as any;
    if (d.dominant_risk) {
      sectorMap[d.sector][d.dominant_risk] = (sectorMap[d.sector][d.dominant_risk] || 0) + 1;
    }
  });
  const heatmapData = Object.entries(sectorMap).map(([sector, risks]) => ({
    sector,
    risks: risks as Record<RiskCategory, number>
  }));

  // --- 3. The "Cowboys" (High Risk Companies) ---
  const cowboys = currentData
    .filter(d => d.mitigation_gap_score > 0.5)
    .sort((a, b) => b.mitigation_gap_score - a.mitigation_gap_score)
    .slice(0, 5);

  return (
    <div className="bg-white min-h-screen font-sans">
      
      {/* Header - Cleanest possible version: just Title and Subtitle */}
      <div className="border-b border-gray-200 bg-white py-8">
        <div className="aisi-container">
           <div>
             <h1 className="text-4xl md:text-5xl font-normal tracking-tight text-slate-900 mb-3">
               AI Risk Observatory
             </h1>
             <p className="text-lg text-slate-500 max-w-3xl font-light">
               Analyzing annual reports to track how firms are disclosing and mitigating Artificial Intelligence risks.
             </p>
           </div>
        </div>
      </div>

      <div className="aisi-container py-12">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-12">
          
          {/* Left Sidebar - Minimal Filters */}
          <div className="hidden lg:block lg:col-span-1 space-y-8">
             <div>
               <h3 className="text-sm font-medium text-slate-900 mb-4 uppercase tracking-wider">Sectors</h3>
               <div className="space-y-2">
                 {['Technology', 'Finance', 'Healthcare', 'Industrial', 'Energy'].map((s) => (
                   <div key={s} className="flex items-center gap-3 group cursor-pointer">
                     <div className="w-4 h-4 border border-gray-300 rounded-sm group-hover:border-slate-900 transition-colors"></div>
                     <span className="text-slate-600 group-hover:text-slate-900 transition-colors">{s}</span>
                   </div>
                 ))}
               </div>
             </div>

             <div>
               <h3 className="text-sm font-medium text-slate-900 mb-4 uppercase tracking-wider">Risk Categories</h3>
               <div className="space-y-2">
                 {['Operational', 'Security', 'Compliance', 'Workforce', 'Ethical', 'Frontier'].map((s) => (
                   <div key={s} className="flex items-center gap-3 group cursor-pointer">
                     <div className="w-4 h-4 border border-gray-300 rounded-sm group-hover:border-slate-900 transition-colors"></div>
                     <span className="text-slate-600 group-hover:text-slate-900 transition-colors">{s}</span>
                   </div>
                 ))}
               </div>
             </div>
          </div>

          {/* Main Grid - Content Cards */}
          <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Card 1: Risk Trend */}
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow bg-white col-span-1 md:col-span-2">
               <div className="mb-6">
                 <h2 className="text-xl font-medium text-slate-900">Risk Evolution (2020-2024)</h2>
               </div>
               <div className="h-[300px]">
                 <TrendChart data={trendChartData} />
               </div>
            </div>

            {/* Card 2: Sector Heatmap */}
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow bg-white col-span-1 md:col-span-2">
               <div className="mb-6">
                 <h2 className="text-xl font-medium text-slate-900">Sector Risk Concentration</h2>
               </div>
               <div className="overflow-x-auto">
                 <SectorHeatmap data={heatmapData} />
               </div>
            </div>

            {/* Card 3: Risk Distribution */}
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow bg-white">
               <div className="mb-6">
                 <h2 className="text-xl font-medium text-slate-900">Primary Risks</h2>
               </div>
               <OverviewCharts data={riskChartData} />
            </div>

            {/* Card 4: High Risk Firms (Cowboys) */}
            <div className="border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow bg-white">
               <div className="mb-6">
                 <h2 className="text-xl font-medium text-slate-900">High Mitigation Gaps</h2>
               </div>
               <div className="space-y-4">
                  {cowboys.map((firm) => (
                    <div key={firm.firm_id} className="pb-4 border-b border-gray-100 last:border-0 last:pb-0">
                      <h4 className="font-medium text-slate-900">{firm.firm_name}</h4>
                      <div className="flex justify-between mt-1">
                        <span className="text-sm text-slate-500">{firm.sector}</span>
                        <span className="text-sm font-medium text-red-600">Gap: {firm.mitigation_gap_score.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
               </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}
