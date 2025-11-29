'use client';

import { 
  BarChart, Bar, AreaChart, Area, ScatterChart, Scatter, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend, ZAxis, ReferenceArea
} from 'recharts';
import { RiskCategory } from '@/data/mockData';

// --- Shared Colors ---
// Keeping these distinct but ensuring they work well on white
export const COLORS: Record<string, string> = {
  'operational_reliability': '#f97316', // Orange
  'security_malicious_use': '#ef4444', // Red
  'legal_regulatory_compliance': '#eab308', // Yellow
  'workforce_human_capital': '#8b5cf6', // Purple
  'societal_ethical_reputational': '#3b82f6', // Blue
  'frontier_systemic': '#ec4899', // Pink
  'default': '#cbd5e1' // Slate 300
};

export const SPEC_COLORS: Record<string, string> = {
  'Concrete': '#0ea5e9', // Sky 500
  'Contextual': '#f59e0b', // Amber 500
  'Boilerplate': '#94a3b8' // Slate 400
};

export const SECTOR_COLORS: Record<string, string> = {
  'Financials': '#0f172a', // Slate 900
  'Technology': '#3b82f6', // Blue 500
  'Healthcare': '#ef4444', // Red 500
  'Industrials': '#f59e0b', // Amber 500
  'Energy': '#10b981', // Emerald 500
  'Consumer Discretionary': '#8b5cf6', // Violet 500
};

const formatLabel = (val: string) => {
  return val.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

// --- Component 1: Bar Chart (Risk Distribution) ---
interface OverviewChartsProps {
  data: {
    name: string;
    value: number;
  }[];
}

export function OverviewCharts({ data }: OverviewChartsProps) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          layout="vertical"
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
          <XAxis type="number" hide />
          <YAxis 
            dataKey="name" 
            type="category" 
            width={100}
            tickFormatter={(val) => val.split('_')[0]} 
            tick={{ fontSize: 12, fill: '#64748b' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            cursor={{ fill: '#f8fafc' }}
            contentStyle={{ 
              backgroundColor: '#fff',
              borderRadius: '6px', 
              border: '1px solid #e2e8f0', 
              boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
              color: '#1e293b'
            }}
            itemStyle={{ color: '#1e293b' }}
            formatter={(value: number) => [value, 'Mentions']}
            labelFormatter={formatLabel}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[entry.name] || COLORS.default} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component 2: Stacked Area Chart (Trends over Time) ---
interface TrendChartProps {
  data: {
    year: number;
    [key: string]: number; // dynamic risk keys
  }[];
}

export function TrendChart({ data }: TrendChartProps) {
  return (
    <div className="h-[350px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey="year" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#64748b', fontSize: 12 }} 
            dy={10}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#64748b', fontSize: 12 }} 
          />
          <Tooltip 
             contentStyle={{ 
               backgroundColor: '#fff',
               borderRadius: '6px', 
               border: '1px solid #e2e8f0', 
               boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
               color: '#1e293b'
             }}
             itemStyle={{ color: '#1e293b' }}
             labelStyle={{ color: '#64748b', marginBottom: '0.25rem' }}
             labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }} 
            iconType="circle"
          />
          {Object.keys(COLORS).filter(k => k !== 'default').map(risk => (
            <Area 
              key={risk}
              type="monotone" 
              dataKey={risk} 
              stackId="1" 
              stroke={COLORS[risk]} 
              fill={COLORS[risk]} 
              name={formatLabel(risk).split(' ')[0]} 
              strokeWidth={0}
              fillOpacity={0.8}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component 3: Heatmap (Sector vs Risk) ---
interface HeatmapProps {
  data: {
    sector: string;
    risks: Record<RiskCategory, number>;
  }[];
  // removed click handlers
}

export function SectorHeatmap({ data }: HeatmapProps) {
  const risks = Object.keys(COLORS).filter(k => k !== 'default') as RiskCategory[];

  // 1. Calculate max value for normalization
  let maxVal = 0;
  data.forEach(row => {
    risks.forEach(r => {
      if (row.risks[r] > maxVal) maxVal = row.risks[r];
    });
  });

  return (
    <div className="w-full">
      {/* We remove the border from the container to let the grid be the structure */}
      <div className="bg-white">
        {/* 
           GRID DEFINITION:
           - Seamless Grid Layout
           - 150px for Sector Labels
           - Remaining space split equally
           - Gap-1 creates the "grid lines" using the background
        */}
        <div className="grid grid-cols-[150px_repeat(6,1fr)] w-full gap-px bg-slate-200 border border-slate-200 rounded-lg overflow-hidden">
          
          {/* Header Row - Darker background for header */}
          <div className="bg-slate-50 p-3 text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center">
            Sector
          </div>
          {risks.map(r => (
            <div key={r} className="bg-slate-50 p-3 text-[10px] font-semibold text-slate-500 text-center uppercase tracking-wider flex items-center justify-center h-12">
              {r.split('_')[0]}
            </div>
          ))}

          {/* Data Rows */}
          {data.map(row => (
            <div key={`row-${row.sector}`} style={{ display: 'contents' }}>
              {/* Sector Name Cell */}
              <div 
                className="bg-white px-3 py-2 text-sm font-medium text-slate-700 flex items-center h-16"
              >
                {row.sector}
              </div>

              {/* Risk Cells - Full bleed color */}
              {risks.map(r => {
                const val = row.risks[r] || 0;
                const intensity = val / maxVal; // 0 to 1
                const baseColor = COLORS[r];
                
                return (
                  <div 
                    key={`${row.sector}-${r}`} 
                    className="bg-white h-16 relative group cursor-default"
                    title={`${row.sector} - ${formatLabel(r)}: ${val} mentions`}
                  >
                    {/* Background Color Layer - Fills the whole cell */}
                    <div 
                      className="absolute inset-0 transition-opacity"
                      style={{ 
                        backgroundColor: baseColor,
                        opacity: intensity === 0 ? 0 : Math.max(0.05, intensity * 0.9),
                      }}
                    />
                    
                    {/* Number Layer */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className={`text-sm font-bold ${intensity > 0.5 ? 'text-white' : 'text-slate-900'} opacity-90`}>
                        {val > 0 ? val : ''}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Component 4: Governance Scatter Plot (The Cowboys) ---
interface ScatterPlotProps {
  data: {
    name: string;
    x: number; // Risk Mentions (Adoption)
    y: number; // Governance Maturity (0-3) with jitter
    rawY: number; // Original integer for display
    z: number; // Mitigation Gap Score (for tooltip)
    sector: string;
  }[];
}

export function GovernanceScatterPlot({ data }: ScatterPlotProps) {
  return (
    <div className="h-[400px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          
          {/* Quadrant Backgrounds */}
          <ReferenceArea x1={15} x2={30} y1={0} y2={1.5} fill="#fee2e2" fillOpacity={0.3} /> {/* Danger Zone */}
          <ReferenceArea x1={15} x2={30} y1={2.5} y2={3.5} fill="#dcfce7" fillOpacity={0.3} /> {/* Safe Zone */}
          
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Risk Adoption" 
            unit="" 
            label={{ value: 'Adoption Intensity (Risk Mentions)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 12 }}
            tick={{ fill: '#64748b', fontSize: 12 }}
            domain={[0, 'auto']}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="Governance" 
            label={{ value: 'Governance Maturity', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }}
            // Custom tick formatter that rounds to nearest integer to show labels correctly despite jitter
            tickFormatter={(val) => ['None', 'Basic', 'Inter.', 'Advanced'][Math.round(val)] || ''}
            domain={[-0.5, 3.5]}
            ticks={[0, 1, 2, 3]}
            tick={{ fill: '#64748b', fontSize: 12 }}
          />
          <ZAxis dataKey="z" range={[20, 100]} name="Gap Score" /> {/* Smaller dots */}
          
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }} 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                const maturityLabel = ['None', 'Basic', 'Intermediate', 'Advanced'][data.rawY] || 'Unknown';
                return (
                  <div className="bg-white p-3 border border-slate-200 shadow-md rounded-md z-50">
                    <p className="font-bold text-slate-900">{data.name}</p>
                    <p className="text-sm text-slate-500">{data.sector}</p>
                    <div className="mt-2 text-xs space-y-1">
                      <p>Adoption: <span className="font-medium">{data.x} mentions</span></p>
                      <p>Maturity: <span className="font-medium">{maturityLabel}</span></p>
                      <p className={data.z > 0.5 ? 'text-red-600 font-bold' : 'text-slate-600'}>
                        Gap Score: {data.z.toFixed(2)}
                      </p>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend iconType="circle" />
          {Object.keys(SECTOR_COLORS).map(sector => (
             <Scatter 
                key={sector} 
                name={sector} 
                data={data.filter(d => d.sector === sector)} 
                fill={SECTOR_COLORS[sector]} 
                fillOpacity={0.6} // Transparency for density
             />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component 5: Quality Trend Chart (Time Series) ---
interface QualityTrendChartProps {
  data: {
    year: number;
    Concrete: number;
    Contextual: number;
    Boilerplate: number;
  }[];
}

export function QualityTrendChart({ data }: QualityTrendChartProps) {
  return (
    <div className="h-[350px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey="year" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#64748b', fontSize: 12 }} 
            dy={10}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#64748b', fontSize: 12 }} 
          />
          <Tooltip 
             contentStyle={{ 
               backgroundColor: '#fff',
               borderRadius: '6px', 
               border: '1px solid #e2e8f0', 
               boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
               color: '#1e293b'
             }}
             itemStyle={{ color: '#1e293b' }}
             labelStyle={{ color: '#64748b', marginBottom: '0.25rem' }}
             labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }} 
            iconType="circle"
          />
          {['Boilerplate', 'Contextual', 'Concrete'].map(key => (
            <Area 
              key={key}
              type="monotone" 
              dataKey={key} 
              stackId="1" 
              stroke={SPEC_COLORS[key]} 
              fill={SPEC_COLORS[key]} 
              name={key} 
              strokeWidth={0}
              fillOpacity={0.8}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
