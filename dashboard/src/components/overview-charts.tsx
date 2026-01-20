'use client';

import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

// --- Shared Colors ---
export const COLORS: Record<string, string> = {
  default: '#cbd5e1',
};

const formatLabel = (val: string) => {
  const overrides: Record<string, string> = {
    llm: 'LLM',
    non_llm: 'Non-LLM',
    general_ambiguous: 'General / Ambiguous',
    third_party_supply_chain: 'Third-Party Supply Chain',
    operational_technical: 'Operational / Technical',
    reputational_ethical: 'Reputational / Ethical',
    information_integrity: 'Information Integrity',
    none: 'Unspecified',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

// --- Component: Stacked Bar Chart ---
interface StackedBarChartProps {
  data: any[];
  xAxisKey: string;
  stackKeys: string[];
  colors?: Record<string, string>;
}

export function StackedBarChart({ data, xAxisKey, stackKeys, colors = COLORS }: StackedBarChartProps) {
  return (
    <div className="h-[500px] w-full rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey={xAxisKey} 
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
            cursor={{ fill: '#f8fafc' }}
            contentStyle={{ 
              backgroundColor: '#fff',
              borderRadius: '6px', 
              border: '1px solid #e2e8f0', 
              boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
              color: '#1e293b'
            }}
            itemStyle={{ color: '#1e293b' }}
            formatter={(value: number, name: string) => [value, formatLabel(name)]}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
          {stackKeys.map((key) => (
            <Bar 
              key={key} 
              dataKey={key} 
              stackId="a" 
              fill={colors[key] || colors.default} 
              name={formatLabel(key)}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component: Generic Heatmap ---
// Re-implementing as a CSS Grid for better control than Recharts Scatter/Heatmap
interface GenericHeatmapProps {
  data: {
    x: string | number; // e.g. Year
    y: string | number; // e.g. Sector
    value: number;
  }[];
  xLabels: (string | number)[];
  yLabels: (string | number)[];
  valueFormatter?: (val: number) => string;
  baseColor?: string;
  xLabelFormatter?: (val: string | number) => string;
  yLabelFormatter?: (val: string | number) => string;
}

export function GenericHeatmap({
  data,
  xLabels,
  yLabels,
  valueFormatter = (v) => v.toString(),
  baseColor = '#0ea5e9',
  xLabelFormatter = (val) => val.toString(),
  yLabelFormatter = (val) => val.toString(),
}: GenericHeatmapProps) {
  // Create a lookup map
  const dataMap = new Map<string, number>();
  let maxValue = 0;
  data.forEach(d => {
    const key = `${d.x}-${d.y}`;
    dataMap.set(key, d.value);
    if (d.value > maxValue) maxValue = d.value;
  });

  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
      <div 
        className="grid gap-px bg-slate-200 border border-slate-200 rounded-lg overflow-hidden"
        style={{ 
          gridTemplateColumns: `150px repeat(${xLabels.length}, minmax(100px, 1fr))` 
        }}
      >
        {/* Header Row */}
        <div className="bg-slate-50 p-3 text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-center">
          {/* Top-Left Corner */}
        </div>
        {xLabels.map(x => (
          <div key={x} className="bg-slate-50 p-3 text-sm font-semibold text-slate-700 text-center flex items-center justify-center h-12">
            {xLabelFormatter(x)}
          </div>
        ))}

        {/* Rows */}
        {yLabels.map(y => (
          <div key={`row-${y}`} className="contents">
            <div className="bg-white px-3 py-2 text-sm font-medium text-slate-700 flex items-center h-16 border-r border-slate-100">
              {yLabelFormatter(y)}
            </div>

            {/* Cells */}
            {xLabels.map(x => {
              const val = dataMap.get(`${x}-${y}`) || 0;
              const intensity = maxValue > 0 ? val / maxValue : 0;
              
              // Color scale: White to Blue (or custom)
              // Using Slate/Blue mix for professional look
              const opacity = Math.max(0.05, intensity * 0.95);
              
              return (
                <div 
                  key={`${x}-${y}`} 
                  className="bg-white h-16 relative group flex items-center justify-center"
                  title={`${y} in ${x}: ${val}`}
                >
                  <div 
                    className="absolute inset-0 transition-all duration-300"
                    style={{ 
                      backgroundColor: baseColor,
                      opacity: val === 0 ? 0 : opacity 
                    }}
                  />
                  <span className={`relative z-10 text-sm font-bold ${intensity > 0.5 ? 'text-white' : 'text-slate-700'}`}>
                    {val > 0 ? valueFormatter(val) : '-'}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
