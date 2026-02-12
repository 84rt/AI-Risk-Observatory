'use client';

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
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
// CSS Grid heatmap with blind spot indicators and row/column totals
interface GenericHeatmapProps {
  data: {
    x: string | number; // e.g. Year or risk label
    y: string | number; // e.g. Sector
    value: number;
  }[];
  xLabels: (string | number)[];
  yLabels: (string | number)[];
  valueFormatter?: (val: number) => string;
  baseColor?: string;
  xLabelFormatter?: (val: string | number) => string;
  yLabelFormatter?: (val: string | number) => string;
  showTotals?: boolean;
  showBlindSpots?: boolean;
  title?: string;
  compact?: boolean;
}

export function GenericHeatmap({
  data,
  xLabels,
  yLabels,
  valueFormatter = (v) => v.toString(),
  baseColor = '#0ea5e9',
  xLabelFormatter = (val) => val.toString(),
  yLabelFormatter = (val) => val.toString(),
  showTotals = true,
  showBlindSpots = true,
  title,
  compact = false,
}: GenericHeatmapProps) {
  // Create lookup map and compute totals
  const dataMap = new Map<string, number>();
  const rowTotals = new Map<string | number, number>();
  const colTotals = new Map<string | number, number>();
  let maxValue = 0;
  let grandTotal = 0;

  // Initialize totals
  yLabels.forEach(y => rowTotals.set(y, 0));
  xLabels.forEach(x => colTotals.set(x, 0));

  data.forEach(d => {
    const key = `${d.x}-${d.y}`;
    dataMap.set(key, d.value);
    if (d.value > maxValue) maxValue = d.value;
    rowTotals.set(d.y, (rowTotals.get(d.y) || 0) + d.value);
    colTotals.set(d.x, (colTotals.get(d.x) || 0) + d.value);
    grandTotal += d.value;
  });

  const labelCol = compact ? 130 : 180;
  const valueCol = compact ? 64 : 70;
  const totalCol = compact ? 52 : 60;
  const gridCols = showTotals
    ? `${labelCol}px repeat(${xLabels.length}, ${valueCol}px) ${totalCol}px`
    : `${labelCol}px repeat(${xLabels.length}, ${valueCol}px)`;

  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
      {title && (
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          {title}
        </h3>
      )}
      <div
        className="inline-grid gap-px bg-slate-200 border border-slate-200 rounded-lg overflow-hidden"
        style={{ gridTemplateColumns: gridCols }}
      >
        {/* Header Row */}
        <div className="bg-slate-50 p-3 text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-center">
          {/* Top-Left Corner */}
        </div>
        {xLabels.map(x => (
          <div key={x} className="bg-slate-50 px-1 py-2 text-[10px] font-semibold text-slate-700 text-center flex items-center justify-center min-h-[60px] leading-tight">
            {xLabelFormatter(x)}
          </div>
        ))}
        {showTotals && (
          <div className="bg-slate-100 px-1 py-2 text-[10px] font-bold text-slate-600 text-center flex items-center justify-center min-h-[60px]">
            Total
          </div>
        )}

        {/* Data Rows */}
        {yLabels.map(y => (
          <div key={`row-${y}`} className="contents">
            <div className="bg-white px-3 py-2 text-sm font-medium text-slate-700 flex items-center h-[44px] border-r border-slate-100 leading-tight">
              {yLabelFormatter(y)}
            </div>

            {/* Cells */}
            {xLabels.map(x => {
              const val = dataMap.get(`${x}-${y}`) || 0;
              const intensity = maxValue > 0 ? val / maxValue : 0;
              const opacity = Math.max(0.08, intensity * 0.92);
              const isBlindSpot = val === 0 && showBlindSpots;

              return (
                <div
                  key={`${x}-${y}`}
                  className={`h-[44px] relative group flex items-center justify-center ${isBlindSpot ? 'bg-slate-50' : 'bg-white'}`}
                  title={`${yLabelFormatter(y)} × ${xLabelFormatter(x)}: ${val}`}
                >
                  {isBlindSpot ? (
                    // Blind spot indicator - diagonal stripes pattern
                    <div
                      className="absolute inset-0 opacity-40"
                      style={{
                        backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 4px, #cbd5e1 4px, #cbd5e1 5px)',
                      }}
                    />
                  ) : (
                    <div
                      className="absolute inset-0 transition-all duration-300"
                      style={{
                        backgroundColor: baseColor,
                        opacity: val === 0 ? 0 : opacity
                      }}
                    />
                  )}
                  {val > 0 && (
                    <span className={`relative z-10 text-sm font-bold ${intensity > 0.5 ? 'text-white' : 'text-slate-700'}`}>
                      {valueFormatter(val)}
                    </span>
                  )}
                </div>
              );
            })}

            {/* Row Total */}
            {showTotals && (
              <div className="bg-slate-50 h-[44px] flex items-center justify-center">
                <span className="text-sm font-semibold text-slate-600">
                  {rowTotals.get(y) || 0}
                </span>
              </div>
            )}
          </div>
        ))}

        {/* Column Totals Row */}
        {showTotals && (
          <div className="contents">
            <div className="bg-slate-100 px-3 py-2 text-sm font-bold text-slate-600 flex items-center h-[44px]">
              Total
            </div>
            {xLabels.map(x => (
              <div key={`total-${x}`} className="bg-slate-50 h-[44px] flex items-center justify-center">
                <span className="text-sm font-semibold text-slate-600">
                  {colTotals.get(x) || 0}
                </span>
              </div>
            ))}
            <div className="bg-slate-100 h-[44px] flex items-center justify-center">
              <span className="text-sm font-bold text-slate-700">
                {grandTotal}
              </span>
            </div>
          </div>
        )}
      </div>

      {showBlindSpots && (
        <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
          <div
            className="w-4 h-4 rounded border border-slate-200"
            style={{
              backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, #cbd5e1 2px, #cbd5e1 3px)',
            }}
          />
          <span>No reports — potential blind spot</span>
        </div>
      )}
    </div>
  );
}
