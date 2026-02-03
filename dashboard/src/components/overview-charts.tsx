'use client';

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
} from 'recharts';
import type { LabelMetric } from '@/lib/golden-set';

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

  const gridCols = showTotals
    ? `180px repeat(${xLabels.length}, 70px) 60px`
    : `180px repeat(${xLabels.length}, 70px)`;

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

// --- Component: Metrics Bar Chart (Precision/Recall/F1) ---
interface MetricsBarChartProps {
  data: LabelMetric[];
  title: string;
}

const metricsColors = {
  precision: '#0ea5e9',
  recall: '#14b8a6',
  f1: '#f97316',
};

export function MetricsBarChart({ data, title }: MetricsBarChartProps) {
  const chartData = data.map(d => ({
    label: formatLabel(d.label),
    Precision: Math.round(d.precision * 100),
    Recall: Math.round(d.recall * 100),
    F1: Math.round(d.f1 * 100),
  }));

  return (
    <div className="w-full rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
        {title}
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
            <XAxis
              type="number"
              domain={[0, 100]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={(v) => `${v}%`}
            />
            <YAxis
              type="category"
              dataKey="label"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 11 }}
              width={95}
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
              formatter={(value: number) => [`${value}%`, '']}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
            <Bar dataKey="Precision" fill={metricsColors.precision} radius={[0, 4, 4, 0]} />
            <Bar dataKey="Recall" fill={metricsColors.recall} radius={[0, 4, 4, 0]} />
            <Bar dataKey="F1" fill={metricsColors.f1} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// --- Component: Agreement Score Card ---
interface AgreementCardProps {
  title: string;
  jaccard: number;
  bestLabel?: string;
  bestF1?: number;
}

export function AgreementCard({ title, jaccard, bestLabel, bestF1 }: AgreementCardProps) {
  const jaccardPercent = Math.round(jaccard * 100);
  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-emerald-600';
    if (score >= 40) return 'text-amber-600';
    return 'text-red-500';
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
      <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
        {title}
      </h3>
      <div className="mt-3 flex items-baseline gap-2">
        <span className={`text-3xl font-semibold ${getScoreColor(jaccardPercent)}`}>
          {jaccardPercent}%
        </span>
        <span className="text-sm text-slate-500">Jaccard</span>
      </div>
      {bestLabel && bestF1 !== undefined && (
        <p className="mt-2 text-sm text-slate-600">
          Best: <span className="font-medium">{formatLabel(bestLabel)}</span>{' '}
          <span className="text-slate-400">(F1: {Math.round(bestF1 * 100)}%)</span>
        </p>
      )}
    </div>
  );
}
