'use client';

import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
  LineChart, Line,
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
  allowLineChart?: boolean;
  title?: string;
  subtitle?: string;
}

export function StackedBarChart({ data, xAxisKey, stackKeys, colors = COLORS, allowLineChart = false, title, subtitle }: StackedBarChartProps) {
  const [chartType, setChartType] = useState<'bar' | 'line'>('bar');

  const sharedAxisProps = {
    xAxis: {
      dataKey: xAxisKey,
      axisLine: false,
      tickLine: false,
      tick: { fill: '#64748b', fontSize: 12 },
      dy: 10,
    } as const,
    yAxis: {
      axisLine: false,
      tickLine: false,
      tick: { fill: '#64748b', fontSize: 12 },
    } as const,
  };

  const tooltipProps = {
    cursor: chartType === 'bar' ? { fill: '#f8fafc' } : { stroke: '#e2e8f0' },
    contentStyle: {
      backgroundColor: '#fff',
      borderRadius: '6px',
      border: '1px solid #e2e8f0',
      boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
      color: '#1e293b',
    },
    itemStyle: { color: '#1e293b' },
    formatter: (value: number, name: string) => [value, formatLabel(name)] as [number, string],
  };

  return (
    <div className="w-full rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm relative">
      {title && (
        <h3 className="mb-1 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          {title}
        </h3>
      )}
      {subtitle && (
        <p className="mb-2 text-xs leading-relaxed text-slate-500">{subtitle}</p>
      )}
      {allowLineChart && (
        <div className="absolute top-3 right-3 z-10 flex rounded-lg border border-slate-200 bg-white overflow-hidden shadow-sm">
          <button
            onClick={() => setChartType('bar')}
            className={`px-2.5 py-1.5 text-xs font-medium transition-colors ${chartType === 'bar' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:text-slate-700'}`}
            title="Bar chart"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="1" y="6" width="3" height="7" rx="0.5" fill="currentColor"/>
              <rect x="5.5" y="3" width="3" height="10" rx="0.5" fill="currentColor"/>
              <rect x="10" y="1" width="3" height="12" rx="0.5" fill="currentColor"/>
            </svg>
          </button>
          <button
            onClick={() => setChartType('line')}
            className={`px-2.5 py-1.5 text-xs font-medium transition-colors ${chartType === 'line' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:text-slate-700'}`}
            title="Line chart"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M1 12L4.5 6L8 8.5L13 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      )}
      <div className="h-[460px]">
      <ResponsiveContainer width="100%" height="100%">
        {chartType === 'line' ? (
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis {...sharedAxisProps.xAxis} />
            <YAxis {...sharedAxisProps.yAxis} />
            <Tooltip {...tooltipProps} />
            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
            {stackKeys.map((key) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={colors[key] || colors.default}
                strokeWidth={2}
                dot={{ r: 4, fill: colors[key] || colors.default }}
                name={formatLabel(key)}
              />
            ))}
          </LineChart>
        ) : (
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis {...sharedAxisProps.xAxis} />
            <YAxis {...sharedAxisProps.yAxis} />
            <Tooltip {...tooltipProps} />
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
        )}
      </ResponsiveContainer>
      </div>
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
  subtitle?: string;
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
  subtitle,
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
  const minGridWidth = showTotals
    ? labelCol + (xLabels.length * valueCol) + totalCol
    : labelCol + (xLabels.length * valueCol);
  const gridCols = showTotals
    ? `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr)) ${totalCol}px`
    : `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr))`;

  const needsScroll = yLabels.length > 15;
  const cellHeight = needsScroll ? 36 : 44;

  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
      {title && (
        <h3 className="mb-1 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          {title}
        </h3>
      )}
      {subtitle && (
        <p className="mb-4 text-xs leading-relaxed text-slate-500">{subtitle}</p>
      )}
      {!subtitle && title && <div className="mb-3" />}
      <div className={needsScroll ? 'max-h-[600px] overflow-y-auto' : ''}>
      <div
        className="grid w-full gap-px overflow-hidden rounded-lg border border-slate-200 bg-slate-200"
        style={{ gridTemplateColumns: gridCols, minWidth: `${minGridWidth}px` }}
      >
        {/* Header Row */}
        <div className={`bg-slate-50 p-3 text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center justify-center ${needsScroll ? 'sticky top-0 z-10' : ''}`}>
          {/* Top-Left Corner */}
        </div>
        {xLabels.map(x => (
          <div key={x} className={`bg-slate-50 px-1 py-2 text-[10px] font-semibold text-slate-700 text-center flex items-center justify-center min-h-[60px] leading-tight ${needsScroll ? 'sticky top-0 z-10' : ''}`}>
            {xLabelFormatter(x)}
          </div>
        ))}
        {showTotals && (
          <div className={`bg-slate-100 px-1 py-2 text-[10px] font-bold text-slate-600 text-center flex items-center justify-center min-h-[60px] ${needsScroll ? 'sticky top-0 z-10' : ''}`}>
            Total
          </div>
        )}

        {/* Data Rows */}
        {yLabels.map(y => (
          <div key={`row-${y}`} className="contents">
            <div className={`bg-white px-3 py-2 text-sm font-medium text-slate-700 flex items-center border-r border-slate-100 leading-tight`} style={{ height: cellHeight }}>
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
                  className={`relative group flex items-center justify-center ${isBlindSpot ? 'bg-slate-50' : 'bg-white'}`}
                  style={{ height: cellHeight }}
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
              <div className="bg-slate-50 flex items-center justify-center" style={{ height: cellHeight }}>
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
            <div className="bg-slate-100 px-3 py-2 text-sm font-bold text-slate-600 flex items-center" style={{ height: cellHeight }}>
              Total
            </div>
            {xLabels.map(x => (
              <div key={`total-${x}`} className="bg-slate-50 flex items-center justify-center" style={{ height: cellHeight }}>
                <span className="text-sm font-semibold text-slate-600">
                  {colTotals.get(x) || 0}
                </span>
              </div>
            ))}
            <div className="bg-slate-100 flex items-center justify-center" style={{ height: cellHeight }}>
              <span className="text-sm font-bold text-slate-700">
                {grandTotal}
              </span>
            </div>
          </div>
        )}
      </div>

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
