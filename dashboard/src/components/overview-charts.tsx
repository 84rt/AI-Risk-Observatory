'use client';

import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
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
    no_ai_mention: 'No AI Mention',
    no_ai_risk_mention: 'No AI Risk Mention',
    none: 'Unspecified',
    openai: 'OpenAI',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

// --- Component: Info Tooltip ---
// Uses position:fixed so the popup escapes any overflow:hidden/auto ancestor.
export function InfoTooltip({ content }: { content: React.ReactNode }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  const handleEnter = (e: React.MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setPos({ x: rect.left + rect.width / 2, y: rect.top });
  };

  return (
    <span className="relative inline-flex align-middle">
      <button
        type="button"
        className="ml-1.5 inline-flex h-[18px] w-[18px] shrink-0 items-center justify-center text-slate-400 transition-colors hover:text-slate-600"
        aria-label="Chart information"
        onMouseEnter={handleEnter}
        onMouseLeave={() => setPos(null)}
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
          <circle cx="5" cy="5" r="4.3" stroke="currentColor" strokeWidth="1.1" />
          <path d="M5 4.4V7.4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          <circle cx="5" cy="2.7" r="0.7" fill="currentColor" />
        </svg>
      </button>
      {pos && (
        <div
          className="pointer-events-none z-[9999] w-72 rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-xs font-normal leading-relaxed text-slate-600 normal-case tracking-normal shadow-lg"
          style={{
            position: 'fixed',
            left: pos.x,
            top: pos.y,
            transform: 'translate(-50%, calc(-100% - 10px))',
          }}
        >
          {content}
          <div className="absolute left-1/2 top-full h-2 w-2 -translate-x-1/2 -translate-y-[5px] rotate-45 border-b border-r border-slate-200 bg-white" />
        </div>
      )}
    </span>
  );
}

// --- Component: Stacked Bar Chart ---
interface StackedBarChartProps {
  data: Array<Record<string, string | number | null | undefined>>;
  xAxisKey: string;
  stackKeys: string[];
  colors?: Record<string, string>;
  allowLineChart?: boolean;
  title?: string;
  subtitle?: string;
  tooltip?: React.ReactNode;
  headerExtra?: React.ReactNode;
  legendPosition?: 'bottom' | 'right';
  legendKeys?: string[];
  activeLegendKey?: string | null;
  onLegendItemClick?: (key: string) => void;
}

export function StackedBarChart({
  data,
  xAxisKey,
  stackKeys,
  colors = COLORS,
  allowLineChart = false,
  title,
  subtitle,
  tooltip,
  headerExtra,
  legendPosition = 'bottom',
  legendKeys,
  activeLegendKey = null,
  onLegendItemClick,
}: StackedBarChartProps) {
  const [chartType, setChartType] = useState<'bar' | 'line'>('bar');
  const showSideLegend = legendPosition === 'right';
  const visibleLegendKeys = [...(legendKeys ?? stackKeys)].reverse();

  const hasMonthAxis = xAxisKey === 'month';
  const sharedAxisProps = {
    xAxis: {
      dataKey: xAxisKey,
      axisLine: false,
      tickLine: false,
      tick: { fill: '#64748b', fontSize: hasMonthAxis ? 10 : 12 },
      dy: 10,
      ...(hasMonthAxis ? { angle: -45, textAnchor: 'end' as const, height: 60 } : {}),
    },
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
        <h3 className="mb-1 flex items-center gap-0.5 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          {title}
          {tooltip && <InfoTooltip content={tooltip} />}
        </h3>
      )}
      {(allowLineChart || headerExtra) && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          {headerExtra}
          {allowLineChart && (
            <div className="flex rounded-lg border border-slate-200 bg-white overflow-hidden shadow-sm">
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
        </div>
      )}
      <div className={showSideLegend ? 'flex flex-col gap-4 lg:flex-row lg:items-start' : ''}>
        <div className={showSideLegend ? 'h-[460px] w-full lg:flex-1' : 'h-[460px]'}>
          <ResponsiveContainer width="100%" height="100%">
            {chartType === 'line' ? (
              <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis {...sharedAxisProps.xAxis} />
                <YAxis {...sharedAxisProps.yAxis} />
                <Tooltip {...tooltipProps} />
                {!showSideLegend && <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />}
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
                {!showSideLegend && <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />}
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
        {showSideLegend && (
          <div className="w-full rounded-lg border border-slate-200 bg-slate-50/70 p-3 lg:mt-4 lg:w-60">
            <div className="space-y-1">
              {visibleLegendKeys.map((key) => {
                const isSelected = activeLegendKey === key;
                const isDimmed = !!activeLegendKey && !isSelected;
                const itemClass = [
                  'flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-xs transition-colors',
                  isSelected ? 'border-slate-300 bg-white font-semibold text-slate-900' : 'border-transparent',
                  isDimmed ? 'text-slate-400' : 'text-slate-700',
                  onLegendItemClick ? 'hover:border-slate-200 hover:bg-white cursor-pointer' : '',
                ].join(' ');

                if (!onLegendItemClick) {
                  return (
                    <div key={key} className={itemClass}>
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: colors[key] || colors.default }}
                      />
                      <span className="truncate">{formatLabel(key)}</span>
                    </div>
                  );
                }

                return (
                  <button
                    key={key}
                    type="button"
                    className={itemClass}
                    onClick={() => onLegendItemClick(key)}
                    title={isSelected ? `Show all risk types` : `Filter to ${formatLabel(key)}`}
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: colors[key] || colors.default }}
                    />
                    <span className="truncate">{formatLabel(key)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
      {subtitle && (
        <p className="mt-3 border-t border-slate-100 pt-3 text-xs leading-relaxed text-slate-400">{subtitle}</p>
      )}
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
  tooltip?: React.ReactNode;
  headerExtra?: React.ReactNode;
  xAxisLabel?: string;
  yAxisLabel?: string;
  compact?: boolean;
  labelColumnWidth?: number;
  rowHeight?: number;
  yLabelClassName?: string;
}

export function GenericHeatmap({
  data,
  xLabels,
  yLabels,
  valueFormatter = (v) => v.toString(),
  baseColor = '#64748b',
  xLabelFormatter = (val) => val.toString(),
  yLabelFormatter = (val) => val.toString(),
  showTotals = true,
  showBlindSpots = true,
  title,
  subtitle,
  tooltip,
  headerExtra,
  xAxisLabel,
  yAxisLabel,
  compact = false,
  labelColumnWidth,
  rowHeight,
  yLabelClassName,
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

  const longestYLabelLength = yLabels.reduce<number>((max, y) => {
    const formatted = String(yLabelFormatter(y));
    return Math.max(max, formatted.length);
  }, 0);
  const inferredLabelCol = compact
    ? 110
    : Math.max(150, Math.min(320, Math.round(longestYLabelLength * 5.2)));
  const labelCol = labelColumnWidth ?? inferredLabelCol;
  const valueCol = compact ? 44 : 60;
  const totalCol = compact ? 44 : 54;
  const minGridWidth = showTotals
    ? labelCol + (xLabels.length * valueCol) + totalCol
    : labelCol + (xLabels.length * valueCol);
  const gridCols = showTotals
    ? `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr)) ${totalCol}px`
    : `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr))`;

  const needsScroll = yLabels.length > 25;
  const inferredCellHeight = compact
    ? 36
    : longestYLabelLength > 44
      ? 62
      : longestYLabelLength > 30
        ? 52
        : 40;
  const cellHeight = rowHeight ?? (needsScroll ? Math.max(36, inferredCellHeight - 4) : inferredCellHeight);

  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm relative">
      {(title || headerExtra) && (
        <div className="mb-3 flex items-start justify-between gap-4">
          {title ? (
            <h3 className="flex items-center gap-0.5 text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
              {title}
              {tooltip && <InfoTooltip content={tooltip} />}
            </h3>
          ) : <div />}
          {headerExtra && <div className="shrink-0">{headerExtra}</div>}
        </div>
      )}
      <div className={needsScroll ? 'max-h-[800px] overflow-y-auto' : ''}>
      <div
        className="grid w-full gap-px overflow-hidden rounded-lg border border-slate-200 bg-slate-200"
        style={{ gridTemplateColumns: gridCols, minWidth: `${minGridWidth}px` }}
      >
        {/* Header Row */}
        <div className={`relative bg-slate-50 min-h-[60px] overflow-hidden ${needsScroll ? 'sticky top-0 z-10' : ''}`}>
          {xAxisLabel && yAxisLabel ? (
            <>
              <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" aria-hidden="true">
                <line x1="0" y1="0" x2="100%" y2="100%" stroke="#cbd5e1" strokeWidth="1" />
              </svg>
              <span className="absolute right-2 top-2 text-[9px] font-semibold uppercase leading-none tracking-wider text-slate-400">
                {xAxisLabel}
              </span>
              <span className="absolute bottom-2 left-2 text-[9px] font-semibold uppercase leading-none tracking-wider text-slate-400">
                {yAxisLabel}
              </span>
            </>
          ) : null}
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
            <div
              className={`bg-white px-3 py-2 font-medium text-slate-700 flex items-center border-r border-slate-100 leading-tight break-words ${
                compact ? 'text-xs' : 'text-sm'
              } ${yLabelClassName || ''}`}
              style={{ height: cellHeight }}
            >
              {yLabelFormatter(y)}
            </div>

            {/* Cells */}
            {xLabels.map(x => {
              const val = dataMap.get(`${x}-${y}`) || 0;
              // Use log scaling so gradients remain readable when totals grow (e.g. 3-year windows).
              const scaledIntensity = maxValue > 0
                ? Math.log1p(val) / Math.log1p(maxValue)
                : 0;
              const opacity = val > 0 ? (0.12 + scaledIntensity * 0.88) : 0;
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
                    <span className={`relative z-10 text-sm font-bold ${scaledIntensity > 0.6 ? 'text-white' : 'text-slate-700'}`}>
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
          <span>No reports containing specified mention</span>
        </div>
      )}
      {subtitle && (
        <p className="mt-3 border-t border-slate-100 pt-3 text-xs leading-relaxed text-slate-400">{subtitle}</p>
      )}
    </div>
  );
}
