'use client';

import { type ReactNode, type RefObject, useEffect, useState } from 'react';
import {
  Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line,
} from 'recharts';

// --- Shared Colors ---
export const COLORS: Record<string, string> = {
  default: '#cbd5e1',
};

const formatLabel = (val: string) => {
  const overrides: Record<string, string> = {
    llm: 'LLM',
    non_llm: 'Traditional AI (non-LLM)',
    risk: 'AI Risk Mentioned',
    general_ambiguous: 'General / Ambiguous',
    third_party_supply_chain: 'Third-Party Supply Chain',
    operational_technical: 'Operational / Technical',
    reputational_ethical: 'Reputational / Ethical',
    information_integrity: 'Information Integrity',
    no_ai_mention: 'No AI Mention',
    no_ai_risk_mention: 'No AI Risk Mention',
    none: 'None / False Positive',
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
  yAxisTickFormatter?: (value: number) => string;
  tooltipValueFormatter?: (value: number, name: string) => string | number;
  allowLineChart?: boolean;
  showChartTypeToggle?: boolean;
  title?: string;
  subtitle?: string;
  tooltip?: React.ReactNode;
  headerExtra?: React.ReactNode;
  legendPosition?: 'bottom' | 'left' | 'right' | 'floating-top-left';
  legendKeys?: string[];
  activeLegendKey?: string | null;
  onLegendItemClick?: (key: string) => void;
  yAxisDomain?: [number, number];
  footerExtra?: React.ReactNode;
  chartType?: 'bar' | 'grouped' | 'line';
  onChartTypeChange?: (type: 'bar' | 'grouped' | 'line') => void;
  showLegend?: boolean;
  exportRef?: RefObject<HTMLDivElement | null>;
  exportWatermark?: ReactNode;
  exportMode?: boolean;
}

type ChartTooltipEntry = {
  color?: string;
  dataKey?: string | number;
  name?: string;
  value?: string | number;
};

export function StackedBarChart({
  data,
  xAxisKey,
  stackKeys = [],
  colors = COLORS,
  yAxisTickFormatter,
  tooltipValueFormatter,
  allowLineChart = false,
  showChartTypeToggle,
  title,
  subtitle,
  tooltip,
  headerExtra,
  legendPosition = 'bottom',
  legendKeys,
  activeLegendKey = null,
  onLegendItemClick,
  yAxisDomain,
  footerExtra,
  chartType,
  onChartTypeChange,
  showLegend = true,
  exportRef,
  exportWatermark,
  exportMode = false,
}: StackedBarChartProps) {
  const [hasMounted, setHasMounted] = useState(false);
  const [internalChartType, setInternalChartType] = useState<'bar' | 'grouped' | 'line'>('bar');

  useEffect(() => {
    setHasMounted(true);
  }, []);

  const resolvedChartType = chartType ?? internalChartType;
  const setResolvedChartType = (nextType: 'bar' | 'grouped' | 'line') => {
    if (onChartTypeChange) {
      onChartTypeChange(nextType);
      return;
    }
    setInternalChartType(nextType);
  };
  const activeChartType = allowLineChart ? resolvedChartType : 'bar';
  const showChartModeToggle = showChartTypeToggle ?? allowLineChart;
  const showLeftLegend = showLegend && legendPosition === 'left';
  const showSideLegend = showLegend && legendPosition === 'right';
  const showFloatingLegend = showLegend && legendPosition === 'floating-top-left';
  const visibleLegendKeys = [...(legendKeys ?? stackKeys ?? [])];
  const isGrouped = activeChartType === 'grouped';
  const showSingleLineArea = activeChartType === 'line' && stackKeys.length === 1;
  const chartInstanceKey = `${activeChartType}-${xAxisKey}-${stackKeys.join('|')}`;
  const renderLegendItems = () =>
    visibleLegendKeys.map((key) => {
      const isSelected = activeLegendKey === key;
      const isDimmed = !!activeLegendKey && !isSelected;
      const itemClass = [
        'flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-xs transition-colors',
        isSelected ? 'border-slate-300 bg-white font-semibold text-slate-900' : 'border-transparent',
        isDimmed ? 'text-slate-400' : 'text-slate-700',
        onLegendItemClick ? 'hover:border-slate-200 hover:bg-secondary/70 cursor-pointer' : '',
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
    });

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
      tickFormatter: yAxisTickFormatter,
      ...(yAxisDomain ? { domain: yAxisDomain } : {}),
    },
  };

  const tooltipProps = {
    cursor: activeChartType === 'bar' ? { fill: '#f3f2f1' } : { stroke: '#b1b4b6' },
    wrapperStyle: { outline: 'none', zIndex: 20 },
    content: ({
      active,
      payload,
      label,
    }: {
      active?: boolean;
      payload?: readonly ChartTooltipEntry[];
      label?: string | number;
    }) => {
      if (!active || !payload || payload.length === 0) return null;

      const rows = payload.filter(
        (entry): entry is Required<Pick<ChartTooltipEntry, 'value'>> & ChartTooltipEntry =>
          entry.value !== undefined && entry.value !== null
      );

      if (rows.length === 0) return null;

      return (
        <div className="min-w-[220px] rounded-lg border border-slate-200 bg-white px-3 py-3 shadow-[0_8px_24px_rgba(15,23,42,0.12)]">
          <div className="mb-2 border-b border-slate-100 pb-2 text-xs font-semibold tracking-[0.08em] text-slate-500">
            {label}
          </div>
          <div className="space-y-2">
            {rows.map((entry, index) => {
              const rawName = entry.name ?? (typeof entry.dataKey === 'string' ? formatLabel(entry.dataKey) : String(entry.dataKey ?? ''));
              const color = entry.color || colors[String(entry.dataKey)] || colors.default;
              const formattedValue =
                typeof entry.value === 'number' && tooltipValueFormatter
                  ? tooltipValueFormatter(entry.value, rawName)
                  : entry.value;

              return (
                <div key={`${String(entry.dataKey)}-${index}`} className="flex items-start justify-between gap-4">
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-sm font-medium leading-tight text-slate-700">
                      {rawName}
                    </span>
                  </div>
                  <span className="shrink-0 text-sm font-semibold leading-none text-slate-900">
                    {formattedValue}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    },
  };

  const chartHeaderControls = !exportMode && (showChartModeToggle || headerExtra) ? (
    <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
      {headerExtra}
      {showChartModeToggle && (
        <div className="flex h-9 overflow-hidden rounded border border-border bg-white p-0.5">
          <button
            onClick={() => setResolvedChartType('bar')}
            className={`h-full rounded-sm px-3 text-[9px] font-bold uppercase tracking-widest transition-all ${activeChartType === 'bar' ? 'bg-primary text-white' : 'text-muted-foreground hover:bg-secondary'}`}
            title="Stacked bar chart"
          >
            Stacked
          </button>
          {allowLineChart && (
            <button
              onClick={() => setResolvedChartType('grouped')}
              className={`h-full rounded-sm px-3 text-[9px] font-bold uppercase tracking-widest transition-all ${activeChartType === 'grouped' ? 'bg-primary text-white' : 'text-muted-foreground hover:bg-secondary'}`}
              title="Grouped bar chart"
            >
              Grouped
            </button>
          )}
          {allowLineChart && (
            <button
              onClick={() => setResolvedChartType('line')}
              className={`h-full rounded-sm px-3 text-[9px] font-bold uppercase tracking-widest transition-all ${activeChartType === 'line' ? 'bg-primary text-white' : 'text-muted-foreground hover:bg-secondary'}`}
              title="Line chart"
            >
              Line
            </button>
          )}
        </div>
      )}
    </div>
  ) : null;

  return (
    <div ref={exportRef} className="relative w-full rounded-lg border border-border bg-white p-5 shadow-[0_1px_2px_rgba(15,23,42,0.05)] sm:p-6">
      {(title || chartHeaderControls) && (
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          {title ? (
            <h3 className="flex min-w-0 items-center gap-2 text-base font-semibold tracking-tight text-primary sm:text-lg">
              <span className="w-1.5 h-1.5 bg-accent" />
              {title}
              {tooltip && <InfoTooltip content={tooltip} />}
            </h3>
          ) : null}
          {chartHeaderControls}
        </div>
      )}
      {showFloatingLegend && (
        <div className="mb-4 mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {renderLegendItems()}
        </div>
      )}
      <div className={showLeftLegend || showSideLegend ? 'flex flex-col gap-3 lg:flex-row lg:items-start' : ''}>
        {showLeftLegend && (
          <div className="w-full rounded border border-border bg-secondary/35 p-3 lg:mt-4 lg:w-56 lg:shrink-0">
            <div className="space-y-1">{renderLegendItems()}</div>
          </div>
        )}
        <div className={`relative ${(showLeftLegend || showSideLegend) ? 'h-[420px] w-full lg:flex-1' : 'h-[420px]'}`}>
          {hasMounted ? (
            <ResponsiveContainer width="100%" height="100%">
              {activeChartType === 'line' ? (
                <LineChart key={chartInstanceKey} data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  {showSingleLineArea && (
                    <defs>
                      {stackKeys.map((key) => (
                        <linearGradient key={key} id={`line-area-${key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={colors[key] || colors.default} stopOpacity={0.22} />
                          <stop offset="65%" stopColor={colors[key] || colors.default} stopOpacity={0.07} />
                          <stop offset="100%" stopColor={colors[key] || colors.default} stopOpacity={0.01} />
                        </linearGradient>
                      ))}
                    </defs>
                  )}
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis {...sharedAxisProps.xAxis} />
                  <YAxis {...sharedAxisProps.yAxis} />
                  <Tooltip {...tooltipProps} />
                  {showLegend && !showSideLegend && !showFloatingLegend && (
                    <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                  )}
                  {showSingleLineArea && stackKeys.map((key) => (
                    <Area
                      key={`${key}-area`}
                      type="monotone"
                      dataKey={key}
                      stroke="none"
                      fill={`url(#line-area-${key})`}
                      isAnimationActive={false}
                    />
                  ))}
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
                <BarChart key={chartInstanceKey} data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }} barCategoryGap={isGrouped ? '20%' : '10%'} barGap={isGrouped ? 2 : 0}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis {...sharedAxisProps.xAxis} />
                  <YAxis {...sharedAxisProps.yAxis} />
                  <Tooltip {...tooltipProps} />
                  {showLegend && !showSideLegend && !showFloatingLegend && (
                    <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                  )}
                  {stackKeys.map((key) => (
                    <Bar
                      key={key}
                      dataKey={key}
                      stackId={isGrouped ? undefined : 'a'}
                      fill={colors[key] || colors.default}
                      name={formatLabel(key)}
                    />
                  ))}
                </BarChart>
              )}
            </ResponsiveContainer>
          ) : (
            <div className="h-full w-full animate-pulse rounded-xl bg-slate-100" aria-hidden="true" />
          )}
        </div>
        {showSideLegend && (
          <div className="w-full rounded border border-border bg-secondary/35 p-3 lg:w-56">
            <div className="space-y-1">{renderLegendItems()}</div>
          </div>
        )}
      </div>
      {!exportMode && footerExtra && <div className="mt-5 border-t border-border pt-4">{footerExtra}</div>}
      {subtitle && (
        <p className="mt-4 border-t border-slate-100 pt-4 text-sm leading-relaxed text-slate-500">{subtitle}</p>
      )}
      {exportMode && exportWatermark ? (
        <div className="pointer-events-none mt-4 border-t border-slate-100 pt-4">
          {exportWatermark}
        </div>
      ) : null}
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
  totalsMode?: 'sum' | 'average';
  totalsLabel?: string;
  totalValueFormatter?: (val: number) => string;
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
  yLabelSubtext?: Record<string | number, string | number>;
  rowGroups?: {
    label: string;
    childKeys: (string | number)[];
  }[];
  expandedRowGroups?: string[];
  onToggleRowGroup?: (groupLabel: string) => void;
  footerExtra?: React.ReactNode;
  exportRef?: RefObject<HTMLDivElement | null>;
  exportWatermark?: ReactNode;
  exportMode?: boolean;
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
  totalsMode = 'sum',
  totalsLabel,
  totalValueFormatter,
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
  yLabelSubtext,
  rowGroups,
  expandedRowGroups = [],
  onToggleRowGroup,
  footerExtra,
  exportRef,
  exportWatermark,
  exportMode = false,
}: GenericHeatmapProps) {
  const rowGroupsByLabel = new Map(
    (rowGroups ?? []).map(group => [String(group.label), group])
  );
  const childToGroupLabel = new Map<string, string>();
  (rowGroups ?? []).forEach(group => {
    group.childKeys.forEach(childKey => {
      childToGroupLabel.set(String(childKey), group.label);
    });
  });
  const expandedRowGroupSet = new Set(expandedRowGroups);

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
  const longestXLabelLength = xLabels.reduce<number>((max, x) => {
    const formatted = String(xLabelFormatter(x));
    return Math.max(max, formatted.length);
  }, 0);
  const inferredLabelCol = compact
    ? 110
    : Math.max(150, Math.min(320, Math.round(longestYLabelLength * 5.2)));
  const labelCol = labelColumnWidth ?? inferredLabelCol;
  const valueCol = compact
    ? 44
    : longestXLabelLength > 20
      ? 106
      : longestXLabelLength > 14
        ? 92
        : longestXLabelLength > 8
          ? 76
          : 60;
  const totalCol = compact ? 44 : 54;
  const columnCount = 1 + xLabels.length + (showTotals ? 1 : 0);
  const gapAndBorderAllowance = Math.max(0, columnCount - 1) + 2;
  const minGridWidth = showTotals
    ? labelCol + (xLabels.length * valueCol) + totalCol + gapAndBorderAllowance
    : labelCol + (xLabels.length * valueCol) + gapAndBorderAllowance;
  const gridCols = showTotals
    ? `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr)) ${totalCol}px`
    : `${labelCol}px repeat(${xLabels.length}, minmax(${valueCol}px, 1fr))`;
  const headerMinHeight = compact
    ? 60
    : longestXLabelLength > 20
      ? 90
      : longestXLabelLength > 14
        ? 82
        : 60;

  const needsScroll = yLabels.length > 25;
  const lockGroupedViewport = Boolean(rowGroups?.length);
  const useVerticalScroll = needsScroll || lockGroupedViewport;
  const inferredCellHeight = compact
    ? 36
    : longestYLabelLength > 44
      ? 62
      : longestYLabelLength > 30
        ? 52
        : 40;
  const cellHeight = rowHeight ?? inferredCellHeight;
  const summaryLabel = totalsLabel ?? (totalsMode === 'average' ? 'Avg' : 'Total');
  const formatSummaryValue = totalValueFormatter ?? valueFormatter;
  const averageOrZero = (sum: number, count: number) => (count > 0 ? sum / count : 0);
  const getRowSummaryValue = (y: string | number) =>
    totalsMode === 'average'
      ? averageOrZero(rowTotals.get(y) || 0, xLabels.length)
      : (rowTotals.get(y) || 0);
  const getColumnSummaryValue = (x: string | number) =>
    totalsMode === 'average'
      ? averageOrZero(colTotals.get(x) || 0, yLabels.length)
      : (colTotals.get(x) || 0);
  const getGrandSummaryValue = () =>
    totalsMode === 'average'
      ? averageOrZero(grandTotal, xLabels.length * yLabels.length)
      : grandTotal;

  return (
    <div ref={exportRef} className="relative w-full rounded-lg border border-border bg-white p-5 shadow-[0_1px_2px_rgba(15,23,42,0.05)] sm:p-6">
      {(title || headerExtra) && (
        <div className="mb-4 flex items-start justify-between gap-4">
          {title ? (
            <h3 className="flex items-center gap-2 text-base font-semibold tracking-tight text-primary sm:text-lg">
              <span className="w-1.5 h-1.5 bg-accent" />
              {title}
              {tooltip && <InfoTooltip content={tooltip} />}
            </h3>
          ) : <div />}
          {!exportMode && headerExtra && <div className="shrink-0">{headerExtra}</div>}
        </div>
      )}
      <div
        className={`overflow-x-auto pb-2 ${
          useVerticalScroll ? 'overflow-y-auto' : ''
        }`}
        style={lockGroupedViewport ? { height: '720px' } : undefined}
      >
      <div
        className="grid w-full gap-px overflow-hidden rounded border border-border bg-border"
        style={{ gridTemplateColumns: gridCols, minWidth: `${minGridWidth}px` }}
      >
        {/* Header Row */}
        <div
          className={`relative bg-secondary overflow-hidden ${useVerticalScroll ? 'sticky top-0 z-10' : ''}`}
          style={{ minHeight: headerMinHeight }}
        >
          {xAxisLabel && yAxisLabel ? (
            <>
              <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" aria-hidden="true">
                <line x1="0" y1="0" x2="100%" y2="100%" stroke="#b1b4b6" strokeWidth="1" />
              </svg>
              <span className="absolute right-2 top-2 text-[9px] font-bold uppercase leading-none tracking-widest text-muted-foreground">
                {xAxisLabel}
              </span>
              <span className="absolute bottom-2 left-2 text-[9px] font-bold uppercase leading-none tracking-widest text-muted-foreground">
                {yAxisLabel}
              </span>
            </>
          ) : null}
        </div>
        {xLabels.map(x => (
          <div
            key={x}
            className={`bg-secondary px-2 py-2 text-[10px] font-bold uppercase tracking-widest text-primary text-center flex items-center justify-center leading-tight ${useVerticalScroll ? 'sticky top-0 z-10' : ''}`}
            style={{ minHeight: headerMinHeight }}
          >
            {xLabelFormatter(x)}
          </div>
        ))}
        {showTotals && (
          <div
            className={`bg-border px-1 py-2 text-[10px] font-bold uppercase tracking-widest text-primary text-center flex items-center justify-center ${useVerticalScroll ? 'sticky top-0 z-10' : ''}`}
            style={{ minHeight: headerMinHeight }}
          >
            {summaryLabel}
          </div>
        )}

        {/* Data Rows */}
        {yLabels.map(y => (
          <div key={`row-${y}`} className="contents">
            {(() => {
              const yKey = String(y);
              const rowGroup = rowGroupsByLabel.get(yKey);
              const parentGroupLabel = childToGroupLabel.get(yKey);
              const isChildRow = Boolean(parentGroupLabel);

              if (rowGroup) {
                const isExpanded = expandedRowGroupSet.has(yKey);

                return (
                  <div
                    className={`border-r border-border px-3 py-2 ${
                      isExpanded
                        ? 'bg-secondary ring-1 ring-inset ring-border'
                        : 'bg-white'
                    } ${
                      compact ? 'text-[11px]' : 'text-xs'
                    } ${yLabelClassName || ''}`}
                    style={{ height: cellHeight }}
                  >
                    {onToggleRowGroup ? (
                      <button
                        type="button"
                        onClick={() => onToggleRowGroup(rowGroup.label)}
                        aria-expanded={isExpanded}
                        className={`flex h-full w-full items-center justify-between gap-3 text-left font-bold uppercase tracking-wider transition ${
                          isExpanded
                            ? 'text-primary'
                            : 'text-muted-foreground hover:text-primary'
                        }`}
                        title={`${isExpanded ? 'Collapse' : 'Expand'} ${rowGroup.label}`}
                      >
                        <span className="min-w-0 break-words leading-tight">{yLabelFormatter(y)}</span>
                        <span className="flex shrink-0 items-center gap-1.5">
                          <span className="rounded-full border border-border bg-white px-1.5 py-0.5 text-[10px] text-muted-foreground">
                            {rowGroup.childKeys.length}
                          </span>
                          <span
                            className={`inline-flex h-5 w-5 items-center justify-center rounded-full border transition-colors ${
                              isExpanded
                                ? 'border-accent/30 bg-accent/8 text-accent'
                                : 'border-border bg-white text-muted-foreground'
                            }`}
                            aria-hidden="true"
                          >
                            <svg
                              viewBox="0 0 16 16"
                              className={`h-3.5 w-3.5 transition-transform duration-200 ${
                                isExpanded ? 'rotate-180' : ''
                              }`}
                              fill="none"
                            >
                              <path
                                d="M4 6.5L8 10L12 6.5"
                                stroke="currentColor"
                                strokeWidth="1.7"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                          </span>
                        </span>
                      </button>
                    ) : (
                      <div className={`flex h-full items-center font-bold uppercase tracking-wider text-primary`}>
                        {yLabelFormatter(y)}
                      </div>
                    )}
                  </div>
                );
              }

              return (
                <div
                  className={`bg-white px-3 py-2 font-bold uppercase tracking-wider text-primary flex items-center border-r border-border leading-tight break-words ${
                    compact ? 'text-[10px]' : 'text-[11px]'
                  } ${
                    isChildRow
                      ? 'border-l-2 border-l-accent bg-secondary/60 pl-6 text-muted-foreground'
                      : ''
                  } ${yLabelClassName || ''}`}
                  style={{ height: cellHeight }}
                >
                  {yLabelSubtext?.[y] !== undefined ? (
                    <div className="flex flex-col gap-0.5">
                      <span>{yLabelFormatter(y)}</span>
                      <span className="text-[9px] font-normal normal-case tracking-normal text-muted-foreground">
                        {yLabelSubtext[y]} {typeof yLabelSubtext[y] === 'number' ? 'companies' : ''}
                      </span>
                    </div>
                  ) : yLabelFormatter(y)}
                </div>
              );
            })()}

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
                  className={`relative group flex items-center justify-center ${isBlindSpot ? 'bg-secondary/40' : 'bg-white'}`}
                  style={{ height: cellHeight }}
                  title={`${yLabelFormatter(y)} × ${xLabelFormatter(x)}: ${valueFormatter(val)}`}
                >
                  {isBlindSpot ? (
                    // Blind spot indicator - diagonal stripes pattern
                    <div
                      className="absolute inset-0 opacity-20"
                      style={{
                        backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 4px, #0b0c0c 4px, #0b0c0c 5px)',
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
                    <span className={`relative z-10 text-xs font-bold ${scaledIntensity > 0.6 ? 'text-white' : 'text-primary'}`}>
                      {valueFormatter(val)}
                    </span>
                  )}
                </div>
              );
            })}

            {/* Row Total */}
            {showTotals && (
              <div className="bg-secondary flex items-center justify-center border-l border-border" style={{ height: cellHeight }}>
                <span className="text-[11px] font-bold text-primary">
                  {formatSummaryValue(getRowSummaryValue(y))}
                </span>
              </div>
            )}
          </div>
        ))}

        {/* Column Totals Row */}
        {showTotals && (
          <div className="contents">
            <div className="bg-border px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-primary flex items-center" style={{ height: cellHeight }}>
              {summaryLabel}
            </div>
            {xLabels.map(x => (
              <div key={`total-${x}`} className="bg-secondary flex items-center justify-center" style={{ height: cellHeight }}>
                <span className="text-[11px] font-bold text-primary">
                  {formatSummaryValue(getColumnSummaryValue(x))}
                </span>
              </div>
            ))}
            <div className="bg-border flex items-center justify-center" style={{ height: cellHeight }}>
              <span className="text-[11px] font-bold text-primary">
                {formatSummaryValue(getGrandSummaryValue())}
              </span>
            </div>
          </div>
        )}
      </div>

      </div>
      {showBlindSpots && (
        <div className="mt-4 flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          <div
            className="w-4 h-4 border border-border"
            style={{
              backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, #b1b4b6 2px, #b1b4b6 3px)',
            }}
          />
          <span>No reports containing specified mention</span>
        </div>
      )}
      {!exportMode && footerExtra && <div className="mt-5 border-t border-border pt-4">{footerExtra}</div>}
      {subtitle && (
        <p className="mt-4 border-t border-border pt-4 text-sm leading-relaxed text-muted">{subtitle}</p>
      )}
      {exportMode && exportWatermark ? (
        <div className="pointer-events-none mt-4 border-t border-border pt-4">
          {exportWatermark}
        </div>
      ) : null}
    </div>
  );
}
