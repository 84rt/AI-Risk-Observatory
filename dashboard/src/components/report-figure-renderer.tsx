'use client';

import React from 'react';
import clsx from 'clsx';
import Link from 'next/link';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  Treemap,
  XAxis,
  YAxis,
} from 'recharts';

import type {
  ReportFigure,
  ReportFigureGroupedBar,
  ReportFigureHeatmap,
  ReportFigureHorizontalBar,
  ReportFigureLine,
  ReportFigureTreemap,
} from '@/lib/report-figures';

type RenderMode = 'preview' | 'export';
type FigureRenderContext = {
  mode: RenderMode;
  chartWidth?: number;
  chartHeight: number;
};

const AXIS_TICK = { fill: '#6f777b', fontSize: 12 };
const GRID_STROKE = '#e5e7eb';

const defaultTooltipContentStyle = {
  borderRadius: '12px',
  borderColor: '#e5e7eb',
  boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
  fontSize: '12px',
};

const figureViewport = (figure: ReportFigure) => {
  switch (figure.id) {
    case 'figure3':
    case 'figure6':
      return { width: 1280, height: 860 };
    case 'figure4':
      return { width: 1280, height: 820 };
    case 'figure7':
      return { width: 1320, height: 900 };
    default:
      return { width: 1280, height: 760 };
  }
};

const chartHeightFor = (figure: ReportFigure, mode: RenderMode) => {
  const multiplier = mode === 'export' ? 1 : 0.88;
  switch (figure.id) {
    case 'figure3':
    case 'figure6':
      return Math.round(620 * multiplier);
    case 'figure4':
      return Math.round(620 * multiplier);
    case 'figure7':
      return mode === 'export' ? 920 : 860;
    default:
      return Math.round(520 * multiplier);
  }
};

const normalizeHexColor = (hex: string) => {
  if (!hex.startsWith('#')) return null;
  if (hex.length === 4) {
    return `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`;
  }
  if (hex.length === 7) {
    return hex.toLowerCase();
  }
  return null;
};

const transparent = (hex: string, alpha: string) => {
  const normalized = normalizeHexColor(hex);
  if (!normalized) return hex;
  return `${normalized}${alpha}`;
};

const hexToRgb = (hex: string) => {
  const normalized = normalizeHexColor(hex);
  if (!normalized) return null;
  return {
    r: Number.parseInt(normalized.slice(1, 3), 16),
    g: Number.parseInt(normalized.slice(3, 5), 16),
    b: Number.parseInt(normalized.slice(5, 7), 16),
  };
};

const toHexByte = (value: number) => value.toString(16).padStart(2, '0');

const mixHexColors = (from: string, to: string, ratio: number) => {
  const fromRgb = hexToRgb(from);
  const toRgb = hexToRgb(to);

  if (!fromRgb || !toRgb) return from;

  const t = Math.min(Math.max(ratio, 0), 1);
  const mixChannel = (start: number, end: number) => Math.round(start + (end - start) * t);

  return `#${toHexByte(mixChannel(fromRgb.r, toRgb.r))}${toHexByte(mixChannel(fromRgb.g, toRgb.g))}${toHexByte(mixChannel(fromRgb.b, toRgb.b))}`;
};

const isLightColor = (hex: string) => {
  const rgb = hexToRgb(hex);
  if (!rgb) return false;

  const luminance = (0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b) / 255;
  return luminance > 0.64;
};

const rankedRiskChangeColor = (index: number, total: number) => {
  if (total <= 1) return '#b91c1c';
  return mixHexColors('#b91c1c', '#fecaca', index / (total - 1));
};

const isFixedSizeRender = (context: FigureRenderContext) =>
  context.mode === 'export' && typeof context.chartWidth === 'number';

const plotHeight = (context: FigureRenderContext, legendSpace = 0) =>
  Math.max(context.chartHeight - legendSpace, 240);

const FigureLegend = ({ series }: { series: { label: string; color: string }[] }) => (
  <div className="flex flex-wrap gap-4 border-t border-slate-100 pt-4">
    {series.map(item => (
      <div key={item.label} className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
        <span className="text-xs font-medium text-slate-600">{item.label}</span>
      </div>
    ))}
  </div>
);

const FigureShell = ({
  figure,
  mode,
  children,
}: {
  figure: ReportFigure;
  mode: RenderMode;
  children: React.ReactNode;
}) => {
  const viewport = figureViewport(figure);
  const chartHeight = chartHeightFor(figure, mode);

  return (
    <article
      className={clsx(
        mode === 'export'
          ? 'w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-none'
          : 'w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-[0_12px_40px_rgba(15,23,42,0.08)]'
      )}
      style={mode === 'export' ? { width: viewport.width } : undefined}
    >
      <div className="border-b border-slate-100 px-6 py-5 sm:px-7">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="mb-1 text-[11px] font-bold uppercase tracking-[0.14em] text-red-500">{figure.id.toUpperCase()}</p>
            <h2 className="text-xl font-semibold tracking-tight text-slate-950">{figure.title}</h2>
          </div>
          {mode === 'preview' ? (
            <Link
              href={`/report-figures/${figure.id}`}
              className="shrink-0 rounded border border-slate-200 px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.08em] text-slate-600 transition-colors hover:border-slate-300 hover:text-slate-900"
            >
              Open
            </Link>
          ) : null}
        </div>
      </div>
      <div className="px-4 py-5 sm:px-6" style={{ height: chartHeight }}>
        {children}
      </div>
      {mode === 'preview' && figure.notes?.length ? (
        <div className="border-t border-slate-100 bg-slate-50/60 px-6 py-4">
          <ul className="space-y-1 text-sm leading-relaxed text-slate-600">
            {figure.notes.map(note => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
};

const Figure1Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureLine;
  context: FigureRenderContext;
}) => {
  const chartData = figure.data.map(row => ({
    ...row,
    gap_area: row.adoption_minus_risk_gap_pp,
  }));
  const fixed = isFixedSizeRender(context);
  const chart = (
    <ComposedChart
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, 60) : undefined}
      data={chartData}
      margin={{ top: 16, right: 18, left: 8, bottom: 8 }}
    >
      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
      <XAxis dataKey="year" axisLine={false} tickLine={false} tick={AXIS_TICK} />
      <YAxis axisLine={false} tickLine={false} tick={AXIS_TICK} tickFormatter={value => `${value}%`} />
      <Tooltip
        contentStyle={defaultTooltipContentStyle}
        formatter={(value: number, name: string) => [`${value.toFixed?.(1) ?? value}%`, name]}
      />
      <Area dataKey="risk_rate_pct" stackId="gap" stroke="none" fill="transparent" isAnimationActive={false} />
      <Area dataKey="gap_area" stackId="gap" stroke="none" fill={transparent('#e63946', '22')} isAnimationActive={false} />
      {figure.series.map(series => (
        <Line
          key={series.key}
          type="monotone"
          dataKey={series.key}
          name={series.label}
          stroke={series.color}
          strokeWidth={3}
          dot={{ r: 3.5, fill: series.color }}
          activeDot={{ r: 5 }}
          isAnimationActive={false}
        />
      ))}
    </ComposedChart>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      <div className="mt-4">
        <FigureLegend series={figure.series} />
      </div>
    </div>
  );
};

const Figure2Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureLine;
  context: FigureRenderContext;
}) => {
  const fixed = isFixedSizeRender(context);
  const chart = (
    <AreaChart
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, 60) : undefined}
      data={figure.data}
      margin={{ top: 18, right: 16, left: 8, bottom: 8 }}
    >
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
        <XAxis dataKey="year" axisLine={false} tickLine={false} tick={AXIS_TICK} />
        <YAxis axisLine={false} tickLine={false} tick={AXIS_TICK} />
        <Tooltip contentStyle={defaultTooltipContentStyle} />
        {figure.series.map(series => (
          <Area
            key={series.key}
            type="monotone"
            dataKey={series.key}
            name={series.label}
            stackId="adoption"
            stroke={series.color}
            fill={transparent(series.color, '70')}
            strokeWidth={2.5}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      <div className="mt-4">
        <FigureLegend series={figure.series} />
      </div>
    </div>
  );
};

const Figure3Or6Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureHorizontalBar;
  context: FigureRenderContext;
}) => {
  const isGrouped = figure.chartType === 'grouped-horizontal-bar';
  const data = figure.data;
  const fixed = isFixedSizeRender(context);
  const chart = (
    <BarChart
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, isGrouped ? 60 : 0) : undefined}
      data={data}
      layout="vertical"
      margin={{ top: 12, right: 28, left: 40, bottom: 8 }}
      barCategoryGap={isGrouped ? '18%' : '24%'}
    >
      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={GRID_STROKE} />
      <XAxis
        type="number"
        axisLine={false}
        tickLine={false}
        tick={AXIS_TICK}
        tickFormatter={value => `${value}%`}
      />
      <YAxis
        type="category"
        dataKey={figure.id === 'figure6' ? 'sector' : 'label'}
        axisLine={false}
        tickLine={false}
        width={190}
        tick={{ fill: '#475569', fontSize: 12 }}
      />
      <Tooltip
        contentStyle={defaultTooltipContentStyle}
        formatter={(value: number, name: string) => [`${value.toFixed?.(1) ?? value}%`, name]}
      />
      {isGrouped && figure.series ? (
        figure.series.map(series => (
          <Bar
            key={series.key}
            dataKey={series.key}
            name={series.label}
            fill={series.color}
            radius={[0, 4, 4, 0]}
            isAnimationActive={false}
          />
        ))
      ) : (
        <Bar
          dataKey="delta_pp"
          name="YoY change"
          fill="#e63946"
          radius={[0, 4, 4, 0]}
          isAnimationActive={false}
        >
          {data.map((entry, index) => (
            <Cell
              key={`${String(entry.key ?? entry.label)}-${index}`}
              fill={figure.id === 'figure3'
                ? rankedRiskChangeColor(index, data.length)
                : typeof entry.color === 'string'
                  ? entry.color
                  : '#e63946'}
            />
          ))}
        </Bar>
      )}
    </BarChart>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      {isGrouped && figure.series ? (
        <div className="mt-4">
          <FigureLegend series={figure.series} />
        </div>
      ) : null}
    </div>
  );
};

const wrapTreemapLabel = (label: string, width: number) => {
  const normalized = label.replace(/\s*\/\s*/g, ' / ');

  if (width < 180 && normalized.includes(' / ')) {
    const splitParts = normalized.split(' / ');
    if (splitParts.length === 2) {
      return [
        `${splitParts[0]} /`,
        splitParts[1],
      ];
    }
  }

  const maxCharsPerLine = Math.max(8, Math.floor((width - 18) / 8.5));
  const words = normalized.split(' ');
  const lines: string[] = [];
  let current = '';
  let consumedWords = 0;

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;

    if (candidate.length <= maxCharsPerLine || current.length === 0) {
      current = candidate;
      consumedWords += 1;
      continue;
    }

    lines.push(current);
    current = word;
    consumedWords += 1;

    if (lines.length === 2) {
      break;
    }
  }

  if (lines.length < 2 && current) {
    lines.push(current);
  }

  const limited = lines.slice(0, 2);
  if (consumedWords < words.length && limited.length) {
    limited[limited.length - 1] = `${limited[limited.length - 1].replace(/[.,;:]$/, '')}…`;
  }

  return limited.length ? limited : [normalized];
};

const TreemapNode = (props: any) => {
  const { depth, x, y, width, height, index, colors, name, value, fill: nodeFill } = props;

  if (depth === 1) {
    return (
      <g>
        <rect x={x} y={y} width={width} height={height} fill="#f8fafc" stroke="#e5e7eb" />
        {width > 100 && height > 30 ? (
          <text x={x + 10} y={y + 18} fontSize={12} fill="#475569" fontWeight={700}>
            {name}
          </text>
        ) : null}
      </g>
    );
  }

  const fill = typeof nodeFill === 'string' ? nodeFill : colors[index % colors.length];
  const lightFill = isLightColor(fill);
  const labelFill = lightFill ? '#0f172a' : '#ffffff';
  const labelStroke = lightFill ? 'rgba(255,255,255,0.92)' : 'rgba(15,23,42,0.66)';
  const valueFill = lightFill ? '#1e293b' : '#f8fafc';
  const valueStroke = lightFill ? 'rgba(255,255,255,0.86)' : 'rgba(15,23,42,0.6)';
  const wrappedLabel = wrapTreemapLabel(String(name), width);
  const labelFontSize = width < 145 ? 12 : 14;
  const valueY = y + 16 + wrappedLabel.length * (labelFontSize - 1) + 8;

  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} stroke="#ffffff" strokeWidth={2} />
      {width > 90 && height > 46 ? (
        <>
          <text
            x={x + 8}
            y={y + 18}
            fontSize={labelFontSize}
            fill={labelFill}
            fontWeight={900}
            stroke={labelStroke}
            strokeWidth={2.2}
            paintOrder="stroke"
            strokeLinejoin="round"
          >
            {wrappedLabel.map((line, lineIndex) => (
              <tspan key={`${String(name)}-${lineIndex}`} x={x + 8} dy={lineIndex === 0 ? 0 : labelFontSize - 1}>
                {line}
              </tspan>
            ))}
          </text>
          <text
            x={x + 8}
            y={valueY}
            fontSize={13}
            fill={valueFill}
            fontWeight={800}
            stroke={valueStroke}
            strokeWidth={1.8}
            paintOrder="stroke"
            strokeLinejoin="round"
          >
            {value}
          </text>
        </>
      ) : null}
    </g>
  );
};


const Figure4Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureTreemap;
  context: FigureRenderContext;
}) => {
  const groups = [
    {
      name: 'Named tracked',
      children: figure.data.filter(item => item.group === 'named_tracked').map(item => ({
        name: item.label,
        size: item.assignments_2025,
        fill: item.color,
      })),
    },
    {
      name: 'Opaque / untracked',
      children: figure.data.filter(item => item.group === 'opaque_or_untracked').map(item => ({
        name: item.label,
        size: item.assignments_2025,
        fill: item.color,
      })),
    },
    {
      name: 'Internal',
      children: figure.data.filter(item => item.group === 'internal').map(item => ({
        name: item.label,
        size: item.assignments_2025,
        fill: item.color,
      })),
    },
  ];

  const palette = figure.data.map(item => String(item.color));
  const fixed = isFixedSizeRender(context);
  const chart = (
    <Treemap
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, 96) : undefined}
      data={groups}
      dataKey="size"
      stroke="#ffffff"
      aspectRatio={4 / 3}
      isAnimationActive={false}
      content={<TreemapNode colors={palette} />}
    />
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      <div className="mt-4">
        <FigureLegend
          series={figure.data.map(item => ({
            label: String(item.label),
            color: String(item.color),
          }))}
        />
      </div>
    </div>
  );
};

const Figure5Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureLine;
  context: FigureRenderContext;
}) => {
  const chartData = figure.data.map(row => ({
    ...row,
    gap_area: row.quality_gap_pp,
  }));
  const fixed = isFixedSizeRender(context);
  const chart = (
    <ComposedChart
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, 60) : undefined}
      data={chartData}
      margin={{ top: 16, right: 18, left: 8, bottom: 8 }}
    >
      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
      <XAxis dataKey="year" axisLine={false} tickLine={false} tick={AXIS_TICK} />
      <YAxis axisLine={false} tickLine={false} tick={AXIS_TICK} tickFormatter={value => `${value}%`} />
      <Tooltip
        contentStyle={defaultTooltipContentStyle}
        formatter={(value: number, name: string) => [`${value.toFixed?.(1) ?? value}%`, name]}
      />
      <Area dataKey="substantive_risk_rate_pct" stackId="quality-gap" stroke="none" fill="transparent" isAnimationActive={false} />
      <Area dataKey="gap_area" stackId="quality-gap" stroke="none" fill="#fecdd3" isAnimationActive={false} />
      {figure.series.map(series => (
        <Line
          key={series.key}
          type="monotone"
          dataKey={series.key}
          name={series.label}
          stroke={series.color}
          strokeWidth={3}
          dot={{ r: 3.5, fill: series.color }}
          activeDot={{ r: 5 }}
          isAnimationActive={false}
        />
      ))}
    </ComposedChart>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      <div className="mt-4">
        <FigureLegend series={figure.series} />
      </div>
    </div>
  );
};

const heatmapCellColor = (value: number, maxAbs: number) => {
  if (maxAbs <= 0) return '#f8fafc';
  const intensity = Math.min(Math.abs(value) / maxAbs, 1);
  if (value >= 0) {
    const red = 239;
    const green = Math.round(68 + (226 - 68) * (1 - intensity));
    const blue = Math.round(68 + (226 - 68) * (1 - intensity));
    return `rgb(${red}, ${green}, ${blue})`;
  }
  const red = Math.round(59 + (219 - 59) * (1 - intensity));
  const green = Math.round(130 + (234 - 130) * (1 - intensity));
  const blue = 246;
  return `rgb(${red}, ${green}, ${blue})`;
};

const Figure7Chart = ({ figure }: { figure: ReportFigureHeatmap }) => {
  const maxAbs = Math.max(...figure.cells.map(cell => Math.abs(cell.delta_pp)), 0);
  const cellMap = new Map(figure.cells.map(cell => [`${cell.sector}-${cell.year}`, cell]));

  return (
    <div className="flex h-full flex-col">
      <div
        className="grid gap-2"
        style={{
          gridTemplateColumns: `180px repeat(${figure.years.length}, minmax(0, 1fr))`,
        }}
      >
        <div />
        {figure.years.map(year => (
          <div key={year} className="pb-1 text-center text-xs font-bold uppercase tracking-[0.08em] text-slate-500">
            {year}
          </div>
        ))}
        {figure.sectors.map(sector => (
          <React.Fragment key={sector}>
            <div className="flex items-center pr-3 text-sm font-medium text-slate-700">{sector}</div>
            {figure.years.map(year => {
              const cell = cellMap.get(`${sector}-${year}`);
              const value = cell?.delta_pp ?? 0;
              return (
                <div
                  key={`${sector}-${year}`}
                  className="flex h-14 items-center justify-center rounded-md border border-white text-sm font-semibold text-slate-900"
                  style={{ backgroundColor: heatmapCellColor(value, maxAbs) }}
                  title={`${sector}, ${year}: ${value >= 0 ? '+' : ''}${value.toFixed(1)} pp`}
                >
                  {value >= 0 ? '+' : ''}
                  {value.toFixed(1)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div className="mt-5 flex items-center justify-end gap-3 border-t border-slate-100 pt-4">
        <span className="text-xs font-medium text-slate-500">YoY change</span>
        <div className="h-3 w-28 rounded-full bg-gradient-to-r from-blue-400 via-slate-100 to-red-500" />
        <div className="flex gap-3 text-xs text-slate-500">
          <span>Lower</span>
          <span>Higher</span>
        </div>
      </div>
    </div>
  );
};

const Figure8Chart = ({
  figure,
  context,
}: {
  figure: ReportFigureGroupedBar;
  context: FigureRenderContext;
}) => {
  const fixed = isFixedSizeRender(context);
  const chart = (
    <BarChart
      width={fixed ? context.chartWidth : undefined}
      height={fixed ? plotHeight(context, 60) : undefined}
      data={figure.data}
      margin={{ top: 16, right: 16, left: 8, bottom: 12 }}
      barCategoryGap="20%"
    >
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
        <XAxis dataKey="segment_label" axisLine={false} tickLine={false} tick={AXIS_TICK} />
        <YAxis axisLine={false} tickLine={false} tick={AXIS_TICK} tickFormatter={value => `${value}%`} />
        <Tooltip
          contentStyle={defaultTooltipContentStyle}
          formatter={(value: number, name: string) => [`${value.toFixed?.(1) ?? value}%`, name]}
        />
        {figure.series.map(series => (
          <Bar
            key={series.key}
            dataKey={series.key}
            name={series.label}
            fill={series.color}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        {fixed ? chart : <ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer>}
      </div>
      <div className="mt-4">
        <FigureLegend series={figure.series} />
      </div>
    </div>
  );
};

const renderFigureChart = (figure: ReportFigure, context: FigureRenderContext) => {
  switch (figure.id) {
    case 'figure1':
      return <Figure1Chart figure={figure as ReportFigureLine} context={context} />;
    case 'figure2':
      return <Figure2Chart figure={figure as ReportFigureLine} context={context} />;
    case 'figure3':
      return <Figure3Or6Chart figure={figure as ReportFigureHorizontalBar} context={context} />;
    case 'figure4':
      return <Figure4Chart figure={figure as ReportFigureTreemap} context={context} />;
    case 'figure5':
      return <Figure5Chart figure={figure as ReportFigureLine} context={context} />;
    case 'figure6':
      return <Figure3Or6Chart figure={figure as ReportFigureHorizontalBar} context={context} />;
    case 'figure7':
      return <Figure7Chart figure={figure as ReportFigureHeatmap} />;
    case 'figure8':
      return <Figure8Chart figure={figure as ReportFigureGroupedBar} context={context} />;
    default:
      return <div className="text-sm text-slate-500">Unsupported figure</div>;
  }
};

export function ReportFigureRenderer({
  figure,
  mode = 'preview',
}: {
  figure: ReportFigure;
  mode?: RenderMode;
}) {
  const viewport = figureViewport(figure);
  const chartWidth = mode === 'export' ? viewport.width - 48 : undefined;
  const chartHeight = Math.max(chartHeightFor(figure, mode) - 40, 240);

  return (
    <FigureShell figure={figure} mode={mode}>
      {renderFigureChart(figure, {
        mode,
        chartWidth,
        chartHeight,
      })}
    </FigureShell>
  );
}
