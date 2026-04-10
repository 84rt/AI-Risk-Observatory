'use client';

import Link from 'next/link';
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface ChartSeries {
  label: string;
  subtitle: string;
  data: { year: number; value: number }[];
  color: string;
  linkHref?: string;
}

interface HeroRiskChartProps {
  series: ChartSeries[];
}

export default function HeroRiskChart({ series }: HeroRiskChartProps) {
  const years = Array.from(
    new Set(series.flatMap(item => item.data.map(point => point.year)))
  ).sort((a, b) => a - b);
  const latestYear = years[years.length - 1];

  const chartData = years.map(year => {
    const row: Record<string, number> = { year };
    series.forEach((item, index) => {
      row[`series-${index}`] = item.data.find(point => point.year === year)?.value ?? 0;
    });
    return row;
  });

  const maxValue = Math.max(
    ...chartData.flatMap(row => series.map((_, index) => row[`series-${index}`] ?? 0))
  );
  const yAxisMax = Math.min(100, Math.ceil((maxValue + 8) / 10) * 10);
  const endpointLabelOffsets = [-16, 0, 16];
  const getEndpointLabel = (label: string) => {
    if (label === 'AI as a cybersecurity threat mentions') return 'AI as a cybersecurity threat';
    if (label === 'LLM adoption mentions') return 'LLM adoption';
    if (label === 'AI risk mentions') return 'AI risk';
    return label;
  };

  return (
    <div className="w-full bg-white p-5 sm:p-6">
      <div className="mx-auto max-w-3xl text-center">
        <h2 className="text-2xl font-bold tracking-tight text-primary sm:text-3xl">
          AI Disclosure Trends Over Time
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-muted sm:text-base">
          Share of UK public-company annual reports mentioning AI risk, LLM adoption, and AI-related cybersecurity by filing year.
        </p>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 border-y border-slate-200/90 py-3">
        {series.map(item => {
          const latestValue = item.data[item.data.length - 1]?.value ?? 0;
          return (
            <div key={item.label} className="flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-sm font-semibold text-slate-800">{item.label}</span>
              {item.linkHref ? (
                <Link
                  href={item.linkHref}
                  className="text-xs font-bold tracking-[0.08em] text-primary underline decoration-border underline-offset-4 transition-colors hover:text-accent"
                >
                  see full data
                </Link>
              ) : (
                <span className="text-xs uppercase tracking-[0.08em] text-muted-foreground">
                  {latestValue}% in {latestYear}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-5 rounded-[24px] border border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#fbfcfe_100%)] p-3 sm:p-4">
        <div className="mb-3 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          <span>Share of UK public-company annual reports</span>
          <span>{years[0]}–{latestYear}</span>
        </div>

        <div className="h-[340px] sm:h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 16, right: 140, bottom: 12, left: 0 }}>
              <defs>
                {series.map((item, index) => (
                  <linearGradient
                    key={item.label}
                    id={`hero-area-${index}`}
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor={item.color} stopOpacity={0.22} />
                    <stop offset="65%" stopColor={item.color} stopOpacity={0.07} />
                    <stop offset="100%" stopColor={item.color} stopOpacity={0.01} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid vertical={false} stroke="#e5e7eb" strokeDasharray="3 3" />
              <XAxis
                dataKey="year"
                tick={{ fontSize: 11, fill: '#6f777b' }}
                axisLine={false}
                tickLine={false}
                dy={8}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#6f777b' }}
                axisLine={false}
                tickLine={false}
                width={48}
                domain={[0, yAxisMax]}
                tickFormatter={(value: number) => `${value}%`}
              />
              <Tooltip
                cursor={{ stroke: '#b1b4b6', strokeWidth: 1 }}
                wrapperStyle={{ outline: 'none' }}
                content={({
                  active,
                  payload,
                  label,
                }: {
                  active?: boolean;
                  payload?: ReadonlyArray<{
                    color?: string;
                    dataKey?: string | number;
                    value?: string | number;
                  }>;
                  label?: string | number;
                }) => {
                  if (!active || !payload || payload.length === 0) return null;

                  const rows = payload
                    .filter(
                      (
                        entry
                      ): entry is {
                        color?: string;
                        dataKey?: string | number;
                        value: string | number;
                      } => entry.value !== undefined && entry.value !== null
                    )
                    .filter((entry, index, entries) => {
                      const key = String(entry.dataKey ?? '');
                      return entries.findIndex(candidate => String(candidate.dataKey ?? '') === key) === index;
                    });

                  if (rows.length === 0) return null;

                  return (
                    <div className="min-w-[220px] rounded-xl border border-border bg-white px-4 py-3 shadow-[0_18px_50px_rgba(11,12,12,0.14)]">
                      <div className="mb-3 border-b border-slate-100 pb-2 text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                        {label}
                      </div>
                      <div className="space-y-2.5">
                        {rows.map((entry, index) => {
                          const seriesIndex =
                            typeof entry.dataKey === 'string'
                              ? Number(entry.dataKey.replace('series-', ''))
                              : -1;
                          const item = series[seriesIndex];

                          if (!item) return null;

                          return (
                            <div
                              key={`${String(entry.dataKey)}-${index}`}
                              className="flex items-start justify-between gap-4"
                            >
                              <div className="flex min-w-0 items-center gap-2">
                                <span className="mt-1 inline-flex shrink-0 items-center">
                                  <span
                                    className="inline-block h-3 w-3 shrink-0 rounded-full border border-white shadow-[0_0_0_1px_rgba(148,163,184,0.35)]"
                                    style={{ backgroundColor: item.color }}
                                  />
                                </span>
                                <span className="text-sm font-medium leading-tight text-slate-700">
                                  {item.label}
                                </span>
                              </div>
                              <span className="shrink-0 text-sm font-semibold leading-none text-slate-900">
                                {entry.value}%
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                }}
              />
              {series.map((item, index) => (
                <Area
                  key={`${item.label}-area`}
                  type="monotone"
                  dataKey={`series-${index}`}
                  stroke="none"
                  fill={`url(#hero-area-${index})`}
                  isAnimationActive={true}
                  animationDuration={900}
                />
              ))}
              {series.map((item, index) => (
                <Line
                  key={item.label}
                  type="monotone"
                  dataKey={`series-${index}`}
                  name={item.label}
                  stroke={item.color}
                  strokeWidth={3.25}
                  dot={({ cx, cy, index: pointIndex }) => {
                    if (cx == null || cy == null) return null;
                    const isLast = pointIndex === chartData.length - 1;

                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isLast ? 5 : 3.25}
                        fill={item.color}
                        stroke="#ffffff"
                        strokeWidth={isLast ? 2.5 : 2}
                      />
                    );
                  }}
                  label={({ x, y, index: pointIndex, value }) => {
                    if (pointIndex !== chartData.length - 1 || x == null || y == null) return null;

                    const endpointLabel = getEndpointLabel(item.label);
                    const isCyberLabel = endpointLabel === 'AI as a cybersecurity threat';
                    const labelX = Number(x) + 6;
                    const labelY = Number(y) + endpointLabelOffsets[index];

                    return (
                      <g>
                        <text
                          x={labelX}
                          y={labelY}
                          fill={item.color}
                          fontSize={11}
                          fontWeight={700}
                          dominantBaseline="hanging"
                        >
                          {isCyberLabel ? (
                            <>
                              <tspan x={labelX} dy="0">AI as a cybersecurity</tspan>
                              <tspan x={labelX} dy="12">threat</tspan>
                              <tspan x={labelX} dy="12" fill="#6f777b" fontWeight={600}>{`${value}%`}</tspan>
                            </>
                          ) : (
                            <>
                              <tspan x={labelX} dy="0">{endpointLabel}</tspan>
                              <tspan x={labelX} dy="12" fill="#6f777b" fontWeight={600}>{`${value}%`}</tspan>
                            </>
                          )}
                        </text>
                      </g>
                    );
                  }}
                  activeDot={{ r: 5, fill: '#ffffff', stroke: item.color, strokeWidth: 2.5 }}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <p className="mt-3 text-center text-xs leading-relaxed text-muted-foreground">
          Values show the proportion of UK public-company annual reports that mention each disclosure type in a given year.
        </p>
      </div>
    </div>
  );
}
