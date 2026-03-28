'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';

interface ChartSeries {
  label: string;
  subtitle: string;
  data: { year: number; value: number }[];
  color: string;
}

interface HeroRiskChartProps {
  series: ChartSeries[];
}

type Phase = 'visible' | 'exiting' | 'entering';

export default function HeroRiskChart({ series }: HeroRiskChartProps) {
  const [displayIndex, setDisplayIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>('visible');
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Resolve the sentinel in the setState
  useEffect(() => {
    if (phase === 'exiting') return;
  }, [phase]);

  // Override goTo to handle cycling
  const handleCycle = useCallback(() => {
    setPhase('exiting');
    setTimeout(() => {
      setDisplayIndex(i => (i + 1) % series.length);
      setPhase('entering');
      setTimeout(() => setPhase('visible'), 50);
    }, 500);
  }, [series.length]);

  const handleDotClick = useCallback((i: number) => {
    if (i === displayIndex) return;
    setPhase('exiting');
    setTimeout(() => {
      setDisplayIndex(i);
      setPhase('entering');
      setTimeout(() => setPhase('visible'), 50);
    }, 500);
  }, [displayIndex]);

  // Auto-cycle timer
  useEffect(() => {
    timerRef.current = setInterval(handleCycle, 7000);
    return () => clearInterval(timerRef.current);
  }, [handleCycle]);

  // Reset timer on manual click
  const onDotClick = useCallback((i: number) => {
    clearInterval(timerRef.current);
    handleDotClick(i);
    timerRef.current = setInterval(handleCycle, 7000);
  }, [handleDotClick, handleCycle]);

  const current = series[displayIndex];
  const gradientId = `heroGrad-${displayIndex}`;

  // Chart area opacity: fade out during exit, invisible during enter (Recharts draws in), full when visible
  const chartOpacity = phase === 'exiting' ? 0 : 1;

  return (
    <div className="w-full max-w-md lg:w-[380px]">
      <div className="relative h-[130px]">
        <div
          className="h-full w-full [&_.recharts-area-area]:![mask:url(#heroEdgeMask)] [&_.recharts-area-curve]:![mask:url(#heroEdgeMask)]"
          style={{
            opacity: chartOpacity,
            transition: phase === 'exiting' ? 'opacity 450ms ease-in' : 'none',
          }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={current.data} margin={{ top: 4, right: 12, bottom: 0, left: -12 }}>
              <defs>
                {/* Vertical fill gradient */}
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={current.color} stopOpacity={0.4} />
                  <stop offset="100%" stopColor={current.color} stopOpacity={0.02} />
                </linearGradient>
                {/* Horizontal edge-fade mask applied to stroke + fill paths */}
                <linearGradient id="heroEdgeFade" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="white" stopOpacity={0} />
                  <stop offset="12%" stopColor="white" stopOpacity={1} />
                  <stop offset="88%" stopColor="white" stopOpacity={1} />
                  <stop offset="100%" stopColor="white" stopOpacity={0} />
                </linearGradient>
                <mask id="heroEdgeMask" maskContentUnits="objectBoundingBox">
                  <rect x="0" y="0" width="1" height="1" fill="url(#heroEdgeFade)" />
                </mask>
              </defs>
              <XAxis
                dataKey="year"
                tick={{ fontSize: 10, fill: '#a8a29e' }}
                axisLine={false}
                tickLine={false}
                dy={4}
              />
              <YAxis
                tick={{ fontSize: 9, fill: '#a09890' }}
                axisLine={false}
                tickLine={false}
                width={40}
                tickCount={4}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 11,
                  borderRadius: 10,
                  border: 'none',
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(12px)',
                  boxShadow: '0 4px 24px rgba(0,0,0,.1)',
                }}
                labelFormatter={label => `${label}`}
                formatter={(value: number) => [value, current.label]}
                cursor={{ stroke: '#d6d3d1', strokeWidth: 1, strokeDasharray: '3 3' }}
              />
              <Area
                key={displayIndex}
                type="monotone"
                dataKey="value"
                stroke={current.color}
                strokeWidth={2.5}
                fill={`url(#${gradientId})`}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, fill: '#fff', stroke: current.color }}
                isAnimationActive={true}
                animationDuration={900}
                animationEasing="ease-out"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Label + dots indicator */}
      <div className="mt-1 flex items-center justify-between">
        <p
          className="text-[11px] text-slate-400"
          style={{
            opacity: phase === 'exiting' ? 0 : 1,
            transition: phase === 'exiting' ? 'opacity 350ms ease-in' : 'opacity 400ms ease-out',
          }}
        >
          {current.subtitle}
        </p>
        <div className="flex gap-1.5">
          {series.map((_, i) => (
            <button
              key={i}
              onClick={() => onDotClick(i)}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === displayIndex
                  ? 'w-4 bg-amber-500'
                  : 'w-1.5 bg-slate-300 hover:bg-slate-400'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
