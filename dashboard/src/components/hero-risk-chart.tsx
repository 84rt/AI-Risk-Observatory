'use client';

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';

interface HeroRiskChartProps {
  data: { year: number; risk: number }[];
}

export default function HeroRiskChart({ data }: HeroRiskChartProps) {
  return (
    <div className="w-full max-w-md lg:w-[380px]">
      <div className="relative h-[130px]">
        {/* Side fades — scoped to chart area only */}
        <div
          className="pointer-events-none absolute z-10"
          style={{ top: 0, bottom: 20, left: 24, width: 48 }}
        >
          <div className="h-full w-full bg-gradient-to-r from-[#f6f3ef] via-[#f6f3ef]/60 to-transparent" />
        </div>
        <div
          className="pointer-events-none absolute z-10"
          style={{ top: 0, bottom: 20, right: 0, width: 48 }}
        >
          <div className="h-full w-full bg-gradient-to-l from-[#f6f3ef] via-[#f6f3ef]/60 to-transparent" />
        </div>

        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -12 }}>
            <defs>
              <linearGradient id="heroRiskGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f97316" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#f97316" stopOpacity={0} />
              </linearGradient>
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
              width={36}
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
              formatter={(value: number) => [value, 'Risk mentions']}
              cursor={{ stroke: '#d6d3d1', strokeWidth: 1, strokeDasharray: '3 3' }}
            />
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#f97316"
              strokeWidth={2.5}
              fill="url(#heroRiskGrad)"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, fill: '#fff', stroke: '#f97316' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-[11px] text-slate-400">
        AI mentioned as a risk by public companies
      </p>
    </div>
  );
}
