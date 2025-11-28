'use client';

import { 
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend
} from 'recharts';
import { RiskCategory } from '@/data/mockData';

// --- Shared Colors ---
// Keeping these distinct but ensuring they work well on white
const COLORS: Record<string, string> = {
  'operational_reliability': '#f97316', // Orange
  'security_malicious_use': '#ef4444', // Red
  'legal_regulatory_compliance': '#eab308', // Yellow
  'workforce_human_capital': '#8b5cf6', // Purple
  'societal_ethical_reputational': '#3b82f6', // Blue
  'frontier_systemic': '#ec4899', // Pink
};

const formatLabel = (val: string) => {
  return val.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

// --- Component 1: Bar Chart (Risk Distribution) ---
interface OverviewChartsProps {
  data: {
    name: string;
    value: number;
  }[];
}

export function OverviewCharts({ data }: OverviewChartsProps) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          layout="vertical"
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
          <XAxis type="number" hide />
          <YAxis 
            dataKey="name" 
            type="category" 
            width={100}
            tickFormatter={(val) => val.split('_')[0]} 
            tick={{ fontSize: 12, fill: '#64748b' }}
            axisLine={false}
            tickLine={false}
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
            formatter={(value: number) => [value, 'Mentions']}
            labelFormatter={formatLabel}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#cbd5e1'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component 2: Line Chart (Trends over Time) ---
interface TrendChartProps {
  data: {
    year: number;
    [key: string]: number; // dynamic risk keys
  }[];
}

export function TrendChart({ data }: TrendChartProps) {
  return (
    <div className="h-[350px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey="year" 
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
             contentStyle={{ 
               backgroundColor: '#fff',
               borderRadius: '6px', 
               border: '1px solid #e2e8f0', 
               boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
               color: '#1e293b'
             }}
             itemStyle={{ color: '#1e293b' }}
             labelStyle={{ color: '#64748b', marginBottom: '0.25rem' }}
             labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }} 
            iconType="circle"
          />
          {Object.keys(COLORS).map(risk => (
            <Line 
              key={risk}
              type="monotone" 
              dataKey={risk} 
              stroke={COLORS[risk]} 
              name={formatLabel(risk).split(' ')[0]} 
              strokeWidth={2.5}
              dot={{ r: 4, strokeWidth: 2, fill: '#fff' }}
              activeDot={{ r: 6, strokeWidth: 0 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Component 3: Heatmap (Sector vs Risk) ---
interface HeatmapProps {
  data: {
    sector: string;
    risks: Record<RiskCategory, number>;
  }[];
}

export function SectorHeatmap({ data }: HeatmapProps) {
  const risks = Object.keys(COLORS) as RiskCategory[];

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-left border-collapse">
        <thead>
          <tr>
            <th className="p-3 text-xs font-medium text-slate-500 uppercase tracking-wider border-b border-gray-200">Sector</th>
            {risks.map(r => (
              <th key={r} className="p-3 font-medium text-xs text-slate-500 text-center border-b border-gray-200 w-24">
                {r.split('_')[0]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map(row => (
            <tr key={row.sector} className="hover:bg-gray-50 transition-colors">
              <td className="p-3 font-medium text-slate-700 border-b border-gray-100">{row.sector}</td>
              {risks.map(r => {
                const val = row.risks[r] || 0;
                // Simple opacity scale based on value (max assumed ~10 for mock)
                const opacity = Math.min(val / 10, 1); 
                return (
                  <td key={r} className="p-1 border-b border-gray-100">
                    <div 
                      className="h-8 w-full rounded-sm flex items-center justify-center text-xs font-bold transition-all cursor-default"
                      style={{ 
                        backgroundColor: COLORS[r], 
                        opacity: Math.max(0.1, opacity),
                        color: opacity > 0.5 ? 'white' : 'black'
                      }}
                      title={`${row.sector} - ${formatLabel(r)}: ${val} mentions`}
                    >
                      {val > 0 ? val : ''}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
