'use client';

import { StackedBarChart } from '@/components/overview-charts';

const mentionColors: Record<string, string> = {
  adoption: '#0ea5e9',
  risk: '#f97316',
  vendor: '#14b8a6',
  general_ambiguous: '#64748b',
  harm: '#ef4444',
};

export function MentionTypesChart({
  data,
  stackKeys,
}: {
  data: Record<string, number>[];
  stackKeys: string[];
}) {
  return (
    <StackedBarChart
      data={data}
      xAxisKey="year"
      stackKeys={stackKeys}
      colors={mentionColors}
      allowLineChart
      title="Mention Types Over Time"
      subtitle="Each bar shows how many reports per year were tagged with each mention type (confidence ≥ 0.2)."
    />
  );
}
