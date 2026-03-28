'use client';

import dynamic from 'next/dynamic';
import type { ReportClassificationFlow } from '@/lib/golden-set';

const ReportClassificationSankey = dynamic(
  () => import('@/components/report-classification-sankey').then(mod => mod.ReportClassificationSankey),
  {
    ssr: false,
    loading: () => (
      <div className="rounded-[1.4rem] border border-slate-200/80 bg-white/90 p-6 shadow-sm">
        <div className="mt-4 h-[640px] animate-pulse rounded-[1.1rem] border border-slate-200 bg-slate-100/80" />
      </div>
    ),
  }
);

export function ReportClassificationSankeyShell({ flow }: { flow: ReportClassificationFlow }) {
  return <ReportClassificationSankey flow={flow} />;
}
