import { Suspense } from 'react';
import DashboardClient from '@/app/dashboard-client';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

export default function DataPage() {
  const data = loadGoldenSetDashboardData();
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#f6f3ef] px-6 py-16 text-slate-600">
          Loading dashboard…
        </div>
      }
    >
      <DashboardClient data={data} />
    </Suspense>
  );
}
