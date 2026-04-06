import DashboardClient from '@/app/dashboard-client';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

export default function DataPage() {
  const data = loadGoldenSetDashboardData();
  return <DashboardClient data={data} renderedAtIso={new Date().toISOString()} />;
}
