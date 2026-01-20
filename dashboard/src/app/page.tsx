import DashboardClient from '@/app/dashboard-client';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

export default function Dashboard() {
  const data = loadGoldenSetDashboardData();
  return <DashboardClient data={data} />;
}
