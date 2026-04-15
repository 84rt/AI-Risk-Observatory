import fs from 'fs';
import DashboardClient from '@/app/dashboard-client';
import { PRECOMPUTED_DASHBOARD_DATA_PATH } from '@/lib/golden-set';

export const dynamic = 'force-dynamic';

const getDashboardDataVersion = () => {
  if (!fs.existsSync(PRECOMPUTED_DASHBOARD_DATA_PATH)) return undefined;
  const stat = fs.statSync(PRECOMPUTED_DASHBOARD_DATA_PATH);
  const deployVersion =
    process.env.VERCEL_GIT_COMMIT_SHA ||
    process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA ||
    Math.floor(stat.mtimeMs).toString();
  return `${stat.size}-${deployVersion}`;
};

export default function DataPage() {
  return <DashboardClient dataVersion={getDashboardDataVersion()} />;
}
