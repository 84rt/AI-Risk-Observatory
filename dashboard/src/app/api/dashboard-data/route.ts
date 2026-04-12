import fs from 'fs';
import { Readable } from 'stream';
import {
  loadGoldenSetDashboardData,
  PRECOMPUTED_DASHBOARD_DATA_PATH,
} from '@/lib/golden-set';

export const runtime = 'nodejs';

const CACHE_CONTROL = 'public, s-maxage=3600, stale-while-revalidate=86400';

export async function GET() {
  const renderedAtIso = new Date().toISOString();

  if (fs.existsSync(PRECOMPUTED_DASHBOARD_DATA_PATH)) {
    const stat = fs.statSync(PRECOMPUTED_DASHBOARD_DATA_PATH);
    const stream = fs.createReadStream(PRECOMPUTED_DASHBOARD_DATA_PATH);

    return new Response(Readable.toWeb(stream) as ReadableStream, {
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': String(stat.size),
        'Cache-Control': CACHE_CONTROL,
        'X-Rendered-At': renderedAtIso,
      },
    });
  }

  return Response.json(loadGoldenSetDashboardData(), {
    headers: {
      'Cache-Control': CACHE_CONTROL,
      'X-Rendered-At': renderedAtIso,
    },
  });
}
