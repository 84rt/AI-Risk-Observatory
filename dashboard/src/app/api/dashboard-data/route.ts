import fs from 'fs';
import { Readable } from 'stream';
import {
  loadGoldenSetDashboardData,
  PRECOMPUTED_DASHBOARD_DATA_PATH,
} from '@/lib/golden-set';

export const runtime = 'nodejs';

const VERSIONED_CACHE_CONTROL = 'public, max-age=31536000, immutable';
const UNVERSIONED_CACHE_CONTROL = 'public, max-age=0, s-maxage=60, must-revalidate';

export async function GET(request: Request) {
  const renderedAtIso = new Date().toISOString();
  const hasVersion = new URL(request.url).searchParams.has('v');
  const cacheControl = hasVersion ? VERSIONED_CACHE_CONTROL : UNVERSIONED_CACHE_CONTROL;

  if (fs.existsSync(PRECOMPUTED_DASHBOARD_DATA_PATH)) {
    const stat = fs.statSync(PRECOMPUTED_DASHBOARD_DATA_PATH);
    const stream = fs.createReadStream(PRECOMPUTED_DASHBOARD_DATA_PATH);

    return new Response(Readable.toWeb(stream) as ReadableStream, {
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': String(stat.size),
        'Cache-Control': cacheControl,
        'X-Rendered-At': renderedAtIso,
      },
    });
  }

  return Response.json(loadGoldenSetDashboardData(), {
    headers: {
      'Cache-Control': cacheControl,
      'X-Rendered-At': renderedAtIso,
    },
  });
}
