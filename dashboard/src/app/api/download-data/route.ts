import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';

const RAW_DATASET_PATH = path.join(process.cwd(), 'data', 'annotations.jsonl');
const PRECOMPUTED_DATASET_PATH = path.join(process.cwd(), 'data', 'dashboard-data.json');

export const runtime = 'nodejs';

export async function GET() {
  const hasRawDataset = fs.existsSync(RAW_DATASET_PATH);
  const filePath = hasRawDataset ? RAW_DATASET_PATH : PRECOMPUTED_DATASET_PATH;

  if (!fs.existsSync(filePath)) {
    return new Response('Dataset not found.', { status: 404 });
  }

  const stream = fs.createReadStream(filePath);
  const stat = fs.statSync(filePath);
  const filename = hasRawDataset
    ? 'ai-risk-observatory-annotations.jsonl'
    : 'ai-risk-observatory-dashboard-data.json';
  const contentType = hasRawDataset
    ? 'application/x-ndjson; charset=utf-8'
    : 'application/json; charset=utf-8';

  return new Response(Readable.toWeb(stream) as ReadableStream, {
    headers: {
      'Content-Type': contentType,
      'Content-Length': String(stat.size),
      'Content-Disposition': `attachment; filename="${filename}"`,
      'Cache-Control': 'no-store',
    },
  });
}
