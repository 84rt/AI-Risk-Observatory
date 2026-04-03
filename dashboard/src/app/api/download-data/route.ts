import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';

export const runtime = 'nodejs';

export async function GET() {
  const filePath = path.join(process.cwd(), 'data', 'annotations.jsonl');

  if (!fs.existsSync(filePath)) {
    return new Response('Dataset not found.', { status: 404 });
  }

  const stream = fs.createReadStream(filePath);
  const stat = fs.statSync(filePath);
  const filename = 'ai-risk-observatory-annotations.jsonl';

  return new Response(Readable.toWeb(stream) as ReadableStream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Content-Length': String(stat.size),
      'Content-Disposition': `attachment; filename="${filename}"`,
      'Cache-Control': 'no-store',
    },
  });
}
