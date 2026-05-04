import { readFile } from 'node:fs/promises';
import path from 'node:path';

export const dynamic = 'force-static';

export async function GET() {
  const reportPath = path.join(process.cwd(), 'public', 'report', 'report-3.html');
  const source = await readFile(reportPath, 'utf8');
  const html = source
    .replaceAll('url("report-bg.jpg")', 'url("/report/report-bg.jpg")')
    .replaceAll('src="figures/', 'src="/report/figures/');

  return new Response(html, {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'public, max-age=300',
    },
  });
}
