import Link from 'next/link';

import { ReportFigureRenderer } from '@/components/report-figure-renderer';
import { getAllReportFigures, loadReportFigures } from '@/lib/report-figures';

export const dynamic = 'force-dynamic';

export default function ReportFiguresPage() {
  const figures = getAllReportFigures();
  const document = loadReportFigures();

  return (
    <main className="min-h-screen bg-slate-100 px-6 py-8 text-slate-950">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="rounded-2xl border border-slate-200 bg-white px-8 py-7 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
          <p className="mb-2 text-[11px] font-bold uppercase tracking-[0.16em] text-red-500">AI Risk Observatory</p>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-950">Report Figures QA</h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-600">
            This page renders the report-specific figures from <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[13px] text-slate-700">report/report-figures.json</code>.
            Use the individual figure routes for export-ready views.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <span>Generated: {document.generatedAt}</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" aria-hidden="true" />
            <span>Base dataset: {document.source.baseDataset}</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" aria-hidden="true" />
            <Link href="/" className="font-semibold text-slate-700 underline underline-offset-4 hover:text-slate-900">
              Back to dashboard
            </Link>
          </div>
        </header>

        <div className="grid gap-6">
          {figures.map(figure => (
            <ReportFigureRenderer key={figure.id} figure={figure} mode="preview" />
          ))}
        </div>
      </div>
    </main>
  );
}
