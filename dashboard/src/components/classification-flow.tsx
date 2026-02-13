import type { LucideIcon } from 'lucide-react';
import {
  BarChart3,
  Database,
  FileCode2,
  FileText,
  ScanText,
  Sparkles,
} from 'lucide-react';

type FlowNode = {
  step: string;
  title: string;
  detail: string;
  signals: string;
  icon: LucideIcon;
  iconClass: string;
};

const FLOW_NODES: FlowNode[] = [
  {
    step: 'Step 1',
    title: 'Ingestion',
    detail:
      'Source raw reports as PDFs (FinancialReports DB) and iXBRL/HTML filings (Companies House API).',
    signals: 'Output: Raw source files + company/year metadata',
    icon: FileText,
    iconClass: 'text-sky-600',
  },
  {
    step: 'Step 2',
    title: 'Preprocessing (Text Extraction)',
    detail:
      'Convert PDFs to layout-aware Markdown and parse iXBRL into text blocks mapped to financial tags.',
    signals: 'Output: Cleaned, structured report text',
    icon: ScanText,
    iconClass: 'text-indigo-600',
  },
  {
    step: 'Step 3',
    title: 'Chunking',
    detail:
      'Split documents into context-heavy chunks (~500-2000 chars); keyword matching isolates AI-relevant segments.',
    signals: 'Output: AI-candidate text chunks',
    icon: FileCode2,
    iconClass: 'text-amber-600',
  },
  {
    step: 'Step 4',
    title: 'AI Classification',
    detail:
      'run_llm_classifier scripts send chunks to an LLM (for example GPT-4 or Gemini).',
    signals:
      'Signals: AI mention, risk vs opportunity, substantiveness + confidence',
    icon: Sparkles,
    iconClass: 'text-emerald-600',
  },
  {
    step: 'Step 5',
    title: 'Database Loading',
    detail:
      'Persist classified chunks, metadata (company, year, sector), and model scores into SQLite via Prisma.',
    signals: 'Output: Queryable analytical tables',
    icon: Database,
    iconClass: 'text-rose-600',
  },
  {
    step: 'Step 6',
    title: 'Visualization',
    detail:
      'Next.js dashboard queries the database and renders trends, sector comparisons, and Golden Set benchmarks.',
    signals: 'Output: End-user charts and filters',
    icon: BarChart3,
    iconClass: 'text-teal-600',
  },
];

export function ClassificationFlowDiagram() {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
      <div className="pointer-events-none absolute -left-16 -top-14 h-44 w-44 rounded-full bg-sky-200/35 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-16 right-0 h-48 w-48 rounded-full bg-amber-200/30 blur-3xl" />

      <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
        Processing Pipeline
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        End-to-end flow from raw filings to chart-ready risk and adoption analytics.
      </p>

      <div className="relative mt-6">
        <div className="absolute bottom-4 left-[18px] top-4 w-px bg-slate-200 md:left-1/2" />
        <div className="space-y-4">
          {FLOW_NODES.map((node, index) => {
            const NodeIcon = node.icon;
            const alignRight = index % 2 === 0;

            return (
              <div
                key={node.title}
                className={`relative flex items-stretch ${alignRight ? 'md:justify-end' : 'md:justify-start'}`}
              >
                <article
                  className={`animate-rise w-full rounded-xl border border-slate-200 bg-white p-4 shadow-sm md:w-[46%] ${
                    alignRight ? 'md:mr-10' : 'md:ml-10'
                  }`}
                  style={{ animationDelay: `${index * 90}ms` }}
                >
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                    {node.step}
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-900">{node.title}</p>
                  <p className="mt-2 text-xs leading-relaxed text-slate-600">{node.detail}</p>
                  <p className="mt-2 border-t border-slate-100 pt-2 text-[11px] font-medium leading-relaxed text-slate-500">
                    {node.signals}
                  </p>
                </article>

                <div className="absolute left-[18px] top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-slate-300 bg-white shadow-sm md:left-1/2">
                  <NodeIcon className={`h-4 w-4 ${node.iconClass}`} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
