import type { LucideIcon } from 'lucide-react';
import {
  BarChart3,
  ChevronRight,
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
  icon: LucideIcon;
  iconClass: string;
};

const FLOW_NODES: FlowNode[] = [
  {
    step: '1',
    title: 'Ingestion',
    detail: 'Source PDFs and iXBRL/HTML filings with company/year metadata.',
    icon: FileText,
    iconClass: 'text-sky-600',
  },
  {
    step: '2',
    title: 'Text Extraction',
    detail: 'Convert PDFs to Markdown; parse iXBRL into structured text blocks.',
    icon: ScanText,
    iconClass: 'text-indigo-600',
  },
  {
    step: '3',
    title: 'Chunking',
    detail: 'Split into ~500-2000 char excerpts; keyword-filter for AI relevance.',
    icon: FileCode2,
    iconClass: 'text-amber-600',
  },
  {
    step: '4',
    title: 'Classification',
    detail: 'LLM classifies each excerpt for mention type, risk, adoption, and confidence.',
    icon: Sparkles,
    iconClass: 'text-emerald-600',
  },
  {
    step: '5',
    title: 'Database',
    detail: 'Persist classified excerpts and scores into SQLite via Prisma.',
    icon: Database,
    iconClass: 'text-rose-600',
  },
  {
    step: '6',
    title: 'Visualization',
    detail: 'Dashboard renders trends, sector comparisons, and quality metrics.',
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
      <p className="mt-1 text-xs leading-relaxed text-slate-500">
        End-to-end flow from raw filings to chart-ready analytics.
      </p>

      <div className="relative mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {FLOW_NODES.map((node, index) => {
          const NodeIcon = node.icon;

          return (
            <div key={node.title} className="relative flex items-stretch">
              <article
                className="animate-rise flex w-full flex-col rounded-xl border border-slate-200 bg-white p-3 shadow-sm"
                style={{ animationDelay: `${index * 60}ms` }}
              >
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-slate-50">
                    <NodeIcon className={`h-3.5 w-3.5 ${node.iconClass}`} />
                  </div>
                  <p className="text-xs font-semibold text-slate-900 leading-tight">{node.title}</p>
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{node.detail}</p>
              </article>
              {index < FLOW_NODES.length - 1 && (
                <ChevronRight className="absolute -right-[9px] top-1/2 z-10 h-3.5 w-3.5 -translate-y-1/2 text-slate-300 hidden lg:block" />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
