import type {
  ReportClassificationBreakdown as ReportClassificationBreakdownData,
  ReportClassificationBreakdownBranch,
  ReportClassificationBreakdownLeaf,
} from '@/lib/golden-set';

const formatNumber = (value: number) => new Intl.NumberFormat('en-GB').format(value);
const formatPercent = (value: number) => `${value.toFixed(1)}%`;

const branchStyles: Record<string, { accent: string; chip: string; bar: string }> = {
  adoption: {
    accent: 'border-sky-200 bg-sky-50/75',
    chip: 'bg-sky-100 text-sky-700',
    bar: 'bg-sky-400',
  },
  general_ambiguous: {
    accent: 'border-slate-200 bg-slate-50/85',
    chip: 'bg-slate-200 text-slate-700',
    bar: 'bg-slate-400',
  },
  risk: {
    accent: 'border-orange-200 bg-orange-50/75',
    chip: 'bg-orange-100 text-orange-700',
    bar: 'bg-orange-400',
  },
  vendor: {
    accent: 'border-teal-200 bg-teal-50/75',
    chip: 'bg-teal-100 text-teal-700',
    bar: 'bg-teal-400',
  },
  harm: {
    accent: 'border-rose-200 bg-rose-50/75',
    chip: 'bg-rose-100 text-rose-700',
    bar: 'bg-rose-400',
  },
};

function NodeCard({
  title,
  count,
  meta,
  tone = 'slate',
}: {
  title: string;
  count: number;
  meta: string;
  tone?: 'slate' | 'amber' | 'sky';
}) {
  const toneClasses: Record<string, string> = {
    slate: 'border-slate-200 bg-white text-slate-900',
    amber: 'border-amber-200 bg-amber-50/80 text-amber-950',
    sky: 'border-sky-200 bg-sky-50/80 text-sky-950',
  };

  return (
    <div className={`rounded-2xl border px-4 py-3 shadow-sm ${toneClasses[tone]}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
        {title}
      </p>
      <div className="mt-2 flex items-end justify-between gap-3">
        <p className="text-2xl font-semibold tracking-tight">{formatNumber(count)}</p>
        <p className="text-right text-xs leading-relaxed text-slate-500">{meta}</p>
      </div>
    </div>
  );
}

function ChildRow({
  item,
  parentCount,
  barClassName,
}: {
  item: ReportClassificationBreakdownLeaf;
  parentCount: number;
  barClassName: string;
}) {
  const width = parentCount > 0 ? `${Math.max((item.count / parentCount) * 100, item.count > 0 ? 4 : 0)}%` : '0%';

  return (
    <div className="rounded-xl border border-white/70 bg-white/75 px-3 py-2.5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-slate-800">{item.label}</p>
          <p className="mt-0.5 text-xs text-slate-500">{formatPercent(item.pctOfParent)} of parent branch</p>
        </div>
        <p className="text-sm font-semibold text-slate-900">{formatNumber(item.count)}</p>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-slate-200/80">
        <div className={`h-full rounded-full ${barClassName}`} style={{ width }} />
      </div>
    </div>
  );
}

function BranchCard({
  branch,
  signalReports,
}: {
  branch: ReportClassificationBreakdownBranch;
  signalReports: number;
}) {
  const style = branchStyles[branch.id] ?? branchStyles.general_ambiguous;
  const children = (branch.children ?? []).filter(child => child.count > 0);
  const branchWidth =
    signalReports > 0 ? `${Math.max((branch.count / signalReports) * 100, branch.count > 0 ? 6 : 0)}%` : '0%';

  return (
    <div className={`rounded-[1.15rem] border p-4 shadow-sm ${style.accent}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-base font-semibold text-slate-900">{branch.label}</p>
          <p className="mt-1 text-xs text-slate-500">{formatPercent(branch.pctOfParent)} of Phase 1 signal reports</p>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${style.chip}`}>
          {formatNumber(branch.count)}
        </span>
      </div>

      <div className="mt-3 h-2 rounded-full bg-white/80">
        <div className={`h-full rounded-full ${style.bar}`} style={{ width: branchWidth }} />
      </div>

      {children.length > 0 ? (
        <div className="mt-4 space-y-2.5 border-l border-white/80 pl-4">
          {children.map(child => (
            <ChildRow key={child.id} item={child} parentCount={branch.count} barClassName={style.bar} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function ReportClassificationBreakdown({
  summary,
}: {
  summary: ReportClassificationBreakdownData;
}) {
  return (
    <section className="rounded-[1.4rem] border border-slate-200/80 bg-white/90 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_6px_20px_rgba(0,0,0,0.03)] sm:p-6">
      <div className="max-w-3xl">
        <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
          Report Classification Diagram
        </h3>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          The top half shows the mutually exclusive flow of reports. The lower half shows the non-exclusive label
          families and their child tags, using the same counts as the breakdown you outlined.
        </p>
      </div>

      <div className="mt-5 rounded-[1.1rem] border border-slate-200 bg-[linear-gradient(180deg,#fffdfa_0%,#fcfbf8_100%)] p-4 sm:p-5">
        <div className="mx-auto max-w-5xl">
          <div className="mx-auto max-w-sm">
            <NodeCard
              title="All Reports"
              count={summary.totalReports}
              meta="100.0% of corpus"
              tone="slate"
            />
          </div>

          <div className="mx-auto h-6 w-px bg-slate-200" />

          <div className="relative grid gap-4 md:grid-cols-2">
            <div className="pointer-events-none absolute left-1/2 top-0 hidden h-px w-[calc(100%-5rem)] -translate-x-1/2 bg-slate-200 md:block" />
            <NodeCard
              title="No AI Chunk Extracted"
              count={summary.noAiChunkExtracted}
              meta={`${formatPercent((summary.noAiChunkExtracted / summary.totalReports) * 100)} of ${formatNumber(summary.totalReports)}`}
              tone="slate"
            />
            <NodeCard
              title="AI Chunk Extracted"
              count={summary.aiChunkExtracted}
              meta={`${formatPercent((summary.aiChunkExtracted / summary.totalReports) * 100)} of ${formatNumber(summary.totalReports)}`}
              tone="amber"
            />
          </div>

          <div className="mx-auto h-6 w-px bg-slate-200" />

          <div className="ml-auto max-w-2xl">
            <div className="grid gap-4 md:grid-cols-2">
              <NodeCard
                title="Phase 1 = None Only"
                count={summary.phase1NoneOnly}
                meta={`${formatPercent((summary.phase1NoneOnly / summary.aiChunkExtracted) * 100)} of ${formatNumber(summary.aiChunkExtracted)}`}
                tone="slate"
              />
              <NodeCard
                title="Phase 1 Signal Reports"
                count={summary.phase1SignalReports}
                meta={`${formatPercent((summary.phase1SignalReports / summary.aiChunkExtracted) * 100)} of ${formatNumber(summary.aiChunkExtracted)}`}
                tone="sky"
              />
            </div>
          </div>

          <div className="mx-auto my-6 h-6 w-px bg-slate-200" />

          <div className="rounded-2xl border border-slate-200 bg-white/70 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                Non-Exclusive Labels Below
              </p>
              <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 font-semibold">
                  Percentages below are of {formatNumber(summary.phase1SignalReports)}
                </span>
                <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 font-semibold">
                  {formatNumber(summary.phase1TotalAssignments)} assignments
                </span>
                <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 font-semibold">
                  {summary.averageLabelsPerSignalReport.toFixed(2)}x avg labels/report
                </span>
              </div>
            </div>

            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              {summary.branches.map(branch => (
                <BranchCard
                  key={branch.id}
                  branch={branch}
                  signalReports={summary.phase1SignalReports}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
