import { MentionTypesChart } from '@/components/mention-types-chart';
import ExampleBrowser from '@/components/example-browser';
import { MethodologyToc } from '@/components/methodology-toc';
import { CollapsibleSection } from '@/components/collapsible-section';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

const sumSeriesKey = (
  rows: Record<string, string | number>[],
  key: string
) => rows.reduce((total, row) => total + Number(row[key] || 0), 0);

function FlowConnectorSplit() {
  return (
    <div className="mx-auto h-14 w-full max-w-lg" aria-hidden="true">
      <svg
        viewBox="0 0 512 56"
        preserveAspectRatio="none"
        className="h-full w-full text-border"
        fill="none"
      >
        <path
          d="M256 0V16M256 16H124V42M256 16H388V42"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M120 38L124 42L128 38" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M384 38L388 42L392 38" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function FlowConnectorMergeLeft() {
  return (
    <div className="mx-auto h-14 w-full max-w-lg" aria-hidden="true">
      <svg
        viewBox="0 0 440 56"
        preserveAspectRatio="none"
        className="h-full w-full text-border"
        fill="none"
      >
        <path
          d="M110 0V18H220V42"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M216 38L220 42L224 38" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function FlowConnectorStraight() {
  return (
    <div className="mx-auto h-12 w-16" aria-hidden="true">
      <svg
        viewBox="0 0 64 48"
        preserveAspectRatio="none"
        className="h-full w-full text-border"
        fill="none"
      >
        <path
          d="M32 0V36"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        <path d="M28 32L32 36L36 32" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function StageFrame({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`border border-border/70 px-5 py-5 sm:px-6 ${className}`}>
      {children}
    </div>
  );
}

export default function AboutPage() {
  const data = loadGoldenSetDashboardData();
  const flow = data.reportClassificationFlow;
  const breakdown = data.reportClassificationBreakdown;
  const summary = data.datasets.perReport.summary;
  const totalChunks = data.datasets.perChunk.summary.totalReports;
  const phaseCardStyles: Record<
    string,
    {
      dotClass: string;
      barClass: string;
      highlightClass: string;
      borderClass: string;
      surfaceClass: string;
    }
  > = {
    adoption: {
      dotClass: 'bg-sky-500',
      barClass: 'bg-sky-500',
      highlightClass: 'bg-sky-500/15',
      borderClass: 'border-sky-200',
      surfaceClass: 'bg-sky-50',
    },
    risk: {
      dotClass: 'bg-orange-500',
      barClass: 'bg-orange-500',
      highlightClass: 'bg-orange-500/15',
      borderClass: 'border-orange-200',
      surfaceClass: 'bg-orange-50',
    },
    vendor: {
      dotClass: 'bg-teal-500',
      barClass: 'bg-teal-500',
      highlightClass: 'bg-teal-500/15',
      borderClass: 'border-teal-200',
      surfaceClass: 'bg-teal-50',
    },
    harm: {
      dotClass: 'bg-rose-500',
      barClass: 'bg-rose-500',
      highlightClass: 'bg-rose-500/15',
      borderClass: 'border-rose-200',
      surfaceClass: 'bg-rose-50',
    },
    general_ambiguous: {
      dotClass: 'bg-slate-500',
      barClass: 'bg-slate-500',
      highlightClass: 'bg-slate-500/15',
      borderClass: 'border-slate-200',
      surfaceClass: 'bg-slate-50',
    },
    none: {
      dotClass: 'bg-zinc-500',
      barClass: 'bg-zinc-500',
      highlightClass: 'bg-zinc-500/15',
      borderClass: 'border-zinc-200',
      surfaceClass: 'bg-zinc-50',
    },
  };
  const phase2TrendRowsById: Record<string, Record<string, string | number>[]> = {
    adoption: data.datasets.perChunk.adoptionTrend as Record<string, string | number>[],
    risk: data.datasets.perChunk.riskTrend as Record<string, string | number>[],
    vendor: data.datasets.perChunk.vendorTrend as Record<string, string | number>[],
  };
  const phase1Cards = ['general_ambiguous', 'none', 'harm', 'adoption', 'risk', 'vendor'].flatMap(id => {
    if (id === 'none') {
      return [
        {
          id: 'none',
          label: 'None or false positive',
          count: flow.phase1NoneOnlyReports,
          pctOfParent: flow.extractedAiReports > 0 ? (flow.phase1NoneOnlyReports / flow.extractedAiReports) * 100 : 0,
          ...phaseCardStyles.none,
          continuesToPhase2: false,
          children: undefined,
        },
      ];
    }

    const branch = breakdown.branches.find(item => item.id === id);
    if (!branch) return [];

    return [
      {
        ...branch,
        label: branch.id === 'general_ambiguous' ? 'General or ambiguous' : branch.label,
        ...phaseCardStyles[id],
        continuesToPhase2: (branch.children?.length ?? 0) > 0,
      },
    ];
  });
  const phase2Cards = phase1Cards
    .filter(card => card.continuesToPhase2)
    .map(card => {
      const children = (card.children ?? [])
        .map(child => ({
          ...child,
          chunkCount: sumSeriesKey(phase2TrendRowsById[card.id] ?? [], child.id),
        }))
        .sort((a, b) => {
          if (b.chunkCount !== a.chunkCount) return b.chunkCount - a.chunkCount;
          if (b.count !== a.count) return b.count - a.count;
          return a.label.localeCompare(b.label, 'en');
        });
      const totalChildChunks = children.reduce((total, child) => total + child.chunkCount, 0);

      return {
        ...card,
        children,
        totalChildChunks,
      };
    });

  return (
    <div className="min-h-screen bg-white text-primary">
      <div className="mx-auto max-w-7xl px-6">

        {/* Header */}
        <div className="py-24 max-w-3xl">
          <span className="aisi-tag">Methodology</span>
          <h1 className="aisi-h1">About the<br />Observatory</h1>
          <p className="mt-8 text-xl leading-relaxed text-muted">
            How we turn annual-report text into the dashboard metrics.
          </p>
        </div>

        {/* Content + ToC */}
        <div className="pb-32 xl:grid xl:grid-cols-[1fr_200px] xl:gap-16">

          {/* Main content */}
          <article className="space-y-24 text-[17px] leading-relaxed text-muted">

            {/* ── Overview ─────────────────────────────────────── */}
            <section id="overview" className="scroll-mt-20 space-y-6">
              <h2 className="aisi-h2 uppercase">Overview</h2>
              <p>
                The AI Risk Observatory processes UK public-company annual reports through a
                two-phase NLP classification pipeline. The current corpus
                spans {summary.totalCompanies.toLocaleString()} companies
                and {flow.totalReports.toLocaleString()} filings
                across {data.years[0]}–{data.years[data.years.length - 1]}.
                Of these, {flow.extractedAiReports.toLocaleString()} filings contain at
                least one AI-relevant passage,
                and {flow.phase1SignalReports.toLocaleString()} carry a real AI signal after
                gating. The pipeline produces {totalChunks.toLocaleString()} annotated text
                chunks in total.
              </p>
              <p>
                The diagram below separates the two classification phases. Phase 1 first
                assigns extracted passages to six top-level outputs, including a
                <em> none / false positive</em> outcome for passages that do not contain a
                real AI mention. Phase 2 applies deeper taxonomies only within the relevant
                Phase 1 branches, so the specific downstream labels appear later in the
                flow. The signal-bearing Phase 1 labels are not exclusive, so those counts
                sum to more than the number of extracted reports.
              </p>

              {/* Pipeline flowchart */}
              <div className="space-y-0">
                {/* Stage 1: Total */}
                <div className="flex justify-center">
                  <div className="border-2 border-primary bg-primary px-8 py-5 text-center text-white">
                    <p className="text-3xl font-bold">{flow.totalReports.toLocaleString()}</p>
                    <p className="text-xs font-bold uppercase tracking-widest opacity-80 mt-1">Filings examined</p>
                  </div>
                </div>

                {/* Connector: split */}
                <FlowConnectorSplit />

                {/* Stage 2: Extraction split */}
                {(() => {
                  const extractPct = Math.round((flow.extractedAiReports / flow.totalReports) * 100);
                  return (
                    <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
                      <div className="border border-border px-5 py-4 text-center">
                        <p className="text-2xl font-bold text-primary">{flow.extractedAiReports.toLocaleString()}</p>
                        <p className="mt-1 text-[13px] font-bold uppercase tracking-[0.14em] text-muted-foreground sm:text-sm">
                          Reports that mention AI
                        </p>
                        <div className="mt-3 h-1.5 w-full rounded-full bg-secondary">
                          <div className="h-full rounded-full bg-primary" style={{ width: `${extractPct}%` }} />
                        </div>
                        <p className="text-xs text-muted mt-1">{extractPct}% of filings</p>
                      </div>
                      <div className="border border-border bg-secondary px-5 py-4 text-center">
                        <p className="text-2xl font-bold text-muted-foreground">{flow.noExtractedAiReports.toLocaleString()}</p>
                        <p className="mt-1 text-[13px] font-bold uppercase tracking-[0.14em] text-muted-foreground sm:text-sm">
                          Don&apos;t mention AI
                        </p>
                        <div className="mt-3 h-1.5 w-full rounded-full bg-white">
                          <div className="h-full rounded-full bg-border" style={{ width: `${100 - extractPct}%` }} />
                        </div>
                        <p className="text-xs text-muted mt-1">{100 - extractPct}% of filings</p>
                      </div>
                    </div>
                  );
                })()}

                {/* Connector: left branch continues */}
                <FlowConnectorMergeLeft />

                {/* Stage 3: Phase 1 classification */}
                <StageFrame className="mx-auto max-w-6xl">
                  <p className="text-sm font-bold uppercase tracking-widest text-muted-foreground mb-2 text-center">
                    Phase 1 — Mention-Type Classification
                  </p>
                  <p className="mb-5 text-center text-sm text-muted">
                    Extracted passages are classified into six high-level outputs. Only the
                    relevant signal branches continue to Phase 2.
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {phase1Cards.map(card => {
                      const reportPct = flow.extractedAiReports > 0
                        ? Math.round((card.count / flow.extractedAiReports) * 100)
                        : 0;

                      return (
                        <div key={card.id} className="border border-border px-5 py-5">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${card.dotClass}`} />
                            <span className="text-base font-bold text-primary">{card.label}</span>
                          </div>
                          <p className="text-sm text-muted">{card.count.toLocaleString()} reports</p>
                          <div className="mt-2 h-1.5 w-full rounded-full bg-secondary">
                            <div className={`h-full rounded-full ${card.barClass}`} style={{ width: `${reportPct}%` }} />
                          </div>
                          <p className="text-xs text-muted mt-1">{reportPct}% of extracted</p>
                          <div className={`mt-4 rounded-sm border px-3 py-2 text-xs font-medium text-muted ${card.borderClass} ${card.surfaceClass}`}>
                            {card.continuesToPhase2 ? 'Continues to Phase 2' : 'Final output at Phase 1'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </StageFrame>

                {/* Connector */}
                <FlowConnectorStraight />

                {/* Stage 4: Phase 2 detailed taxonomies */}
                <StageFrame>
                  <p className="text-sm font-bold uppercase tracking-widest text-muted-foreground mb-2 text-center">
                    Phase 2 — Detailed Taxonomies
                  </p>
                  <p className="mb-4 text-center text-sm text-muted">
                    Only the Phase 1 categories with dedicated taxonomies continue below.
                  </p>
                  <div className="grid gap-4 sm:grid-cols-3">
                    {phase2Cards.map(card => {
                      return (
                        <div key={card.id} className="border border-border px-5 py-5">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${card.dotClass}`} />
                            <span className="text-base font-bold text-primary">{card.label}</span>
                          </div>
                          <div className={`mt-4 pl-3 border-l-2 ${card.borderClass} space-y-1`}>
                            {card.children.map(child => {
                              const chunkShare = card.totalChildChunks > 0
                                ? (child.chunkCount / card.totalChildChunks) * 100
                                : 0;

                              return (
                                <div
                                  key={child.id}
                                  className="relative flex items-baseline justify-between gap-3 overflow-hidden rounded-sm px-2 py-1 text-sm text-muted"
                                  title={`${child.count.toLocaleString()} reports; ${child.chunkCount.toLocaleString()} chunks tagged ${child.label}`}
                                >
                                  <div
                                    className={`absolute inset-y-1 left-0 rounded-sm ${card.highlightClass}`}
                                    style={{ width: `${chunkShare}%` }}
                                  />
                                  <span className="relative z-10">{child.label}</span>
                                  <span className="relative z-10 tabular-nums font-medium text-primary">
                                    {child.count.toLocaleString()}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </StageFrame>
              </div>
            </section>

            {/* ── Data ─────────────────────────────────────────── */}
            <section id="data" className="scroll-mt-20 space-y-16">
              <h2 className="aisi-h2 uppercase">1. Data</h2>

              <div id="data-scope" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Scope</h3>
                <p>
                  The dataset covers annual reports published by FTSE 350 companies and major
                  UK Critical National Infrastructure (CNI) operators across Finance, Energy,
                  Transport, and Health. Reports span financial years 2020 through 2026,
                  providing a longitudinal view of how AI disclosure has evolved before,
                  during, and after the large-language-model inflection point.
                </p>
              </div>

              <div id="data-decisions" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Decisions & Rationale</h3>
                <p>
                  <strong className="text-primary">Why FTSE 350 and CNI?</strong>{' '}
                  These companies are systematically important to the UK economy and subject
                  to heightened regulatory scrutiny. Their annual reports are a standardised,
                  legally required disclosure vehicle — the most reliable longitudinal corpus
                  for studying AI risk narrative.
                </p>
                <p>
                  <strong className="text-primary">Why 2020–2026?</strong>{' '}
                  This window spans the pre-LLM era, the ChatGPT inflection (late 2022), and
                  the subsequent rapid adoption cycle. It lets us track how AI disclosure
                  language shifted from boilerplate to substantive across the same population
                  of companies.
                </p>
                <p>
                  <strong className="text-primary">Why annual reports?</strong>{' '}
                  Unlike earnings calls or press releases, annual reports are audited,
                  structured, and published on a consistent cadence — suitable for
                  longitudinal analysis without noise from reactive media cycles.
                </p>
              </div>

              <div id="data-acknowledgements" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Acknowledgements</h3>
                <p>
                  The machine-readable Markdown versions of the annual reports were provided
                  by <strong className="text-primary">FR</strong>. Converting PDF filings to
                  clean, structured text is a significant undertaking, and their contribution
                  made it possible to build the extraction pipeline without spending the bulk
                  of the project on document parsing.
                </p>
              </div>
            </section>

            {/* ── Pre-processing ───────────────────────────────── */}
            <section id="preprocessing" className="scroll-mt-20 space-y-16">
              <h2 className="aisi-h2 uppercase">2. Pre-processing</h2>

              <div id="preprocessing-approach" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Chunking Approach</h3>
                <p>
                  Each annual report is converted to clean Markdown text and split into
                  passages (chunks) using a sliding-window approach that respects paragraph
                  and section boundaries. A keyword filter isolates sections that explicitly
                  mention AI or closely related techniques; only those sections are retained
                  for annotation. Each chunk carries metadata: company identifier, reporting
                  year, release month, report section (e.g. <em>Risk Factors</em>,{' '}
                  <em>Strategy</em>), and a stable chunk ID for traceability.
                </p>
              </div>

              <div id="preprocessing-results" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Chunking Results</h3>
                <p>
                  After chunking, each filing is classified as having at least one
                  AI-relevant passage or not. The table below shows filings with AI
                  mentions and the number of AI-relevant chunks (excerpts) extracted
                  per year.
                </p>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b-2 border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary">Year</th>
                      <th className="py-2 pr-4 font-bold text-primary text-right">Filings</th>
                      <th className="py-2 pr-4 font-bold text-primary text-right">With AI Mention</th>
                      <th className="py-2 pr-4 font-bold text-primary text-right">% AI</th>
                      <th className="py-2 font-bold text-primary text-right">Excerpts that Mention AI</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    {data.datasets.perReport.blindSpotTrend.map((reportRow: Record<string, number>, i: number) => {
                      const chunkRow = data.datasets.perChunk.blindSpotTrend[i] as Record<string, number> | undefined;
                      return (
                        <tr key={reportRow.year} className="border-b border-border/50">
                          <td className="py-2 pr-4 font-medium text-primary">{reportRow.year}</td>
                          <td className="py-2 pr-4 text-right">{Number(reportRow.total_reports).toLocaleString()}</td>
                          <td className="py-2 pr-4 text-right">{Number(reportRow.ai_mention).toLocaleString()}</td>
                          <td className="py-2 pr-4 text-right">
                            {Number(reportRow.total_reports) > 0
                              ? `${Math.round((Number(reportRow.ai_mention) / Number(reportRow.total_reports)) * 100)}%`
                              : '—'}
                          </td>
                          <td className="py-2 text-right">{chunkRow ? Number(chunkRow.ai_mention).toLocaleString() : '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                  <tfoot>
                    <tr className="border-t-2 border-border font-bold text-primary">
                      <td className="py-2 pr-4">Total</td>
                      <td className="py-2 pr-4 text-right">{flow.totalReports.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-right">{flow.phase1SignalReports.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-right">
                        {Math.round((flow.phase1SignalReports / flow.totalReports) * 100)}%
                      </td>
                      <td className="py-2 text-right">{data.datasets.perChunk.summary.aiSignalReports.toLocaleString()}</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </section>

            {/* ── Processing ───────────────────────────────────── */}
            <section id="processing" className="scroll-mt-20 space-y-16">
              <h2 className="aisi-h2 uppercase">3. Processing</h2>

              {/* Phase 1 */}
              <div id="phase-1" className="scroll-mt-20 space-y-6">
                <h3 className="text-lg font-bold text-primary">Phase 1: Mention-Type Classification</h3>
                <p>
                  Phase 1 is a gating step. Each chunk is passed to a language-model
                  classifier that decides whether the passage contains a genuine AI mention
                  and, if so, assigns one or more mention-type labels. Chunks labelled
                  exclusively as <em>none</em> are filtered out before Phase 2.
                </p>

                <p>The mention-type taxonomy:</p>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary w-48">Label</th>
                      <th className="py-2 font-bold text-primary">Definition</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Adoption</td><td className="py-2">Current use, rollout, pilot, implementation, or delivery of AI systems.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Risk</td><td className="py-2">AI described as a downside or exposure for the company.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Harm</td><td className="py-2">AI described as causing or enabling harm (misinformation, fraud, abuse, safety incidents).</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Vendor reference</td><td className="py-2">A named AI model, vendor, or platform provider is referenced.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">General or ambiguous</td><td className="py-2">AI mentioned but too high-level or vague for the above categories.</td></tr>
                    <tr><td className="py-2 pr-4 font-bold text-primary">None</td><td className="py-2">No real AI mention / false positive. Exclusive — cannot co-occur with others.</td></tr>
                  </tbody>
                </table>

                <div className="p-6">
                  <MentionTypesChart
                    data={data.datasets.perReport.mentionTrend}
                    stackKeys={data.labels.mentionTypes}
                  />
                </div>

                <CollapsibleSection title="Phase 1 Prompts">
                  <p className="text-sm text-muted leading-relaxed">
                    The classification prompts and structured output schemas used in Phase 1
                    are versioned in the project repository. They will be linked here once the
                    repository is public.
                  </p>
                </CollapsibleSection>
              </div>

              {/* Phase 2 */}
              <div id="phase-2" className="scroll-mt-20 space-y-6">
                <h3 className="text-lg font-bold text-primary">Phase 2: Deep-Taxonomy Classification</h3>
                <p>
                  Chunks that passed Phase 1 are processed through specialised sub-classifiers
                  depending on their mention types. Adoption-labelled chunks are classified
                  by adoption type; risk-labelled chunks by risk category; vendor-labelled
                  chunks by provider; and all signal-bearing chunks receive a
                  substantiveness score.
                </p>

                {/* Adoption */}
                <h4 className="text-sm font-bold uppercase tracking-widest text-primary pt-4">Adoption Taxonomy</h4>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary w-48">Label</th>
                      <th className="py-2 font-bold text-primary">Definition</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Traditional AI / ML</td><td className="py-2">Traditional AI/ML — predictive models, computer vision, detection/classification systems.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">LLM / GenAI</td><td className="py-2">Large language model / GenAI use (GPT, Gemini, Claude, Copilot-style deployments).</td></tr>
                    <tr><td className="py-2 pr-4 font-bold text-primary">Agentic systems</td><td className="py-2">Autonomous or agent-based workflows with limited human intervention.</td></tr>
                  </tbody>
                </table>

                {/* Risk */}
                <h4 className="text-sm font-bold uppercase tracking-widest text-primary pt-4">Risk Taxonomy</h4>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary w-48">Label</th>
                      <th className="py-2 font-bold text-primary">Definition</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Strategic / competitive</td><td className="py-2">Competitive disadvantage, disruption, or failure to adapt.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Operational / technical</td><td className="py-2">Reliability/accuracy/model-risk failures degrading operations.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Cybersecurity</td><td className="py-2">AI-enabled attacks, fraud, breach pathways, or adversarial abuse.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Workforce impacts</td><td className="py-2">Displacement, skills gaps, or risky employee AI usage.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Regulatory / compliance</td><td className="py-2">Legal, regulatory, privacy, or IP liability and compliance burden.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Information integrity</td><td className="py-2">Misinformation, deepfakes, or authenticity manipulation.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Reputational / ethical</td><td className="py-2">Trust, fairness, ethics, or rights concerns.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Third-party / supply chain</td><td className="py-2">Dependency on external AI vendors and concentration exposure.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Environmental impact</td><td className="py-2">Energy, carbon, or resource-burden risk.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">National security</td><td className="py-2">Geopolitical destabilisation or critical-systems exposure.</td></tr>
                    <tr><td className="py-2 pr-4 font-bold text-primary">None</td><td className="py-2">No attributable risk category (or too vague to assign one).</td></tr>
                  </tbody>
                </table>

                {/* Vendor */}
                <h4 className="text-sm font-bold uppercase tracking-widest text-primary pt-4">Vendor Taxonomy</h4>
                <p>
                  Amazon, Google, Microsoft, OpenAI, Anthropic, Meta, internal (in-house),
                  undisclosed (implied but unnamed), and other (named provider outside the
                  predefined list, captured as free text).
                </p>

                {/* Substantiveness */}
                <h4 className="text-sm font-bold uppercase tracking-widest text-primary pt-4">Substantiveness</h4>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary w-48">Level</th>
                      <th className="py-2 font-bold text-primary">Definition</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Boilerplate</td><td className="py-2">Generic AI language; could appear in many reports unchanged.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Moderate</td><td className="py-2">Specific area identified but limited mechanism, metrics, or mitigation detail.</td></tr>
                    <tr><td className="py-2 pr-4 font-bold text-primary">Substantive</td><td className="py-2">Concrete mechanism, tangible action, commitment, metric, or timeline.</td></tr>
                  </tbody>
                </table>

                {/* Examples */}
                {data.exampleChunks.length > 0 && (
                  <div className="-mx-6 mt-8">
                    <ExampleBrowser exampleChunks={data.exampleChunks} />
                  </div>
                )}

                <CollapsibleSection title="Phase 2 Prompts">
                  <p className="text-sm text-muted leading-relaxed">
                    The Phase 2 prompts — adoption, risk, vendor, and substantiveness
                    classifier templates — are versioned in the project repository alongside
                    the annotations. They will be linked here once the repository is public.
                  </p>
                </CollapsibleSection>
              </div>
            </section>

            {/* ── Quality Assurance ────────────────────────────── */}
            <section id="quality-assurance" className="scroll-mt-20 space-y-6">
              <h2 className="aisi-h2 uppercase">4. Quality Assurance</h2>

              <p>
                We use schema-constrained outputs, deterministic settings, and explicit
                validation to reduce noise and improve reproducibility. The checks applied
                to the current dataset:
              </p>

              <ul className="space-y-3">
                <li><strong className="text-primary">Structured outputs</strong> — classifiers write to strict JSON response schemas; malformed or out-of-vocabulary labels are retried or flagged.</li>
                <li><strong className="text-primary">Conservative prompting</strong> — prompts require explicit AI attribution and discourage over-labelling; the default is <em>none</em> or <em>general_ambiguous</em>.</li>
                <li><strong className="text-primary">Temperature zero</strong> — all calls use deterministic settings for reproducibility.</li>
                <li><strong className="text-primary">Chunk-level traceability</strong> — every annotation maps back to a company, year, and report section via a stable chunk ID.</li>
                <li><strong className="text-primary">QA scripts</strong> — repository scripts cross-check label distributions, flag implausible co-occurrences, and surface low-confidence cases.</li>
                <li><strong className="text-primary">Human review</strong> — a sample of annotations was reviewed by a human annotator; disagreements were used to refine prompts and taxonomy definitions.</li>
              </ul>

              <div className="rounded border border-amber-200 bg-amber-50 p-6">
                <p className="text-sm font-bold uppercase tracking-widest text-amber-700 mb-2">
                  Incomplete section
                </p>
                <p className="text-sm text-amber-800">
                  @BART Write this section — include pass/fail rates from QA scripts,
                  human-review sample size, inter-annotator agreement scores, and known
                  limitations or edge cases.
                </p>
              </div>

              <div className="border-t border-border pt-10 mt-10 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Review the Data Yourself</h3>
                <p>
                  The full annotated dataset — every chunk, label, confidence score, and
                  metadata field — will be available for download. We encourage researchers,
                  policy analysts, and practitioners to examine the annotations, reproduce
                  the QA checks, and flag errors or improvements.
                </p>
                <div className="flex flex-wrap gap-4 pt-2">
                  <span className="inline-flex items-center rounded border border-border bg-secondary px-5 py-2.5 text-sm font-bold uppercase tracking-widest text-muted-foreground cursor-not-allowed">
                    Download Dataset — Coming Soon
                  </span>
                  <span className="inline-flex items-center rounded border border-border bg-secondary px-5 py-2.5 text-sm font-bold uppercase tracking-widest text-muted-foreground cursor-not-allowed">
                    GitHub Repository — Coming Soon
                  </span>
                </div>
              </div>
            </section>

          </article>

          {/* Sticky ToC */}
          <aside className="hidden xl:block">
            <div className="sticky top-8 pt-4">
              <MethodologyToc />
            </div>
          </aside>

        </div>
      </div>
    </div>
  );
}
