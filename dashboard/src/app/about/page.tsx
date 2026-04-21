import { MentionTypesChart } from '@/components/mention-types-chart';
import ExampleBrowser from '@/components/example-browser';
import { MethodologyToc } from '@/components/methodology-toc';
import { CollapsibleSection } from '@/components/collapsible-section';
import { InfoTooltip } from '@/components/overview-charts';
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
    <div className={`rounded-lg border border-border/70 px-5 py-5 sm:px-6 ${className}`}>
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
  const phase2LabelMap: Record<string, string> = {
    // Adoption
    non_llm: 'Traditional AI (non-LLM)',
    llm: 'LLM',
    agentic: 'Agentic AI',
    agentic_ai: 'Agentic AI',
    // Risk
    regulatory_compliance: 'Regulatory/compliance',
    strategic_competitive: 'Strategic/competitive',
    operational_technical: 'Operational/technical',
    cybersecurity: 'Cybersecurity',
    reputational_ethical: 'Reputational/ethical',
    third_party_supply_chain: 'Third party/supply chain',
    workforce_impacts: 'Workforce impacts',
    information_integrity: 'Information integrity',
    national_security: 'National security',
    environmental_impact: 'Environmental impact',
    // Vendor
    openai: 'OpenAI',
    microsoft: 'Microsoft',
    google: 'Google',
    amazon: 'Amazon / AWS',
    nvidia: 'Nvidia',
    salesforce: 'Salesforce',
    databricks: 'Databricks',
    ibm: 'IBM',
    snowflake: 'Snowflake',
    meta: 'Meta',
    anthropic: 'Anthropic',
    xai: 'xAI / Grok',
    palantir: 'Palantir',
    arm: 'Arm',
    mistral: 'Mistral',
    uk_ai: 'UK AI Institutions',
    open_source_model: 'Open-Source Model',
    internal: 'Internal',
    other: 'Other',
    undisclosed: 'Undisclosed',
  };
  const phase1Cards = ['general_ambiguous', 'none', 'harm', 'adoption', 'risk', 'vendor'].flatMap(id => {
    if (id === 'none') {
      return [
        {
          id: 'none',
          label: 'None (including false positive)',
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
            This page explains how we turn annual reports from UK-listed public companies into the data
            powering the dashboard, and the decisions behind each step in the pipeline.
            For a deeper dive, the full technical report is coming soon.
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
                The AI Risk Observatory processes annual reports from UK-listed public companies through
                a two-stage AI classification pipeline. The dataset spans all annual reports
                published between {data.years[0]} and {data.years[data.years.length - 1]} by{' '}
                {summary.totalCompanies.toLocaleString()} companies, totalling{' '}
                {flow.totalReports.toLocaleString()} filings. Of these,{' '}
                {flow.extractedAiReports.toLocaleString()} filings contain at least one
                AI-relevant mention, and after quality filters{' '}
                {flow.phase1SignalReports.toLocaleString()} carry meaningful AI signal. Because
                annual reports can run to hundreds of pages, we extract only the relevant
                AI mentions and their surrounding context — giving us {totalChunks.toLocaleString()} annotated text chunks in
                total.
              </p>
              <p>The pipeline follows three stages:</p>
              <ol className="list-decimal list-inside space-y-1 pl-2">
                <li>Extract all relevant AI mentions from each filing.</li>
                <li>Broadly classify the type of AI mentioned into six categories: Adoption, Risk, Harm, Vendor, General or ambiguous, or False Positive.</li>
                <li>For each Adoption, Risk, and Vendor mention, classify it into a detailed sub-taxonomy.</li>
              </ol>
              <p>
                We also run a substantiveness classifier to measure the depth of each mention,
                rating it on a scale from boilerplate to substantive.
              </p>
              <p>
                The pipeline is illustrated below. Phase 1 labels are not mutually exclusive,
                so those counts sum to more than the number of extracted reports.
              </p>

              {/* Pipeline flowchart */}
              <div className="space-y-0">
                {/* Stage 1: Total */}
                <div className="flex justify-center">
                  <div className="rounded-lg border-2 border-primary bg-primary px-8 py-5 text-center text-white">
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
                      <div className="rounded-lg border border-border px-5 py-4 text-center">
                        <p className="text-2xl font-bold text-primary">{flow.extractedAiReports.toLocaleString()}</p>
                        <p className="mt-1 text-[13px] font-bold uppercase tracking-[0.14em] text-muted-foreground sm:text-sm">
                          Reports that mention AI
                        </p>
                        <div className="mt-3 h-1.5 w-full rounded-full bg-secondary">
                          <div className="h-full rounded-full bg-primary" style={{ width: `${extractPct}%` }} />
                        </div>
                        <p className="text-xs text-muted mt-1">{extractPct}% of filings</p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary px-5 py-4 text-center">
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
                    Phase 1: Classify the Type of AI Mention
                  </p>
                  <p className="mb-5 text-center text-sm text-muted">
                    AI mentions are classified into six categories.
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {phase1Cards.map(card => {
                      const reportPct = flow.extractedAiReports > 0
                        ? Math.round((card.count / flow.extractedAiReports) * 100)
                        : 0;

                      return (
                        <div key={card.id} className="rounded-lg border border-border px-5 py-5">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${card.dotClass}`} />
                            <span className="text-base font-bold text-primary">{card.label}</span>
                            {card.id === 'none' ? (
                              <InfoTooltip content="This is the only mutually exclusive Phase 1 tag. It is used as a quality filter when an initially extracted AI mention does not actually discuss AI, including false positives." />
                            ) : null}
                          </div>
                          <p className="text-sm text-muted">{card.count.toLocaleString()} reports</p>
                          <div className="mt-2 h-1.5 w-full rounded-full bg-secondary">
                            <div className={`h-full rounded-full ${card.barClass}`} style={{ width: `${reportPct}%` }} />
                          </div>
                          <p className="text-xs text-muted mt-1">{reportPct}% of extracted</p>
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
                    Phase 2: Detailed Taxonomies
                  </p>
                  <p className="mb-4 text-center text-sm text-muted">
                    From Phase 1, only Adoption, Risk, and Vendor are processed further into the following subcategories.
                  </p>
                  <div className="grid gap-4 sm:grid-cols-3">
                    {phase2Cards.map(card => {
                      return (
                        <div key={card.id} className="rounded-lg border border-border px-5 py-5">
                          <p className="text-base font-bold text-primary mb-3">{card.label}</p>
                          <div className={`pl-3 border-l-2 ${card.borderClass} space-y-1`}>
                            {card.children.map(child => {
                              const chunkShare = card.totalChildChunks > 0
                                ? (child.chunkCount / card.totalChildChunks) * 100
                                : 0;
                              const displayLabel = phase2LabelMap[child.id] ?? child.label;
                              return (
                                <div
                                  key={child.id}
                                  className="relative flex items-baseline justify-between gap-3 overflow-hidden rounded-sm px-2 py-1 text-sm text-muted"
                                  title={`${child.count.toLocaleString()} reports; ${child.chunkCount.toLocaleString()} AI mentions tagged ${displayLabel}`}
                                >
                                  <div
                                    className={`absolute inset-y-1 left-0 rounded-sm ${card.highlightClass}`}
                                    style={{ width: `${chunkShare}%` }}
                                  />
                                  <span className="relative z-10">{displayLabel}</span>
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
                  To measure AI risk, adoption, and vendor dependence across the UK economy, we
                  process all annual reports published by all public companies in the UK. There are
                  1,660 public companies listed on UK markets (LSE Main Market, AIM Market, and AQSE).
                  After excluding companies not registered in the UK (e.g. Irish or Canadian
                  companies listed on these exchanges) and firms without filings available via
                  Companies House, our working universe is approximately 1,362 companies. Each company
                  files, on average, one annual report per year.<sup className="text-xs align-super"><a href="#fn-1" className="hover:text-primary">1</a></sup>
                </p>
              </div>

              <div id="data-decisions" className="scroll-mt-20 space-y-6">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Decisions & Rationale</h3>
                <div className="space-y-3">
                  <p>
                    <strong className="text-primary">Why annual reports?</strong>{' '}
                    Unlike earnings calls, press releases, or public media, annual reports are
                    audited, structured, and published on a consistent cadence — making them a
                    reliable, high-signal source of information. UK public companies must
                    publish annual accounts, a strategic report, a directors&apos; report, and an
                    auditor&apos;s report under the Companies Act 2006. All listed companies share
                    that statutory core, but Main Market issuers face tighter deadlines and more
                    detailed disclosure rules than AIM and AQSE companies.<sup className="text-xs align-super"><a href="#fn-5" className="hover:text-primary">5</a></sup>
                  </p>
                  <p>
                    This makes annual reports well suited to tracking trends across the UK economy
                    over time. There are two
                    primary limitations: (1) they are inherently backward-looking, often with a
                    significant delay; and (2) their highly regulated nature means many statements
                    are boilerplate and contain little real information.<sup className="text-xs align-super"><a href="#fn-2" className="hover:text-primary">2</a></sup>
                  </p>
                </div>
                <p>
                  <strong className="text-primary">Why 2020–2026?</strong>{' '}
                  We chose this window to capture a pre-ChatGPT baseline (before the late-2022
                  inflection) and the rapid adoption cycle that followed.
                </p>
                <div className="space-y-3">
                  <p>
                    <strong className="text-primary">How do we map to CNI?</strong>{' '}
                    The Critical National Infrastructure in the{' '}
                    <a href="https://www.npsa.gov.uk/about-npsa/critical-national-infrastructure" className="underline underline-offset-2 hover:text-primary transition-colors" target="_blank" rel="noopener noreferrer">UK has 13 distinct sectors</a>.
                    Each company in our database has an{' '}
                    <a href="https://en.wikipedia.org/wiki/International_Standard_Industrial_Classification" className="underline underline-offset-2 hover:text-primary transition-colors" target="_blank" rel="noopener noreferrer">ISIC sector code</a>{' '}
                    that only partially maps to CNI sectors. We take a conservative approach,
                    using an LLM classifier to assign CNI sectors to companies that
                    do not map directly from ISIC; when no assignment can be made, we use an
                    &ldquo;Other&rdquo; CNI category.<sup className="text-xs align-super"><a href="#fn-3" className="hover:text-primary">3</a></sup>{' '}
                    A major limitation of CNI analysis via annual reports is that some sectors —
                    such as Space, Emergency Services, or Civil Nuclear — have few public companies
                    or suppliers represented.<sup className="text-xs align-super"><a href="#fn-4" className="hover:text-primary">4</a></sup>
                  </p>
                </div>
              </div>

              <div id="data-acknowledgements" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Data Provider Acknowledgment</h3>
                <p>
                  Converting PDFs to clean, structured text is technically demanding, and doing
                  so at that scale would have exceeded our compute budget. We partnered with{' '}
                  <a href="https://financialreports.eu" className="underline underline-offset-2 hover:text-primary transition-colors" target="_blank" rel="noopener noreferrer">FinancialReports.eu</a>,
                  a third-party financial data provider, to obtain all annual reports in our scope
                  in Markdown format. Their filings API and generous support made this project
                  possible.
                </p>
              </div>
            </section>

            {/* ── Pre-processing ───────────────────────────────── */}
            <section id="preprocessing" className="scroll-mt-20 space-y-16">
              <h2 className="aisi-h2 uppercase">2. Pre-processing</h2>

              <div id="preprocessing-approach" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Chunking Approach</h3>
                <p>
                  Once each annual report is in structured Markdown text, we split it into
                  chunks using a sliding-window approach that respects paragraph and
                  section boundaries, with generous padding around each AI mention. An AI keyword filter isolates sections that explicitly mention AI
                  or closely related techniques; only those sections are retained for further
                  annotation as AI mentions. Each chunk carries metadata: company identifier, reporting year,
                  release month, report section (e.g. <em>Risk Factors</em>,{' '}
                  <em>Strategy</em>), and a stable chunk ID for traceability.
                </p>
              </div>

              <div id="preprocessing-results" className="scroll-mt-20 space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary">Chunking Results</h3>
                <p>
                  The table below shows filings with AI mentions and the number of
                  AI mentions extracted per year.
                </p>
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b-2 border-border text-left">
                      <th className="py-2 pr-4 font-bold text-primary">Year</th>
                      <th className="py-2 pr-4 font-bold text-primary text-right">Number of Filings</th>
                      <th className="py-2 pr-4 font-bold text-primary text-right">Filings with AI Mention (% of total)</th>
                      <th className="py-2 font-bold text-primary text-right">Count of AI mentions</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted">
                    {data.datasets.perReport.blindSpotTrend.map((reportRow: Record<string, number>, i: number) => {
                      const chunkRow = data.datasets.perChunk.blindSpotTrend[i] as Record<string, number> | undefined;
                      const pct = Number(reportRow.total_reports) > 0
                        ? Math.round((Number(reportRow.ai_mention) / Number(reportRow.total_reports)) * 100)
                        : null;
                      return (
                        <tr key={reportRow.year} className="border-b border-border/50">
                          <td className="py-2 pr-4 font-medium text-primary">{reportRow.year}</td>
                          <td className="py-2 pr-4 text-right">{Number(reportRow.total_reports).toLocaleString()}</td>
                          <td className="py-2 pr-4 text-right">
                            {Number(reportRow.ai_mention).toLocaleString()}
                            {pct !== null && <span className="text-muted-foreground ml-1">({pct}%)</span>}
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
                      <td className="py-2 pr-4 text-right">
                        {flow.phase1SignalReports.toLocaleString()}
                        <span className="text-muted-foreground font-normal ml-1">({Math.round((flow.phase1SignalReports / flow.totalReports) * 100)}%)</span>
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
                  First, each chunk is passed to an LLM classifier that decides whether the
                  text contains a genuine AI mention and, if so, assigns one or more
                  mention-type labels. Chunks assigned only the <em>None</em> label are
                  filtered out as false positives before Phase 2.
                </p>

                <p>The Phase 1 classifier uses the following taxonomy:</p>
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

                <div className="space-y-2 pt-2">
                  <p className="text-sm font-bold uppercase tracking-widest text-primary">Phase 1 Label Distribution Over Time</p>
                  <p className="text-sm text-muted">
                    Distribution of Phase 1 mention-type labels across all AI-mentioning filings, by year. Labels are not mutually exclusive, so a single filing can contribute to multiple categories.
                  </p>
                  <MentionTypesChart
                    data={data.datasets.perReport.mentionTrend}
                    stackKeys={[...data.labels.mentionTypes, 'none'].filter((v, i, a) => a.indexOf(v) === i)}
                  />
                </div>

                <CollapsibleSection title="Phase 1 classifier prompt">
                  <div className="space-y-4">
                    <p className="text-sm text-muted leading-relaxed">
                      All prompts and structured output schemas used in Phase 1 are visible in the{' '}
                      <a href="https://github.com/84rt/ai-risk-observatory" className="underline underline-offset-2 hover:text-primary transition-colors" target="_blank" rel="noopener noreferrer">project repository</a>.
                      The prompt used for the Phase 1 classification of the data visible on the dashboard is:
                    </p>
                    <pre className="overflow-x-auto rounded border border-border bg-secondary p-4 text-xs leading-relaxed text-muted whitespace-pre-wrap">
{`You are an expert analyst labeling AI mentions from company annual reports.

## TASK
Assign ALL mention types that apply to the excerpt. Types are NOT mutually exclusive except for "none".
If the excerpt contains no AI mention, return only "none". Only tag content that is explicitly about AI in the excerpt; ignore unrelated sentences.

## RULES
1. AI EXPLICITNESS GATE: If the excerpt does NOT explicitly mention AI/ML/LLM/GenAI or a clearly AI-specific technique (e.g., machine learning, neural networks, computer vision), assign the tag "none". Terms like "data analytics" or "digital tools" generally are NOT considered AI under our definition. The tag "none" is used when there is no AI mention, a false positive, or unrelated automation not clearly AI. Only consider terms like "autonomous or virtual assistant" as AI if it can be clearly attributed to AI. Otherwise, use the following tags in a non-mutually exclusive manner: adoption (the current usage of AI technology by the company), risk (AI as a risk: risks that are directly coming from AI), harm (past harms that were caused by AI), vendor (any mention of a provider of AI technology), general_ambiguous (any statement about AI that does not fit into the other tags). Here are more details on each tag:
   - adoption: must directly describe real current deployment, implementation, rollout, pilot, or use of AI by the company or for its clients. Generic statements about intent/strategy/roadmaps/plans (adoption in the future) are NOT considered adoption. Treat "exploring", "piloting", or "investigating" AI use as adoption ONLY when it refers to an initiative currently underway (e.g. "current trial resulted in..."). Delivering AI systems directly or indirectly for clients does count as adoption; pure consulting/advice without deployment does NOT.
   - risk: must directly attribute AI as the source of a risk or downside (i.e. strategic & competitive, operational & technical, cybersecurity, workforce impacts, regulatory & compliance, information integrity, reputational & ethical, third-party & supply chain, environmental impact, and national security etc.). The excerpt might contain a sentence on risk and a separate sentence on AI; make sure to only assign the "risk" tag if AI is mentioned as the source of the risk. Generic risk language without explicitly mentioning AI is NOT risk from AI. However, the risk section might outline downstream risks or effects from AI technologies in an indirect way; these should be classified as risk from AI.
   - harm: must describe past harms that were caused by AI (misinformation, fraud, abuse, safety incidents).
   - vendor: must explicitly name a third-party AI vendor/platform that provides the AI technology (i.e. Microsoft, Google, OpenAI, AWS, or explicitly developed in-house). We primarily want to tag text that mentions any information about what AI models are used by the company (i.e. GPT or Google Gemini).
   - general_ambiguous: vague AI strategy, high-level plans, or AI mentions that don't have enough context or are not specific enough. If AI is explicitly mentioned but does not meet adoption/risk/harm/vendor, use general_ambiguous. The tag general_ambiguous should only be added when the excerpt clearly talks about AI but does not meet the other tag definitions.
2. Assign confidence scores to each tag. Confidence scores always indicate how likely the label applies (including "none").

## CONFIDENCE GUIDANCE (0.0–1.0)
- 0.2: faint/implicit signal; could be this type but hard to tell
- 0.5: uncertain — weak evidence
- 0.8: likely YES — strong signal, but not fully explicit
- 0.95–1.0: confident YES — explicit, unambiguous mention

## EXAMPLES
- "We deployed an AI chatbot for customer support." → adoption ~0.9
- "We are exploring AI opportunities." → general_ambiguous ~0.7
- "We are piloting AI to automate invoice processing." → adoption ~0.8
- "We use data analytics and predictive analytics to optimize routing." → none ~0.6 (unless AI/ML explicitly stated)
- "AI could increase misinformation risks." → risk ~0.8
- "AI is a long-term megatrend, being widely adopted within the industry; we are evaluating any risks associated with it." → risk ~0.7 (no "adoption" tag, as no evidence of adoption by company)
- "We partner with Microsoft for AI tooling." → vendor ~0.9, adoption ~0.6
- "We partnered with OpenAI to deliver AI systems for clients in 2024." → vendor ~0.9, adoption ~0.8
- "Automation of customer service tasks improved our..." → general_ambiguous ~0.2 (not necessarily AI)
- "Address: FT-AI 4810 Shangh'ai', is where the..." → none ~0.9 (a false positive)
- "AI-generated misinformation has damaged our brand reputation." → harm ~0.9

## OUTPUT CONSTRAINTS
- mention_types must be non-empty.
- If "none" is present, it must be the only label.
- Provide a confidence score for EVERY label in mention_types.
- Do NOT include confidence scores for labels not in mention_types.

## EXCERPT CONTEXT
Company: {firm_name} | Sector: {sector} | Report Year: {report_year} | Report Section: {report_section}`}
                    </pre>
                  </div>
                </CollapsibleSection>
              </div>

              {/* Phase 2 */}
              <div id="phase-2" className="scroll-mt-20 space-y-6">
                <h3 className="text-lg font-bold text-primary">Phase 2: Deep-Taxonomy Classification</h3>
                <p>
                  Chunks that passed Phase 1 are processed by dedicated classifiers
                  depending on their mention types. We process three of the Phase 1 mention
                  types — adoption, risk, and vendor — each through its own LLM classifier.
                  Chunks tagged as Risk are also scored for substantiveness. The taxonomies
                  used are as follows:
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
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Traditional AI/ML</td><td className="py-2">Traditional AI/ML — predictive models, computer vision, detection/classification systems.</td></tr>
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">LLM/GenAI</td><td className="py-2">Large language model/GenAI use (GPT, Gemini, Claude, Copilot-style deployments).</td></tr>
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
                  Vendors are tagged against a predefined list of named providers: OpenAI, Microsoft,
                  Google, Amazon / AWS, Nvidia, Salesforce, Databricks, IBM, Snowflake, Meta,
                  Anthropic, xAI / Grok, Palantir, Arm, Mistral, and UK AI institutions (e.g. DSIT,
                  AISI, Alan Turing Institute). Additional categories cover open-source models
                  (named frameworks without a commercial vendor), internal (in-house AI development),
                  undisclosed (implied but unnamed provider), and other (any named provider outside
                  the predefined list, captured as free text).
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
                    <tr className="border-b border-border/50"><td className="py-2 pr-4 font-bold text-primary">Moderate</td><td className="py-2">A specific area is identified, but without concrete mechanisms, metrics, or mitigation steps.</td></tr>
                    <tr><td className="py-2 pr-4 font-bold text-primary">Substantive</td><td className="py-2">Concrete mechanism, tangible action, commitment, metric, or timeline.</td></tr>
                  </tbody>
                </table>

                <CollapsibleSection title="Phase 2 classifier prompts">
                  <div className="space-y-4">
                    <p className="text-sm text-muted leading-relaxed">
                      The Phase 2 classifier prompts, as well as all other prompts used in the pipeline, are available in the repository at{' '}
                      <a
                        href="https://github.com/84rt/ai-risk-observatory/blob/main/pipeline/prompts/classifiers.yaml"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-block underline underline-offset-2 hover:text-primary transition-colors"
                      >
                        <code className="rounded bg-secondary px-1 py-0.5 font-mono text-xs">
                          /pipeline/prompts/classifiers.yaml
                        </code>
                      </a>.
                    </p>
                  </div>
                </CollapsibleSection>
              </div>
            </section>

            {/* ── Quality Assurance ────────────────────────────── */}
            <section id="quality-assurance" className="scroll-mt-20 space-y-6">
              <h2 className="aisi-h2 uppercase">4. Quality Assurance</h2>

              <p>
                We enforce structured outputs and explicit validation rules to reduce noise and
                improve reproducibility. We apply the following checks:
              </p>

              <ul className="space-y-4">
                <li><strong className="text-primary">Structured outputs</strong> — classifiers write to strict JSON response schemas; malformed or labels outside the permitted set are retried or flagged.</li>
                <li><strong className="text-primary">Conservative prompting</strong> — prompts require explicit AI attribution and discourage over-labelling; the default outcome is <em>none</em> or <em>general_ambiguous</em>.</li>
                <li><strong className="text-primary">Temperature zero</strong> — all classifier calls use temperature zero for deterministic, reproducible outputs.</li>
                <li><strong className="text-primary">Chunk-level traceability</strong> — every annotation maps back to a company, year, and report section via a stable chunk ID.</li>
                <li>
                  <strong className="text-primary">QA scripts</strong> — we run QA tests across each pipeline stage, checking primarily for anomalies and out-of-distribution outputs:
                  <ul className="mt-2 space-y-1 pl-4 list-disc text-muted">
                    <li>Document size, length, duplication, fiscal-year-match, and text anomalies (non-Markdown formatting, unexpected characters).</li>
                    <li>Outlier analysis on the distribution of Phase 1 and Phase 2 labels per company, report, and year; AI mentions extracted per report; and chunk creation keywords.</li>
                  </ul>
                  <p className="mt-2">All flagged outputs were manually reviewed.</p>
                </li>
                <li><strong className="text-primary">Human review</strong> — the dataset is vast, and while we have made every effort to audit anomalies arising from data processing, some errors and misclassifications may remain. Our data is available for download. If you spot an issue, please file it on the repository.</li>
              </ul>

              <div className="flex flex-wrap gap-4">
                <a
                  href="https://github.com/84rt/AI-Risk-Observatory/releases/download/dataset-v1.0/airo-dataset-v1.0.zip"
                  className="inline-flex items-center gap-2 rounded border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
                >
                  Download Dataset
                </a>
                <a href="https://github.com/84rt/ai-risk-observatory" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 rounded border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors">
                  GitHub Repository
                </a>
              </div>

              {data.exampleChunks.length > 0 && (
                <div className="-mx-6">
                  <ExampleBrowser exampleChunks={data.exampleChunks} />
                </div>
              )}
            </section>

            {/* ── Footnotes ────────────────────────────────────── */}
            <section id="footnotes" className="scroll-mt-20 border-t border-border pt-10 space-y-3 text-sm text-muted">
              <h2 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Footnotes</h2>
              <ol className="list-decimal list-inside space-y-3">
                <li id="fn-1">
                  Some companies have multiple subsidiaries with separate filings, while others
                  were recently listed or spun off and therefore have fewer years of filings
                  available. This means the per-company filing count is not uniform across the
                  dataset.
                </li>
                <li id="fn-2">
                  To address the boilerplate problem we apply a substantiveness classifier (see
                  Phase 2 above) that rates each mention on a scale from boilerplate to
                  substantive, allowing users to filter to high-signal disclosures.
                </li>
                <li id="fn-3">
                  The ISIC-to-CNI mapping follows two steps: a direct lookup for ISIC codes that
                  clearly correspond to a CNI sector, followed by an LLM classifier for ambiguous
                  cases. Companies that cannot be assigned to any CNI sector are labelled
                  &ldquo;Other&rdquo;.
                </li>
                <li id="fn-4">
                  The following CNI sectors have particularly low public-company representation
                  in our dataset: Space (0), Emergency Services (0), Civil Nuclear (2), Water
                  (18), Defence (20), Government (20), Data Infrastructure (22), Communications
                  (28), Chemicals (34). Conclusions drawn about these sectors should be treated
                  with caution.
                </li>
                <li id="fn-5">
                  Main Market issuers are generally subject to FCA disclosure and listing rules,
                  including a four-month reporting deadline, while AIM and AQSE companies
                  typically have up to six months. The auditor&apos;s formal opinion covers the
                  financial statements, not the annual report narrative as a whole.
                </li>
              </ol>
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
