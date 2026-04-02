type FlowStep = {
  label: string;
  detail: string;
};

type FlowStage = {
  stageNumber: string;
  title: string;
  summary: string;
  output: string;
  accentClass: string;
  surfaceClass: string;
  borderClass: string;
  badgeClass: string;
  steps: FlowStep[];
};

const FLOW_STAGES: FlowStage[] = [
  {
    stageNumber: '01',
    title: 'Pre-processing',
    summary:
      'Build the company-year universe, fetch the reports, normalize the documents, and isolate the passages that are genuinely relevant to AI.',
    output: 'Output: normalized report text plus AI-relevant excerpts with company, year, and document metadata.',
    accentClass: 'bg-[#e63946]',
    surfaceClass: 'bg-[#fff3f4]',
    borderClass: 'border-[#f1c6ca]',
    badgeClass: 'text-[#8f2732]',
    steps: [
      {
        label: 'Define the corpus',
        detail: 'Select the companies, reporting years, and metadata needed for consistent longitudinal analysis.',
      },
      {
        label: 'Fetch annual reports',
        detail: 'Pull the relevant filings into a single processable corpus rather than handling reports one by one.',
      },
      {
        label: 'Normalize document text',
        detail: 'Convert reports into clean Markdown/text so the rest of the pipeline can work on a stable representation.',
      },
      {
        label: 'Extract AI excerpts',
        detail: 'Keep only the passages that explicitly mention AI or clearly AI-specific techniques.',
      },
    ],
  },
  {
    stageNumber: '02',
    title: 'Classification',
    summary:
      'Run a staged classifier stack that separates vague AI language from concrete adoption, risk, vendor, and disclosure-quality signals.',
    output: 'Output: structured labels for mention type, adoption type, risk taxonomy, vendor references, and substantiveness.',
    accentClass: 'bg-[#0f6cbd]',
    surfaceClass: 'bg-[#f1f7fd]',
    borderClass: 'border-[#c9def3]',
    badgeClass: 'text-[#24598d]',
    steps: [
      {
        label: 'Gate the excerpt',
        detail: 'First determine whether the passage is a real AI mention, a false positive, or only a high-level ambiguous reference.',
      },
      {
        label: 'Assign mention type',
        detail: 'Label adoption, risk, harm, vendor references, or general ambiguous AI discussion.',
      },
      {
        label: 'Run phase-two tagging',
        detail: 'For signal-bearing excerpts, classify adoption type, risk category, and vendor/provider information.',
      },
      {
        label: 'Score disclosure quality',
        detail: 'Separate boilerplate AI language from more substantive reporting with concrete mechanisms or actions.',
      },
    ],
  },
  {
    stageNumber: '03',
    title: 'Aggregation',
    summary:
      'Roll excerpt-level labels up into report, sector, and year views so the corpus becomes usable as a monitoring system rather than a pile of documents.',
    output: 'Output: dashboard-ready trends, heatmaps, report-level metrics, and a traceable structured dataset.',
    accentClass: 'bg-[#00703c]',
    surfaceClass: 'bg-[#eefaf4]',
    borderClass: 'border-[#c8e7d6]',
    badgeClass: 'text-[#26634a]',
    steps: [
      {
        label: 'Aggregate across excerpts',
        detail: 'Combine labels at the report level so one company filing can be understood as a whole rather than as isolated snippets.',
      },
      {
        label: 'Compute metrics',
        detail: 'Generate counts, percentages, and breakdowns by year, sector, and market segment.',
      },
      {
        label: 'Identify patterns',
        detail: 'Surface adoption trends, risk concentration, vendor visibility, disclosure blind spots, and changes over time.',
      },
      {
        label: 'Publish to the dashboard',
        detail: 'Turn the processed corpus into charts and tables that can be explored without re-reading the source reports manually.',
      },
    ],
  },
];

export function ClassificationFlowDiagram() {
  return (
    <section className="space-y-5">
      {FLOW_STAGES.map(stage => (
        <article
          key={stage.title}
          className={`border ${stage.borderClass} bg-white shadow-sm`}
        >
          <div className="grid lg:grid-cols-[280px_minmax(0,1fr)]">
            <div className={`border-b ${stage.borderClass} ${stage.surfaceClass} p-6 lg:border-b-0 lg:border-r`}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className={`inline-flex text-[11px] font-bold uppercase tracking-[0.18em] ${stage.badgeClass}`}>
                    Stage {stage.stageNumber}
                  </span>
                  <h4 className="mt-3 text-2xl font-bold tracking-tight text-primary">{stage.title}</h4>
                </div>
                <span className={`mt-1 h-3 w-3 shrink-0 ${stage.accentClass}`} aria-hidden="true" />
              </div>
              <p className="mt-4 text-sm leading-6 text-muted">{stage.summary}</p>
              <p className="mt-5 border-t border-black/10 pt-4 text-[12px] font-medium leading-5 text-primary">
                {stage.output}
              </p>
            </div>

            <ol className="grid gap-px bg-border sm:grid-cols-2 xl:grid-cols-4">
              {stage.steps.map((step, index) => (
                <li key={`${stage.title}-${step.label}`} className="bg-white p-5">
                  <div className="flex items-start gap-4">
                    <span
                      className={`inline-flex h-8 w-8 shrink-0 items-center justify-center text-[11px] font-bold ${stage.surfaceClass} ${stage.badgeClass}`}
                    >
                      {stage.stageNumber}.{index + 1}
                    </span>
                    <div>
                      <h5 className="text-sm font-bold uppercase tracking-[0.08em] text-primary">{step.label}</h5>
                      <p className="mt-2 text-sm leading-6 text-muted">{step.detail}</p>
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </article>
      ))}
    </section>
  );
}
