type FlowStage = {
  sectionNumber: number;
  title: string;
  panelClass: string;
  labelClass: string;
  arrowClass: string;
  badgeClass: string;
  steps: string[];
};

const FLOW_STAGES: FlowStage[] = [
  {
    sectionNumber: 1,
    title: 'Pre Processing',
    panelClass: 'border-[#dfbcc7] bg-[#f9eef2]',
    labelClass: 'text-[#7d5a67]',
    arrowClass: 'text-[#c990a2]',
    badgeClass: 'border-[#d8b0bd] text-[#7d5a67] bg-white/95',
    steps: [
      'List companies and years to analyse with metadata',
      'Fetch annual reports',
      'Convert reports to Markdown',
      'Extract all excerpts that mention AI',
    ],
  },
  {
    sectionNumber: 2,
    title: 'Processing',
    panelClass: 'border-[#bdd3ea] bg-[#eef5fd]',
    labelClass: 'text-[#4f6f92]',
    arrowClass: 'text-[#93b5da]',
    badgeClass: 'border-[#a7c4e3] text-[#4f6f92] bg-white/95',
    steps: [
      'For all AI mention excerpts',
      'Run a mention type classifier',
      'Run Phase 2 classifiers (risk, adoption, vendor)',
      'Run the boilerplate level classifier',
    ],
  },
  {
    sectionNumber: 3,
    title: 'Post Processing',
    panelClass: 'border-[#b8ddd2] bg-[#edf8f4]',
    labelClass: 'text-[#4e7a6f]',
    arrowClass: 'text-[#8fc4b6]',
    badgeClass: 'border-[#9ecfbe] text-[#4e7a6f] bg-white/95',
    steps: [
      'Aggregate classifications across chunks and reports',
      'Compute metrics and trends',
      'Produce a structured dataset',
      'Visualize on the dashboard',
    ],
  },
];

export function ClassificationFlowDiagram() {
  return (
    <section className="overflow-x-auto rounded-md border border-slate-200 bg-white/90 p-4 shadow-sm sm:p-5">
      <div className="min-w-[1040px] space-y-4">
        {FLOW_STAGES.map((stage) => (
          <div key={stage.title} className={`relative rounded-md border-2 p-5 pt-8 ${stage.panelClass}`}>
            <p className={`absolute -top-3 left-3 rounded-md bg-white px-2 py-0.5 text-[11px] font-semibold uppercase tracking-[0.14em] ${stage.labelClass}`}>
              {stage.title}
            </p>
            <div className="flex items-center gap-2">
              {stage.steps.map((step, index) => (
                <div key={`${stage.title}-${index}`} className="flex items-center gap-2">
                  <div className="relative flex h-[110px] w-[220px] shrink-0 items-center justify-center rounded-md border border-slate-200 bg-white px-4 text-center text-sm font-semibold leading-tight text-slate-700 shadow-sm">
                    <span className={`absolute left-2 top-2 rounded border px-1.5 py-0.5 text-[10px] font-semibold leading-none ${stage.badgeClass}`}>
                      {stage.sectionNumber}.{index + 1}
                    </span>
                    {step}
                  </div>
                  {index < stage.steps.length - 1 ? (
                    <span className={`text-3xl font-bold leading-none ${stage.arrowClass}`}>→</span>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
