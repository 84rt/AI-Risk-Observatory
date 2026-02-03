'use client';

import { useMemo, useState } from 'react';
import { GenericHeatmap, StackedBarChart } from '@/components/overview-charts';
import type { GoldenDashboardData } from '@/lib/golden-set';

type View = {
  id: number;
  title: string;
  description: string;
};

const VIEWS: View[] = [
  {
    id: 1,
    title: 'Signal Mix',
    description: 'AI-related mention types extracted from annual reports in the golden set sample.',
  },
  {
    id: 2,
    title: 'Adoption & Vendors',
    description: 'AI adoption maturity levels and vendor references identified in disclosures.',
  },
  {
    id: 3,
    title: 'Risk Taxonomy',
    description: 'Risk categories by sector, showing how AI-related risks cluster across CNI industries.',
  },
  {
    id: 4,
    title: 'Quality Signals',
    description: 'Confidence and substantiveness scores measuring annotation reliability and disclosure depth.',
  },
];

const mentionColors: Record<string, string> = {
  adoption: '#0ea5e9',
  risk: '#f97316',
  vendor: '#14b8a6',
  general_ambiguous: '#64748b',
  harm: '#ef4444',
};

const adoptionColors: Record<string, string> = {
  non_llm: '#0f766e',
  llm: '#38bdf8',
  agentic: '#f59e0b',
};

const vendorColors: Record<string, string> = {
  openai: '#0ea5e9',
  microsoft: '#1d4ed8',
  google: '#f97316',
  internal: '#0f766e',
  other: '#64748b',
  undisclosed: '#cbd5e1',
};

const riskColors: Record<string, string> = {
  cybersecurity: '#ef4444',
  operational_technical: '#f97316',
  regulatory: '#f59e0b',
  reputational_ethical: '#14b8a6',
  information_integrity: '#0ea5e9',
  third_party_supply_chain: '#22c55e',
  strategic_market: '#84cc16',
  workforce: '#0f766e',
  environmental: '#10b981',
};

const formatNumber = (value: number) =>
  new Intl.NumberFormat('en-GB').format(value);

const formatLabel = (val: string | number) => {
  if (typeof val === 'number') return val.toString();
  const overrides: Record<string, string> = {
    llm: 'LLM',
    non_llm: 'Non-LLM',
    general_ambiguous: 'General / Ambiguous',
    third_party_supply_chain: 'Third-Party Supply Chain',
    operational_technical: 'Operational / Technical',
    reputational_ethical: 'Reputational / Ethical',
    information_integrity: 'Information Integrity',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

type DatasetKey = 'human' | 'llm';

const datasetLabels: Record<DatasetKey, string> = {
  human: 'Human Annotations',
  llm: 'LLM Annotations',
};

export default function DashboardClient({ data }: { data: GoldenDashboardData }) {
  const [activeView, setActiveView] = useState(1);
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('human');

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const activeData = data.datasets[datasetKey];

  const mentionStackKeys = useMemo(() => data.labels.mentionTypes, [data.labels.mentionTypes]);
  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const vendorStackKeys = useMemo(() => data.labels.vendorTags, [data.labels.vendorTags]);
  const riskStackKeys = useMemo(() => data.labels.riskLabels, [data.labels.riskLabels]);

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      <div className="border-b border-amber-300/60 bg-amber-100/80 px-6 py-4 text-center text-sm font-semibold uppercase tracking-[0.2em] text-amber-900">
        WIP — Data and labels are in active iteration. Do not treat as final.
      </div>
      <header className="relative overflow-hidden border-b border-slate-200/70">
        <div className="absolute inset-0">
          <div className="absolute -top-24 left-10 h-64 w-64 rounded-full bg-amber-200/70 blur-3xl" />
          <div className="absolute top-10 right-0 h-72 w-72 rounded-full bg-sky-200/70 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 h-48 w-48 rounded-full bg-emerald-200/60 blur-3xl" />
        </div>
        <div className="relative mx-auto max-w-7xl px-6 py-12">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="animate-rise">
              <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.2em] text-slate-500">
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">Golden Set</span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {datasetLabels[datasetKey]}
                </span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {data.years[0]}-{data.years[data.years.length - 1]}
                </span>
              </div>
              <h1 className="mt-4 text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
                AI Risk Observatory Calibration Dashboard
              </h1>
              <p className="mt-3 max-w-2xl text-base text-slate-600 sm:text-lg">
                Live view of the golden set used to calibrate the taxonomy choices in the report.
                Charts display per-report aggregated labels across the priority CNI sample.
              </p>
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex flex-col gap-2">
                <label htmlFor="dataset-select" className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  Display Dataset
                </label>
                <select
                  id="dataset-select"
                  value={datasetKey}
                  onChange={event => setDatasetKey(event.target.value as DatasetKey)}
                  className="rounded-xl border border-slate-200 bg-white/90 px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm"
                >
                  <option value="human">Human Annotations</option>
                  <option value="llm">LLM Annotations</option>
                </select>
              </div>
              <div className="animate-rise animate-rise-delay-1 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Reports</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(activeData.summary.totalReports)}
                </p>
              </div>
              <div className="animate-rise animate-rise-delay-2 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Companies</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(activeData.summary.totalCompanies)}
                </p>
              </div>
              <div className="animate-rise animate-rise-delay-3 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">AI Signal Reports</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(activeData.summary.aiSignalReports)}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600">
            <p>
              Switch between human and LLM outputs to audit model drift and over-tagging.
            </p>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {VIEWS.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`animate-rise rounded-2xl border px-5 py-[18px] text-left transition-all ${
                  activeView === item.id
                    ? 'border-slate-900 bg-slate-900 text-white shadow-lg'
                    : 'border-slate-200 bg-white/80 text-slate-700 hover:-translate-y-0.5 hover:border-slate-400'
                }`}
              >
                <p className="text-xs uppercase tracking-[0.2em] opacity-70">View {item.id}</p>
                <p className="mt-2 text-lg font-semibold">{item.title}</p>
                <p className="mt-2 text-sm leading-relaxed opacity-70">{item.description}</p>
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-12">
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-slate-900">{view.title}</h2>
          <p className="mt-2 max-w-3xl text-slate-600">{view.description}</p>
        </div>

        {activeView === 1 && (
          <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
            <StackedBarChart
              data={activeData.mentionTrend}
              xAxisKey="year"
              stackKeys={mentionStackKeys}
              colors={mentionColors}
            />
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
                <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Signal Coverage
                </h3>
                <p className="mt-3 text-3xl font-semibold text-slate-900">
                  {formatNumber(activeData.summary.adoptionReports + activeData.summary.riskReports)}
                </p>
                <p className="mt-2 text-sm text-slate-600">
                  Reports with adoption or risk signals in the golden set.
                </p>
                <div className="mt-6 space-y-3 text-sm text-slate-600">
                  <div className="flex items-center justify-between">
                    <span>Adoption reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.adoptionReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Risk reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.riskReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Vendor reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.vendorReports)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
                <p className="font-semibold text-slate-900">Category Definitions</p>
                <ul className="mt-2 space-y-1 leading-relaxed">
                  <li><span className="font-medium text-slate-800">Adoption:</span> Mentions of AI technology usage, deployment, or implementation.</li>
                  <li><span className="font-medium text-slate-800">Risk:</span> Discussions of AI-related risks, threats, or concerns.</li>
                  <li><span className="font-medium text-slate-800">Vendor:</span> References to specific AI vendors or providers.</li>
                  <li><span className="font-medium text-slate-800">General/Ambiguous:</span> AI mentions lacking clear adoption or risk context.</li>
                  <li><span className="font-medium text-slate-800">Harm:</span> Mentions of actual or potential AI-related harm incidents.</li>
                </ul>
                <p className="mt-3 leading-relaxed">
                  <span className="font-medium text-slate-800">Data source:</span> Labels aggregated per report from human or LLM annotations.
                  A report receives a tag if any text chunk has confidence ≥0.2.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
            <StackedBarChart
              data={activeData.adoptionTrend}
              xAxisKey="year"
              stackKeys={adoptionStackKeys}
              colors={adoptionColors}
            />
            <StackedBarChart
              data={activeData.vendorTrend}
              xAxisKey="year"
              stackKeys={vendorStackKeys}
              colors={vendorColors}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Category Definitions</p>
              <p className="mt-2 font-medium text-slate-800">Adoption Types:</p>
              <ul className="mt-1 space-y-1 leading-relaxed">
                <li><span className="font-medium text-slate-800">Non-LLM:</span> Traditional AI/ML systems (computer vision, predictive analytics, RPA).</li>
                <li><span className="font-medium text-slate-800">LLM:</span> Large language model applications (chatbots, content generation, summarization).</li>
                <li><span className="font-medium text-slate-800">Agentic:</span> Autonomous AI systems with decision-making capabilities.</li>
              </ul>
              <p className="mt-3 font-medium text-slate-800">Vendor Tags:</p>
              <ul className="mt-1 space-y-1 leading-relaxed">
                <li><span className="font-medium text-slate-800">OpenAI, Microsoft, Google:</span> Named major AI providers.</li>
                <li><span className="font-medium text-slate-800">Internal:</span> In-house developed AI solutions.</li>
                <li><span className="font-medium text-slate-800">Other:</span> Other named vendors not in the major category.</li>
                <li><span className="font-medium text-slate-800">Undisclosed:</span> Vendor not specified in disclosure.</li>
              </ul>
              <p className="mt-3 leading-relaxed">
                <span className="font-medium text-slate-800">Data source:</span> Extracted from annual report text via NLP classification.
              </p>
            </div>
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            <StackedBarChart
              data={activeData.riskTrend}
              xAxisKey="year"
              stackKeys={riskStackKeys}
              colors={riskColors}
            />
            <GenericHeatmap
              data={activeData.riskBySector}
              xLabels={data.labels.riskLabels}
              yLabels={data.sectors}
              baseColor="#f97316"
              valueFormatter={value => `${value}`}
              xLabelFormatter={formatLabel}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Risk Category Definitions</p>
              <ul className="mt-2 space-y-1 leading-relaxed text-xs">
                <li><span className="font-medium text-slate-800">Cybersecurity:</span> Data breaches, AI-enabled attacks, security vulnerabilities.</li>
                <li><span className="font-medium text-slate-800">Operational/Technical:</span> System failures, integration issues, performance degradation.</li>
                <li><span className="font-medium text-slate-800">Regulatory:</span> Compliance obligations, legal liability, regulatory uncertainty.</li>
                <li><span className="font-medium text-slate-800">Reputational/Ethical:</span> Brand damage, bias concerns, ethical considerations.</li>
                <li><span className="font-medium text-slate-800">Information Integrity:</span> Misinformation, hallucinations, data quality issues.</li>
                <li><span className="font-medium text-slate-800">Third-Party Supply Chain:</span> Vendor dependencies, API reliance, supplier risks.</li>
                <li><span className="font-medium text-slate-800">Strategic/Market:</span> Competitive displacement, market disruption, innovation pressure.</li>
                <li><span className="font-medium text-slate-800">Workforce:</span> Job displacement, skills gaps, labor relations.</li>
                <li><span className="font-medium text-slate-800">Environmental:</span> Energy consumption, carbon footprint, resource usage.</li>
              </ul>
              <p className="mt-3 leading-relaxed">
                <span className="font-medium text-slate-800">Heatmap:</span> Rows are CNI sectors, columns are risk categories.
                Darker cells indicate higher report counts.
              </p>
            </div>
          </div>
        )}

        {activeView === 4 && (
          <div className="grid gap-8 lg:grid-cols-2">
            <GenericHeatmap
              data={activeData.confidenceHeatmap}
              xLabels={data.years}
              yLabels={data.labels.confidenceBands}
              baseColor="#0ea5e9"
              valueFormatter={value => `${value} reports`}
              yLabelFormatter={formatLabel}
            />
            <GenericHeatmap
              data={activeData.substantivenessHeatmap}
              xLabels={data.years}
              yLabels={data.labels.substantivenessBands}
              baseColor="#14b8a6"
              valueFormatter={value => `${value} reports`}
              yLabelFormatter={formatLabel}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm lg:col-span-2">
              <p className="font-semibold text-slate-900">Quality Metric Definitions</p>
              <div className="mt-2 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="font-medium text-slate-800">Confidence (left heatmap):</p>
                  <p className="mt-1 leading-relaxed">
                    Average classifier confidence across adoption and risk labels per report.
                    Higher confidence indicates stronger classifier certainty in label assignment.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">High:</span> ≥67% average confidence</li>
                    <li><span className="font-medium">Medium:</span> 34-66% average confidence</li>
                    <li><span className="font-medium">Low:</span> &lt;34% average confidence</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-slate-800">Substantiveness (right heatmap):</p>
                  <p className="mt-1 leading-relaxed">
                    Measures disclosure depth—distinguishing concrete AI disclosures from
                    boilerplate or vague references.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">High:</span> Specific, actionable AI details</li>
                    <li><span className="font-medium">Medium:</span> Some concrete information</li>
                    <li><span className="font-medium">Low:</span> Generic or boilerplate mentions</li>
                  </ul>
                </div>
              </div>
              <p className="mt-3 leading-relaxed">
                <span className="font-medium text-slate-800">Data source:</span> Scores computed from annotation metadata averaged per report.
              </p>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
