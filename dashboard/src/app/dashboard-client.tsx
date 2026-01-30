'use client';

import { useMemo, useState } from 'react';
import { GenericHeatmap, StackedBarChart, MetricsBarChart, AgreementCard } from '@/components/overview-charts';
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
    description: 'How the golden set classifies AI-related mentions across the two-year sample.',
  },
  {
    id: 2,
    title: 'Adoption & Vendors',
    description: 'Adoption type coverage and the vendor footprint surfaced in reports.',
  },
  {
    id: 3,
    title: 'Risk Taxonomy',
    description: 'Risk labels by sector, highlighting where risks cluster in the calibration sample.',
  },
  {
    id: 4,
    title: 'Quality Signals',
    description: 'Confidence and substantiveness bands for the annotated risk/adoption mentions.',
  },
  {
    id: 5,
    title: 'Model Evaluation',
    description: 'LLM vs Human annotation agreement metrics. Measures classifier reliability.',
  },
];

const mentionColors: Record<string, string> = {
  adoption: '#0ea5e9',
  risk: '#f97316',
  vendor: '#14b8a6',
  general_ambiguous: '#64748b',
  harm: '#ef4444',
  none: '#e2e8f0',
};

const adoptionColors: Record<string, string> = {
  non_llm: '#0f766e',
  llm: '#38bdf8',
  agentic: '#f59e0b',
  none: '#cbd5e1',
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
  none: '#cbd5e1',
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
    none: 'Unspecified',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export default function DashboardClient({ data }: { data: GoldenDashboardData }) {
  const [activeView, setActiveView] = useState(1);

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];

  const mentionStackKeys = useMemo(() => data.labels.mentionTypes, [data.labels.mentionTypes]);
  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const vendorStackKeys = useMemo(() => data.labels.vendorTags, [data.labels.vendorTags]);
  const riskStackKeys = useMemo(() => data.labels.riskLabels, [data.labels.riskLabels]);

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
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
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">Human Annotations</span>
                {activeView === 5 && (
                  <span className="rounded-full bg-amber-100 px-3 py-1 font-semibold text-amber-700">LLM Comparison</span>
                )}
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
            <div className="flex flex-wrap gap-3">
              <div className="animate-rise animate-rise-delay-1 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Reports</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(data.summary.totalReports)}
                </p>
              </div>
              <div className="animate-rise animate-rise-delay-2 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Companies</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(data.summary.totalCompanies)}
                </p>
              </div>
              <div className="animate-rise animate-rise-delay-3 rounded-2xl border border-slate-900/10 bg-white/90 px-5 py-4 shadow-sm">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">AI Signal Reports</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {formatNumber(data.summary.aiSignalReports)}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {VIEWS.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`animate-rise rounded-2xl border px-5 py-4 text-left transition-all ${
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
              data={data.mentionTrend}
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
                  {formatNumber(data.summary.adoptionReports + data.summary.riskReports)}
                </p>
                <p className="mt-2 text-sm text-slate-600">
                  Reports with adoption or risk signals in the golden set.
                </p>
                <div className="mt-6 space-y-3 text-sm text-slate-600">
                  <div className="flex items-center justify-between">
                    <span>Adoption reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(data.summary.adoptionReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Risk reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(data.summary.riskReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Vendor reports</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(data.summary.vendorReports)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
                <p className="font-semibold text-slate-900">Interpretation</p>
                <p className="mt-2 leading-relaxed">
                  Labels are aggregated per report: a report receives a tag if any chunk has
                  confidence ≥0.2 for that label. This provides a company-year level view of
                  AI disclosure patterns.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
            <StackedBarChart
              data={data.adoptionTrend}
              xAxisKey="year"
              stackKeys={adoptionStackKeys}
              colors={adoptionColors}
            />
            <StackedBarChart
              data={data.vendorTrend}
              xAxisKey="year"
              stackKeys={vendorStackKeys}
              colors={vendorColors}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Context</p>
              <p className="mt-2 leading-relaxed">
                Adoption is scoped to type-of-AI only (LLM, non-LLM, agentic), per the calibrated
                taxonomy in the choices report. Vendor tags are captured without normalization,
                preserving the requested early signal on concentration.
              </p>
            </div>
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            <StackedBarChart
              data={data.riskTrend}
              xAxisKey="year"
              stackKeys={riskStackKeys}
              colors={riskColors}
            />
            <GenericHeatmap
              data={data.riskBySector}
              xLabels={data.labels.riskLabels.filter(l => l !== 'none')}
              yLabels={data.sectors}
              baseColor="#f97316"
              valueFormatter={value => `${value}`}
              xLabelFormatter={formatLabel}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Read the heatmap</p>
              <p className="mt-2 leading-relaxed">
                Rows show CNI sectors; columns show the risk taxonomy categories.
                Cell values show the number of reports; darker cells indicate higher counts.
              </p>
            </div>
          </div>
        )}

        {activeView === 4 && (
          <div className="grid gap-8 lg:grid-cols-2">
            <GenericHeatmap
              data={data.confidenceHeatmap}
              xLabels={data.years}
              yLabels={data.labels.confidenceBands}
              baseColor="#0ea5e9"
              valueFormatter={value => `${value} reports`}
              yLabelFormatter={formatLabel}
            />
            <GenericHeatmap
              data={data.substantivenessHeatmap}
              xLabels={data.years}
              yLabels={data.labels.substantivenessBands}
              baseColor="#14b8a6"
              valueFormatter={value => `${value} reports`}
              yLabelFormatter={formatLabel}
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm lg:col-span-2">
              <p className="font-semibold text-slate-900">Quality Lens</p>
              <p className="mt-2 leading-relaxed">
                Confidence bands reflect the average confidence score across adoption and risk
                labels in each report. Substantiveness is averaged across chunks to separate
                boilerplate-heavy reports from those with more concrete disclosures.
              </p>
            </div>
          </div>
        )}

        {activeView === 5 && (
          <div className="space-y-8">
            {/* Summary Cards */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
                <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Coverage
                </h3>
                <p className="mt-3 text-3xl font-semibold text-slate-900">
                  {formatNumber(data.comparison.coverage.commonChunks)}
                </p>
                <p className="mt-2 text-sm text-slate-600">
                  Common chunks compared
                </p>
              </div>
              <AgreementCard
                title="Mention Types"
                jaccard={data.comparison.mentionTypes.avgJaccard}
                bestLabel={data.comparison.mentionTypes.metrics.sort((a, b) => b.f1 - a.f1)[0]?.label}
                bestF1={data.comparison.mentionTypes.metrics.sort((a, b) => b.f1 - a.f1)[0]?.f1}
              />
              <AgreementCard
                title="Adoption Types"
                jaccard={data.comparison.adoptionTypes.avgJaccard}
                bestLabel={data.comparison.adoptionTypes.metrics.sort((a, b) => b.f1 - a.f1)[0]?.label}
                bestF1={data.comparison.adoptionTypes.metrics.sort((a, b) => b.f1 - a.f1)[0]?.f1}
              />
              <AgreementCard
                title="Vendor Tags"
                jaccard={data.comparison.vendorTags.avgJaccard}
                bestLabel={data.comparison.vendorTags.metrics.sort((a, b) => b.f1 - a.f1)[0]?.label}
                bestF1={data.comparison.vendorTags.metrics.sort((a, b) => b.f1 - a.f1)[0]?.f1}
              />
            </div>

            {/* Metrics Charts */}
            <div className="grid gap-8 lg:grid-cols-2">
              <MetricsBarChart
                data={data.comparison.mentionTypes.metrics}
                title="Mention Types - Precision / Recall / F1"
              />
              <MetricsBarChart
                data={data.comparison.adoptionTypes.metrics}
                title="Adoption Types - Precision / Recall / F1"
              />
              <MetricsBarChart
                data={data.comparison.riskTaxonomy.metrics}
                title="Risk Taxonomy - Precision / Recall / F1"
              />
              <MetricsBarChart
                data={data.comparison.vendorTags.metrics}
                title="Vendor Tags - Precision / Recall / F1"
              />
            </div>

            {/* Interpretation */}
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Interpretation</p>
              <p className="mt-2 leading-relaxed">
                <strong>Jaccard Index</strong> measures overall label set agreement (intersection / union).
                Higher is better. <strong>Precision</strong> shows how often the LLM is correct when it predicts a label.
                <strong> Recall</strong> shows how often the LLM catches labels that humans annotated.
                <strong> F1</strong> is the harmonic mean of precision and recall.
              </p>
              <p className="mt-3 leading-relaxed">
                <span className="font-medium text-emerald-600">Vendor tags (93%)</span> show strong agreement.
                <span className="font-medium text-amber-600"> Adoption types (57%)</span> have moderate agreement with high recall but low precision (LLM over-predicts).
                <span className="font-medium text-red-500"> Mention types (32%)</span> show the most disagreement, particularly on &ldquo;general/ambiguous&rdquo; classification.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
