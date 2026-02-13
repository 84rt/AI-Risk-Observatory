'use client';

import { useMemo, useState } from 'react';
import { GenericHeatmap, StackedBarChart } from '@/components/overview-charts';
import type { GoldenDashboardData } from '@/lib/golden-set';

// Filter type for risk distribution view
type RiskFilter = 'all' | string;

type View = {
  id: number;
  title: string;
  description: string;
};

const VIEWS: View[] = [
  {
    id: 1,
    title: 'Risk Distribution',
    description: 'Which AI risk categories appear most often in each CNI sector — and where the blind spots are.',
  },
  {
    id: 2,
    title: 'Mention Types',
    description: 'How each AI-related text chunk was classified: adoption, risk, vendor reference, general/ambiguous, or harm.',
  },
  {
    id: 3,
    title: 'Adoption & Vendors',
    description: 'AI adoption maturity (non-LLM, LLM, agentic) and which technology vendors companies name in their reports.',
  },
  {
    id: 4,
    title: 'Quality Signals',
    description: 'How explicit and substantive each disclosure is — from concrete detail to boilerplate language.',
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
  regulatory_compliance: '#f59e0b',
  reputational_ethical: '#14b8a6',
  information_integrity: '#0ea5e9',
  third_party_supply_chain: '#22c55e',
  strategic_competitive: '#84cc16',
  workforce_impacts: '#0f766e',
  environmental_impact: '#10b981',
  national_security: '#7c3aed',
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
    regulatory_compliance: 'Regulatory / Compliance',
    reputational_ethical: 'Reputational / Ethical',
    information_integrity: 'Information Integrity',
    workforce_impacts: 'Workforce Impacts',
    environmental_impact: 'Environmental Impact',
    national_security: 'National Security',
    '3-explicit': '3: Explicit',
    '2-strong_implicit': '2: Strong Implicit',
    '1-weak_implicit': '1: Weak Implicit',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

type DatasetKey = 'perReport' | 'perChunk';

const datasetLabels: Record<DatasetKey, string> = {
  perReport: 'Per Report',
  perChunk: 'Per Chunk',
};

export default function DashboardClient({ data }: { data: GoldenDashboardData }) {
  const [activeView, setActiveView] = useState(1);
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('perReport');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const activeData = data.datasets[datasetKey];

  const mentionStackKeys = useMemo(() => data.labels.mentionTypes, [data.labels.mentionTypes]);
  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const vendorStackKeys = useMemo(() => data.labels.vendorTags, [data.labels.vendorTags]);
  const riskStackKeys = useMemo(() => data.labels.riskLabels, [data.labels.riskLabels]);

  // Filter risk data by selected risk type
  const filteredRiskBySector = useMemo(() => {
    if (riskFilter === 'all') return activeData.riskBySector;
    return activeData.riskBySector.filter(d => d.x === riskFilter);
  }, [activeData.riskBySector, riskFilter]);

  // Filter risk trend for single risk type view
  const filteredRiskTrend = useMemo(() => {
    if (riskFilter === 'all') return activeData.riskTrend;
    return activeData.riskTrend.map(row => ({
      year: row.year,
      [riskFilter]: row[riskFilter] || 0,
    }));
  }, [activeData.riskTrend, riskFilter]);

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      {/* Hero section */}
      <header className="relative overflow-hidden border-b border-slate-200/70">
        <div className="absolute inset-0">
          <div className="absolute -top-24 left-10 h-64 w-64 rounded-full bg-amber-200/70 blur-3xl" />
          <div className="absolute top-10 right-0 h-72 w-72 rounded-full bg-sky-200/70 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 h-48 w-48 rounded-full bg-emerald-200/60 blur-3xl" />
        </div>
        <div className="relative mx-auto max-w-7xl px-6 py-12">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="animate-rise">
              <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
                AI Risk Observatory
              </h1>
              <p className="mt-3 max-w-2xl text-base text-slate-600 sm:text-lg">
                Tracking how UK Critical National Infrastructure companies disclose AI-related risks, adoption, and vendor dependencies in their annual reports.
              </p>
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {activeData.summary.totalCompanies} Companies
                </span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {activeData.summary.totalReports} Reports
                </span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {data.years[0]}–{data.years[data.years.length - 1]}
                </span>
              </div>
            </div>
            <div className="flex flex-wrap items-end gap-3">
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
            </div>
          </div>
        </div>
      </header>

      {/* Sticky control bar — only interactive controls */}
      <div className="sticky top-0 z-20 border-b border-slate-200 bg-[#f6f3ef]/70 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
          {/* View tabs */}
          <div className="flex items-center gap-1">
            {VIEWS.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-all ${
                  activeView === item.id
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-slate-500 hover:bg-white hover:text-slate-900'
                }`}
              >
                {item.title}
              </button>
            ))}
          </div>

          {/* Dataset toggle */}
          <select
            value={datasetKey}
            onChange={event => setDatasetKey(event.target.value as DatasetKey)}
            className="rounded-lg border border-slate-200 bg-white/90 px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm"
          >
            <option value="perReport">Per Report</option>
            <option value="perChunk">Per Chunk</option>
          </select>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-6 py-12">
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-slate-900">{view.title}</h2>
          <p className="mt-2 max-w-3xl text-slate-600">{view.description}</p>
        </div>

        {activeView === 1 && (
          <div className="space-y-8">
            {/* Risk Type Filter */}
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex flex-col gap-2">
                <label htmlFor="risk-filter" className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  Focus on Risk Type
                </label>
                <select
                  id="risk-filter"
                  value={riskFilter}
                  onChange={e => setRiskFilter(e.target.value)}
                  className="rounded-xl border border-slate-200 bg-white/90 px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm"
                >
                  <option value="all">All Risk Types</option>
                  {data.labels.riskLabels.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
              </div>
              {riskFilter !== 'all' && (
                <button
                  onClick={() => setRiskFilter('all')}
                  className="mt-6 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-600 hover:bg-slate-50"
                >
                  Clear Filter
                </button>
              )}
            </div>

            {/* Risk Trend Over Time */}
            <StackedBarChart
              data={filteredRiskTrend}
              xAxisKey="year"
              stackKeys={riskFilter === 'all' ? riskStackKeys : [riskFilter]}
              colors={riskColors}
              allowLineChart
              title="Risk Trend Over Time"
              subtitle={`Each bar shows the total ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} per year mentioning a given risk category. A single ${datasetKey === 'perReport' ? 'report' : 'chunk'} can appear in multiple categories.`}
            />

            {/* Risk by Sector Heatmap */}
            <GenericHeatmap
              data={filteredRiskBySector}
              xLabels={riskFilter === 'all' ? data.labels.riskLabels : [riskFilter]}
              yLabels={data.sectors}
              baseColor="#f97316"
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'chunks'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title="Risk Distribution by Sector"
              subtitle={`Each cell shows the count of ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} in that sector mentioning that risk category. Darker = more mentions. Striped = zero mentions (potential blind spots).`}
            />

            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Risk Category Definitions</p>
              <ul className="mt-2 space-y-1 leading-relaxed text-xs">
                <li><span className="font-medium text-slate-800">Cybersecurity:</span> Data breaches, AI-enabled attacks, security vulnerabilities.</li>
                <li><span className="font-medium text-slate-800">Operational/Technical:</span> System failures, integration issues, performance degradation.</li>
                <li><span className="font-medium text-slate-800">Regulatory/Compliance:</span> Compliance obligations, legal liability, regulatory uncertainty.</li>
                <li><span className="font-medium text-slate-800">Reputational/Ethical:</span> Brand damage, bias concerns, ethical considerations.</li>
                <li><span className="font-medium text-slate-800">Information Integrity:</span> Misinformation, hallucinations, data quality issues.</li>
                <li><span className="font-medium text-slate-800">Third-Party Supply Chain:</span> Vendor dependencies, API reliance, supplier risks.</li>
                <li><span className="font-medium text-slate-800">Strategic/Competitive:</span> Competitive displacement, market disruption, innovation pressure.</li>
                <li><span className="font-medium text-slate-800">Workforce Impacts:</span> Job displacement, skills gaps, labor relations.</li>
                <li><span className="font-medium text-slate-800">Environmental Impact:</span> Energy consumption, carbon footprint, resource usage.</li>
                <li><span className="font-medium text-slate-800">National Security:</span> Critical systems, geopolitical exposure, security-of-state concerns.</li>
              </ul>
            </div>
          </div>
        )}

        {activeView === 2 && (
          <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
            <div className="space-y-4">
              <StackedBarChart
                data={activeData.mentionTrend}
                xAxisKey="year"
                stackKeys={mentionStackKeys}
                colors={mentionColors}
                allowLineChart
                title="Mention Types Over Time"
                subtitle={`Each bar shows how many ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} per year were tagged with each mention type (confidence ≥ 0.2).`}
              />
            </div>
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
                <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Signal Coverage
                </h3>
                <p className="mt-3 text-3xl font-semibold text-slate-900">
                  {formatNumber(activeData.summary.adoptionReports + activeData.summary.riskReports)}
                </p>
                <p className="mt-2 text-sm text-slate-600">
                  {datasetKey === 'perReport' ? 'Reports' : 'Chunks'} with at least one adoption or risk signal.
                </p>
                <div className="mt-6 space-y-3 text-sm text-slate-600">
                  <div className="flex items-center justify-between">
                    <span>With adoption signals</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.adoptionReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>With risk signals</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.riskReports)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>With vendor references</span>
                    <span className="font-semibold text-slate-900">
                      {formatNumber(activeData.summary.vendorReports)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
                <p className="font-semibold text-slate-900">What are mention types?</p>
                <p className="mt-2 leading-relaxed">Every AI-related text passage is classified into one or more of these categories based on what the company is talking about:</p>
                <ul className="mt-2 space-y-1 leading-relaxed">
                  <li><span className="font-medium text-slate-800">Adoption:</span> The company describes using, deploying, or implementing AI technology.</li>
                  <li><span className="font-medium text-slate-800">Risk:</span> The company discusses threats, concerns, or negative outcomes tied to AI.</li>
                  <li><span className="font-medium text-slate-800">Vendor:</span> A specific AI vendor or provider is named (e.g. Microsoft, OpenAI).</li>
                  <li><span className="font-medium text-slate-800">General/Ambiguous:</span> AI is mentioned but without a clear adoption or risk context.</li>
                  <li><span className="font-medium text-slate-800">Harm:</span> An actual or potential AI-related harm incident is described.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            {/* Adoption Trends */}
            <StackedBarChart
              data={activeData.adoptionTrend}
              xAxisKey="year"
              stackKeys={adoptionStackKeys}
              colors={adoptionColors}
              allowLineChart
              title="Adoption Maturity Over Time"
              subtitle={`Count of ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} per year by maturity level: Non-LLM (traditional AI/ML), LLM (large language models), Agentic (autonomous AI systems).`}
            />

            {/* Adoption by Sector Heatmap */}
            <GenericHeatmap
              data={activeData.adoptionBySector}
              xLabels={data.labels.adoptionTypes}
              yLabels={data.sectors}
              baseColor="#0ea5e9"
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'chunks'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title="Adoption Intensity by Sector"
              subtitle={`Count of ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} per sector at each adoption maturity level. Striped = zero mentions.`}
            />

            {/* Vendor Trends */}
            <StackedBarChart
              data={activeData.vendorTrend}
              xAxisKey="year"
              stackKeys={vendorStackKeys}
              colors={vendorColors}
              allowLineChart
              title="Vendor References Over Time"
              subtitle={`Count of ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} naming each AI vendor per year. "Internal" = built in-house. "Undisclosed" = AI mentioned but no vendor named.`}
            />

            {/* Vendor by Sector Heatmap */}
            <GenericHeatmap
              data={activeData.vendorBySector}
              xLabels={data.labels.vendorTags}
              yLabels={data.sectors}
              baseColor="#14b8a6"
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'chunks'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title="Vendor Concentration by Sector"
              subtitle={`Count of ${datasetKey === 'perReport' ? 'reports' : 'text chunks'} naming each vendor per sector. Striped = no mentions.`}
            />
          </div>
        )}

        {activeView === 4 && (
          <div className="space-y-8">
            <div className="grid gap-8 lg:grid-cols-3">
              <GenericHeatmap
                data={activeData.riskSignalHeatmap}
                xLabels={data.years}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#f97316"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Risk Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                compact={true}
              />
              <GenericHeatmap
                data={activeData.adoptionSignalHeatmap}
                xLabels={data.years}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#0ea5e9"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Adoption Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                compact={true}
              />
              <GenericHeatmap
                data={activeData.vendorSignalHeatmap}
                xLabels={data.years}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#14b8a6"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Vendor Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                compact={true}
              />
            </div>
            <GenericHeatmap
              data={activeData.substantivenessHeatmap}
              xLabels={data.years}
              yLabels={data.labels.substantivenessBands}
              baseColor="#14b8a6"
              valueFormatter={value => `${value}`}
              yLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={false}
              title="Risk Substantiveness Distribution"
              subtitle="Disclosure quality for AI-risk language. Substantive = concrete mechanism + tangible mitigation. Moderate = specific risk area, limited detail. Boilerplate = generic language without concrete detail."
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Quality Metric Definitions</p>
              <div className="mt-2 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="font-medium text-slate-800">Signal Strength (Risk / Adoption / Vendor):</p>
                  <p className="mt-1 leading-relaxed">
                    How explicitly each mention evidences its classification.
                    Each cell counts individual label mentions across all chunks.
                    Per label per chunk, the strongest signal from any source wins.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">3 Explicit:</span> direct, named, concrete statement</li>
                    <li><span className="font-medium">2 Strong implicit:</span> clear but inferential link</li>
                    <li><span className="font-medium">1 Weak implicit:</span> plausible but lightly supported</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-slate-800">Substantiveness:</p>
                  <p className="mt-1 leading-relaxed">
                    Disclosure quality for AI-risk language per report.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">Substantive:</span> concrete mechanism + tangible mitigation/action</li>
                    <li><span className="font-medium">Moderate:</span> specific risk area, limited detail</li>
                    <li><span className="font-medium">Boilerplate:</span> generic risk language without concrete detail</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
