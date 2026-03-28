import Link from 'next/link';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';
import HeroRiskChart from '@/components/hero-risk-chart';
import { ReportClassificationSankeyShell } from '@/components/report-classification-sankey-shell';

export default function HomePage() {
  const data = loadGoldenSetDashboardData();
  const perReportSummary = data.datasets.perReport.summary;
  const perChunkSummary = data.datasets.perChunk.summary;
  const yearRange =
    data.years.length > 1
      ? `${data.years[0]}–${data.years[data.years.length - 1]}`
      : `${data.years[0] ?? 'N/A'}`;

  const riskLabels = data.labels.riskLabels;

  const heroSeries = [
    {
      label: 'Risk mentions',
      subtitle: 'AI mentioned as a risk by public companies',
      color: '#f97316',
      data: data.datasets.perReport.riskTrend.map(row => ({
        year: Number(row.year),
        value: riskLabels.reduce((sum, key) => sum + (Number(row[key]) || 0), 0),
      })),
    },
    {
      label: 'Workforce impact',
      subtitle: 'Companies reporting workforce impact from AI',
      color: '#8b5cf6',
      data: data.datasets.perReport.riskTrend.map(row => ({
        year: Number(row.year),
        value: Number(row.workforce_impacts) || 0,
      })),
    },
    {
      label: 'LLM adoption',
      subtitle: 'Companies reporting LLM adoption',
      color: '#0ea5e9',
      data: data.datasets.perReport.adoptionTrend.map(row => ({
        year: Number(row.year),
        value: Number(row.llm) || 0,
      })),
    },
  ];

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      {/* Hero */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute -top-24 left-10 h-64 w-64 rounded-full bg-amber-200/70 blur-3xl" />
          <div className="absolute top-10 right-0 h-72 w-72 rounded-full bg-sky-200/70 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 h-48 w-48 rounded-full bg-emerald-200/60 blur-3xl" />
        </div>
        <div className="relative mx-auto max-w-5xl px-6 pt-20 pb-16">
          <div className="flex flex-col items-center gap-10 lg:flex-row lg:items-center lg:justify-between">
            {/* Left — text & stats */}
            <div className="text-center lg:text-left lg:max-w-lg">
              <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
                AI Risk Observatory
              </h1>
              <p className="mt-5 max-w-2xl text-lg text-slate-600 sm:text-xl">
                Tracking how UK Critical National Infrastructure companies disclose AI-related risks, adoption, and vendor dependencies in their{' '}
                <a
                  href="https://en.wikipedia.org/wiki/Annual_report"
                  className="underline decoration-slate-400 hover:text-slate-700"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  annual reports
                </a>.
              </p>

              {/* Stats */}
              <div className="mt-8 flex flex-wrap justify-center gap-3 text-xs uppercase tracking-[0.2em] text-slate-500 lg:justify-start">
                <span className="rounded-full border border-amber-200/60 bg-white/80 px-4 py-1.5 font-semibold shadow-sm">
                  <span className="text-amber-600">{perReportSummary.totalCompanies}</span> Companies
                </span>
                <span className="rounded-full border border-amber-200/60 bg-white/80 px-4 py-1.5 font-semibold shadow-sm">
                  <span className="text-amber-600">{perReportSummary.totalReports}</span> Annual Reports
                </span>
                <span className="rounded-full border border-amber-200/60 bg-white/80 px-4 py-1.5 font-semibold shadow-sm">
                  <span className="text-amber-600">{perChunkSummary.totalReports}</span> Extracted Chunks
                </span>
                <span className="rounded-full border border-amber-200/60 bg-white/80 px-4 py-1.5 font-semibold shadow-sm">
                  <span className="text-amber-600">{yearRange}</span>
                </span>
              </div>
            </div>

            {/* Right — chart */}
            <div className="flex-shrink-0">
              <HeroRiskChart series={heroSeries} />
            </div>
          </div>
        </div>
      </header>

      {/* Description */}
      <section className="mx-auto max-w-3xl px-6 py-12 text-center">
        <p className="text-base leading-relaxed text-slate-600">
          This project uses an NLP pipeline to analyse the annual reports of UK Critical National Infrastructure companies for AI-related disclosures. Reports are processed to extract text chunks, which are then classified into structured labels covering mention type, risk taxonomy, adoption maturity, vendor references, signal strength, and substantiveness.
        </p>
        <p className="mt-4 text-sm text-slate-500">
          The full dataset, processing pipeline, and documentation are open source.
        </p>
        <a
          href="https://github.com/84rt/AI-Risk-Observatory"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-4 inline-flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-slate-800 hover:shadow-md"
        >
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
          </svg>
          View on GitHub
        </a>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="mb-6 max-w-3xl">
          <h2 className="text-2xl font-semibold text-slate-900">How Reports Move Through The Pipeline</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 sm:text-base">
            This is a Sankey-style flow from the full 1,000-report corpus through extraction, Phase 1 classification,
            and the downstream Phase 2 labels.
          </p>
        </div>
        <ReportClassificationSankeyShell flow={data.reportClassificationFlow} />
      </section>

      {/* Navigation cards */}
      <section className="mx-auto max-w-3xl px-6 pb-20">
        <div className="grid gap-6 sm:grid-cols-2">
          <Link
            href="/data"
            className="group rounded-2xl border border-slate-200 bg-white/80 p-8 shadow-sm transition-all hover:border-amber-300 hover:shadow-md"
          >
            <h2 className="text-xl font-semibold text-slate-900 group-hover:text-slate-700">
              Data
            </h2>
            <p className="mt-2 text-sm text-slate-500">
              Interactive charts and heatmaps covering AI risk categories, adoption types, vendor references, signal quality, and disclosure blind spots across sectors and years.
            </p>
            <span className="mt-4 inline-block text-sm font-medium text-amber-600 group-hover:underline">
              Explore the data &rarr;
            </span>
          </Link>
          <Link
            href="/about"
            className="group rounded-2xl border border-slate-200 bg-white/80 p-8 shadow-sm transition-all hover:border-amber-300 hover:shadow-md"
          >
            <h2 className="text-xl font-semibold text-slate-900 group-hover:text-slate-700">
              About
            </h2>
            <p className="mt-2 text-sm text-slate-500">
              How the pipeline works — from keyword extraction and chunk classification to the taxonomies and quality controls behind the data.
            </p>
            <span className="mt-4 inline-block text-sm font-medium text-amber-600 group-hover:underline">
              Read the methodology &rarr;
            </span>
          </Link>
        </div>
      </section>
    </div>
  );
}
