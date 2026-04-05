import Link from 'next/link';
import Image from 'next/image';
import HeroRiskChart from '@/components/hero-risk-chart';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

export default function HomePage() {
  const data = loadGoldenSetDashboardData();
  const perReportSummary = data.datasets.perReport.summary;
  const yearRange =
    data.years.length > 1
      ? `${data.years[0]}–${data.years[data.years.length - 1]}`
      : `${data.years[0] ?? 'N/A'}`;

  // Build a per-year totals map from the report universe (correct denominator for % calculations)
  const reportTotalsByYear = new Map(
    data.datasets.perReport.blindSpotTrend.map(row => [
      Number(row.year),
      Number(row.total_reports) || 0,
    ])
  );

  const heroSeries = [
    {
      label: 'AI risk mentions',
      subtitle: 'Share of UK public-company annual reports mentioning AI as a corporate risk',
      color: '#e63946', // AISI Signal Red
      data: data.datasets.perReport.blindSpotTrend.map(row => {
        const total = Number(row.total_reports) || 0;
        return {
          year: Number(row.year),
          value: total > 0 ? Math.round((Number(row.ai_risk_mention) / total) * 100) : 0,
        };
      }),
    },
    {
      label: 'LLM adoption mentions',
      subtitle: 'Share of UK public-company annual reports mentioning LLM adoption',
      color: '#0b0c0c',
      data: data.datasets.perReport.adoptionTrend.map(row => {
        const year = Number(row.year);
        const total = reportTotalsByYear.get(year) || 0;
        return {
          year,
          value: total > 0 ? Math.round((Number(row.llm) / total) * 100) : 0,
        };
      }),
    },
    {
      label: 'AI as a cybersecurity threat mentions',
      subtitle: 'Share of UK public-company annual reports mentioning AI as a cybersecurity threat to the business',
      color: '#0ea5e9',
      data: data.datasets.perReport.riskTrend.map(row => {
        const year = Number(row.year);
        const total = reportTotalsByYear.get(year) || 0;
        return {
          year,
          value: total > 0 ? Math.round((Number(row.cybersecurity) / total) * 100) : 0,
        };
      }),
    },
  ];

  return (
    <div className="min-h-screen bg-white text-primary">
      {/* Hero */}
      <header id="overview" className="relative overflow-hidden border-b border-border bg-white">
        <div aria-hidden="true" className="pointer-events-none absolute inset-0">
          <div className="absolute -top-24 left-10 h-96 w-96 rounded-full bg-red-200/60 blur-[100px]" />
        </div>
        <div className="relative mx-auto max-w-7xl px-6 py-20 lg:py-24">
          <div className="max-w-4xl">
            <h1 className="aisi-h1 leading-[0.9]">
              AI Risk <br />Observatory
            </h1>
            <p className="mt-8 text-xl font-medium leading-relaxed text-muted">
              Tracking AI-related risks, adoption, and vendor dependencies across{' '}
              <span className="text-primary">UK public-company</span>{' '}
              <a
                href="https://en.wikipedia.org/wiki/Annual_report"
                className="underline decoration-border underline-offset-4 hover:text-primary transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                annual reports
              </a>
              {' '}to help monitor{' '}
              <a
                href="https://www.npsa.gov.uk/about-npsa/critical-national-infrastructure"
                className="underline decoration-border underline-offset-4 hover:text-primary transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                Critical National Infrastructure
              </a>{' '}
              sectors.
            </p>

            <div
              id="coverage"
              className="mt-8 grid gap-x-8 gap-y-4 border-t border-border/80 pt-5 sm:grid-cols-2 xl:grid-cols-4"
            >
              <div>
                <div className="text-2xl font-bold leading-none text-primary">{perReportSummary.totalCompanies}</div>
                <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Companies</div>
              </div>
              <div>
                <div className="text-2xl font-bold leading-none text-primary">{perReportSummary.totalReports}</div>
                <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Reports</div>
              </div>
              <div>
                <div className="text-2xl font-bold leading-none text-primary">{yearRange}</div>
                <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Scope</div>
              </div>
            </div>

            <div className="mt-10 flex flex-wrap gap-4">
              <span className="inline-flex items-center gap-2 rounded border border-border bg-secondary px-6 py-3 text-sm font-bold uppercase tracking-widest text-muted-foreground cursor-not-allowed">
                Full Report Coming Soon
              </span>
              <Link
                href="/data"
                className="inline-flex items-center gap-2 rounded border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
              >
                Explore Data
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Sponsors & Partners Bar */}
      <section id="partners" className="border-b border-border bg-white py-8">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid gap-0 md:grid-cols-2">
            <div className="flex justify-center py-4 md:border-r md:border-border md:px-8">
              <div className="group flex items-center gap-4 opacity-80 transition-opacity hover:opacity-100">
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Main Sponsor</span>
                <a
                  href="https://www.aisi.gov.uk/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center transition-colors group-hover:text-accent"
                >
                  <svg
                    viewBox="0 0 840 180"
                    className="h-10 w-auto"
                    aria-label="AI Safety Institute"
                    role="img"
                  >
                    <g fill="none" fillRule="evenodd">
                      <text
                        x="0"
                        y="120"
                        fill="#e63946"
                        fontFamily="Arial, Helvetica, sans-serif"
                        fontSize="132"
                        fontWeight="700"
                        letterSpacing="-6"
                      >
                        AISI
                      </text>
                      <rect x="390" y="26" width="6" height="122" fill="#0b0c0c" />
                      <text
                        x="426"
                        y="78"
                        fill="#0b0c0c"
                        fontFamily="Arial, Helvetica, sans-serif"
                        fontSize="54"
                        fontWeight="400"
                        letterSpacing="1"
                      >
                        AI SECURITY
                      </text>
                      <text
                        x="426"
                        y="146"
                        fill="#0b0c0c"
                        fontFamily="Arial, Helvetica, sans-serif"
                        fontSize="54"
                        fontWeight="400"
                        letterSpacing="1"
                      >
                        INSTITUTE
                      </text>
                    </g>
                  </svg>
                </a>
              </div>
            </div>
            <div className="flex justify-center py-4 md:px-8">
              <div className="group flex items-center gap-4 opacity-80 transition-opacity hover:opacity-100">
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Data Provider</span>
                <a
                  href="https://financialreports.eu/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center transition-colors"
                >
                  <Image
                    src="/fr-logo.svg"
                    alt="Financial Reports"
                    width={210}
                    height={29}
                    className="h-6 w-auto"
                  />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Primary chart */}
      <section className="border-b border-border bg-white">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <HeroRiskChart series={heroSeries} />
        </div>
      </section>

      {/* Description */}
      <section id="mission" className="border-b border-border bg-secondary">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <div className="max-w-3xl">
            <span className="aisi-tag">Mission</span>
            <p className="text-xl leading-relaxed text-muted">
              AI Risk Observatory is an attempt to better understand patterns in the UK economy, especially across Critical National Infrastructure sectors, by applying an NLP pipeline to public-company annual reports. The goal is to strengthen societal resilience by identifying where AI-related risk, adoption, vendor dependence, and disclosure gaps are emerging across sectors. The main limitation is that this signal is necessarily retrospective: annual reports are shaped by legal, regulatory, and reporting incentives, so using them as an information source yields a limited but high-signal view of underlying risk and potential systemic problems.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <a
                href="https://github.com/84rt/AI-Risk-Observatory"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded bg-primary px-6 py-3 text-sm font-bold uppercase tracking-widest text-white hover:bg-muted transition-colors"
              >
                View on GitHub
              </a>
              <Link
                href="/about"
                className="inline-flex items-center gap-2 rounded border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
              >
                Methodology
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Open data CTA */}
      <section id="explore" className="border-b border-border bg-white">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <div className="mx-auto flex max-w-2xl flex-col items-center gap-4 text-center">
            <p className="text-sm text-muted">Browse the full dataset.</p>
            <Link
              href="/data"
              className="inline-flex items-center gap-2 rounded border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary transition-colors hover:bg-secondary"
            >
              Open Data Dashboard
              <span aria-hidden="true">&rarr;</span>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
