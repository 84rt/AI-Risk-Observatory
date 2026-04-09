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
      color: '#f59e0b',
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
              Built for policymakers and resilience researchers, the Observatory tracks
              AI-related risk, adoption, and third-party AI exposure across{' '}
              <span className="text-primary">UK public-company</span>{' '}
              <a
                href="https://en.wikipedia.org/wiki/Annual_report"
                className="underline decoration-border underline-offset-4 hover:text-primary transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                annual reports
              </a>
              {' '}— with a focus on{' '}
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
                <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Years covered</div>
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

      {/* Primary chart */}
      <section className="border-b border-border bg-white">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <HeroRiskChart series={heroSeries} />
        </div>
      </section>

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

      {/* Description */}
      <section id="mission" className="border-b border-border bg-secondary">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <div className="max-w-3xl">
            <span className="aisi-tag">Mission</span>
            <p className="text-xl leading-relaxed text-muted">
              The Observatory gives policymakers and resilience researchers a clearer picture
              of how AI-related risk and adoption are distributed across the UK economy —
              and where meaningful disclosure is notably absent, particularly across{' '}
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
            <p className="mt-5 text-xl leading-relaxed text-muted">
              We use annual reports as our primary source because they are audited, legally
              mandated, and published on a consistent schedule — making them a reliable
              baseline for tracking real corporate AI exposure at scale.
            </p>
            <div className="mt-10">
              <Link
                href="/data"
                className="inline-flex items-center gap-2 rounded bg-primary px-6 py-3 text-sm font-bold uppercase tracking-widest text-white transition-colors hover:bg-muted"
              >
                View The Dashboard
                <span aria-hidden="true">&rarr;</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Secondary actions */}
      <section id="explore" className="border-b border-border bg-white">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <div className="grid gap-10 md:grid-cols-2">
            <div>
              <p className="text-sm font-bold uppercase tracking-widest text-primary">Methodology</p>
              <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted">
                Read how we collect, process, and classify the data — including the decisions and trade-offs behind the pipeline.
              </p>
              <div className="mt-4">
                <Link
                  href="/about"
                  className="inline-flex items-center gap-2 rounded border border-border bg-white px-5 py-2.5 text-sm font-bold uppercase tracking-widest text-primary transition-colors hover:bg-secondary"
                >
                  Read the Methodology
                </Link>
              </div>
            </div>
            <div>
              <p className="text-sm font-bold uppercase tracking-widest text-primary">Repository</p>
              <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted">
                Browse the source code, data pipeline, and full project structure on GitHub.
              </p>
              <div className="mt-4">
                <a
                  href="https://github.com/84rt/AI-Risk-Observatory"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded border border-border bg-white px-5 py-2.5 text-sm font-bold uppercase tracking-widest text-primary transition-colors hover:bg-secondary"
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 24 24"
                    className="h-4 w-4 fill-current"
                  >
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58 0-.28-.01-1.04-.02-2.04-3.34.73-4.04-1.61-4.04-1.61-.55-1.38-1.33-1.75-1.33-1.75-1.09-.74.08-.72.08-.72 1.2.09 1.84 1.24 1.84 1.24 1.08 1.84 2.82 1.31 3.5 1 .1-.78.42-1.31.76-1.62-2.67-.3-5.47-1.34-5.47-5.94 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.8 5.64-5.48 5.94.43.37.82 1.1.82 2.22 0 1.6-.01 2.89-.01 3.28 0 .32.21.69.83.57A12 12 0 0 0 24 12c0-6.63-5.37-12-12-12Z" />
                  </svg>
                  View Source on GitHub
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
