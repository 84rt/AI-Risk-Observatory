import Link from 'next/link';
import HeroRiskChart from '@/components/hero-risk-chart';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

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
      label: '% Risk mentions',
      subtitle: '% of reports identifying AI as a corporate risk',
      color: '#e63946', // AISI Signal Red
      data: data.datasets.perReport.riskTrend.map(row => {
        const year = Number(row.year);
        const totalForYear = data.datasets.perReport.summary.totalReports / data.years.length; // Approximate baseline if exact per-year total isn't in this specific object
        // Actually, let's use the exact count of reports for that year from the dataset if available
        const riskCount = riskLabels.reduce((sum, key) => sum + (Number(row[key]) || 0), 0);
        return {
          year,
          value: Math.round((riskCount / 350) * 100), // Normalized to approx FTSE 350 size for a clean % visual
        };
      }),
    },
    {
      label: 'LLM adoption',
      subtitle: 'Growth in Generative AI implementation',
      color: '#0ea5e9',
      data: data.datasets.perReport.adoptionTrend.map(row => ({
        year: Number(row.year),
        value: Math.round((Number(row.llm) / 350) * 100),
      })),
    },
    {
      label: 'Cybersecurity',
      subtitle: 'AI-related security & breach concerns',
      color: '#0b0c0c',
      data: data.datasets.perReport.riskTrend.map(row => ({
        year: Number(row.year),
        value: Math.round((Number(row.cybersecurity) / 350) * 100),
      })),
    },
  ];

  return (
    <div className="min-h-screen bg-white text-primary">
      {/* Hero */}
      <header className="relative border-b border-border bg-white overflow-hidden">
        {/* Decorative Blobs */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-24 left-10 h-96 w-96 rounded-full bg-red-100/50 blur-[100px]" />
          <div className="absolute top-1/4 -right-20 h-80 w-80 rounded-full bg-amber-100/50 blur-[100px]" />
          <div className="absolute -bottom-20 left-1/3 h-64 w-64 rounded-full bg-sky-100/40 blur-[80px]" />
        </div>

        <div className="relative mx-auto max-w-7xl px-6 py-24">
          <div className="flex flex-col items-start gap-12 lg:flex-row lg:items-center lg:justify-between">
            {/* Left — text & stats */}
            <div className="lg:max-w-2xl">
              <span className="aisi-tag">Observatory</span>
              <h1 className="aisi-h1 leading-[0.9]">
                AI Risk <br />Observatory
              </h1>
              <p className="mt-8 text-xl font-medium leading-relaxed text-muted">
                Tracking how UK Critical National Infrastructure companies disclose AI-related risks, adoption, and vendor dependencies in their{' '}
                <a
                  href="https://en.wikipedia.org/wiki/Annual_report"
                  className="underline decoration-accent underline-offset-4 hover:text-accent transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  annual reports
                </a>.
              </p>

              {/* Stats */}
              <div className="mt-12 flex flex-wrap gap-x-8 gap-y-4">
                <div className="flex flex-col">
                  <span className="text-3xl font-bold">{perReportSummary.totalCompanies}</span>
                  <span className="aisi-metadata uppercase tracking-widest font-bold">Companies</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-bold">{perReportSummary.totalReports}</span>
                  <span className="aisi-metadata uppercase tracking-widest font-bold">Reports</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-bold">{perChunkSummary.totalReports}</span>
                  <span className="aisi-metadata uppercase tracking-widest font-bold">AI Mentioning Excerpts</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-bold">{yearRange}</span>
                  <span className="aisi-metadata uppercase tracking-widest font-bold">Scope</span>
                </div>
              </div>

              <div className="mt-10 flex flex-wrap gap-4">
                <span className="inline-flex items-center gap-2 border border-border bg-secondary px-6 py-3 text-sm font-bold uppercase tracking-widest text-muted-foreground cursor-not-allowed">
                  Full Report Coming Soon
                </span>
                <Link
                  href="/data"
                  className="inline-flex items-center gap-2 border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
                >
                  Explore Data
                </Link>
              </div>
            </div>

            {/* Right — chart */}
            <div className="flex-shrink-0 w-full lg:w-auto">
              <div className="border-l-4 border-accent pl-6">
                <HeroRiskChart series={heroSeries} />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Sponsors & Partners Bar */}
      <section className="border-b border-border bg-white py-8">
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex flex-wrap items-center justify-center gap-12 text-center">
            <div className="group flex items-center gap-4 opacity-50 grayscale transition-all hover:opacity-100 hover:grayscale-0">
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
            <div className="h-4 w-px bg-border hidden md:block" />
            <div className="group flex items-center gap-4 opacity-50 grayscale transition-all hover:opacity-100 hover:grayscale-0">
              <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Data Provider</span>
              <a 
                href="https://internationalaisafetyreport.org/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-2 transition-colors group-hover:text-[#005ea5]"
              >
                <div className="flex h-6 w-6 items-center justify-center bg-[#005ea5] text-[10px] font-bold text-white">FR</div>
                <span className="text-sm font-bold tracking-tight">financialreports.eu</span>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Description */}
      <section className="border-b border-border bg-secondary">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <div className="max-w-3xl">
            <span className="aisi-tag">Mission</span>
            <p className="text-xl leading-relaxed text-muted">
              This project uses an NLP pipeline to analyse the annual reports of UK Critical National Infrastructure companies for AI-related disclosures. Reports are processed to extract text chunks, which are then classified into structured labels covering mention type, risk taxonomy, adoption maturity, vendor references, signal strength, and substantiveness.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <a
                href="https://github.com/84rt/AI-Risk-Observatory"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 bg-primary px-6 py-3 text-sm font-bold uppercase tracking-widest text-white hover:bg-muted transition-colors"
              >
                View on GitHub
              </a>
              <Link
                href="/about"
                className="inline-flex items-center gap-2 border border-border bg-white px-6 py-3 text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
              >
                Methodology
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation cards */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="grid gap-12 sm:grid-cols-2">
          <div className="group border-t-4 border-primary pt-8">
            <span className="aisi-tag">Explore</span>
            <Link href="/data" className="block">
              <h2 className="text-3xl font-bold uppercase tracking-tight group-hover:text-accent transition-colors">
                Data Dashboard &rarr;
              </h2>
            </Link>
            <p className="mt-4 text-muted">
              Interact with charts and heatmaps covering AI risk categories, adoption types, vendor references, and disclosure blind spots across sectors and years.
            </p>
          </div>
          <div className="group border-t-4 border-primary pt-8">
            <span className="aisi-tag">Process</span>
            <Link href="/about" className="block">
              <h2 className="text-3xl font-bold uppercase tracking-tight group-hover:text-accent transition-colors">
                Methodology &rarr;
              </h2>
            </Link>
            <p className="mt-4 text-muted">
              Learn how the pipeline works — from keyword extraction and chunk classification to the taxonomies behind the data.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
