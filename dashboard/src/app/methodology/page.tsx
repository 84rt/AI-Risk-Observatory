import { ClassificationFlowDiagram } from '@/components/classification-flow';

export default function MethodologyPage() {
  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-4xl font-semibold tracking-tight text-slate-900">
          Methodology
        </h1>
        <p className="mt-3 text-lg text-slate-600">
          How the AI Risk Observatory collects, processes, and classifies data.
        </p>

        <div className="mt-10 space-y-6 text-base leading-relaxed text-slate-700">
          <section>
            <h2 className="text-xl font-semibold text-slate-900">Data Collection</h2>
            <p className="mt-2">
              We source annual reports from UK Critical National Infrastructure (CNI) companies across sectors like energy, water, telecoms, and finance.
              Inputs come from PDF filings (FinancialReports database) and iXBRL/HTML filings (Companies House API), which are then normalized into structured text for downstream chunking and model analysis.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Classification &amp; Confidence</h2>
            <p className="mt-2">
              Chunks are classified by an LLM pipeline (for example GPT-4 or Gemini via the <code>run_llm_classifier</code> scripts) for AI mention presence, risk vs opportunity framing, and disclosure substantiveness.
              Each prediction carries a confidence score (0&ndash;1), and only labels with confidence &ge; 0.2 are included.
              <strong> Per Report</strong> aggregates all chunks from one company-year into a single row.
              <strong> Per Chunk</strong> treats every individual text passage as its own data point &mdash; giving you finer-grained detail but higher counts.
            </p>
            <div className="mt-5">
              <ClassificationFlowDiagram />
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Reading the Charts</h2>
            <p className="mt-2">
              Values in charts and heatmaps are <em>counts</em> &mdash; how many reports (or chunks) received a given label.
              A cell showing &ldquo;3&rdquo; means 3 reports in that sector mentioned that category. Darker cells = higher counts. Striped cells = zero reports (potential blind spots).
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Risk Categories</h2>
            <ul className="mt-2 space-y-2">
              <li><span className="font-medium text-slate-900">Cybersecurity:</span> Data breaches, AI-enabled attacks, security vulnerabilities.</li>
              <li><span className="font-medium text-slate-900">Operational / Technical:</span> System failures, integration issues, performance degradation.</li>
              <li><span className="font-medium text-slate-900">Regulatory / Compliance:</span> Compliance obligations, legal liability, regulatory uncertainty.</li>
              <li><span className="font-medium text-slate-900">Reputational / Ethical:</span> Brand damage, bias concerns, ethical considerations.</li>
              <li><span className="font-medium text-slate-900">Information Integrity:</span> Misinformation, hallucinations, data quality issues.</li>
              <li><span className="font-medium text-slate-900">Third-Party Supply Chain:</span> Vendor dependencies, API reliance, supplier risks.</li>
              <li><span className="font-medium text-slate-900">Strategic / Competitive:</span> Competitive displacement, market disruption, innovation pressure.</li>
              <li><span className="font-medium text-slate-900">Workforce Impacts:</span> Job displacement, skills gaps, labor relations.</li>
              <li><span className="font-medium text-slate-900">Environmental Impact:</span> Energy consumption, carbon footprint, resource usage.</li>
              <li><span className="font-medium text-slate-900">National Security:</span> Critical systems, geopolitical exposure, security-of-state concerns.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Quality Metrics</h2>
            <div className="mt-2 space-y-3">
              <div>
                <p className="font-medium text-slate-900">Signal Strength (Risk / Adoption / Vendor)</p>
                <p className="mt-1">
                  How explicitly each mention evidences its classification. Per label per chunk, the strongest signal from any source wins.
                </p>
                <ul className="mt-1 space-y-0.5 text-sm">
                  <li><span className="font-medium">3 Explicit:</span> direct, named, concrete statement</li>
                  <li><span className="font-medium">2 Strong implicit:</span> clear but inferential link</li>
                  <li><span className="font-medium">1 Weak implicit:</span> plausible but lightly supported</li>
                </ul>
              </div>
              <div>
                <p className="font-medium text-slate-900">Substantiveness</p>
                <p className="mt-1">Disclosure quality for AI-risk language per report.</p>
                <ul className="mt-1 space-y-0.5 text-sm">
                  <li><span className="font-medium">Substantive:</span> concrete mechanism + tangible mitigation/action</li>
                  <li><span className="font-medium">Moderate:</span> specific risk area, limited detail</li>
                  <li><span className="font-medium">Boilerplate:</span> generic risk language without concrete detail</li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
