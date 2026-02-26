import { ClassificationFlowDiagram } from '@/components/classification-flow';
import { MentionTypesChart } from '@/components/mention-types-chart';
import { loadGoldenSetDashboardData } from '@/lib/golden-set';

const mentionTypeTaxonomy = [
  {
    label: 'adoption',
    definition: 'Current use, rollout, pilot, implementation, or delivery of AI systems by the company (or for clients).',
  },
  {
    label: 'risk',
    definition: 'AI is described as a downside or exposure for the company.',
  },
  {
    label: 'harm',
    definition: 'AI is described as causing or enabling harm (for example misinformation, fraud, abuse, safety incidents).',
  },
  {
    label: 'vendor',
    definition: 'A named AI model/vendor/platform provider is referenced.',
  },
  {
    label: 'general_ambiguous',
    definition: 'AI is mentioned, but the text is too high-level or vague for adoption/risk/harm/vendor.',
  },
  {
    label: 'none',
    definition: 'No real AI mention / false positive. This label is exclusive (it should not co-occur with others).',
  },
];

const adoptionTaxonomy = [
  {
    label: 'non_llm',
    definition: 'Traditional AI/ML (non-LLM), such as predictive models, computer vision, detection/classification systems.',
  },
  {
    label: 'llm',
    definition: 'Large language model / GenAI use (for example GPT/ChatGPT/Gemini/Claude/Copilot-style deployments).',
  },
  {
    label: 'agentic',
    definition: 'Autonomous or agent-based workflows with limited human intervention (can co-occur with llm).',
  },
];

const riskTaxonomy = [
  { label: 'strategic_competitive', definition: 'AI-driven competitive disadvantage, disruption, or failure to adapt.' },
  { label: 'operational_technical', definition: 'Reliability/accuracy/model-risk failures that degrade operations or decisions.' },
  { label: 'cybersecurity', definition: 'AI-enabled attacks/fraud/breach pathways or adversarial AI abuse.' },
  { label: 'workforce_impacts', definition: 'AI-related displacement, skills gaps, or risky employee AI usage.' },
  { label: 'regulatory_compliance', definition: 'AI-specific legal/regulatory/privacy/IP liability and compliance burden.' },
  { label: 'information_integrity', definition: 'AI-enabled misinformation, deepfakes, or authenticity manipulation.' },
  { label: 'reputational_ethical', definition: 'AI-linked trust, fairness, ethics, or rights concerns.' },
  { label: 'third_party_supply_chain', definition: 'Dependency on external AI vendors/providers and concentration exposure.' },
  { label: 'environmental_impact', definition: 'AI-related energy, carbon, or resource-burden risk.' },
  { label: 'national_security', definition: 'AI-linked geopolitical/security destabilization or critical-systems exposure.' },
  { label: 'none', definition: 'No attributable AI-risk category (or too vague to assign one).' },
];

const vendorTaxonomy = [
  { label: 'amazon', definition: 'Amazon / AWS / Bedrock / Titan / related Amazon AI model platforms.' },
  { label: 'google', definition: 'Google / Vertex AI / Gemini / DeepMind / related Google AI model platforms.' },
  { label: 'microsoft', definition: 'Microsoft / Azure AI / Copilot / Azure OpenAI Service.' },
  { label: 'openai', definition: 'OpenAI / GPT / ChatGPT references.' },
  { label: 'anthropic', definition: 'Anthropic / Claude references.' },
  { label: 'meta', definition: 'Meta AI / Llama references.' },
  { label: 'internal', definition: 'Explicitly in-house or proprietary model development/deployment.' },
  { label: 'undisclosed', definition: 'Third-party AI provider is implied but not named.' },
  { label: 'other', definition: 'Named provider outside the predefined list (with free-text vendor name in metadata).' },
];

const substantivenessLevels = [
  {
    label: 'boilerplate',
    definition: 'Generic AI language with low information density; could appear in many reports unchanged.',
  },
  {
    label: 'moderate',
    definition: 'Specific area is identified, but with limited mechanism, metrics, or mitigation detail.',
  },
  {
    label: 'substantive',
    definition: 'Concrete mechanism and/or tangible action, commitment, metric, system detail, or timeline.',
  },
];

export default function AboutPage() {
  const data = loadGoldenSetDashboardData();
  const mentionTrend = data.datasets.perReport.mentionTrend;
  const mentionTypes = data.labels.mentionTypes;
  const perReportSummary = data.datasets.perReport.summary;
  const perChunkSummary = data.datasets.perChunk.summary;
  const yearRange =
    data.years.length > 1
      ? `${data.years[0]}-${data.years[data.years.length - 1]}`
      : `${data.years[0] ?? 'N/A'}`;

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-4xl font-semibold tracking-tight text-slate-900">
          Methodology
        </h1>
        <p className="mt-3 text-lg text-slate-600">
          This page explains, in plain language, how we turn annual-report text into the dashboard metrics.
        </p>
        <p className="mt-2 max-w-5xl text-base leading-relaxed text-slate-700">
          This dashboard currently uses {perReportSummary.totalReports} company-year reports ({yearRange}), covering{' '}
          {perReportSummary.totalCompanies} companies and {data.sectors.length} sectors. The pipeline extracted{' '}
          {perChunkSummary.totalReports} AI-related text chunks from {perChunkSummary.totalCompanies} companies, then
          classified those chunks into structured labels.
        </p>
        <p className="mt-2 max-w-5xl text-base leading-relaxed text-slate-700">
          For full implementation details (exact prompts, scripts, schemas, and run artifacts), see the repository:{' '}
          <a
            href="https://github.com/84rt/AI-Risk-Observatory"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-slate-800 underline decoration-slate-400 underline-offset-2 hover:text-slate-900"
          >
            github.com/84rt/AI-Risk-Observatory
          </a>
          .
        </p>

        <div className="mt-10 space-y-8 text-base leading-relaxed text-slate-700">
          <section>
            <h2 className="text-xl font-semibold text-slate-900">Method Summary</h2>
            <p className="mt-2">
              The method is intentionally staged: find potential AI text first, then classify what that text is about,
              then aggregate labels to report-level trends.
            </p>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-xl border border-slate-200 bg-white/80 p-4">
                <p className="text-sm text-slate-500">Company-Year Reports</p>
                <p className="text-2xl font-semibold text-slate-900">{perReportSummary.totalReports}</p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white/80 p-4">
                <p className="text-sm text-slate-500">Extracted Chunks</p>
                <p className="text-2xl font-semibold text-slate-900">{perChunkSummary.totalReports}</p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white/80 p-4">
                <p className="text-sm text-slate-500">Reports With AI Signal</p>
                <p className="text-2xl font-semibold text-slate-900">{perReportSummary.aiSignalReports}</p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white/80 p-4">
                <p className="text-sm text-slate-500">Reports With AI Risk Signal</p>
                <p className="text-2xl font-semibold text-slate-900">{perReportSummary.riskReports}</p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Methodology in a Nutshell</h2>
            <p className="mt-2">
              The pipeline has three stages: pre-processing, processing, and post-processing. The diagram below shows
              the end-to-end flow.
            </p>
            <div className="mt-5">
              <ClassificationFlowDiagram />
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">1. Pre-processing</h2>
            <p className="mt-2">
              We collect annual reports, convert them to normalized markdown text, then detect AI keyword hits and build
              chunk windows around those hits.
            </p>
            <div className="mt-3 space-y-2">
              <p>
                <span className="font-medium text-slate-900">1.1 Candidate retrieval is recall-first:</span> keyword
                matching is intentionally broad (for example AI, artificial intelligence, machine learning, LLM, GPT,
                GenAI, Copilot). This catches more candidates, including false positives.
              </p>
              <p>
                <span className="font-medium text-slate-900">1.2 Context windows are merged:</span> overlapping hits
                are deduplicated into a single chunk so nearby mentions are analyzed together.
              </p>
              <p>
                <span className="font-medium text-slate-900">1.3 Long/noisy blocks are cleaned:</span> very long table
                rows and formatting noise are reduced so classifiers see readable text.
              </p>
              <p>
                <span className="font-medium text-slate-900">1.4 Traceability is preserved:</span> each chunk keeps
                source report identifiers, section hints, offsets, and matched keywords.
              </p>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">2. Processing</h2>
            <p className="mt-2">
              Processing is two-phase. Phase 1 decides what type of AI mention is in a chunk. Phase 2 adds deeper labels.
            </p>
            <div className="mt-3 space-y-2">
              <p>
                <span className="font-medium text-slate-900">2.1 Mention type labels:</span> adoption, risk, harm,
                vendor, general_ambiguous, none. These are non-mutually-exclusive except that <code>none</code> stands
                alone and means no real AI mention / false positive.
              </p>
              <p>
                This chart is shown here because mention type is the Phase 1 gate: it determines how chunks are routed
                to downstream classifiers, so changes here flow through the rest of the pipeline.
              </p>
              <div className="mt-2">
                <MentionTypesChart data={mentionTrend} stackKeys={mentionTypes} />
              </div>
              <p>
                <span className="font-medium text-slate-900">2.2 Routing logic:</span> the risk classifier only runs on
                chunks tagged risk, adoption classifier only on adoption chunks, and vendor classifier only on vendor
                chunks. If Phase 1 misses a tag, Phase 2 for that branch will not run.
              </p>
              <p>
                <span className="font-medium text-slate-900">2.3 Taxonomies used:</span> adoption type = non_llm, llm,
                agentic. Risk taxonomy = strategic_competitive, operational_technical, cybersecurity, workforce_impacts,
                regulatory_compliance, information_integrity, reputational_ethical, third_party_supply_chain,
                environmental_impact, national_security, none. Vendor tags include amazon, google, microsoft, openai,
                anthropic, meta, internal, undisclosed, other.
              </p>
              <p>
                <span className="font-medium text-slate-900">2.4 Signal and substantiveness:</span> adoption uses 0-3
                signals, while risk and vendor use 1-3 signals (weak implicit to explicit). Risk also gets a
                substantiveness label (boilerplate/moderate/substantive).
              </p>
            </div>
            <div className="mt-6 space-y-5">
              <h3 className="text-lg font-semibold text-slate-900">Exact Taxonomy Reference (Canonical Labels)</h3>
              <p className="text-sm text-slate-600">
                Labels are shown exactly as stored in classifier outputs and dataset fields for transparency.
              </p>

              <div>
                <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-600">Mention Type Taxonomy</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-slate-700">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Label</th>
                        <th className="px-3 py-2 font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mentionTypeTaxonomy.map((row) => (
                        <tr key={row.label} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2 font-mono text-xs text-slate-900">{row.label}</td>
                          <td className="px-3 py-2 text-slate-700">{row.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-600">Adoption Taxonomy</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-slate-700">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Label</th>
                        <th className="px-3 py-2 font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adoptionTaxonomy.map((row) => (
                        <tr key={row.label} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2 font-mono text-xs text-slate-900">{row.label}</td>
                          <td className="px-3 py-2 text-slate-700">{row.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Adoption signal scale: <code>0</code> absent, <code>1</code> weak implicit, <code>2</code> strong
                  implicit, <code>3</code> explicit.
                </p>
              </div>

              <div>
                <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-600">Risk Taxonomy</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-slate-700">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Label</th>
                        <th className="px-3 py-2 font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {riskTaxonomy.map((row) => (
                        <tr key={row.label} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2 font-mono text-xs text-slate-900">{row.label}</td>
                          <td className="px-3 py-2 text-slate-700">{row.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Risk signal scale: <code>1</code> weak implicit, <code>2</code> strong implicit, <code>3</code>{' '}
                  explicit.
                </p>
              </div>

              <div>
                <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-600">Vendor Taxonomy</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-slate-700">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Label</th>
                        <th className="px-3 py-2 font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {vendorTaxonomy.map((row) => (
                        <tr key={row.label} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2 font-mono text-xs text-slate-900">{row.label}</td>
                          <td className="px-3 py-2 text-slate-700">{row.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Vendor signal scale: <code>1</code> weak implicit, <code>2</code> strong implicit, <code>3</code>{' '}
                  explicit.
                </p>
              </div>

              <div>
                <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-600">Substantiveness Levels</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white/90">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-slate-700">
                      <tr>
                        <th className="px-3 py-2 font-semibold">Label</th>
                        <th className="px-3 py-2 font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {substantivenessLevels.map((row) => (
                        <tr key={row.label} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2 font-mono text-xs text-slate-900">{row.label}</td>
                          <td className="px-3 py-2 text-slate-700">{row.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">3. Post-processing</h2>
            <p className="mt-2">
              Chunk outputs are normalized and aggregated into both per-chunk and per-report views.
            </p>
            <div className="mt-3 space-y-2">
              <p>
                <span className="font-medium text-slate-900">3.1 Confidence handling:</span> report-level adoption/risk
                trend counts use confidence thresholds (default 0.2) where confidence maps exist; explicit risk signal
                entries are retained. Signal heatmaps bin values into weak/strong/explicit.
              </p>
              <p>
                <span className="font-medium text-slate-900">3.2 Legacy compatibility:</span> older risk labels
                (for example <code>regulatory</code>, <code>workforce</code>) are mapped to current canonical labels so
                longitudinal charts stay comparable.
              </p>
              <p>
                <span className="font-medium text-slate-900">3.3 Report denominator is explicit:</span> we keep
                no-signal reports in the report-level dataset to show blind spots, not just positive cases.
              </p>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-slate-900">Quality Controls</h2>
            <p className="mt-2">
              We use schema-constrained outputs, deterministic settings, and explicit validation/reconciliation tools to
              reduce noise and improve reproducibility.
            </p>
            <div className="mt-3 space-y-2">
              <p>
                <span className="font-medium text-slate-900">Structured outputs:</span> classifiers write to strict
                response schemas (Pydantic + JSON schema), reducing malformed labels.
              </p>
              <p>
                <span className="font-medium text-slate-900">Conservative prompting:</span> prompts require explicit AI
                attribution and discourage category over-assignment.
              </p>
              <p>
                <span className="font-medium text-slate-900">Testing and reconciliation:</span> repo scripts support QA
                checks, human-vs-LLM disagreement review, and merge-back of reconciled labels.
              </p>
              <p>
                <span className="font-medium text-slate-900">Known limitations:</span> this release is still primarily
                LLM-labeled and keyword-seeded, so it can miss subtle non-keyword AI references and can still include some
                ambiguous cases.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
