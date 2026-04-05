import { ClassificationFlowDiagram } from '@/components/classification-flow';
import { MentionTypesChart } from '@/components/mention-types-chart';
import { ReportClassificationSankeyShell } from '@/components/report-classification-sankey-shell';
import ExampleBrowser from '@/components/example-browser';
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

  return (
    <div className="min-h-screen bg-white text-primary">
      <div className="mx-auto max-w-7xl px-6 py-24">
        <span className="aisi-tag">Methodology</span>
        <h1 className="aisi-h1">About the <br />Observatory</h1>
        <p className="mt-8 max-w-3xl text-xl font-medium leading-relaxed text-muted">
          This page describes the dataset and explains, in plain language, how we turn annual-report text into the dashboard metrics.
        </p>
        <div className="mt-8">
          <span className="inline-flex items-center gap-2 rounded border border-border bg-secondary px-6 py-3 text-sm font-bold uppercase tracking-widest text-muted-foreground cursor-not-allowed">
            Download Coming Soon
          </span>
        </div>

        <div className="mt-20 space-y-24">
          <section id="dataset" className="grid gap-12 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <span className="aisi-tag">01</span>
              <h2 className="aisi-h2 uppercase">Dataset</h2>
            </div>
            <div className="lg:col-span-2 text-lg leading-relaxed text-muted">
              <p>
                The AI Risk Observatory dataset includes a metadata.csv file that details the mapping
                between company, year, report, excerpt, and other metadata. It also provides a list of excerpts,
                each annotated with labels assigned by classifiers. Every excerpt is labeled with mention type,
                risk taxonomy, adoption maturity, and vendor references.
              </p>
              <p className="mt-6">
                This project focuses on the FTSE 350 and major UK Critical National Infrastructure (CNI) companies across sectors such as Finance, Energy, Transport, and Health.
              </p>
            </div>
          </section>

          <section id="pipeline" className="border-y border-border bg-secondary -mx-6 px-6 py-20">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 max-w-3xl">
                <span className="aisi-tag">Pipeline</span>
                <h2 className="aisi-h2 uppercase">Flow of Information</h2>
                <p className="mt-4 text-lg text-muted">
                  This represents a complete view of our corpus and how each document moves through our classification pipeline.
                </p>
              </div>
              <div className="rounded-lg bg-white p-8">
                <ReportClassificationSankeyShell flow={data.reportClassificationFlow} />
              </div>
              <div className="mt-12">
                <div className="mb-6 max-w-3xl">
                  <h3 className="mb-3 text-sm font-bold uppercase tracking-widest text-primary">Functional Logic</h3>
                  <p className="text-base leading-relaxed text-muted">
                    This is the operational workflow behind the observatory: how raw annual reports become structured AI monitoring data.
                  </p>
                </div>
                <ClassificationFlowDiagram />
              </div>
            </div>
          </section>

          {data.exampleChunks.length > 0 && (
            <section id="examples" className="-mx-6">
              <ExampleBrowser exampleChunks={data.exampleChunks} />
            </section>
          )}

          <section id="taxonomies" className="grid gap-12 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <span className="aisi-tag">02</span>
              <h2 className="aisi-h2 uppercase">Taxonomies</h2>
            </div>
            <div className="lg:col-span-2 space-y-16">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary mb-6">Mention Types</h3>
                <div className="grid gap-4">
                  {mentionTypeTaxonomy.map(item => (
                    <div key={item.label} className="rounded bg-secondary p-6">
                      <span className="aisi-pill pill-slate mb-2">{item.label}</span>
                      <p className="text-muted text-sm">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary mb-6">Adoption Taxonomy</h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  {adoptionTaxonomy.map(item => (
                    <div key={item.label} className="rounded bg-secondary p-6">
                      <span className="aisi-pill pill-sky mb-2">{item.label}</span>
                      <p className="text-muted text-sm">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary mb-6">Risk Taxonomy</h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  {riskTaxonomy.map(item => (
                    <div key={item.label} className="rounded border border-border p-6">
                      <span className="aisi-pill pill-red mb-2">{item.label}</span>
                      <p className="text-muted text-[13px]">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary mb-6">Vendor Taxonomy</h3>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {vendorTaxonomy.map(item => (
                    <div key={item.label} className="rounded border border-border p-4">
                      <span className="aisi-pill pill-teal mb-2">{item.label}</span>
                      <p className="text-muted text-[12px]">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold uppercase tracking-widest text-primary mb-6">Substantiveness</h3>
                <div className="grid gap-4 sm:grid-cols-3">
                  {substantivenessLevels.map(item => (
                    <div key={item.label} className="rounded border border-border p-4">
                      <span className="aisi-pill pill-amber mb-2">{item.label}</span>
                      <p className="text-muted text-[13px]">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section id="quality-controls" className="grid gap-12 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <span className="aisi-tag">03</span>
              <h2 className="aisi-h2 uppercase">Quality Controls</h2>
            </div>
            <div className="lg:col-span-2 space-y-8 text-lg text-muted leading-relaxed">
              <p>
                We use schema-constrained outputs, deterministic settings, and explicit validation/reconciliation tools to
                reduce noise and improve reproducibility.
              </p>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <span className="text-accent font-bold">/</span>
                  <p><span className="font-bold text-primary">Structured outputs:</span> Classifiers write to strict response schemas, reducing malformed labels.</p>
                </div>
                <div className="flex gap-4">
                  <span className="text-accent font-bold">/</span>
                  <p><span className="font-bold text-primary">Conservative prompting:</span> Prompts require explicit AI attribution and discourage category over-assignment.</p>
                </div>
                <div className="flex gap-4">
                  <span className="text-accent font-bold">/</span>
                  <p><span className="font-bold text-primary">Testing and reconciliation:</span> Repo scripts support QA checks and human-vs-LLM disagreement review.</p>
                </div>
              </div>
            </div>
          </section>

          <section id="baseline-analysis" className="border-t border-border pt-20">
            <div className="mb-12">
              <span className="aisi-tag">Summary</span>
              <h2 className="aisi-h2 uppercase">Baseline Analysis</h2>
            </div>
            <div className="rounded-lg bg-secondary p-8">
              <MentionTypesChart
                data={data.datasets.perReport.mentionTrend}
                stackKeys={data.labels.mentionTypes}
              />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
