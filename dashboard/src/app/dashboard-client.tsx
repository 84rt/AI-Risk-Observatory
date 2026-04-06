'use client';

import { toPng } from 'html-to-image';
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { GenericHeatmap, StackedBarChart, InfoTooltip } from '@/components/overview-charts';
import { buildIsicSectorGroups, type HeatmapRowGroup } from '@/lib/isic';
import type { GoldenDashboardData } from '@/lib/golden-set';

// Filter type for risk distribution view
type RiskFilter = 'all' | string;
type AdoptionFilter = 'all' | string;

type View = {
  id: number;
  title: string;
  heading: string;
  description: string;
};

type RiskInfoPanelKey = 'definitions' | 'cite' | 'download';

type RiskInfoPanelItem = {
  value: RiskInfoPanelKey;
  label: string;
  title: string;
  content: ReactNode;
};

const DEFAULT_VIEW_ID = 1;
const DEFAULT_DASHBOARD_BASE_URL = 'https://www.riskobservatory.ai/data';

const getViewHash = (viewId: number) => {
  switch (viewId) {
    case 1:
      return 'risk';
    case 2:
      return 'adoption';
    case 3:
      return 'vendors';
    case 4:
      return 'signal-quality';
    case 5:
      return 'blind-spots';
    default:
      return 'risk';
  }
};

const getViewIdFromHash = (hash: string) => {
  switch (hash.replace(/^#/, '')) {
    case 'risk':
      return 1;
    case 'adoption':
      return 2;
    case 'vendors':
      return 3;
    case 'signal-quality':
      return 4;
    case 'blind-spots':
      return 5;
    default:
      return 1;
  }
};

const VIEWS: View[] = [
  {
    id: 1,
    title: 'Risk',
    heading: 'Reports mentioning risk from AI',
    description: 'AI risk categories over time and across sectors.',
  },
  {
    id: 2,
    title: 'Adoption',
    heading: 'Reports mentioning adoption of AI',
    description: 'AI adoption type (non-LLM, LLM, agentic) across sectors and over time.',
  },
  {
    id: 3,
    title: 'Vendors',
    heading: 'Mentions of AI Vendors in Annual Reports',
    description: 'Which technology vendors companies name in their reports, and how that varies by sector.',
  },
  {
    id: 5,
    title: 'Blind Spots',
    heading: 'Reports that did not mention AI',
    description: 'Where disclosures are absent: reports that do not mention AI at all, and reports that do not mention AI risk.',
  },
  {
    id: 4,
    title: 'Signal Quality',
    heading: 'Metrics of findings quality and strength',
    description: 'How explicit and substantive each disclosure is — from concrete detail to boilerplate language.',
  },
];

const adoptionColors: Record<string, string> = {
  non_llm: '#64748b',     // slate-500
  llm: '#3b82f6',         // blue-500
  agentic: '#f59e0b',     // amber-500
};

const vendorColors: Record<string, string> = {
  openai: '#e63946',      // AISI red
  microsoft: '#3b82f6',   // blue-500
  google: '#16a34a',      // green-600
  amazon: '#f59e0b',      // amber-500
  meta: '#1e3a8a',        // blue-900
  anthropic: '#d97706',   // amber-600
  internal: '#0b0c0c',    // near-black
  other: '#64748b',       // slate-500
  undisclosed: '#e2e8f0', // slate-200
};

const riskColors: Record<string, string> = {
  strategic_competitive:    '#1d4ed8', // deep blue
  cybersecurity:            '#3b82f6', // blue
  operational_technical:    '#93c5fd', // light blue
  regulatory_compliance:    '#dbeafe', // pale blue
  reputational_ethical:     '#fecdd3', // soft rose
  third_party_supply_chain: '#fca5a5', // light red
  information_integrity:    '#ef4444', // red
  workforce_impacts:        '#f87171', // red-rose
  environmental_impact:     '#7f1d1d', // darkest red
  national_security:        '#b91c1c', // deep red
};

const blindSpotColors: Record<string, string> = {
  no_ai_mention:      '#b91c1c', // deep red
  no_ai_risk_mention: '#e63946', // AISI red
};

const explicitnessSignalOptions = [
  { value: 'risk_signal', label: 'AI Risk Signal Strength' },
  { value: 'adoption_signal', label: 'AI Adoption Signal Strength' },
  { value: 'vendor_signal', label: 'AI Vendor Signal Strength' },
] as const;

const riskCategoryDefinitions = [
  {
    label: 'Strategic / Competitive',
    definition: 'AI-driven competitive disadvantage, disruption, or failure to adapt.',
  },
  {
    label: 'Operational / Technical',
    definition: 'Reliability, accuracy, integration, or model-risk failures that degrade operations or decision-making.',
  },
  {
    label: 'Cybersecurity',
    definition: 'AI-enabled attacks, fraud, breach pathways, or adversarial AI abuse.',
  },
  {
    label: 'Workforce Impacts',
    definition: 'AI-related displacement, skills gaps, or risky employee AI usage.',
  },
  {
    label: 'Regulatory / Compliance',
    definition: 'AI-specific legal, regulatory, privacy, intellectual-property, or compliance exposure.',
  },
  {
    label: 'Information Integrity',
    definition: 'AI-enabled misinformation, deepfakes, hallucinations, or authenticity manipulation.',
  },
  {
    label: 'Reputational / Ethical',
    definition: 'AI-linked trust, fairness, ethics, or rights concerns.',
  },
  {
    label: 'Third-Party Supply Chain',
    definition: 'Dependency on external AI vendors, providers, APIs, or concentrated supplier exposure.',
  },
  {
    label: 'Environmental Impact',
    definition: 'AI-related energy, carbon, or resource-burden risk.',
  },
  {
    label: 'National Security',
    definition: 'AI-linked geopolitical or critical-systems exposure with wider security implications.',
  },
];

const adoptionTypeDefinitions = [
  {
    label: 'Non-LLM',
    definition: 'Traditional AI or machine-learning systems that are not based on large language models.',
  },
  {
    label: 'LLM',
    definition: 'Large language model usage, including generative AI assistants and model-based workflows.',
  },
  {
    label: 'Agentic',
    definition: 'More autonomous AI systems, often coordinating multi-step tasks with limited human intervention.',
  },
];

const vendorTagDefinitions = [
  {
    label: 'OpenAI',
    definition: 'OpenAI is explicitly named in the source disclosure.',
  },
  {
    label: 'Microsoft',
    definition: 'Microsoft is explicitly named in the source disclosure.',
  },
  {
    label: 'Google',
    definition: 'Google is explicitly named in the source disclosure.',
  },
  {
    label: 'Amazon / AWS',
    definition: 'Amazon or AWS is explicitly named in the source disclosure.',
  },
  {
    label: 'Meta',
    definition: 'Meta is explicitly named in the source disclosure.',
  },
  {
    label: 'Anthropic',
    definition: 'Anthropic is explicitly named in the source disclosure.',
  },
  {
    label: 'Internal',
    definition: 'The company describes AI capabilities as built or operated in-house.',
  },
  {
    label: 'Other',
    definition: 'A named provider outside the primary tracked vendor set.',
  },
];

const signalQualityDefinitions = [
  {
    label: 'Signal Strength',
    definition: 'How directly the text supports a classification: explicit, strong implicit, or weak implicit.',
  },
  {
    label: 'Substantiveness',
    definition: 'How concrete and detailed AI-risk disclosure is at report level: substantive, moderate, or boilerplate.',
  },
];

const blindSpotDefinitions = [
  {
    label: 'No AI Mention',
    definition: 'No AI disclosure signal appears anywhere in the report.',
  },
  {
    label: 'No AI Risk Mention',
    definition: 'AI may be mentioned, but no AI-risk disclosure is present.',
  },
];

const formatAccessedOnLabel = (date: Date) =>
  new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(date);

const TrendChartIcon = ({ className = 'h-[18px] w-[18px]' }: { className?: string }) => (
  <svg viewBox="0 0 20 20" className={className} fill="none" aria-hidden="true">
    <path
      d="M4.25 4.5V15.75H15.5"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M6 12.5L9 9.5L11.4 11L15 7.25"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="9" cy="9.5" r="1.1" fill="currentColor" />
    <circle cx="11.4" cy="11" r="1.1" fill="currentColor" />
    <circle cx="15" cy="7.25" r="1.1" fill="currentColor" />
  </svg>
);

const SectorHeatmapIcon = ({ className = 'h-[18px] w-[18px]' }: { className?: string }) => (
  <svg viewBox="0 0 20 20" className={className} fill="none" aria-hidden="true">
    <rect x="4" y="4" width="5" height="5" rx="1" fill="currentColor" opacity="0.28" />
    <rect x="11" y="4" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <rect x="4" y="11" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <rect x="11" y="11" width="5" height="5" rx="1" fill="currentColor" />
  </svg>
);

const segmentedButtonClass = 'rounded-sm px-4 py-1.5 text-[11px] font-semibold tracking-normal normal-case transition';
const segmentedButtonTallClass = 'rounded-sm px-4 py-2 text-[11px] font-semibold tracking-normal normal-case transition';
const tabButtonClass = 'rounded border px-4 py-2 text-[11px] font-semibold tracking-normal normal-case transition-colors';
const clearButtonClass = 'w-full rounded border border-border bg-white px-3 py-2 text-[11px] font-semibold tracking-normal text-muted-foreground transition hover:bg-secondary';
const actionButtonClass = 'inline-flex h-9 items-center justify-center gap-2 rounded border border-border bg-white px-3 text-[11px] font-semibold tracking-normal text-primary transition hover:bg-secondary';
const subtleActionButtonClass = 'inline-flex h-9 items-center justify-center gap-2 rounded bg-white px-3 text-[11px] font-semibold tracking-normal text-primary transition hover:bg-secondary';
const infoTabButtonClass = 'flex items-center rounded border px-3 py-2 text-left text-[11px] font-semibold tracking-normal normal-case transition-colors lg:w-full';
const inlinePanelButtonClass = 'inline-flex items-center justify-center rounded border border-slate-300 bg-white px-4 py-2 text-xs font-semibold tracking-normal text-slate-900 transition-colors hover:border-slate-900 hover:bg-slate-50';
const inlinePanelLinkClass = 'inline-flex items-center justify-center border border-transparent px-1 py-2 text-xs font-semibold tracking-normal text-accent underline underline-offset-4 transition-colors hover:text-primary';

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
    '3-explicit': 'Explicit',
    '2-strong_implicit': 'Strong Implicit',
    '1-weak_implicit': 'Weak Implicit',
    no_ai_mention: 'No AI Mention',
    no_ai_risk_mention: 'No AI Risk Mention',
    openai: 'OpenAI',
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

type DatasetKey = 'perReport' | 'perChunk';
type TrendTimeAxis = 'year' | 'month';
type RiskSectorView = 'cni' | 'isic';
type SignalQualityMode = 'explicitness' | 'substantiveness';
type ExplicitnessSignalFilter = 'risk_signal' | 'adoption_signal' | 'vendor_signal';
type BlindSpotFilter = 'all' | 'no_ai_mention' | 'no_ai_risk_mention';
type BlindSpotHeatmapSelection = 'no_ai_mention' | 'no_ai_risk_mention';
type MetricMode = 'count' | 'pct_reports';
type ChartRow = Record<string, string | number | null | undefined>;
type HeatmapCell = { x: string | number; y: string | number; value: number };
type VisualizationExport = {
  title: string;
  fileBase: string;
  csv: string;
};

const toNumber = (value: string | number | null | undefined) => Number(value) || 0;
const toPercent = (value: number, total: number) => (total > 0 ? (value / total) * 100 : 0);
const formatPercent = (value: number) => `${value.toFixed(1)}%`;
const escapeCsvValue = (value: string | number | null | undefined) => {
  const stringValue = value == null ? '' : String(value);
  if (!/[",\n]/.test(stringValue)) return stringValue;
  return `"${stringValue.replace(/"/g, '""')}"`;
};
const toCsv = (rows: Array<Record<string, string | number | null | undefined>>) => {
  if (rows.length === 0) return '';
  const headers = Object.keys(rows[0]);
  const headerLine = headers.map(escapeCsvValue).join(',');
  const valueLines = rows.map(row => headers.map(header => escapeCsvValue(row[header])).join(','));
  return [headerLine, ...valueLines].join('\n');
};
const heatmapCellsToCsv = (cells: HeatmapCell[]) =>
  toCsv(cells.map(cell => ({ x: cell.x, y: cell.y, value: cell.value })));
const slugify = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
const filterLabelsWithTotals = (
  labels: (string | number)[],
  totals: Map<string, number>
) => labels.filter(label => (totals.get(String(label)) || 0) > 0);
const filterHeatmapRows = (
  cells: HeatmapCell[],
  yLabels: (string | number)[]
) => {
  const allowedLabels = new Set(yLabels.map(label => String(label)));
  return cells.filter(cell => allowedLabels.has(String(cell.y)));
};
const convertTrendRowsToPercent = (
  rows: ChartRow[],
  axisKey: 'year' | 'month',
  keys: string[],
  totals: Map<string | number, number>
): ChartRow[] =>
  rows.map(row => {
    const axisValue = row[axisKey];
    const total = totals.get(axisValue as string | number) || 0;
    const nextRow: ChartRow = { [axisKey]: axisValue };
    keys.forEach(key => {
      nextRow[key] = toPercent(toNumber(row[key]), total);
    });
    return nextRow;
  });
const convertHeatmapToPercent = (
  cells: HeatmapCell[],
  denominatorForCell: (cell: HeatmapCell) => number
): HeatmapCell[] =>
  cells.map(cell => ({
    ...cell,
    value: toPercent(cell.value, denominatorForCell(cell)),
  }));
const aggregateHeatmapCellsByRowGroups = (
  cells: HeatmapCell[],
  rowGroups: HeatmapRowGroup[]
): HeatmapCell[] => {
  const childToGroupLabel = new Map<string, string>();
  rowGroups.forEach(group => {
    group.childKeys.forEach(childKey => {
      childToGroupLabel.set(String(childKey), group.label);
    });
  });

  const groupedCells = new Map<string, HeatmapCell>();
  cells.forEach(cell => {
    const groupLabel = childToGroupLabel.get(String(cell.y));
    if (!groupLabel) return;

    const groupedKey = `${String(cell.x)}|||${groupLabel}`;
    const existing = groupedCells.get(groupedKey);
    if (existing) {
      existing.value += cell.value;
      return;
    }

    groupedCells.set(groupedKey, {
      x: cell.x,
      y: groupLabel,
      value: cell.value,
    });
  });

  return Array.from(groupedCells.values());
};
const aggregateLabelTotalsByRowGroups = (
  totals: Map<string, number>,
  rowGroups: HeatmapRowGroup[]
) => {
  const childToGroupLabel = new Map<string, string>();
  rowGroups.forEach(group => {
    group.childKeys.forEach(childKey => {
      childToGroupLabel.set(String(childKey), group.label);
    });
  });

  const nextTotals = new Map(totals);
  rowGroups.forEach(group => {
    nextTotals.set(group.label, 0);
  });

  totals.forEach((value, key) => {
    const groupLabel = childToGroupLabel.get(key);
    if (!groupLabel) return;
    nextTotals.set(groupLabel, (nextTotals.get(groupLabel) || 0) + value);
  });

  return nextTotals;
};
const aggregateYearLabelTotalsByRowGroups = (
  totals: Map<string, number>,
  rowGroups: HeatmapRowGroup[]
) => {
  const childToGroupLabel = new Map<string, string>();
  rowGroups.forEach(group => {
    group.childKeys.forEach(childKey => {
      childToGroupLabel.set(String(childKey), group.label);
    });
  });

  const nextTotals = new Map(totals);
  totals.forEach((value, key) => {
    const separatorIndex = key.indexOf('|||');
    if (separatorIndex < 0) return;

    const xKey = key.slice(0, separatorIndex);
    const yKey = key.slice(separatorIndex + 3);
    const groupLabel = childToGroupLabel.get(yKey);
    if (!groupLabel) return;

    const groupedKey = `${xKey}|||${groupLabel}`;
    nextTotals.set(groupedKey, (nextTotals.get(groupedKey) || 0) + value);
  });

  return nextTotals;
};

const buildVisibleGroupedYLabels = (
  rowGroups: HeatmapRowGroup[],
  ungroupedLabels: string[],
  expandedGroupSet: Set<string>
) => {
  const groupedLabels = rowGroups.flatMap(group =>
    expandedGroupSet.has(group.label)
      ? [group.label, ...group.childKeys]
      : [group.label]
  );

  return [...groupedLabels, ...ungroupedLabels];
};

export default function DashboardClient({
  data,
  renderedAtIso,
}: {
  data: GoldenDashboardData;
  renderedAtIso: string;
}) {
  const hasInitializedHashSync = useRef(false);
  const hasSkippedInitialHashWrite = useRef(false);
  const [activeView, setActiveView] = useState(DEFAULT_VIEW_ID);
  const [visualizationMode, setVisualizationMode] = useState<'chart' | 'heatmap'>('chart');
  const [dashboardBaseUrl, setDashboardBaseUrl] = useState(DEFAULT_DASHBOARD_BASE_URL);
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('perReport');
  const [trendTimeAxis, setTrendTimeAxis] = useState<TrendTimeAxis>('year');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [adoptionFilter, setAdoptionFilter] = useState<AdoptionFilter>('all');
  const [riskSectorView, setRiskSectorView] = useState<RiskSectorView>('cni');
  const [adoptionSectorView, setAdoptionSectorView] = useState<RiskSectorView>('cni');
  const [vendorSectorView, setVendorSectorView] = useState<RiskSectorView>('cni');
  const [vendorFilter, setVendorFilter] = useState<string>('all');
  const [infoPanelSelections, setInfoPanelSelections] = useState<Record<number, RiskInfoPanelKey>>({
    1: 'cite',
    2: 'cite',
    3: 'cite',
    4: 'cite',
    5: 'cite',
  });
  const [signalQualityMode, setSignalQualityMode] = useState<SignalQualityMode>('explicitness');
  const [explicitnessSignalFilter, setExplicitnessSignalFilter] = useState<ExplicitnessSignalFilter>('risk_signal');
  const [blindSpotFilter, setBlindSpotFilter] = useState<BlindSpotFilter>('all');
  const [blindSpotHeatmapSelection, setBlindSpotHeatmapSelection] =
    useState<BlindSpotHeatmapSelection>('no_ai_mention');
  const [metricMode, setMetricMode] = useState<MetricMode>('pct_reports');
  const [marketSegmentFilter, setMarketSegmentFilter] = useState<string>('all');
  const [expandedIsicGroups, setExpandedIsicGroups] = useState<string[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);
  const [chartDisplayType, setChartDisplayType] = useState<'bar' | 'line'>('bar');
  const [showChartLegend, setShowChartLegend] = useState(true);
  const [shareButtonLabel, setShareButtonLabel] = useState('Share');
  const [copiedReferenceKey, setCopiedReferenceKey] = useState<string | null>(null);
  const [isExportingVisualization, setIsExportingVisualization] = useState(false);
  const visualizationExportRef = useRef<HTMLDivElement | null>(null);
  const defaultYearRange = {
    start: 0,
    end: Math.max(data.years.length - 1, 0),
  };

  const resetDashboardSettings = () => {
    setDatasetKey('perReport');
    setTrendTimeAxis('year');
    setRiskFilter('all');
    setAdoptionFilter('all');
    setRiskSectorView('cni');
    setAdoptionSectorView('cni');
    setVendorSectorView('cni');
    setVendorFilter('all');
    setSignalQualityMode('explicitness');
    setExplicitnessSignalFilter('risk_signal');
    setBlindSpotFilter('all');
    setBlindSpotHeatmapSelection('no_ai_mention');
    setMetricMode('pct_reports');
    setMarketSegmentFilter('all');
    setExpandedIsicGroups([]);
    setChartDisplayType('bar');
    setShowChartLegend(true);
    setYearRangeIndices(defaultYearRange);
  };

  useEffect(() => {
    const syncViewFromHash = () => {
      const nextView = getViewIdFromHash(window.location.hash);
      setActiveView(prev => (prev === nextView ? prev : nextView));
      setVisualizationMode(nextView === 4 ? 'heatmap' : 'chart');
    };

    setDashboardBaseUrl(`${window.location.origin}/data`);
    syncViewFromHash();
    hasInitializedHashSync.current = true;
    window.addEventListener('hashchange', syncViewFromHash);
    return () => window.removeEventListener('hashchange', syncViewFromHash);
  }, []);

  useEffect(() => {
    if (!hasInitializedHashSync.current) return;
    if (!hasSkippedInitialHashWrite.current) {
      hasSkippedInitialHashWrite.current = true;
      return;
    }

    const nextHash = `#${getViewHash(activeView)}`;
    if ((window.location.hash || activeView !== DEFAULT_VIEW_ID) && window.location.hash !== nextHash) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${nextHash}`);
    }
  }, [activeView]);

  const currentViewUrl = `${dashboardBaseUrl}${activeView === DEFAULT_VIEW_ID ? '' : `#${getViewHash(activeView)}`}`;

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const resolvedDatasets =
    marketSegmentFilter === 'all'
      ? data.datasets
      : (data.byMarketSegment[marketSegmentFilter] ?? data.datasets);
  const activeData = resolvedDatasets[datasetKey];
  const reportBaselineData = resolvedDatasets.perReport;
  const canShowReportShare =
    activeView !== 4 && (activeView === 5 || datasetKey === 'perReport');
  const effectiveMetricMode: MetricMode = canShowReportShare ? metricMode : 'count';
  const isReportShareMode = effectiveMetricMode === 'pct_reports';
  const availableYears = activeData.years;
  const maxYearIndex = Math.max(availableYears.length - 1, 0);
  const fullDatasetYearSpan = data.years.length > 0
    ? `${data.years[0]} to ${data.years[data.years.length - 1]}`
    : 'the full available period';

  const datasetSummaryByView = useMemo(() => {
    const totalReports = data.datasets.perReport.blindSpotTrend.reduce(
      (sum, row) => sum + (Number(row.total_reports) || 0),
      0
    );
    const riskMentionReports = data.datasets.perReport.blindSpotTrend.reduce(
      (sum, row) => sum + (Number(row.ai_risk_mention) || 0),
      0
    );
    const excerptRiskMentions = data.datasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.risk) || 0),
      0
    );
    const adoptionMentionReports = data.datasets.perReport.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.adoption) || 0),
      0
    );
    const excerptAdoptionMentions = data.datasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.adoption) || 0),
      0
    );
    const vendorMentionReports = data.datasets.perReport.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.vendor) || 0),
      0
    );
    const excerptVendorMentions = data.datasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.vendor) || 0),
      0
    );
    const noAiMention = data.datasets.perReport.blindSpotTrend.reduce(
      (sum, row) => sum + (Number(row.no_ai_mention) || 0),
      0
    );
    const noAiRiskMention = data.datasets.perReport.blindSpotTrend.reduce(
      (sum, row) => sum + (Number(row.no_ai_risk_mention) || 0),
      0
    );
    const riskSignalTotal = data.datasets.perReport.riskSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );
    const adoptionSignalTotal = data.datasets.perReport.adoptionSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );
    const vendorSignalTotal = data.datasets.perReport.vendorSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );

    return {
      1: `Across ${formatNumber(totalReports)} annual reports from ${fullDatasetYearSpan}, ${formatNumber(riskMentionReports)} mention AI risk across ${formatNumber(excerptRiskMentions)} tagged passages.`,
      2: `Across ${formatNumber(totalReports)} annual reports from ${fullDatasetYearSpan}, ${formatNumber(adoptionMentionReports)} mention AI adoption across ${formatNumber(excerptAdoptionMentions)} tagged passages.`,
      3: `Across ${formatNumber(totalReports)} annual reports from ${fullDatasetYearSpan}, ${formatNumber(vendorMentionReports)} name AI vendors across ${formatNumber(excerptVendorMentions)} tagged passages.`,
      4: `Across ${formatNumber(totalReports)} annual reports from ${fullDatasetYearSpan}, the dataset records ${formatNumber(riskSignalTotal)} risk, ${formatNumber(adoptionSignalTotal)} adoption, and ${formatNumber(vendorSignalTotal)} vendor signal labels.`,
      5: `Across ${formatNumber(totalReports)} annual reports from ${fullDatasetYearSpan}, ${formatNumber(noAiMention)} contain no AI mention and ${formatNumber(noAiRiskMention)} contain no AI risk disclosure.`,
    } as Record<number, string>;
  }, [data, fullDatasetYearSpan]);

  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const vendorStackKeys = useMemo(
    () => data.labels.vendorTags.filter(tag => tag !== 'undisclosed'),
    [data.labels.vendorTags]
  );
  const riskStackKeys = useMemo(() => {
    const canonicalOrder = data.labels.riskLabels;
    const totals = new Map<string, number>();

    data.datasets.perReport.riskTrend.forEach(row => {
      canonicalOrder.forEach(label => {
        totals.set(label, (totals.get(label) || 0) + (Number(row[label]) || 0));
      });
    });

    return [...canonicalOrder].sort((a, b) => {
      const delta = (totals.get(b) || 0) - (totals.get(a) || 0);
      return delta !== 0 ? delta : canonicalOrder.indexOf(a) - canonicalOrder.indexOf(b);
    });
  }, [data.datasets.perReport.riskTrend, data.labels.riskLabels]);
  const effectiveVendorFilter =
    vendorFilter !== 'all' && !vendorStackKeys.includes(vendorFilter) ? 'all' : vendorFilter;

  const [yearRangeIndices, setYearRangeIndices] = useState(() => ({
    start: 0,
    end: Math.max(data.years.length - 1, 0),
  }));

  const startIndex = Math.min(yearRangeIndices.start, maxYearIndex);
  const endIndex = Math.min(Math.max(yearRangeIndices.end, startIndex), maxYearIndex);

  const selectedStartYear = availableYears[startIndex] ?? 0;
  const selectedEndYear = availableYears[endIndex] ?? selectedStartYear;

  const filteredYears = useMemo(
    () => availableYears.slice(startIndex, endIndex + 1),
    [availableYears, startIndex, endIndex]
  );

  const selectedLeftPct =
    availableYears.length <= 1 ? 0 : (startIndex / maxYearIndex) * 100;
  const selectedRightPct =
    availableYears.length <= 1 ? 0 : 100 - (endIndex / maxYearIndex) * 100;
  const toggleIsicGroup = (groupLabel: string) => {
    setExpandedIsicGroups(prev =>
      prev.includes(groupLabel)
        ? prev.filter(label => label !== groupLabel)
        : [...prev, groupLabel]
    );
  };

  const updateYearRangeFromTrackClick = (clientX: number, element: HTMLDivElement) => {
    if (maxYearIndex <= 0) return;
    const rect = element.getBoundingClientRect();
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    const clickedIndex = Math.round(ratio * maxYearIndex);

    setYearRangeIndices(prev => {
      if (prev.start === prev.end) {
        if (clickedIndex <= prev.start) {
          return { start: clickedIndex, end: prev.end };
        }
        return { start: prev.start, end: clickedIndex };
      }

      const startDistance = Math.abs(clickedIndex - prev.start);
      const endDistance = Math.abs(clickedIndex - prev.end);

      if (startDistance <= endDistance) {
        return { start: Math.min(clickedIndex, prev.end), end: prev.end };
      }
      return { start: prev.start, end: Math.max(clickedIndex, prev.start) };
    });
  };

  const adoptionTrendInRange = useMemo(
    () =>
      activeData.adoptionTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [activeData.adoptionTrend, selectedStartYear, selectedEndYear]
  );

  const riskTrendInRange = useMemo(
    () =>
      activeData.riskTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [activeData.riskTrend, selectedStartYear, selectedEndYear]
  );

  const vendorTrendInRange = useMemo(
    () =>
      activeData.vendorTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [activeData.vendorTrend, selectedStartYear, selectedEndYear]
  );

  const riskBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.riskBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.riskBySectorYear, selectedStartYear, selectedEndYear]
  );

  const riskByIsicSectorYearInRange = useMemo(
    () =>
      reportBaselineData.riskByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.riskByIsicSectorYear, selectedStartYear, selectedEndYear]
  );

  const riskBySectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      riskBySectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [riskBySectorYearInRange]
  );

  const riskByIsicSectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      riskByIsicSectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [riskByIsicSectorYearInRange]
  );

  const adoptionBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.adoptionBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.adoptionBySectorYear, selectedStartYear, selectedEndYear]
  );

  const adoptionBySectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      adoptionBySectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [adoptionBySectorYearInRange]
  );

  const adoptionByIsicSectorYearInRange = useMemo(
    () =>
      reportBaselineData.adoptionByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.adoptionByIsicSectorYear, selectedStartYear, selectedEndYear]
  );

  const adoptionByIsicSectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      adoptionByIsicSectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [adoptionByIsicSectorYearInRange]
  );

  const vendorBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.vendorBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.vendorBySectorYear, selectedStartYear, selectedEndYear]
  );

  const vendorBySectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      vendorBySectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [vendorBySectorYearInRange]
  );

  const vendorByIsicSectorYearInRange = useMemo(
    () =>
      reportBaselineData.vendorByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [reportBaselineData.vendorByIsicSectorYear, selectedStartYear, selectedEndYear]
  );

  const vendorByIsicSectorInRange = useMemo(
    () => {
      const counts = new Map<string, number>();
      vendorByIsicSectorYearInRange.forEach(cell => {
        const key = `${cell.x}|||${cell.y}`;
        counts.set(key, (counts.get(key) || 0) + cell.value);
      });
      return Array.from(counts.entries()).map(([key, value]) => {
        const [x, y] = key.split('|||');
        return { x, y, value };
      });
    },
    [vendorByIsicSectorYearInRange]
  );

  // Filter risk trend for single risk type view
  const filteredRiskTrend = useMemo(() => {
    if (riskFilter === 'all') return riskTrendInRange;
    return riskTrendInRange.map(row => ({
      year: row.year,
      [riskFilter]: row[riskFilter] || 0,
    }));
  }, [riskTrendInRange, riskFilter]);

  const filteredAdoptionTrend = useMemo(() => {
    if (adoptionFilter === 'all') return adoptionTrendInRange;
    return adoptionTrendInRange.map(row => ({
      year: row.year,
      [adoptionFilter]: row[adoptionFilter] || 0,
    }));
  }, [adoptionTrendInRange, adoptionFilter]);

  const filteredVendorTrend = useMemo(() => {
    if (effectiveVendorFilter === 'all') {
      return vendorTrendInRange.map(row => {
        const nextRow: Record<string, string | number | null | undefined> = { year: row.year };
        vendorStackKeys.forEach(key => {
          nextRow[key] = row[key] || 0;
        });
        return nextRow;
      });
    }
    return vendorTrendInRange.map(row => ({
      year: row.year,
      [effectiveVendorFilter]: row[effectiveVendorFilter] || 0,
    }));
  }, [effectiveVendorFilter, vendorTrendInRange, vendorStackKeys]);

  // Monthly trend data filtered by year range
  const monthlyRiskTrendInRange = useMemo(() => {
    const startMonth = `${selectedStartYear}-01`;
    const endMonth = `${selectedEndYear}-12`;
    return activeData.riskTrendMonthly.filter(row => {
      const m = row.month as string;
      return m >= startMonth && m <= endMonth;
    });
  }, [activeData.riskTrendMonthly, selectedStartYear, selectedEndYear]);

  const monthlyAdoptionTrendInRange = useMemo(() => {
    const startMonth = `${selectedStartYear}-01`;
    const endMonth = `${selectedEndYear}-12`;
    return activeData.adoptionTrendMonthly.filter(row => {
      const m = row.month as string;
      return m >= startMonth && m <= endMonth;
    });
  }, [activeData.adoptionTrendMonthly, selectedStartYear, selectedEndYear]);

  const monthlyVendorTrendInRange = useMemo(() => {
    const startMonth = `${selectedStartYear}-01`;
    const endMonth = `${selectedEndYear}-12`;
    return activeData.vendorTrendMonthly.filter(row => {
      const m = row.month as string;
      return m >= startMonth && m <= endMonth;
    });
  }, [activeData.vendorTrendMonthly, selectedStartYear, selectedEndYear]);

  const monthlyBlindSpotTrendInRange = useMemo(() => {
    const startMonth = `${selectedStartYear}-01`;
    const endMonth = `${selectedEndYear}-12`;
    return reportBaselineData.blindSpotTrendMonthly.filter(row => {
      const m = row.month as string;
      return m >= startMonth && m <= endMonth;
    });
  }, [reportBaselineData.blindSpotTrendMonthly, selectedStartYear, selectedEndYear]);

  const reportTotalsByYear = useMemo(
    () =>
      new Map(
        reportBaselineData.blindSpotTrend.map(row => [
          Number(row.year),
          Number(row.total_reports) || 0,
        ])
      ),
    [reportBaselineData.blindSpotTrend]
  );

  const reportTotalsByMonth = useMemo(
    () =>
      new Map(
        reportBaselineData.blindSpotTrendMonthly.map(row => [
          String(row.month),
          Number(row.total_reports) || 0,
        ])
      ),
    [reportBaselineData.blindSpotTrendMonthly]
  );

  const reportTotalsBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.reportCountBySectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [reportBaselineData.reportCountBySectorYear, selectedStartYear, selectedEndYear]
  );

  const reportTotalsByIsicSectorYearInRange = useMemo(
    () =>
      reportBaselineData.reportCountByIsicSectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [reportBaselineData.reportCountByIsicSectorYear, selectedStartYear, selectedEndYear]
  );

  const reportTotalsBySectorYearMap = useMemo(
    () =>
      new Map(
        reportTotalsBySectorYearInRange.map(cell => [`${cell.x}|||${cell.y}`, cell.value])
      ),
    [reportTotalsBySectorYearInRange]
  );

  const reportTotalsByIsicSectorYearMap = useMemo(
    () =>
      new Map(
        reportTotalsByIsicSectorYearInRange.map(cell => [`${cell.x}|||${cell.y}`, cell.value])
      ),
    [reportTotalsByIsicSectorYearInRange]
  );

  const reportTotalsBySectorInRange = useMemo(() => {
    const counts = new Map<string, number>();
    reportTotalsBySectorYearInRange.forEach(cell => {
      counts.set(cell.y, (counts.get(cell.y) || 0) + cell.value);
    });
    return counts;
  }, [reportTotalsBySectorYearInRange]);

  const reportTotalsByIsicSectorInRange = useMemo(() => {
    const counts = new Map<string, number>();
    reportTotalsByIsicSectorYearInRange.forEach(cell => {
      counts.set(cell.y, (counts.get(cell.y) || 0) + cell.value);
    });
    return counts;
  }, [reportTotalsByIsicSectorYearInRange]);
  const visibleIsicSectorLabels = useMemo(
    () => filterLabelsWithTotals(data.isicSectors, reportTotalsByIsicSectorInRange).map(String),
    [data.isicSectors, reportTotalsByIsicSectorInRange]
  );
  const { groups: isicRowGroups, ungroupedLabels: ungroupedIsicSectorLabels } = useMemo(
    () => buildIsicSectorGroups(visibleIsicSectorLabels, data.isicSectorParents),
    [visibleIsicSectorLabels, data.isicSectorParents]
  );
  const expandedIsicGroupSet = useMemo(
    () => new Set(expandedIsicGroups),
    [expandedIsicGroups]
  );
  const isicHeatmapYLabels = useMemo(
    () => buildVisibleGroupedYLabels(isicRowGroups, ungroupedIsicSectorLabels, expandedIsicGroupSet),
    [isicRowGroups, ungroupedIsicSectorLabels, expandedIsicGroupSet]
  );

  const filteredMonthlyRiskTrend = useMemo(() => {
    if (riskFilter === 'all') return monthlyRiskTrendInRange;
    return monthlyRiskTrendInRange.map(row => ({
      month: row.month,
      [riskFilter]: row[riskFilter] || 0,
    }));
  }, [monthlyRiskTrendInRange, riskFilter]);

  const filteredMonthlyAdoptionTrend = useMemo(() => {
    if (adoptionFilter === 'all') return monthlyAdoptionTrendInRange;
    return monthlyAdoptionTrendInRange.map(row => ({
      month: row.month,
      [adoptionFilter]: row[adoptionFilter] || 0,
    }));
  }, [monthlyAdoptionTrendInRange, adoptionFilter]);

  const filteredMonthlyVendorTrend = useMemo(() => {
    if (effectiveVendorFilter === 'all') {
      return monthlyVendorTrendInRange.map(row => {
        const nextRow: Record<string, string | number | null | undefined> = { month: row.month };
        vendorStackKeys.forEach(key => {
          nextRow[key] = row[key] || 0;
        });
        return nextRow;
      });
    }
    return monthlyVendorTrendInRange.map(row => ({
      month: row.month,
      [effectiveVendorFilter]: row[effectiveVendorFilter] || 0,
    }));
  }, [effectiveVendorFilter, monthlyVendorTrendInRange, vendorStackKeys]);

  const filteredMonthlyBlindSpotTrend = useMemo(() => {
    if (blindSpotFilter === 'all') return monthlyBlindSpotTrendInRange;
    return monthlyBlindSpotTrendInRange.map(row => ({
      month: row.month,
      [blindSpotFilter]: row[blindSpotFilter] || 0,
    }));
  }, [blindSpotFilter, monthlyBlindSpotTrendInRange]);

  const displayRiskTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyRiskTrend : filteredRiskTrend;
    if (!isReportShareMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      riskFilter === 'all' ? riskStackKeys : [riskFilter],
      trendTimeAxis === 'month' ? reportTotalsByMonth : reportTotalsByYear
    );
  }, [
    trendTimeAxis,
    filteredMonthlyRiskTrend,
    filteredRiskTrend,
    isReportShareMode,
    riskFilter,
    riskStackKeys,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);

  const displayAdoptionTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyAdoptionTrend : filteredAdoptionTrend;
    if (!isReportShareMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      adoptionFilter === 'all' ? adoptionStackKeys : [adoptionFilter],
      trendTimeAxis === 'month' ? reportTotalsByMonth : reportTotalsByYear
    );
  }, [
    trendTimeAxis,
    filteredMonthlyAdoptionTrend,
    filteredAdoptionTrend,
    isReportShareMode,
    adoptionFilter,
    adoptionStackKeys,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);

  const displayVendorTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyVendorTrend : filteredVendorTrend;
    if (!isReportShareMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      effectiveVendorFilter === 'all' ? vendorStackKeys : [effectiveVendorFilter],
      trendTimeAxis === 'month' ? reportTotalsByMonth : reportTotalsByYear
    );
  }, [
    trendTimeAxis,
    filteredMonthlyVendorTrend,
    filteredVendorTrend,
    isReportShareMode,
    effectiveVendorFilter,
    vendorStackKeys,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);

  const riskSignalHeatmapInRange = useMemo(
    () =>
      activeData.riskSignalHeatmap.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.riskSignalHeatmap, selectedStartYear, selectedEndYear]
  );

  const adoptionSignalHeatmapInRange = useMemo(
    () =>
      activeData.adoptionSignalHeatmap.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.adoptionSignalHeatmap, selectedStartYear, selectedEndYear]
  );

  const vendorSignalHeatmapInRange = useMemo(
    () =>
      activeData.vendorSignalHeatmap.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.vendorSignalHeatmap, selectedStartYear, selectedEndYear]
  );

  const substantivenessHeatmapInRange = useMemo(
    () =>
      activeData.substantivenessHeatmap.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.substantivenessHeatmap, selectedStartYear, selectedEndYear]
  );

  const blindSpotTrendInRange = useMemo(
    () =>
      reportBaselineData.blindSpotTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [reportBaselineData.blindSpotTrend, selectedStartYear, selectedEndYear]
  );

  const noAiBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.noAiBySectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [reportBaselineData.noAiBySectorYear, selectedStartYear, selectedEndYear]
  );

  const noAiRiskBySectorYearInRange = useMemo(
    () =>
      reportBaselineData.noAiRiskBySectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [reportBaselineData.noAiRiskBySectorYear, selectedStartYear, selectedEndYear]
  );

  const blindSpotYearsInRange = useMemo(
    () =>
      reportBaselineData.years.filter(
        year => year >= selectedStartYear && year <= selectedEndYear
      ),
    [reportBaselineData.years, selectedStartYear, selectedEndYear]
  );

  const riskHeatmapYLabels = useMemo(
    () =>
      riskSectorView === 'cni'
        ? data.sectors
        : isicHeatmapYLabels,
    [riskSectorView, data.sectors, isicHeatmapYLabels]
  );
  const riskHeatmapAxisSectorLabel = riskSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector';
  const riskHeatmapTaxonomyDataInRange =
    riskSectorView === 'cni' ? riskBySectorInRange : riskByIsicSectorInRange;
  const riskHeatmapTaxonomyYearDataInRange =
    riskSectorView === 'cni' ? riskBySectorYearInRange : riskByIsicSectorYearInRange;

  const riskHeatmapData = useMemo(() => {
    if (riskFilter === 'all') return riskHeatmapTaxonomyDataInRange;
    return riskHeatmapTaxonomyYearDataInRange
      .filter(cell => cell.x === riskFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [riskFilter, riskHeatmapTaxonomyDataInRange, riskHeatmapTaxonomyYearDataInRange]);

  const riskHeatmapXLabels = riskFilter === 'all' ? data.labels.riskLabels : filteredYears;
  const riskHeatmapBaseColor = riskFilter === 'all'
    ? '#e63946'
    : (riskColors[riskFilter] || '#e63946');
  const visibleRiskHeatmapData = useMemo(() => {
    if (riskSectorView === 'cni') {
      const displayData = riskFilter === 'all'
        ? convertHeatmapToPercent(
            riskHeatmapData,
            cell => reportTotalsBySectorInRange.get(String(cell.y)) || 0
          )
        : convertHeatmapToPercent(
            riskHeatmapData,
            cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
          );

      return filterHeatmapRows(displayData, riskHeatmapYLabels);
    }

    const groupedHeatmapData = [
      ...riskHeatmapData,
      ...aggregateHeatmapCellsByRowGroups(riskHeatmapData, isicRowGroups),
    ];
    const groupedTotalsBySector = aggregateLabelTotalsByRowGroups(
      reportTotalsByIsicSectorInRange,
      isicRowGroups
    );
    const groupedTotalsBySectorYear = aggregateYearLabelTotalsByRowGroups(
      reportTotalsByIsicSectorYearMap,
      isicRowGroups
    );
    const displayData = riskFilter === 'all'
      ? convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySector.get(String(cell.y)) || 0
        )
      : convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySectorYear.get(`${cell.x}|||${cell.y}`) || 0
        );

    return filterHeatmapRows(displayData, riskHeatmapYLabels);
  }, [
    riskFilter,
    riskHeatmapData,
    riskHeatmapYLabels,
    riskSectorView,
    isicRowGroups,
    reportTotalsByIsicSectorInRange,
    reportTotalsByIsicSectorYearMap,
    reportTotalsBySectorInRange,
    reportTotalsBySectorYearMap,
  ]);
  const adoptionHeatmapYLabels = useMemo(
    () =>
      adoptionSectorView === 'cni'
        ? data.sectors
        : isicHeatmapYLabels,
    [adoptionSectorView, data.sectors, isicHeatmapYLabels]
  );
  const adoptionHeatmapSectorData = adoptionSectorView === 'cni' ? adoptionBySectorInRange : adoptionByIsicSectorInRange;
  const adoptionHeatmapSectorYearData = adoptionSectorView === 'cni' ? adoptionBySectorYearInRange : adoptionByIsicSectorYearInRange;
  const adoptionHeatmapData = useMemo(() => {
    if (adoptionFilter === 'all') return adoptionHeatmapSectorData;
    return adoptionHeatmapSectorYearData
      .filter(cell => cell.x === adoptionFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [adoptionFilter, adoptionHeatmapSectorData, adoptionHeatmapSectorYearData]);
  const adoptionHeatmapXLabels = adoptionFilter === 'all' ? data.labels.adoptionTypes : filteredYears;
  const adoptionHeatmapBaseColor = adoptionFilter === 'all'
    ? '#3b82f6'
    : (adoptionColors[adoptionFilter] || '#3b82f6');
  const visibleAdoptionHeatmapData = useMemo(() => {
    if (adoptionSectorView === 'cni') {
      const displayData = adoptionFilter === 'all'
        ? convertHeatmapToPercent(
            adoptionHeatmapData,
            cell => reportTotalsBySectorInRange.get(String(cell.y)) || 0
          )
        : convertHeatmapToPercent(
            adoptionHeatmapData,
            cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
          );

      return filterHeatmapRows(displayData, adoptionHeatmapYLabels);
    }

    const groupedHeatmapData = [
      ...adoptionHeatmapData,
      ...aggregateHeatmapCellsByRowGroups(adoptionHeatmapData, isicRowGroups),
    ];
    const groupedTotalsBySector = aggregateLabelTotalsByRowGroups(
      reportTotalsByIsicSectorInRange,
      isicRowGroups
    );
    const groupedTotalsBySectorYear = aggregateYearLabelTotalsByRowGroups(
      reportTotalsByIsicSectorYearMap,
      isicRowGroups
    );
    const displayData = adoptionFilter === 'all'
      ? convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySector.get(String(cell.y)) || 0
        )
      : convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySectorYear.get(`${cell.x}|||${cell.y}`) || 0
        );

    return filterHeatmapRows(displayData, adoptionHeatmapYLabels);
  }, [
    adoptionFilter,
    adoptionHeatmapData,
    adoptionHeatmapYLabels,
    adoptionSectorView,
    isicRowGroups,
    reportTotalsByIsicSectorInRange,
    reportTotalsByIsicSectorYearMap,
    reportTotalsBySectorInRange,
    reportTotalsBySectorYearMap,
  ]);
  const vendorHeatmapYLabels = useMemo(
    () =>
      vendorSectorView === 'cni'
        ? data.sectors
        : isicHeatmapYLabels,
    [vendorSectorView, data.sectors, isicHeatmapYLabels]
  );
  const vendorHeatmapSectorData = vendorSectorView === 'cni' ? vendorBySectorInRange : vendorByIsicSectorInRange;
  const vendorHeatmapSectorYearData = vendorSectorView === 'cni' ? vendorBySectorYearInRange : vendorByIsicSectorYearInRange;
  const vendorHeatmapData = useMemo(() => {
    if (effectiveVendorFilter === 'all') {
      return vendorHeatmapSectorData.filter(cell => vendorStackKeys.includes(cell.x));
    }
    return vendorHeatmapSectorYearData
      .filter(cell => cell.x === effectiveVendorFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [effectiveVendorFilter, vendorHeatmapSectorData, vendorHeatmapSectorYearData, vendorStackKeys]);
  const vendorHeatmapXLabels = effectiveVendorFilter === 'all' ? vendorStackKeys : filteredYears;
  const vendorHeatmapBaseColor = effectiveVendorFilter === 'all'
    ? '#64748b'
    : (vendorColors[effectiveVendorFilter] || '#64748b');
  const visibleVendorHeatmapData = useMemo(() => {
    if (vendorSectorView === 'cni') {
      const displayData = effectiveVendorFilter === 'all'
        ? convertHeatmapToPercent(
            vendorHeatmapData,
            cell => reportTotalsBySectorInRange.get(String(cell.y)) || 0
          )
        : convertHeatmapToPercent(
            vendorHeatmapData,
            cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
          );

      return filterHeatmapRows(displayData, vendorHeatmapYLabels);
    }

    const groupedHeatmapData = [
      ...vendorHeatmapData,
      ...aggregateHeatmapCellsByRowGroups(vendorHeatmapData, isicRowGroups),
    ];
    const groupedTotalsBySector = aggregateLabelTotalsByRowGroups(
      reportTotalsByIsicSectorInRange,
      isicRowGroups
    );
    const groupedTotalsBySectorYear = aggregateYearLabelTotalsByRowGroups(
      reportTotalsByIsicSectorYearMap,
      isicRowGroups
    );
    const displayData = effectiveVendorFilter === 'all'
      ? convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySector.get(String(cell.y)) || 0
        )
      : convertHeatmapToPercent(
          groupedHeatmapData,
          cell => groupedTotalsBySectorYear.get(`${cell.x}|||${cell.y}`) || 0
        );

    return filterHeatmapRows(displayData, vendorHeatmapYLabels);
  }, [
    effectiveVendorFilter,
    vendorHeatmapData,
    vendorHeatmapYLabels,
    vendorSectorView,
    isicRowGroups,
    reportTotalsByIsicSectorInRange,
    reportTotalsByIsicSectorYearMap,
    reportTotalsBySectorInRange,
    reportTotalsBySectorYearMap,
  ]);
  const filteredBlindSpotTrend = useMemo(() => {
    if (blindSpotFilter === 'all') return blindSpotTrendInRange;
    return blindSpotTrendInRange.map(row => ({
      year: row.year,
      [blindSpotFilter]: row[blindSpotFilter] || 0,
    }));
  }, [blindSpotFilter, blindSpotTrendInRange]);
  const displayBlindSpotTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyBlindSpotTrend : filteredBlindSpotTrend;
    if (!isReportShareMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      blindSpotFilter === 'all'
        ? ['no_ai_mention', 'no_ai_risk_mention']
        : [blindSpotFilter],
      trendTimeAxis === 'month' ? reportTotalsByMonth : reportTotalsByYear
    );
  }, [
    trendTimeAxis,
    filteredMonthlyBlindSpotTrend,
    filteredBlindSpotTrend,
    isReportShareMode,
    blindSpotFilter,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);
  const blindSpotHeatmapData = blindSpotHeatmapSelection === 'no_ai_mention'
    ? noAiBySectorYearInRange
    : noAiRiskBySectorYearInRange;
  const displayBlindSpotHeatmapData = useMemo(() => {
    return convertHeatmapToPercent(
      blindSpotHeatmapData,
      cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
    );
  }, [blindSpotHeatmapData, reportTotalsBySectorYearMap]);
  const blindSpotHeatmapTitle = blindSpotHeatmapSelection === 'no_ai_mention'
    ? 'No AI Mention by Sector and Year'
    : 'No AI Risk Mention by Sector and Year';
  const blindSpotHeatmapSubtitle = blindSpotHeatmapSelection === 'no_ai_mention'
    ? 'Heatmap of annual reports containing no AI mention, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with zero AI mention signal; colour intensity encodes relative frequency.'
    : 'Heatmap of annual reports containing no AI risk disclosure, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with no AI risk mention; colour intensity encodes relative frequency.';
  const blindSpotHeatmapTooltip = blindSpotHeatmapSelection === 'no_ai_mention'
    ? 'Each cell is the share of reports in that sector-year that do not mention AI at all. The Avg column and row show simple averages across the displayed years and sectors.'
    : 'Each cell is the share of reports in that sector-year with no AI risk mention. The Avg column and row show simple averages across the displayed years and sectors.';
  const blindSpotHeatmapColor = blindSpotHeatmapSelection === 'no_ai_mention' ? '#b91c1c' : '#e63946';
  const selectedSignalQualityHeatmap = useMemo(() => {
    if (signalQualityMode === 'substantiveness') {
      return {
        data: substantivenessHeatmapInRange,
        yLabels: data.labels.substantivenessBands,
        baseColor: '#f59e0b',
        title: 'AI Risk Substantiveness Distribution',
        subtitle:
          'Heatmap of report-level risk-disclosure quality by substantiveness band (rows: Substantive, Moderate, Boilerplate) and fiscal year (columns). Each cell counts the number of reports whose AI-risk language was classified into that quality tier in a given year; colour intensity encodes relative frequency.',
        tooltip:
          'Substantiveness measures depth and specificity of risk disclosure at report level. Substantive disclosures include concrete mechanisms and mitigation/action detail, while boilerplate disclosures remain generic.',
        yAxisLabel: 'Quality Band',
        compact: false,
      };
    }

    switch (explicitnessSignalFilter) {
      case 'risk_signal':
        return {
          data: riskSignalHeatmapInRange,
          yLabels: data.labels.riskSignalLevels,
          baseColor: '#e63946',
          title: 'AI Risk Signal Strength',
          subtitle:
            'Heatmap of risk-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level risk classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.',
          tooltip:
            'Risk signal strength scores how directly the text supports a risk classification. 3 = explicit statement; 2 = strong implicit evidence; 1 = weak implicit evidence. Each cell counts label-level outcomes, not unique reports.',
          yAxisLabel: 'Signal Level',
          compact: true,
        };
      case 'adoption_signal':
        return {
          data: adoptionSignalHeatmapInRange,
          yLabels: data.labels.riskSignalLevels,
          baseColor: '#3b82f6',
          title: 'AI Adoption Signal Strength',
          subtitle:
            'Heatmap of adoption-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level adoption classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.',
          tooltip:
            'Applies the same signal-strength rubric to AI adoption mentions. Higher rows indicate clearer, more directly supported adoption disclosures, while lower rows reflect softer inferential language.',
          yAxisLabel: 'Signal Level',
          compact: true,
        };
      case 'vendor_signal':
        return {
          data: vendorSignalHeatmapInRange,
          yLabels: data.labels.riskSignalLevels,
          baseColor: '#64748b',
          title: 'AI Vendor Signal Strength',
          subtitle:
            'Heatmap of vendor-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level vendor classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.',
          tooltip:
            'Measures how directly a vendor relationship is stated in the text. Low explicitness can indicate more opaque supplier disclosure; higher explicit counts suggest clearer provider attribution.',
          yAxisLabel: 'Signal Level',
          compact: true,
        };
    }
  }, [
    signalQualityMode,
    explicitnessSignalFilter,
    riskSignalHeatmapInRange,
    adoptionSignalHeatmapInRange,
    vendorSignalHeatmapInRange,
    substantivenessHeatmapInRange,
    data.labels.riskSignalLevels,
    data.labels.substantivenessBands,
  ]);
  const riskSelectedYearSpan = filteredYears.length > 0
    ? `${selectedStartYear}–${selectedEndYear}`
    : 'N/A';
  const stackedChartYAxisFormatter = isReportShareMode
    ? (value: number) => `${Math.round(value)}%`
    : undefined;
  const stackedChartTooltipFormatter = (value: number) =>
    isReportShareMode ? formatPercent(value) : formatNumber(value);
  const riskHeatmapValueFormatter = (value: number) => formatPercent(value);
  const blindSpotHeatmapValueFormatter = (value: number) => formatPercent(value);
  const dashboardUpdatedLabel = 'Dataset updated 03.04.2026';
  const currentVisualizationExport = useMemo<VisualizationExport>(() => {
    if (activeView === 1) {
      if (visualizationMode === 'chart') {
        return {
          title: 'AI Risk Mentioned Over Time',
          fileBase: 'ai-risk-mentioned-over-time',
          csv: toCsv(displayRiskTrend),
        };
      }

      return {
        title: riskFilter === 'all' ? 'Risk Distribution by Sector' : `${formatLabel(riskFilter)} Risk Mentions by Sector and Year`,
        fileBase: riskFilter === 'all' ? 'risk-distribution-by-sector' : `${slugify(formatLabel(riskFilter))}-risk-mentions-by-sector-and-year`,
        csv: heatmapCellsToCsv(visibleRiskHeatmapData),
      };
    }

    if (activeView === 2) {
      if (visualizationMode === 'chart') {
        return {
          title: 'AI Adoption Mentioned Over Time',
          fileBase: 'ai-adoption-mentioned-over-time',
          csv: toCsv(displayAdoptionTrend),
        };
      }

      return {
        title: adoptionFilter === 'all' ? 'Adoption Intensity by Sector' : `${formatLabel(adoptionFilter)} Mentions by Sector and Year`,
        fileBase: adoptionFilter === 'all' ? 'adoption-intensity-by-sector' : `${slugify(formatLabel(adoptionFilter))}-mentions-by-sector-and-year`,
        csv: heatmapCellsToCsv(visibleAdoptionHeatmapData),
      };
    }

    if (activeView === 3) {
      if (visualizationMode === 'chart') {
        return {
          title: 'AI Vendor Mentions Over Time',
          fileBase: 'ai-vendor-mentions-over-time',
          csv: toCsv(displayVendorTrend),
        };
      }

      return {
        title: effectiveVendorFilter === 'all' ? 'Vendor Mentions by Sector' : `${formatLabel(effectiveVendorFilter)} Mentions by Sector and Year`,
        fileBase: effectiveVendorFilter === 'all' ? 'vendor-mentions-by-sector' : `${slugify(formatLabel(effectiveVendorFilter))}-mentions-by-sector-and-year`,
        csv: heatmapCellsToCsv(visibleVendorHeatmapData),
      };
    }

    if (activeView === 4) {
      return {
        title: selectedSignalQualityHeatmap.title,
        fileBase: slugify(selectedSignalQualityHeatmap.title),
        csv: heatmapCellsToCsv(selectedSignalQualityHeatmap.data),
      };
    }

    if (visualizationMode === 'chart') {
      return {
        title: trendTimeAxis === 'month' ? 'Blind Spots by Month' : 'Blind Spots by Year',
        fileBase: trendTimeAxis === 'month' ? 'blind-spots-by-month' : 'blind-spots-by-year',
        csv: toCsv(displayBlindSpotTrend),
      };
    }

    return {
      title: blindSpotHeatmapTitle,
      fileBase: slugify(blindSpotHeatmapTitle),
      csv: heatmapCellsToCsv(displayBlindSpotHeatmapData),
    };
  }, [
    activeView,
    visualizationMode,
    displayRiskTrend,
    riskFilter,
    visibleRiskHeatmapData,
    displayAdoptionTrend,
    adoptionFilter,
    visibleAdoptionHeatmapData,
    displayVendorTrend,
    effectiveVendorFilter,
    visibleVendorHeatmapData,
    selectedSignalQualityHeatmap,
    trendTimeAxis,
    displayBlindSpotTrend,
    blindSpotHeatmapTitle,
    displayBlindSpotHeatmapData,
  ]);

  const handleDownloadVisualization = () => {
    if (!currentVisualizationExport.csv) return;

    const blob = new Blob([currentVisualizationExport.csv], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${currentVisualizationExport.fileBase}-${selectedStartYear}-${selectedEndYear}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportWatermark = (
    <div className="flex flex-col items-end gap-1 text-right">
      <p className="text-sm font-semibold tracking-tight text-slate-800">
        {currentVisualizationExport.title}
      </p>
      <div className="flex items-center gap-2 text-[11px] leading-relaxed text-slate-500">
        <span className="font-semibold tracking-[0.08em] text-slate-600">AI Risk Observatory</span>
        <span className="h-1 w-1 rounded-full bg-slate-300" aria-hidden="true" />
        <span>{currentViewUrl}</span>
      </div>
    </div>
  );

  const handleDownloadVisualizationImage = async () => {
    if (!visualizationExportRef.current) return;

    setIsExportingVisualization(true);

    try {
      await new Promise(resolve => requestAnimationFrame(() => resolve(null)));
      await new Promise(resolve => requestAnimationFrame(() => resolve(null)));

      const dataUrl = await toPng(visualizationExportRef.current, {
        cacheBust: true,
        pixelRatio: 2,
        backgroundColor: '#ffffff',
      });

      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = `${currentVisualizationExport.fileBase}-${selectedStartYear}-${selectedEndYear}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } finally {
      setIsExportingVisualization(false);
    }
  };

  const handleShareVisualization = async () => {
    const shareUrl = currentViewUrl;
    const shareText = `${currentVisualizationExport.title} (${riskSelectedYearSpan})`;

    try {
      if (navigator.share) {
        await navigator.share({
          title: currentVisualizationExport.title,
          text: shareText,
          url: shareUrl,
        });
        setShareButtonLabel('Shared');
        setTimeout(() => setShareButtonLabel('Share'), 2000);
        return;
      }

      await navigator.clipboard.writeText(`${shareText}\n${shareUrl}`);
      setShareButtonLabel('Copied Link');
      setTimeout(() => setShareButtonLabel('Share'), 2000);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      setShareButtonLabel('Share Failed');
      setTimeout(() => setShareButtonLabel('Share'), 2000);
    }
  };

  const handleCopyReferenceText = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedReferenceKey(key);
      setTimeout(() => {
        setCopiedReferenceKey(prev => (prev === key ? null : prev));
      }, 2000);
    } catch {
      setCopiedReferenceKey(null);
    }
  };

  const makeSectorToggle = (
    current: RiskSectorView,
    setter: (v: RiskSectorView) => void
  ) => (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setter('cni')}
        className={`${segmentedButtonClass} ${
          current === 'cni'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        CNI
      </button>
      <button
        type="button"
        onClick={() => setter('isic')}
        className={`${segmentedButtonClass} ${
          current === 'isic'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        ISIC
      </button>
    </div>
  );

  const trendTimeToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setTrendTimeAxis('year')}
        className={`${segmentedButtonClass} ${
          trendTimeAxis === 'year'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Year
      </button>
      <button
        type="button"
        onClick={() => setTrendTimeAxis('month')}
        className={`${segmentedButtonClass} ${
          trendTimeAxis === 'month'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Month
      </button>
    </div>
  );

  const metricModeToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => {
          if (!canShowReportShare) return;
          setMetricMode('pct_reports');
        }}
        disabled={!canShowReportShare}
        className={`${segmentedButtonClass} ${
          effectiveMetricMode === 'pct_reports'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        } ${!canShowReportShare ? 'cursor-not-allowed opacity-40 hover:bg-white' : ''}`}
      >
        % of Reports
      </button>
      <button
        type="button"
        onClick={() => setMetricMode('count')}
        className={`${segmentedButtonClass} ${
          effectiveMetricMode === 'count'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Count
      </button>
    </div>
  );

  const showVisualizationToggle = activeView !== 4;
  const showSignalQualityToggle = activeView === 4;
  const canUseLineChart = visualizationMode === 'chart' && activeView !== 4 && trendTimeAxis === 'year';
  const activeChartDisplayType = canUseLineChart ? chartDisplayType : 'bar';

  const chartTypeToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setChartDisplayType('bar')}
        className={`${segmentedButtonClass} ${
          activeChartDisplayType === 'bar'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Bar
      </button>
      <button
        type="button"
        onClick={() => {
          if (!canUseLineChart) return;
          setChartDisplayType('line');
        }}
        disabled={!canUseLineChart}
        className={`${segmentedButtonClass} ${
          activeChartDisplayType === 'line'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        } ${!canUseLineChart ? 'cursor-not-allowed opacity-40 hover:bg-white' : ''}`}
      >
        Line
      </button>
    </div>
  );

  const legendVisibilityToggle = (
    <button
      type="button"
      role="switch"
      aria-checked={showChartLegend}
      onClick={() => setShowChartLegend(prev => !prev)}
      className="inline-flex items-center rounded-full transition"
      title={showChartLegend ? 'Hide legend' : 'Show legend'}
    >
      <span
        className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
          showChartLegend ? 'bg-primary' : 'bg-slate-300'
        }`}
        aria-hidden="true"
      >
        <span
          className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
            showChartLegend ? 'translate-x-3.5' : 'translate-x-0.5'
          }`}
        />
      </span>
    </button>
  );

  const settingsPanelToggle = (
    <button
      type="button"
      onClick={() => setIsSettingsOpen(prev => !prev)}
      className="inline-flex h-9 w-9 items-center justify-center rounded border border-border bg-white text-primary transition hover:bg-secondary"
      aria-label={isSettingsOpen ? 'Collapse settings panel' : 'Expand settings panel'}
      title={isSettingsOpen ? 'Collapse settings' : 'Expand settings'}
    >
      <svg viewBox="0 0 20 20" className="h-[18px] w-[18px]" fill="none" aria-hidden="true">
        <path
          d="M10 3.25L11.05 1.75L13.15 2.45L13.1 4.28C13.55 4.55 13.95 4.88 14.29 5.26L16.06 4.78L17.12 6.72L15.82 7.99C15.88 8.32 15.91 8.66 15.91 9C15.91 9.34 15.88 9.68 15.82 10.01L17.12 11.28L16.06 13.22L14.29 12.74C13.95 13.12 13.55 13.45 13.1 13.72L13.15 15.55L11.05 16.25L10 14.75C9.66 14.79 9.34 14.79 9 14.75L7.95 16.25L5.85 15.55L5.9 13.72C5.45 13.45 5.05 13.12 4.71 12.74L2.94 13.22L1.88 11.28L3.18 10.01C3.12 9.68 3.09 9.34 3.09 9C3.09 8.66 3.12 8.32 3.18 7.99L1.88 6.72L2.94 4.78L4.71 5.26C5.05 4.88 5.45 4.55 5.9 4.28L5.85 2.45L7.95 1.75L9 3.25C9.34 3.21 9.66 3.21 10 3.25Z"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="9.5" cy="9" r="2.35" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    </button>
  );

  const renderSettingsSectionHeading = (
    title: string,
    tooltip?: ReactNode,
    action?: ReactNode
  ) => (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-start gap-1.5">
        <p className="text-sm font-semibold leading-tight text-primary">{title}</p>
        {tooltip ? <InfoTooltip content={tooltip} /> : null}
      </div>
      {action ? <div className="ml-auto shrink-0">{action}</div> : null}
    </div>
  );

  const renderSettingsPanel = () => (
    <div className="overflow-hidden rounded-lg border border-border bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <h3 className="text-[11px] font-bold uppercase tracking-[0.18em] text-primary">Settings</h3>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={resetDashboardSettings}
            className="inline-flex h-8 w-8 items-center justify-center rounded border border-border bg-white text-muted-foreground transition hover:bg-secondary hover:text-primary"
            aria-label="Reset settings"
            title="Reset settings"
          >
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" aria-hidden="true">
              <path
                d="M16 10a6 6 0 1 1-1.76-4.24"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M16 4.5v3.5h-3.5"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => setIsSettingsOpen(false)}
            className="inline-flex h-8 w-8 items-center justify-center rounded border border-border bg-white text-muted-foreground transition hover:bg-secondary hover:text-primary"
            aria-label="Collapse settings panel"
            title="Collapse settings"
          >
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" aria-hidden="true">
              <path
                d="M7.5 6L12 10L7.5 14"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
      <div className="p-5 [&>*+*]:mt-6">
        {visualizationMode === 'chart' && activeView !== 4 && (
          <div>
            <div className="flex items-center gap-2">
              {legendVisibilityToggle}
              <p className="text-sm font-semibold leading-tight text-primary">Show Legend</p>
            </div>
          </div>
        )}

        {visualizationMode === 'chart' && activeView !== 4 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Chart Style',
              'Bar chart stacks all categories. Line chart shows each category as a separate trend line. Line requires Year axis and is disabled otherwise.'
            )}
            {chartTypeToggle}
          </div>
        )}

        <div className="space-y-3">
          {renderSettingsSectionHeading('Market Segment')}
          <select
            value={marketSegmentFilter}
            onChange={event => {
              setMarketSegmentFilter(event.target.value);
              setYearRangeIndices({ start: 0, end: Math.max(data.years.length - 1, 0) });
            }}
            className="w-full aisi-select"
          >
            <option value="all">All Companies</option>
            {data.marketSegments.map(segment => (
              <option key={segment} value={segment}>{segment}</option>
            ))}
          </select>
        </div>

        {visualizationMode === 'chart' && activeView !== 4 && (
          <>
            <div className="space-y-3">
              {renderSettingsSectionHeading(
                'Show Values As',
                '% of Reports shows what share of annual filings contain this label in each period. Count shows the raw number of matching reports or excerpts.'
              )}
              {metricModeToggle}
            </div>
          </>
        )}

        {activeView !== 5 && activeView !== 4 && effectiveMetricMode === 'count' && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Aggregate Data By',
              'Count by report (one annual filing = one data point) or by excerpt (each tagged passage counts separately). Per Report is the default and avoids double-counting a single filing.'
            )}
            {datasetToggle}
          </div>
        )}

        {visualizationMode === 'chart' && activeView !== 4 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Group The Data By')}
            {trendTimeToggle}
          </div>
        )}

        {visualizationMode === 'heatmap' && activeView === 1 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Sector Classification',
              'CNI groups companies by UK Critical National Infrastructure sector. ISIC uses standard industry codes for a broader cross-sector comparison.'
            )}
            {makeSectorToggle(riskSectorView, setRiskSectorView)}
          </div>
        )}

        {visualizationMode === 'heatmap' && activeView === 2 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Sector Classification',
              'CNI groups companies by UK Critical National Infrastructure sector. ISIC uses standard industry codes for a broader cross-sector comparison.'
            )}
            {makeSectorToggle(adoptionSectorView, setAdoptionSectorView)}
          </div>
        )}

        {visualizationMode === 'heatmap' && activeView === 3 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Sector Classification',
              'CNI groups companies by UK Critical National Infrastructure sector. ISIC uses standard industry codes for a broader cross-sector comparison.'
            )}
            {makeSectorToggle(vendorSectorView, setVendorSectorView)}
          </div>
        )}

        {activeView === 1 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Focus On One Risk Type')}
            <select
              id="risk-filter"
              value={riskFilter}
              onChange={e => setRiskFilter(e.target.value)}
              className="w-full aisi-select"
            >
              <option value="all">All Risk Types</option>
              {riskStackKeys.map(label => (
                <option key={label} value={label}>{formatLabel(label)}</option>
              ))}
            </select>
            {riskFilter !== 'all' && (
              <button
                onClick={() => setRiskFilter('all')}
                className={clearButtonClass}
              >
                Clear
              </button>
            )}
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Focus On One Adoption Type')}
            <select
              id="adoption-filter"
              value={adoptionFilter}
              onChange={e => setAdoptionFilter(e.target.value)}
              className="w-full aisi-select"
            >
              <option value="all">All Adoption Types</option>
              {data.labels.adoptionTypes.map(label => (
                <option key={label} value={label}>{formatLabel(label)}</option>
              ))}
            </select>
            {adoptionFilter !== 'all' && (
              <button
                onClick={() => setAdoptionFilter('all')}
                className={clearButtonClass}
              >
                Clear
              </button>
            )}
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Focus On One Vendor')}
            <select
              id="vendor-filter"
              value={effectiveVendorFilter}
              onChange={e => setVendorFilter(e.target.value)}
              className="w-full aisi-select"
            >
              <option value="all">All Vendors</option>
              {vendorStackKeys.map(label => (
                <option key={label} value={label}>{formatLabel(label)}</option>
              ))}
            </select>
            {effectiveVendorFilter !== 'all' && (
              <button
                onClick={() => setVendorFilter('all')}
                className={clearButtonClass}
              >
                Clear
              </button>
            )}
          </div>
        )}

        {activeView === 4 && signalQualityMode === 'explicitness' && (
          <div className="space-y-3">
            {renderSettingsSectionHeading(
              'Explicitness Summary',
              'Choose which aspect of explicitness to measure: the share of substantive mentions, the share of boilerplate, or the average score across all levels.'
            )}
            <select
              id="signal-quality-metric"
              value={explicitnessSignalFilter}
              onChange={e => setExplicitnessSignalFilter(e.target.value as ExplicitnessSignalFilter)}
              className="w-full aisi-select"
            >
              {explicitnessSignalOptions.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
        )}

        {activeView === 5 && visualizationMode === 'chart' && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Focus On One Blind-Spot Type')}
            <select
              id="blind-spot-filter"
              value={blindSpotFilter}
              onChange={e => setBlindSpotFilter(e.target.value as BlindSpotFilter)}
              className="w-full aisi-select"
            >
              <option value="all">All Blind Spots</option>
              <option value="no_ai_mention">No AI Mention</option>
              <option value="no_ai_risk_mention">No AI Risk Mention</option>
            </select>
            {blindSpotFilter !== 'all' && (
              <button
                onClick={() => setBlindSpotFilter('all')}
                className={clearButtonClass}
              >
                Clear
              </button>
            )}
          </div>
        )}

        {activeView === 5 && visualizationMode === 'heatmap' && (
          <div className="space-y-3">
            {renderSettingsSectionHeading('Blind-Spot Heatmap Mode')}
            {blindSpotHeatmapToggle}
          </div>
        )}
      </div>
    </div>
  );

  const signalQualityModeToggle = (
    <div className="inline-flex shrink-0 items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setSignalQualityMode('explicitness')}
        className={`${segmentedButtonTallClass} ${
          signalQualityMode === 'explicitness'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Signal Strength
      </button>
      <button
        type="button"
        onClick={() => setSignalQualityMode('substantiveness')}
        className={`${segmentedButtonTallClass} ${
          signalQualityMode === 'substantiveness'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Substantiveness
      </button>
    </div>
  );

  const accessedOnLabel = useMemo(
    () => formatAccessedOnLabel(new Date(renderedAtIso)),
    [renderedAtIso]
  );
  const citationTargetUrl = currentViewUrl;
  const plainCitation = `AI Risk Observatory. "${currentVisualizationExport.title}" dashboard view, UK annual-report AI disclosure dataset. Accessed ${accessedOnLabel}. ${citationTargetUrl}`;
  const bibtexCitation = `@misc{AIRiskObservatory${slugify(currentVisualizationExport.title).replace(/-/g, '')},
  title = {${currentVisualizationExport.title}},
  author = {{AI Risk Observatory}},
  year = {2026},
  note = {Dashboard view, accessed ${accessedOnLabel}},
  url = {${citationTargetUrl}}
}`;

  const renderCopyableReferenceBlock = (
    key: string,
    heading: string,
    value: string
  ) => (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-primary">{heading}</h4>
      <div className="relative">
        <button
          type="button"
          onClick={() => handleCopyReferenceText(key, value)}
          className="absolute right-3 top-3 inline-flex h-8 w-8 items-center justify-center rounded border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:bg-slate-100 hover:text-primary"
          aria-label={copiedReferenceKey === key ? `${heading} copied` : `Copy ${heading}`}
          title={copiedReferenceKey === key ? 'Copied' : `Copy ${heading}`}
        >
          {copiedReferenceKey === key ? (
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" aria-hidden="true">
              <path
                d="M4.5 10.5L8 14L15.5 6.5"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" aria-hidden="true">
              <rect x="7" y="4" width="9" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
              <path d="M4 7.5V15C4 15.55 4.45 16 5 16H11.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          )}
        </button>
        <pre className="overflow-x-auto whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 pr-14 font-mono text-xs leading-relaxed text-slate-700">
        {value}
        </pre>
      </div>
    </div>
  );

  const sharedCitationItem: RiskInfoPanelItem = {
    value: 'cite',
    label: 'Cite',
    title: `How To Cite The ${view.title} View`,
    content: (
      <div className="space-y-5 text-sm leading-relaxed text-slate-600">
        <p>
          Use the citation format that fits your workflow. Each block can be copied directly.
        </p>
        {renderCopyableReferenceBlock('citation-plain', 'Citation', plainCitation)}
        {renderCopyableReferenceBlock('citation-bibtex', 'BibTeX Citation', bibtexCitation)}
        {renderCopyableReferenceBlock('citation-link', 'Direct Link', citationTargetUrl)}
      </div>
    ),
  };

  const sharedDownloadItem: RiskInfoPanelItem = {
    value: 'download',
    label: 'Download',
    title: 'Download And Reuse',
    content: (
      <div className="space-y-3 text-sm leading-relaxed text-slate-600">
        <p>
          Download the plotted data as CSV, or download the full data bundle used to render this deployment for offline use.
        </p>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleDownloadVisualization}
            className={inlinePanelButtonClass}
          >
            Download CSV
          </button>
          <a
            href="/api/download-data"
            download
            className={inlinePanelButtonClass}
          >
            Download Dataset
          </a>
          <a
            href="/about"
            className={inlinePanelLinkClass}
          >
            Read Methodology
          </a>
        </div>
      </div>
    ),
  };

  const riskInfoPanelItems: RiskInfoPanelItem[] = [
    sharedCitationItem,
    {
      value: 'definitions',
      label: 'Definitions',
      title: 'Risk Category Definitions',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            A single report or excerpt can be tagged with more than one category when the disclosure covers multiple AI
            risk dimensions.
          </p>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
            {riskCategoryDefinitions.map(item => (
              <li key={item.label}>
                <span className="font-medium text-slate-800">{item.label}:</span> {item.definition}
              </li>
            ))}
          </ul>
        </>
      ),
    },
    sharedDownloadItem,
  ];

  const adoptionInfoPanelItems: RiskInfoPanelItem[] = [
    sharedCitationItem,
    {
      value: 'definitions',
      label: 'Definitions',
      title: 'Adoption Type Definitions',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            Adoption labels describe the maturity of AI deployment referenced in each disclosure.
          </p>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
            {adoptionTypeDefinitions.map(item => (
              <li key={item.label}>
                <span className="font-medium text-slate-800">{item.label}:</span> {item.definition}
              </li>
            ))}
          </ul>
        </>
      ),
    },
    sharedDownloadItem,
  ];

  const vendorInfoPanelItems: RiskInfoPanelItem[] = [
    sharedCitationItem,
    {
      value: 'definitions',
      label: 'Definitions',
      title: 'Vendor Tag Definitions',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            Vendor tags indicate which provider is named or implied in the disclosure text.
          </p>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
            {vendorTagDefinitions.map(item => (
              <li key={item.label}>
                <span className="font-medium text-slate-800">{item.label}:</span> {item.definition}
              </li>
            ))}
          </ul>
        </>
      ),
    },
    sharedDownloadItem,
  ];

  const signalQualityInfoPanelItems: RiskInfoPanelItem[] = [
    sharedCitationItem,
    {
      value: 'definitions',
      label: 'Definitions',
      title: 'Quality Metric Definitions',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            This view evaluates the quality of AI disclosure, not just the volume of mentions.
          </p>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
            {signalQualityDefinitions.map(item => (
              <li key={item.label}>
                <span className="font-medium text-slate-800">{item.label}:</span> {item.definition}
              </li>
            ))}
          </ul>
        </>
      ),
    },
    sharedDownloadItem,
  ];

  const blindSpotInfoPanelItems: RiskInfoPanelItem[] = [
    sharedCitationItem,
    {
      value: 'definitions',
      label: 'Definitions',
      title: 'Blind Spot Definitions',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            Blind spot metrics are calculated from report-level coverage, including reports with zero extracted AI
            passages.
          </p>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
            {blindSpotDefinitions.map(item => (
              <li key={item.label}>
                <span className="font-medium text-slate-800">{item.label}:</span> {item.definition}
              </li>
            ))}
          </ul>
        </>
      ),
    },
    sharedDownloadItem,
  ];

  const activeInfoPanelItems =
    activeView === 1
      ? riskInfoPanelItems
      : activeView === 2
        ? adoptionInfoPanelItems
        : activeView === 3
          ? vendorInfoPanelItems
          : activeView === 4
            ? signalQualityInfoPanelItems
            : blindSpotInfoPanelItems;

  const selectedInfoPanelKey = infoPanelSelections[activeView] ?? 'cite';
  const selectedInfoPanel =
    activeInfoPanelItems.find(item => item.value === selectedInfoPanelKey) ?? activeInfoPanelItems[0];

  const infoPanelSection = (
    <section className="border-t border-border pt-8">
      <div className="grid gap-6 lg:grid-cols-[180px_minmax(0,1fr)] lg:gap-8">
        <div className="lg:pr-2">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
            Reference
          </p>
          <div className="mt-4 flex flex-wrap gap-2 lg:flex-col lg:gap-1.5">
            {activeInfoPanelItems.map(item => (
              <button
                key={item.value}
                type="button"
                onClick={() =>
                  setInfoPanelSelections(prev => ({
                    ...prev,
                    [activeView]: item.value,
                  }))
                }
                className={`${infoTabButtonClass} ${
                  selectedInfoPanelKey === item.value
                    ? 'border-primary bg-primary text-white'
                    : 'border-border bg-secondary text-primary hover:bg-white'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="lg:border-l lg:border-border lg:pl-8">
          <div className="h-[540px] overflow-y-auto pr-1 sm:h-[600px]">
            <span className="aisi-tag">{view.title}</span>
            <h3 className="mt-3 text-base font-semibold text-primary">{selectedInfoPanel.title}</h3>
            <div className="mt-4 text-sm leading-relaxed text-muted">{selectedInfoPanel.content}</div>
          </div>
        </div>
      </div>
    </section>
  );

  const datasetToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => {
          const nextYears = data.datasets.perReport.years;
          setDatasetKey('perReport');
          setYearRangeIndices({ start: 0, end: Math.max(nextYears.length - 1, 0) });
        }}
        className={`${segmentedButtonClass} ${
          datasetKey === 'perReport'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Per Report
      </button>
      <button
        type="button"
        onClick={() => {
          const nextYears = data.datasets.perChunk.years;
          setDatasetKey('perChunk');
          setYearRangeIndices({ start: 0, end: Math.max(nextYears.length - 1, 0) });
        }}
        className={`${segmentedButtonClass} ${
          datasetKey === 'perChunk'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Per Excerpt
      </button>
    </div>
  );

  const blindSpotHeatmapToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setBlindSpotHeatmapSelection('no_ai_mention')}
        className={`${segmentedButtonClass} ${
          blindSpotHeatmapSelection === 'no_ai_mention'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        No AI
      </button>
      <button
        type="button"
        onClick={() => setBlindSpotHeatmapSelection('no_ai_risk_mention')}
        className={`${segmentedButtonClass} ${
          blindSpotHeatmapSelection === 'no_ai_risk_mention'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        No AI Risk
      </button>
    </div>
  );

  const yearRangeSelector = (
    <div className="px-1">
      <div className="flex items-center justify-between gap-4">
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
          Date Range
        </p>
        <div className="text-xs font-semibold text-slate-700">
          {selectedStartYear} - {selectedEndYear}
        </div>
      </div>
      <div className="mt-4">
        <div
          className="relative h-3"
          onMouseDown={event => {
            if (maxYearIndex === 0) return;
            if (event.target instanceof HTMLInputElement) return;
            updateYearRangeFromTrackClick(event.clientX, event.currentTarget);
          }}
          onTouchStart={event => {
            if (maxYearIndex === 0) return;
            if (event.target instanceof HTMLInputElement) return;
            const touch = event.touches[0];
            if (!touch) return;
            updateYearRangeFromTrackClick(touch.clientX, event.currentTarget);
          }}
        >
          <div className="absolute left-0 right-0 top-1/2 h-0.5 -translate-y-1/2 bg-secondary" />
          <div
            className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-accent"
            style={{ left: `${selectedLeftPct}%`, right: `${selectedRightPct}%` }}
          />
          <input
            type="range"
            min={0}
            max={maxYearIndex}
            step={1}
            value={startIndex}
            onChange={event => {
              const nextStart = Number(event.target.value);
              setYearRangeIndices(prev => ({
                start: Math.min(nextStart, prev.end),
                end: prev.end,
              }));
            }}
            className="year-range-slider pointer-events-none absolute inset-0 h-3 w-full appearance-none bg-transparent"
            aria-label="Start year"
            disabled={maxYearIndex === 0}
          />
          <input
            type="range"
            min={0}
            max={maxYearIndex}
            step={1}
            value={endIndex}
            onChange={event => {
              const nextEnd = Number(event.target.value);
              setYearRangeIndices(prev => ({
                start: prev.start,
                end: Math.max(nextEnd, prev.start),
              }));
            }}
            className="year-range-slider pointer-events-none absolute inset-0 h-3 w-full appearance-none bg-transparent"
            aria-label="End year"
            disabled={maxYearIndex === 0}
          />
        </div>
        <div className="relative mt-2 h-3 text-[9px] font-bold uppercase tracking-[0.1em] text-muted-foreground">
          {availableYears.length === 0 ? (
            <span className="block text-center">-</span>
          ) : (
            availableYears.map((year, index) => {
              const leftPct = maxYearIndex <= 0 ? 50 : (index / maxYearIndex) * 100;
              const edgeClass =
                index === 0
                  ? 'translate-x-0 text-left'
                  : index === availableYears.length - 1
                    ? '-translate-x-full text-right'
                    : '-translate-x-1/2 text-center';
              return (
                <span
                  key={year}
                  className={`absolute top-0 transition-colors ${edgeClass} ${
                    index >= startIndex && index <= endIndex ? 'text-primary' : 'text-muted-foreground'
                  }`}
                  style={{ left: `${leftPct}%` }}
                >
                  {year}
                </span>
              );
            })
          )}
        </div>
      </div>
    </div>
  );

  const visualizationFooter = yearRangeSelector;

  const visualizationActions = (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <button
        type="button"
        onClick={handleShareVisualization}
        className={subtleActionButtonClass}
        title="Share the current dashboard page"
      >
        <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="none" aria-hidden="true">
          <path
            d="M7 10L13 6.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M7 10L13 13.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle
            cx="5"
            cy="10"
            r="2"
            stroke="currentColor"
            strokeWidth="1.6"
          />
          <circle
            cx="15"
            cy="5.5"
            r="2"
            stroke="currentColor"
            strokeWidth="1.6"
          />
          <circle
            cx="15"
            cy="14.5"
            r="2"
            stroke="currentColor"
            strokeWidth="1.6"
          />
        </svg>
        {shareButtonLabel}
      </button>
      <button
        type="button"
        onClick={handleDownloadVisualizationImage}
        className={actionButtonClass}
        title="Download current visualization as PNG"
      >
        <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="none" aria-hidden="true">
          <path
            d="M10 3.5V11.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M6.75 8.75L10 12L13.25 8.75"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M4 15.5H16"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        Download Image
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-white text-primary">
      <main className="mx-auto max-w-[1320px] px-6 py-10 sm:py-14">
        <div aria-hidden="true" className="h-0 overflow-hidden">
          {VIEWS.map(item => (
            <div key={item.id} id={getViewHash(item.id)} />
          ))}
        </div>
        <div className="pb-3">
          <div className="mb-5 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
            {dashboardUpdatedLabel}
          </div>

          <div>
            <h2 className="aisi-h2 uppercase">
              <span className="mr-3 inline-block h-6 w-1.5 bg-accent align-middle" />
              {view.heading}
            </h2>
            <div className="mt-3 max-w-full overflow-x-auto">
              <p className="inline-block min-w-max whitespace-nowrap text-sm leading-relaxed text-muted sm:text-base">
                {datasetSummaryByView[activeView] ?? view.description}
              </p>
            </div>
          </div>

          <div className="mt-8 flex items-start justify-between gap-3 overflow-x-auto">
            <div className="flex shrink-0 items-center gap-2.5">
              {VIEWS.map(item => (
                <div key={item.id} className="flex items-center gap-2.5">
                  {item.id === 4 && <span aria-hidden="true" className="h-5 w-px bg-border" />}
                  <button
                    onClick={() => {
                      setActiveView(item.id);
                      setVisualizationMode(item.id === 4 ? 'heatmap' : 'chart');
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                    className={`${tabButtonClass} ${
                      activeView === item.id
                        ? 'border-primary bg-primary text-white'
                        : 'border-border bg-secondary text-primary hover:bg-white'
                    }`}
                  >
                    {item.title}
                  </button>
                </div>
              ))}
            </div>

            {showVisualizationToggle && (
              <div className="flex shrink-0 items-center gap-2">
                <span className="hidden text-[9px] font-bold uppercase tracking-[0.2em] text-accent sm:block">View</span>
                <div className="inline-flex shrink-0 items-center overflow-hidden rounded border border-border bg-white p-0.5">
                  <button
                    type="button"
                    onClick={() => setVisualizationMode('chart')}
                    className={`inline-flex items-center gap-2 ${segmentedButtonTallClass} ${
                      visualizationMode === 'chart'
                        ? 'bg-primary text-white'
                        : 'text-muted-foreground hover:bg-secondary'
                    }`}
                  >
                    <TrendChartIcon />
                    Trend Chart
                  </button>
                  <button
                    type="button"
                    onClick={() => setVisualizationMode('heatmap')}
                    className={`inline-flex items-center gap-2 ${segmentedButtonTallClass} ${
                      visualizationMode === 'heatmap'
                        ? 'bg-primary text-white'
                        : 'text-muted-foreground hover:bg-secondary'
                    }`}
                  >
                    <SectorHeatmapIcon />
                    Sector Heatmap
                  </button>
                </div>
              </div>
            )}

            {showSignalQualityToggle && (
              <div className="flex shrink-0 items-center gap-2">
                <span className="hidden text-[9px] font-bold uppercase tracking-[0.2em] text-accent sm:block">Measure</span>
                {signalQualityModeToggle}
              </div>
            )}
          </div>
        </div>

        <div
          className={`mt-3 grid gap-5 xl:items-start ${
            isSettingsOpen
              ? 'min-[960px]:grid-cols-[minmax(0,1fr)_280px]'
              : 'min-[960px]:grid-cols-1'
          }`}
        >
          <div className="min-w-0">

        {activeView === 1 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`risk-trend-${trendTimeAxis}`}
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayRiskTrend}
                  xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
                  stackKeys={riskFilter === 'all' ? riskStackKeys : [riskFilter]}
                  colors={riskColors}
                  yAxisTickFormatter={stackedChartYAxisFormatter}
                  tooltipValueFormatter={stackedChartTooltipFormatter}
                  yAxisDomain={isReportShareMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={settingsPanelToggle}
                  legendKeys={[...riskStackKeys].reverse()}
                  activeLegendKey={riskFilter === 'all' ? null : riskFilter}
                  onLegendItemClick={(key) => setRiskFilter(prev => (prev === key ? 'all' : key))}
                  footerExtra={visualizationFooter}
                subtitle={
                  isReportShareMode
                      ? `Stacked bar chart showing the percentage of annual reports mentioning each AI risk category (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). Each coloured segment is the share of reports in that period tagged with the category. Because a single report can carry multiple risk labels, stacked totals can exceed 100%.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} mentioning each AI risk category (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). Each colour represents one risk category; bars are additive because a single ${datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple categories. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    <>
                      <p>
                        {isReportShareMode
                          ? 'Each segment shows the share of reports in that period tagged with the risk category.'
                          : 'Each bar is stacked by risk category: the total height is the sum of all risk-category mentions that year, and each colour represents one category.'}
                      </p>
                      <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple risk categories and therefore contribute to several coloured segments within the same year&apos;s bar; segments are not mutually exclusive.</p>
                      <p className="mt-2">Year-on-year growth may also reflect shifts in disclosure requirements or reporting culture rather than changes in actual risk levels — see the About page for more detail.</p>
                      <p className="mt-2">Click a legend item to isolate a single category.</p>
                    </>
                  }
                />
                {visualizationActions}
              </div>
            ) : (
              <div className="space-y-4">
                <GenericHeatmap
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={visibleRiskHeatmapData}
                  xLabels={riskHeatmapXLabels}
                  yLabels={riskHeatmapYLabels}
                  baseColor={riskHeatmapBaseColor}
                  valueFormatter={riskHeatmapValueFormatter}
                  xLabelFormatter={formatLabel}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={true}
                  title={riskFilter === 'all' ? 'Risk Distribution by Sector' : `${formatLabel(riskFilter)} Risk Mentions by Sector and Year`}
                  subtitle={
                    riskFilter === 'all'
                      ? `Heatmap of report-share percentages by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and risk category (columns). Each cell shows the percentage of reports in that sector, across the selected years, that mention the risk type at least once.`
                      : `Heatmap of ${formatLabel(riskFilter)} report-share percentages by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(riskFilter)}.`
                  }
                  tooltip={
                    riskFilter === 'all'
                      ? <>
                          <p>Each cell is normalised by the total number of reports in that sector across the selected years, so sectors of different sizes can be compared directly.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed categories and sectors.</p>
                          <p className="mt-2">Select one risk type in the settings panel to switch the columns from risk categories to years.</p>
                        </>
                      : <>
                          <p>Each cell is the share of reports in that sector-year mentioning the selected risk category.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                          <p className="mt-2">Clearing the risk-type filter returns the categorical sector view.</p>
                        </>
                  }
                  xAxisLabel={riskFilter === 'all' ? 'Risk Type' : 'Year'}
                  yAxisLabel={riskHeatmapAxisSectorLabel}
                  rowGroups={riskSectorView === 'isic' ? isicRowGroups : undefined}
                  expandedRowGroups={riskSectorView === 'isic' ? expandedIsicGroups : undefined}
                  onToggleRowGroup={riskSectorView === 'isic' ? toggleIsicGroup : undefined}
                  labelColumnWidth={riskSectorView === 'isic' ? 330 : undefined}
                  rowHeight={riskSectorView === 'isic' ? 58 : undefined}
                  yLabelClassName={riskSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
                  headerExtra={settingsPanelToggle}
                  footerExtra={visualizationFooter}
                />
                {visualizationActions}
              </div>
            )}

            {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}

            {infoPanelSection}
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`adoption-trend-${trendTimeAxis}`}
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayAdoptionTrend}
                  xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
                  stackKeys={adoptionFilter === 'all' ? adoptionStackKeys : [adoptionFilter]}
                  colors={adoptionColors}
                  yAxisTickFormatter={stackedChartYAxisFormatter}
                  tooltipValueFormatter={stackedChartTooltipFormatter}
                  yAxisDomain={isReportShareMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={settingsPanelToggle}
                  legendKeys={adoptionStackKeys}
                  activeLegendKey={adoptionFilter === 'all' ? null : adoptionFilter}
                  onLegendItemClick={(key) => setAdoptionFilter(prev => (prev === key ? 'all' : key))}
                  footerExtra={visualizationFooter}
                  title="AI Adoption Mentioned Over Time"
                  subtitle={
                    isReportShareMode
                      ? `Stacked bar chart showing the percentage of annual reports referencing each AI adoption maturity level — Non-LLM, LLM, and Agentic — (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single report may be tagged with multiple adoption types, so stacked totals can exceed 100%.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} referencing each AI adoption maturity level — Non-LLM, LLM, and Agentic — (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'passage'} may be tagged with multiple adoption types. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    <>
                      <p>{isReportShareMode ? 'Each segment shows the share of reports in that period tagged with the adoption type.' : 'Each bar is stacked by adoption type: Non-LLM, LLM, and Agentic.'}</p>
                      <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple adoption types and can therefore contribute to more than one segment within the same year.</p>
                      <p className="mt-2">Click a legend item to isolate one adoption type.</p>
                    </>
                  }
                />
                {visualizationActions}
              </div>
            ) : (
              <div className="space-y-4">
                <GenericHeatmap
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={visibleAdoptionHeatmapData}
                  xLabels={adoptionHeatmapXLabels}
                  yLabels={adoptionHeatmapYLabels}
                  baseColor={adoptionHeatmapBaseColor}
                  valueFormatter={riskHeatmapValueFormatter}
                  xLabelFormatter={formatLabel}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={true}
                  title={adoptionFilter === 'all' ? 'Adoption Intensity by Sector' : `${formatLabel(adoptionFilter)} Mentions by Sector and Year`}
                  subtitle={
                    adoptionFilter === 'all'
                      ? `Heatmap of report-share percentages by ${adoptionSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and adoption type (columns). Each cell shows the percentage of reports in that sector, across the selected years, mentioning the adoption type at least once.`
                      : `Heatmap of ${formatLabel(adoptionFilter)} report-share percentages by ${adoptionSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(adoptionFilter)}.`
                  }
                  tooltip={
                    adoptionFilter === 'all'
                      ? <>
                          <p>Each cell is normalised by the total number of reports in that sector across the selected years.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed adoption types and sectors.</p>
                          <p className="mt-2">Select one adoption type in the settings panel to switch the columns from adoption types to years.</p>
                        </>
                      : <>
                          <p>This filtered view shows the share of reports in each sector-year mentioning one adoption type.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                          <p className="mt-2">Clearing the filter returns the categorical sector view.</p>
                        </>
                  }
                  xAxisLabel={adoptionFilter === 'all' ? 'Adoption Type' : 'Year'}
                  yAxisLabel={adoptionSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
                  rowGroups={adoptionSectorView === 'isic' ? isicRowGroups : undefined}
                  expandedRowGroups={adoptionSectorView === 'isic' ? expandedIsicGroups : undefined}
                  onToggleRowGroup={adoptionSectorView === 'isic' ? toggleIsicGroup : undefined}
                  labelColumnWidth={adoptionSectorView === 'isic' ? 330 : undefined}
                  rowHeight={adoptionSectorView === 'isic' ? 58 : undefined}
                  yLabelClassName={adoptionSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
                  headerExtra={settingsPanelToggle}
                  footerExtra={visualizationFooter}
                />
                {visualizationActions}
              </div>
            )}

            {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}

            {infoPanelSection}
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`vendor-trend-${trendTimeAxis}`}
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayVendorTrend}
                  xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
                  stackKeys={effectiveVendorFilter === 'all' ? vendorStackKeys : [effectiveVendorFilter]}
                  colors={vendorColors}
                  yAxisTickFormatter={stackedChartYAxisFormatter}
                  tooltipValueFormatter={stackedChartTooltipFormatter}
                  yAxisDomain={isReportShareMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={settingsPanelToggle}
                  legendKeys={vendorStackKeys}
                  activeLegendKey={effectiveVendorFilter === 'all' ? null : effectiveVendorFilter}
                  onLegendItemClick={(key) => setVendorFilter(prev => (prev === key ? 'all' : key))}
                  footerExtra={visualizationFooter}
                  title="AI Vendors Mentioned Over Time"
                  subtitle={
                    isReportShareMode
                      ? `Stacked bar chart showing the percentage of annual reports referencing each AI vendor or provider tag (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single report may reference multiple vendors, so stacked totals can exceed 100%.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} referencing each AI vendor or provider tag (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'passage'} may reference multiple vendors. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    <>
                      <p>{isReportShareMode ? 'Each segment shows the share of reports in that period tagged with the vendor reference.' : 'Each bar is stacked by vendor tag (OpenAI, Microsoft, Google, Amazon / AWS, Meta, Anthropic, Internal, Other).'}</p>
                      <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can include multiple vendor tags, so one item may contribute to more than one segment.</p>
                      <p className="mt-2"><span className="font-medium">Internal</span> means in-house AI development.</p>
                      <p className="mt-2">Click a legend item to isolate one vendor tag.</p>
                    </>
                  }
                />
                {visualizationActions}
              </div>
            ) : (
              <div className="space-y-4">
                <GenericHeatmap
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={visibleVendorHeatmapData}
                  xLabels={vendorHeatmapXLabels}
                  yLabels={vendorHeatmapYLabels}
                  baseColor={vendorHeatmapBaseColor}
                  valueFormatter={riskHeatmapValueFormatter}
                  xLabelFormatter={formatLabel}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={true}
                  title={effectiveVendorFilter === 'all' ? 'Vendor Concentration by Sector' : `${formatLabel(effectiveVendorFilter)} Mentions by Sector and Year`}
                  subtitle={
                    effectiveVendorFilter === 'all'
                      ? `Heatmap of report-share percentages by ${vendorSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and vendor tag (columns). Each cell shows the percentage of reports in that sector, across the selected years, naming the vendor tag at least once.`
                      : `Heatmap of ${formatLabel(effectiveVendorFilter)} report-share percentages by ${vendorSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(effectiveVendorFilter)}.`
                  }
                  tooltip={
                    effectiveVendorFilter === 'all'
                      ? <>
                          <p>Each cell is normalised by the total number of reports in that sector across the selected years.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed vendor tags and sectors.</p>
                          <p className="mt-2">Select one vendor in the settings panel to switch columns from vendor tags to years.</p>
                        </>
                      : <>
                          <p>This filtered view shows the share of reports in each sector-year mentioning one vendor tag.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                          <p className="mt-2">Clear the filter to return to the all-vendor categorical view.</p>
                        </>
                  }
                  xAxisLabel={effectiveVendorFilter === 'all' ? 'Vendor' : 'Year'}
                  yAxisLabel={vendorSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
                  rowGroups={vendorSectorView === 'isic' ? isicRowGroups : undefined}
                  expandedRowGroups={vendorSectorView === 'isic' ? expandedIsicGroups : undefined}
                  onToggleRowGroup={vendorSectorView === 'isic' ? toggleIsicGroup : undefined}
                  labelColumnWidth={vendorSectorView === 'isic' ? 330 : undefined}
                  rowHeight={vendorSectorView === 'isic' ? 58 : undefined}
                  yLabelClassName={vendorSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
                  headerExtra={settingsPanelToggle}
                  footerExtra={visualizationFooter}
                />
                {visualizationActions}
              </div>
            )}

            {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}

            {infoPanelSection}
          </div>
        )}

        {activeView === 4 && (
          <div className="space-y-8">
            <div className="space-y-4">
              <GenericHeatmap
                exportRef={visualizationExportRef}
                exportMode={isExportingVisualization}
                exportWatermark={exportWatermark}
                data={selectedSignalQualityHeatmap.data}
                xLabels={filteredYears}
                yLabels={selectedSignalQualityHeatmap.yLabels}
                baseColor={selectedSignalQualityHeatmap.baseColor}
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title={selectedSignalQualityHeatmap.title}
                subtitle={selectedSignalQualityHeatmap.subtitle}
                tooltip={selectedSignalQualityHeatmap.tooltip}
                xAxisLabel="Year"
                yAxisLabel={selectedSignalQualityHeatmap.yAxisLabel}
                compact={selectedSignalQualityHeatmap.compact}
                headerExtra={settingsPanelToggle}
                footerExtra={visualizationFooter}
              />
              {visualizationActions}
            </div>

            {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}
            {infoPanelSection}
          </div>
        )}

        {activeView === 5 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`blind-spot-trend-${trendTimeAxis}`}
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayBlindSpotTrend}
                  xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
                  stackKeys={blindSpotFilter === 'all' ? ['no_ai_mention', 'no_ai_risk_mention'] : [blindSpotFilter]}
                  colors={blindSpotColors}
                  yAxisTickFormatter={stackedChartYAxisFormatter}
                  tooltipValueFormatter={stackedChartTooltipFormatter}
                  yAxisDomain={isReportShareMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={settingsPanelToggle}
                  legendKeys={['no_ai_mention', 'no_ai_risk_mention']}
                  activeLegendKey={blindSpotFilter === 'all' ? null : blindSpotFilter}
                  onLegendItemClick={(key) =>
                    setBlindSpotFilter(prev => (prev === key ? 'all' : (key as BlindSpotFilter)))
                  }
                  footerExtra={visualizationFooter}
                  title={trendTimeAxis === 'month' ? 'Blind Spots by Month' : 'Blind Spots by Year'}
                  subtitle={
                    isReportShareMode
                      ? blindSpotFilter === 'all'
                        ? `Stacked bar chart showing the percentage of annual reports with no AI mention (red) and no AI risk mention (amber) on the y-axis, plotted across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} on the x-axis. The two series are not mutually exclusive, so stacked totals can exceed 100%.`
                        : `Bar chart showing the percentage of annual reports classified as ${formatLabel(blindSpotFilter)} (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis).`
                      : blindSpotFilter === 'all'
                        ? `Stacked bar chart showing the number of annual reports with no AI mention (red) and no AI risk mention (amber) on the y-axis, plotted across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} on the x-axis. The two series are not mutually exclusive: a report with no AI mention is also counted under no AI risk mention. The y-axis scale adjusts dynamically to the data shown.`
                        : `Bar chart showing the number of annual reports classified as ${formatLabel(blindSpotFilter)} (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    blindSpotFilter === 'all'
                      ? isReportShareMode
                        ? 'This report-level view compares the share of reports exhibiting two absence patterns: no AI mention at all vs. no AI risk mention. Use the legend or the settings panel to isolate one series.'
                        : 'This report-level view compares two absence patterns: no AI mention at all vs. no AI risk mention. Use the legend or the settings panel to isolate one series.'
                      : `${formatLabel(blindSpotFilter)} is currently isolated. Clear the filter to compare both blind-spot types together.`
                  }
                />
                {visualizationActions}
              </div>
            ) : (
              <div className="space-y-4">
                <GenericHeatmap
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayBlindSpotHeatmapData}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor={blindSpotHeatmapColor}
                  valueFormatter={blindSpotHeatmapValueFormatter}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={false}
                  title={blindSpotHeatmapTitle}
                  subtitle={blindSpotHeatmapSubtitle}
                  tooltip={blindSpotHeatmapTooltip}
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                  headerExtra={settingsPanelToggle}
                  footerExtra={visualizationFooter}
                />
                {visualizationActions}
              </div>
            )}

            {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}

            {infoPanelSection}
          </div>
        )}
          </div>

          {isSettingsOpen && <aside className="hidden self-start min-[960px]:block">{renderSettingsPanel()}</aside>}
        </div>
      </main>
    </div>
  );
}
