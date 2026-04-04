'use client';

import { type ReactNode, useMemo, useState } from 'react';
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

type RiskInfoPanelKey = 'definitions' | 'method' | 'cite' | 'download';

type RiskInfoPanelItem = {
  value: RiskInfoPanelKey;
  label: string;
  title: string;
  content: ReactNode;
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
  google: '#f59e0b',      // amber-500
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
  information_integrity:    '#f87171', // red-rose
  workforce_impacts:        '#ef4444', // red
  environmental_impact:     '#b91c1c', // deep red
  national_security:        '#7f1d1d', // darkest red
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
    label: 'OpenAI / Microsoft / Google',
    definition: 'The provider is explicitly named in the source disclosure.',
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

export default function DashboardClient({ data }: { data: GoldenDashboardData }) {
  const [activeView, setActiveView] = useState(1);
  const [visualizationMode, setVisualizationMode] = useState<'chart' | 'heatmap'>('chart');
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('perReport');
  const [trendTimeAxis, setTrendTimeAxis] = useState<TrendTimeAxis>('year');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [adoptionFilter, setAdoptionFilter] = useState<AdoptionFilter>('all');
  const [riskSectorView, setRiskSectorView] = useState<RiskSectorView>('cni');
  const [adoptionSectorView, setAdoptionSectorView] = useState<RiskSectorView>('cni');
  const [vendorSectorView, setVendorSectorView] = useState<RiskSectorView>('cni');
  const [vendorFilter, setVendorFilter] = useState<string>('all');
  const [infoPanelSelections, setInfoPanelSelections] = useState<Record<number, RiskInfoPanelKey>>({
    1: 'definitions',
    2: 'definitions',
    3: 'definitions',
    4: 'definitions',
    5: 'definitions',
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
  const [shareButtonLabel, setShareButtonLabel] = useState('Share');

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
    () => buildIsicSectorGroups(visibleIsicSectorLabels),
    [visibleIsicSectorLabels]
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

  const blindSpotTotalsInRange = useMemo(
    () =>
      blindSpotTrendInRange.reduce(
        (acc, row) => {
          acc.totalReports += Number(row.total_reports) || 0;
          acc.noAiMention += Number(row.no_ai_mention) || 0;
          acc.noAiRiskMention += Number(row.no_ai_risk_mention) || 0;
          return acc;
        },
        { totalReports: 0, noAiMention: 0, noAiRiskMention: 0 }
      ),
    [blindSpotTrendInRange]
  );

  const blindSpotOverviewStats = useMemo(() => {
    const { totalReports, noAiMention, noAiRiskMention } = blindSpotTotalsInRange;
    const safePct = (value: number) =>
      totalReports > 0 ? (value / totalReports) * 100 : 0;

    return {
      totalReports,
      noAiMention,
      noAiRiskMention,
      noAiMentionPct: safePct(noAiMention),
      noAiRiskMentionPct: safePct(noAiRiskMention),
    };
  }, [blindSpotTotalsInRange]);

  const riskOverviewStats = useMemo(() => {
    const reportTotals = blindSpotTrendInRange.reduce(
      (acc, row) => {
        acc.totalReports += Number(row.total_reports) || 0;
        acc.riskMentionReports += Number(row.ai_risk_mention) || 0;
        return acc;
      },
      { totalReports: 0, riskMentionReports: 0 }
    );

    const excerptRiskMentions = resolvedDatasets.perChunk.mentionTrend.reduce((sum, row) => {
      const year = Number(row.year);
      if (year < selectedStartYear || year > selectedEndYear) return sum;
      return sum + (Number(row.risk) || 0);
    }, 0);

    return {
      totalReports: reportTotals.totalReports,
      riskMentionReports: reportTotals.riskMentionReports,
      excerptRiskMentions,
      selectedYearCount: filteredYears.length,
    };
  }, [
    blindSpotTrendInRange,
    resolvedDatasets.perChunk.mentionTrend,
    selectedStartYear,
    selectedEndYear,
    filteredYears.length,
  ]);

  const adoptionOverviewStats = useMemo(() => {
    const totalReports = blindSpotTrendInRange.reduce(
      (sum, row) => sum + (Number(row.total_reports) || 0),
      0
    );

    const adoptionMentionReports = reportBaselineData.mentionTrend.reduce((sum, row) => {
      const year = Number(row.year);
      if (year < selectedStartYear || year > selectedEndYear) return sum;
      return sum + (Number(row.adoption) || 0);
    }, 0);

    const excerptAdoptionMentions = resolvedDatasets.perChunk.mentionTrend.reduce((sum, row) => {
      const year = Number(row.year);
      if (year < selectedStartYear || year > selectedEndYear) return sum;
      return sum + (Number(row.adoption) || 0);
    }, 0);

    return {
      totalReports,
      adoptionMentionReports,
      excerptAdoptionMentions,
      selectedYearCount: filteredYears.length,
    };
  }, [
    blindSpotTrendInRange,
    reportBaselineData.mentionTrend,
    resolvedDatasets.perChunk.mentionTrend,
    selectedStartYear,
    selectedEndYear,
    filteredYears.length,
  ]);

  const vendorOverviewStats = useMemo(() => {
    const totalReports = blindSpotTrendInRange.reduce(
      (sum, row) => sum + (Number(row.total_reports) || 0),
      0
    );

    const vendorMentionReports = reportBaselineData.mentionTrend.reduce((sum, row) => {
      const year = Number(row.year);
      if (year < selectedStartYear || year > selectedEndYear) return sum;
      return sum + (Number(row.vendor) || 0);
    }, 0);

    const excerptVendorMentions = resolvedDatasets.perChunk.mentionTrend.reduce((sum, row) => {
      const year = Number(row.year);
      if (year < selectedStartYear || year > selectedEndYear) return sum;
      return sum + (Number(row.vendor) || 0);
    }, 0);

    return {
      totalReports,
      vendorMentionReports,
      excerptVendorMentions,
      selectedYearCount: filteredYears.length,
    };
  }, [
    blindSpotTrendInRange,
    reportBaselineData.mentionTrend,
    resolvedDatasets.perChunk.mentionTrend,
    selectedStartYear,
    selectedEndYear,
    filteredYears.length,
  ]);

  const signalQualityOverviewStats = useMemo(() => {
    const sumValues = (rows: { value: number }[]) =>
      rows.reduce((sum, row) => sum + (Number(row.value) || 0), 0);

    const totalReports = blindSpotTrendInRange.reduce(
      (sum, row) => sum + (Number(row.total_reports) || 0),
      0
    );

    const riskSignalTotal = sumValues(riskSignalHeatmapInRange);
    const adoptionSignalTotal = sumValues(adoptionSignalHeatmapInRange);
    const vendorSignalTotal = sumValues(vendorSignalHeatmapInRange);
    const substantivenessTotal = sumValues(substantivenessHeatmapInRange);

    const explicitRisk = riskSignalHeatmapInRange
      .filter(row => row.y === '3-explicit')
      .reduce((sum, row) => sum + (Number(row.value) || 0), 0);
    const substantiveRisk = substantivenessHeatmapInRange
      .filter(row => row.y === 'substantive')
      .reduce((sum, row) => sum + (Number(row.value) || 0), 0);

    return {
      totalReports,
      riskSignalTotal,
      adoptionSignalTotal,
      vendorSignalTotal,
      substantivenessTotal,
      explicitRisk,
      substantiveRisk,
    };
  }, [
    blindSpotTrendInRange,
    riskSignalHeatmapInRange,
    adoptionSignalHeatmapInRange,
    vendorSignalHeatmapInRange,
    substantivenessHeatmapInRange,
  ]);

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

  const handleShareVisualization = async () => {
    const shareUrl = window.location.href;
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

  const makeSectorToggle = (
    current: RiskSectorView,
    setter: (v: RiskSectorView) => void
  ) => (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setter('cni')}
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
          trendTimeAxis === 'month'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Month
      </button>
    </div>
  );

  const metricModeToggle = canShowReportShare ? (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setMetricMode('count')}
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
          effectiveMetricMode === 'count'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Count
      </button>
      <button
        type="button"
        onClick={() => setMetricMode('pct_reports')}
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
          effectiveMetricMode === 'pct_reports'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        % of Reports
      </button>
    </div>
  ) : null;

  const showVisualizationToggle = activeView !== 4;
  const canUseLineChart = visualizationMode === 'chart' && activeView !== 4 && trendTimeAxis === 'year';
  const activeChartDisplayType = canUseLineChart ? chartDisplayType : 'bar';

  const chartTypeToggle = (
    <div className="inline-flex items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setChartDisplayType('bar')}
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
          activeChartDisplayType === 'line'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        } ${!canUseLineChart ? 'cursor-not-allowed opacity-40 hover:bg-white' : ''}`}
      >
        Line
      </button>
    </div>
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

  const signalQualityModeToggle = (
    <div className="inline-flex shrink-0 items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setSignalQualityMode('explicitness')}
        className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition ${
          signalQualityMode === 'substantiveness'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Substantiveness
      </button>
    </div>
  );

  const sharedCitationItem: RiskInfoPanelItem = {
    value: 'cite',
    label: 'Cite',
    title: `How To Cite The ${view.title} View`,
    content: (
      <div className="space-y-3 text-sm leading-relaxed text-slate-600">
        <p>
          When citing a specific dashboard view, include the active visualization title and the date you accessed it.
        </p>
        <p className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 font-mono text-xs text-slate-700">
          AI Risk Observatory. &quot;{currentVisualizationExport.title}&quot; dashboard view, UK annual-report AI disclosure dataset, accessed [insert date].
        </p>
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
          Export the current visualization as CSV, or download the full dashboard annotations dataset for offline use.
        </p>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleDownloadVisualization}
            className="inline-flex items-center justify-center rounded border border-slate-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-900 transition-colors hover:border-slate-900 hover:bg-slate-50"
          >
            Download This View
          </button>
          <a
            href="/api/download-data"
            download
            className="inline-flex items-center justify-center rounded border border-slate-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-900 transition-colors hover:border-slate-900 hover:bg-slate-50"
          >
            Download Dataset
          </a>
          <a
            href="/about"
            className="inline-flex items-center justify-center border border-transparent px-1 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-accent underline underline-offset-4 transition-colors hover:text-primary"
          >
            Read Methodology
          </a>
        </div>
      </div>
    ),
  };

  const riskInfoPanelItems: RiskInfoPanelItem[] = [
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
    {
      value: 'method',
      label: 'Method',
      title: 'How Risk Categories Are Assigned',
      content: (
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Risk categories are assigned by an LLM-assisted classifier using a structured AI-risk taxonomy. Labels are
            applied at excerpt level and then rolled up into report-level and sector-level views.
          </p>
          <p>
            Categories are not mutually exclusive. One disclosure can be counted in multiple risk categories if it
            covers more than one mechanism of risk.
          </p>
        </div>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
  ];

  const adoptionInfoPanelItems: RiskInfoPanelItem[] = [
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
    {
      value: 'method',
      label: 'Method',
      title: 'How Adoption Mentions Are Classified',
      content: (
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            The adoption view groups disclosures by implementation maturity: non-LLM AI, LLM-based use, and more
            agentic systems. Classification is applied at excerpt level and can retain more than one adoption tag.
          </p>
          <p>
            The chart view shows how often those adoption patterns appear over time; the heatmap shows where they are
            concentrated across sectors.
          </p>
        </div>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
  ];

  const vendorInfoPanelItems: RiskInfoPanelItem[] = [
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
    {
      value: 'method',
      label: 'Method',
      title: 'How Vendor Mentions Are Tagged',
      content: (
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Vendor tags are assigned when a disclosure explicitly names a provider or clearly indicates in-house AI
            development. A single excerpt can carry multiple vendor tags if more than one provider is referenced.
          </p>
          <p>
            The trend view surfaces changes in provider mention frequency over time, while the heatmap shows which
            sectors most often cite each provider.
          </p>
        </div>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
  ];

  const signalQualityInfoPanelItems: RiskInfoPanelItem[] = [
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
    {
      value: 'method',
      label: 'Method',
      title: 'How Quality Metrics Are Calculated',
      content: (
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Signal-strength views count label-level outcomes and keep the strongest available evidence tier for each
            classified label. Substantiveness is a report-level quality assessment for AI-risk disclosure.
          </p>
          <p>
            The settings panel lets you switch between risk, adoption, and vendor signal strength, or move to
            substantiveness as a separate quality lens.
          </p>
        </div>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
  ];

  const blindSpotInfoPanelItems: RiskInfoPanelItem[] = [
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
    {
      value: 'method',
      label: 'Method',
      title: 'How Blind Spots Are Computed',
      content: (
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Blind spots are measured on a report-year basis. This makes it possible to distinguish complete AI silence
            from cases where AI is discussed but AI risk is not.
          </p>
          <p>
            In chart mode, the series show change over time. In heatmap mode, the same absence patterns are mapped
            across sectors and years.
          </p>
        </div>
      ),
    },
    sharedCitationItem,
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

  const selectedInfoPanelKey = infoPanelSelections[activeView] ?? 'definitions';
  const selectedInfoPanel =
    activeInfoPanelItems.find(item => item.value === selectedInfoPanelKey) ?? activeInfoPanelItems[0];

  const infoPanelSection = (
    <section className="overflow-hidden rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
      <div className="grid lg:grid-cols-[220px_minmax(0,1fr)]">
        <div className="border-b border-slate-100 bg-slate-50/80 p-5 lg:border-b-0 lg:border-r">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
            Reference
          </p>
          <div className="mt-4 space-y-1.5">
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
                className={`flex w-full items-center justify-between rounded border px-3 py-2 text-left text-[11px] font-bold uppercase tracking-[0.14em] transition ${
                  selectedInfoPanelKey === item.value
                    ? 'border-primary bg-primary text-white'
                    : 'border-transparent bg-white text-slate-600 hover:border-slate-200 hover:bg-slate-100'
                }`}
              >
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="p-6">
          <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
            {view.title}
          </p>
          <h3 className="mt-2 text-base font-semibold text-slate-900">{selectedInfoPanel.title}</h3>
          <div className="mt-4">{selectedInfoPanel.content}</div>
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
        className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest transition ${
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
          onClick={handleDownloadVisualization}
          className="inline-flex h-9 items-center justify-center rounded border border-border bg-white px-3 text-[10px] font-bold uppercase tracking-widest text-primary transition hover:bg-secondary"
          title="Download current visualization data as CSV"
        >
          Download
        </button>
        <button
          type="button"
          onClick={handleShareVisualization}
          className="inline-flex h-9 items-center justify-center rounded border border-border bg-white px-3 text-[10px] font-bold uppercase tracking-widest text-primary transition hover:bg-secondary"
          title="Share the current dashboard page"
        >
          {shareButtonLabel}
        </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-white text-primary">
      <main className="mx-auto max-w-[1320px] px-6 py-10 sm:py-14">
        <div className="border-b border-border pb-7">
          <div className="mb-5 text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
            {dashboardUpdatedLabel}
          </div>

          <div>
            <h2 className="aisi-h2 uppercase">
              <span className="mr-3 inline-block h-6 w-1.5 bg-accent align-middle" />
              {view.heading}
            </h2>
            <p className={`mt-3 max-w-3xl text-sm leading-relaxed text-muted sm:text-base ${
              activeView === 5 ? 'xl:whitespace-nowrap' : ''
            }`}>
              {view.description}
            </p>
          </div>

          <div className="mt-8 flex items-start justify-between gap-3 overflow-x-auto">
            <div className="flex shrink-0 items-center gap-2.5">
              {VIEWS.map(item => (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveView(item.id);
                    setVisualizationMode(item.id === 4 ? 'heatmap' : 'chart');
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                  className={`rounded border px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                    activeView === item.id
                      ? 'border-primary bg-primary text-white'
                      : 'border-border bg-secondary text-muted-foreground hover:bg-white hover:text-primary'
                  }`}
                >
                  {item.title}
                </button>
              ))}
            </div>

            {showVisualizationToggle && (
              <div className="inline-flex shrink-0 items-center overflow-hidden rounded border border-border bg-white p-1">
                <button
                  type="button"
                  onClick={() => setVisualizationMode('chart')}
                  className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition ${
                    visualizationMode === 'chart'
                      ? 'bg-primary text-white'
                      : 'text-muted-foreground hover:bg-secondary'
                  }`}
                >
                  Chart
                </button>
                <button
                  type="button"
                  onClick={() => setVisualizationMode('heatmap')}
                  className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition ${
                    visualizationMode === 'heatmap'
                      ? 'bg-primary text-white'
                      : 'text-muted-foreground hover:bg-secondary'
                  }`}
                >
                  Sector Heatmap
                </button>
              </div>
            )}
          </div>
        </div>

        <div
          className={`mt-8 grid gap-8 xl:items-start ${
            isSettingsOpen
              ? 'xl:grid-cols-[minmax(0,1fr)_280px]'
              : 'xl:grid-cols-1'
          }`}
        >
          <div className="min-w-0">

        {activeView === 1 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`risk-trend-${trendTimeAxis}`}
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

            <div className="space-y-4 text-base leading-relaxed text-muted sm:text-lg font-medium">
              <p>
                <span className="font-bold text-primary">{formatNumber(riskOverviewStats.riskMentionReports)}</span>{' '}
                of the{' '}
                <span className="font-bold text-primary">{formatNumber(riskOverviewStats.totalReports)}</span>{' '}
                annual reports examined between{' '}
                <span className="font-bold text-primary">{riskSelectedYearSpan}</span>{' '}
                include at least one AI risk disclosure, across{' '}
                <span className="font-bold text-primary">{formatNumber(riskOverviewStats.excerptRiskMentions)}</span>{' '}
                individual passages.
              </p>
              <p>
                <span className="font-medium text-slate-800">Per Report</span> counts each company filing once —
                useful for measuring how broadly a risk is disclosed across the market.{' '}
                <span className="font-medium text-slate-800">Per Excerpt</span> counts every individual passage where
                a risk from AI appears — useful for visualising the depth of emphasis companies place on that risk.
              </p>
              <div>
                Use the date range slider below the visualization together with the settings panel to focus on a
                specific time window, isolate a risk category, or switch the sector taxonomy between{' '}
                <span className="font-medium text-slate-800">CNI</span> (Critical National Infrastructure) and{' '}
                <span className="font-medium text-slate-800">ISIC</span> (international standard industry codes).
                <InfoTooltip content="Sector classifications for companies that do not fall clearly within a CNI sector have been approximated using an LLM-assisted mapping process. Full details are available on the About page." />
              </div>
              <p>
                The <span className="font-medium text-slate-800">risk trend chart</span> shows how often each risk
                category has been mentioned, year by year. The <span className="font-medium text-slate-800">heatmap</span>{' '}
                shows where those mentions concentrate across sectors. With all risk types selected, heatmap columns
                represent risk categories; selecting a single category switches the columns to years, letting you track
                how that risk has evolved within each sector over time.
              </p>
            </div>

            {infoPanelSection}
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`adoption-trend-${trendTimeAxis}`}
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

            <div className="space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
              <p>
                <span className="font-semibold text-slate-900">{formatNumber(adoptionOverviewStats.adoptionMentionReports)}</span>{' '}
                of the{' '}
                <span className="font-semibold text-slate-900">{formatNumber(adoptionOverviewStats.totalReports)}</span>{' '}
                annual reports examined between{' '}
                <span className="font-semibold text-slate-900">{riskSelectedYearSpan}</span>{' '}
                include at least one AI adoption disclosure, across{' '}
                <span className="font-semibold text-slate-900">{formatNumber(adoptionOverviewStats.excerptAdoptionMentions)}</span>{' '}
                individual passages.
              </p>
              <p>
                <span className="font-medium text-slate-800">Per Report</span> counts whether each company-year filing
                discloses AI adoption at least once.{' '}
                <span className="font-medium text-slate-800">Per Excerpt</span> counts every individual passage classified
                as adoption, giving a more granular view of how heavily companies discuss implementation.
              </p>
              <p>
                Use the date range slider below the visualization and the settings panel to focus on a single adoption type (
                <span className="font-medium text-slate-800">Non-LLM</span>,{' '}
                <span className="font-medium text-slate-800">LLM</span>, or{' '}
                <span className="font-medium text-slate-800">Agentic</span>). Selecting one type switches the heatmap
                from a categorical view (adoption types) to a time view (years), so you can track how that adoption mode
                evolves within each sector.
              </p>
              <p>
                The trend chart highlights maturity shifts over time, while the heatmap shows where each adoption pattern
                is concentrated across sectors.
              </p>
            </div>

            {infoPanelSection}
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`vendor-trend-${trendTimeAxis}`}
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
                      <p>{isReportShareMode ? 'Each segment shows the share of reports in that period tagged with the vendor reference.' : 'Each bar is stacked by vendor tag (OpenAI, Microsoft, Google, Internal, Other).'}</p>
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

            <div className="space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
              <p>
                Of the{' '}
                <span className="font-semibold text-slate-900">{formatNumber(vendorOverviewStats.totalReports)}</span>{' '}
                annual reports examined across{' '}
                <span className="font-semibold text-slate-900">{riskSelectedYearSpan}</span>,{' '}
                <span className="font-semibold text-slate-900">{formatNumber(vendorOverviewStats.vendorMentionReports)}</span>{' '}
                contain at least one AI vendor reference, spanning{' '}
                <span className="font-semibold text-slate-900">{formatNumber(vendorOverviewStats.excerptVendorMentions)}</span>{' '}
                individual passages.
              </p>
              <p>
                <span className="font-medium text-slate-800">Per Report</span> shows how many filings mention a given
                vendor at least once; <span className="font-medium text-slate-800">Per Excerpt</span> shows the full
                volume of vendor-tagged passages across all reports and therefore the depth of mention.
              </p>
              <p>
                Use the settings panel to focus on one vendor tag (
                <span className="font-medium text-slate-800">OpenAI</span>,{' '}
                <span className="font-medium text-slate-800">Microsoft</span>,{' '}
                <span className="font-medium text-slate-800">Google</span>,{' '}
                <span className="font-medium text-slate-800">Internal</span>,{' '}
                <span className="font-medium text-slate-800">Other</span>). Selecting one vendor switches the heatmap
                from categorical columns (vendor tags) to yearly columns.
              </p>
              <p>
                Read the trend chart for time patterns and the heatmap for sector concentration. Together they show both
                which vendors are cited and where those dependencies appear most strongly.
              </p>
            </div>

            {infoPanelSection}
          </div>
        )}

        {activeView === 4 && (
          <div className="space-y-8">
            <div className="space-y-4">
              <GenericHeatmap
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
            <div className="space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
              <p>
                This section evaluates disclosure quality, not just volume. Across{' '}
                <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.totalReports)}</span>{' '}
                reports in <span className="font-semibold text-slate-900">{riskSelectedYearSpan}</span>, the dataset
                contains <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.riskSignalTotal)}</span>{' '}
                risk-signal labels, <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.adoptionSignalTotal)}</span>{' '}
                adoption-signal labels, and <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.vendorSignalTotal)}</span>{' '}
                vendor-signal labels.
              </p>
              <p>
                Signal strength asks how explicitly a disclosure supports its classification: weak implicit, strong
                implicit, or explicit. In the current window,{' '}
                <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.explicitRisk)}</span>{' '}
                risk labels are scored as explicit.
              </p>
              <p>
                Substantiveness is a separate quality lens for risk language, measuring whether disclosures are
                boilerplate, moderate, or substantive. In this range,{' '}
                <span className="font-semibold text-slate-900">{formatNumber(signalQualityOverviewStats.substantiveRisk)}</span>{' '}
                reports are classified as substantive.
              </p>
              <p>
                Use the settings panel to switch between AI risk signal strength, AI adoption signal strength,
                AI vendor signal strength, and risk substantiveness within the same view.
              </p>
            </div>
            {infoPanelSection}
          </div>
        )}

        {activeView === 5 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              <div className="space-y-4">
                <StackedBarChart
                  key={`blind-spot-trend-${trendTimeAxis}`}
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

            <div className="space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
              <p>
                In the selected period (<span className="font-semibold text-slate-900">{riskSelectedYearSpan}</span>),
                the dataset includes <span className="font-semibold text-slate-900">{formatNumber(blindSpotOverviewStats.totalReports)}</span>{' '}
                report-year filings. Of those,{' '}
                <span className="font-semibold text-slate-900">{formatNumber(blindSpotOverviewStats.noAiMention)}</span>{' '}
                (<span className="font-semibold text-slate-900">{blindSpotOverviewStats.noAiMentionPct.toFixed(1)}%</span>)
                contain no AI mention at all, and{' '}
                <span className="font-semibold text-slate-900">{formatNumber(blindSpotOverviewStats.noAiRiskMention)}</span>{' '}
                (<span className="font-semibold text-slate-900">{blindSpotOverviewStats.noAiRiskMentionPct.toFixed(1)}%</span>)
                contain no AI risk mention.
              </p>
              <p>
                This view helps distinguish two different gaps: complete AI silence vs. AI being discussed without any
                associated risk disclosure.
              </p>
              <p>
                Use the settings panel to focus the trend chart on one blind-spot type or switch the heatmap between
                {' '}<span className="font-medium text-slate-800">No AI Mention</span> and{' '}
                <span className="font-medium text-slate-800">No AI Risk Mention</span>.
              </p>
            </div>

            {infoPanelSection}
          </div>
        )}
          </div>

          {isSettingsOpen && (
            <aside className="self-start">
              <div className="overflow-hidden rounded border border-border bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
                <div className="border-b border-border px-5 py-4">
                  <h3 className="text-[11px] font-bold uppercase tracking-[0.18em] text-primary">Settings</h3>
                </div>
                <div className="p-5 [&>*+*]:border-t [&>*+*]:border-border [&>*+*]:pt-5">
                {activeView !== 5 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Dataset
                    </p>
                    {datasetToggle}
                  </div>
                )}

                <div className="space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                    Coverage
                  </p>
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
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Chart Settings
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {chartTypeToggle}
                      {trendTimeToggle}
                      {metricModeToggle}
                    </div>
                  </div>
                )}

                {visualizationMode === 'heatmap' && activeView === 1 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Sector Taxonomy
                    </p>
                    {makeSectorToggle(riskSectorView, setRiskSectorView)}
                  </div>
                )}

                {visualizationMode === 'heatmap' && activeView === 2 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Sector Taxonomy
                    </p>
                    {makeSectorToggle(adoptionSectorView, setAdoptionSectorView)}
                  </div>
                )}

                {visualizationMode === 'heatmap' && activeView === 3 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Sector Taxonomy
                    </p>
                    {makeSectorToggle(vendorSectorView, setVendorSectorView)}
                  </div>
                )}

                {activeView === 1 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Risk Filter
                    </p>
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
                        className="w-full rounded border border-border bg-white px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground transition hover:bg-secondary"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                )}

                {activeView === 2 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Adoption Filter
                    </p>
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
                        className="w-full rounded border border-border bg-white px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground transition hover:bg-secondary"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                )}

	                {activeView === 3 && (
	                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Vendor Filter
                    </p>
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
                        className="w-full rounded border border-border bg-white px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground transition hover:bg-secondary"
                      >
                        Clear
                      </button>
                    )}
	                  </div>
	                )}

	                {activeView === 4 && (
	                  <div className="space-y-3">
	                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
	                      Quality View
	                    </p>
	                    {signalQualityModeToggle}
                      {signalQualityMode === 'explicitness' && (
                        <div className="space-y-3">
                          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                            Signal Metric
                          </p>
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
	                  </div>
	                )}

	                {activeView === 5 && visualizationMode === 'chart' && (
	                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Blind Spot Filter
                    </p>
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
                        className="w-full rounded border border-border bg-white px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground transition hover:bg-secondary"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                )}

                {activeView === 5 && visualizationMode === 'heatmap' && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                      Heatmap View
                    </p>
                    {blindSpotHeatmapToggle}
                  </div>
                )}
                </div>
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
