'use client';

import { toPng } from 'html-to-image';
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { GenericHeatmap, StackedBarChart, InfoTooltip } from '@/components/overview-charts';
import { CollapsibleSection } from '@/components/collapsible-section';
import { buildIsicSectorGroups, type HeatmapRowGroup } from '@/lib/isic';
import type { GoldenDashboardData, GoldenDataset } from '@/lib/golden-set';

// Filter type for risk distribution view
type RiskFilter = 'all' | string;
type AdoptionFilter = 'all' | string;

type View = {
  id: number;
  title: string;
  heading: string;
  description: string;
};

type RiskInfoPanelKey = 'definitions' | 'cite' | 'download' | 'faq';

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
    description: 'AI adoption type (Traditional AI (non-LLM), LLM, agentic) across sectors and over time.',
  },
  {
    id: 3,
    title: 'Vendors',
    heading: 'AI Vendor References',
    description: 'Which technology vendors companies name in their reports, and how that varies by sector.',
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

const phase1RiskColors: Record<string, string> = {
  risk: '#e63946',
};

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
    label: 'Traditional AI (non-LLM)',
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
    definition: 'How clearly AI is mentioned in the disclosure text — ranging from explicit statements to weak or indirect references.',
  },
  {
    label: 'Substantiveness',
    definition: 'How concrete and detailed AI-risk disclosure is at report level: substantive, moderate, or boilerplate.',
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
const inlineClearButtonClass = 'inline-flex h-10 shrink-0 items-center justify-center rounded border border-slate-950 bg-slate-950 px-3 text-[11px] font-semibold tracking-normal text-white transition hover:bg-slate-800';
const actionButtonClass = 'inline-flex h-9 items-center justify-center gap-2 rounded border border-border bg-white px-3 text-[11px] font-semibold tracking-normal text-primary transition hover:bg-secondary';
const subtleActionButtonClass = 'inline-flex h-9 items-center justify-center gap-2 rounded bg-white px-3 text-[11px] font-semibold tracking-normal text-primary transition hover:bg-secondary';
const infoTabButtonClass = 'flex items-center rounded border px-3 py-2.5 text-left text-xs font-semibold tracking-normal normal-case transition-colors lg:w-full';
const inlinePanelButtonClass = 'inline-flex items-center justify-center rounded border border-slate-300 bg-white px-4 py-2 text-xs font-semibold tracking-normal text-slate-900 transition-colors hover:border-slate-900 hover:bg-slate-50';

const formatNumber = (value: number) =>
  new Intl.NumberFormat('en-GB').format(value);

const formatLabel = (val: string | number) => {
  if (typeof val === 'number') return val.toString();
  const overrides: Record<string, string> = {
    llm: 'LLM',
    non_llm: 'Traditional AI (non-LLM)',
    risk: 'AI Risk Mentioned',
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
type RiskBreakdownMode = 'categories' | 'phase1';
type RiskSectorView = 'cni' | 'isic';
type SignalQualityMode = 'explicitness' | 'substantiveness';
type MetricMode = 'count' | 'pct_reports' | 'pct_ai_reports';
type SignalQualityMetricMode = 'count' | 'pct_year_total';
type DashboardPreset = 'ai-risk-line' | 'llm-adoption-line' | 'cyber-risk-line';
type ChartRow = Record<string, string | number | null | undefined>;
type HeatmapCell = { x: string | number; y: string | number; value: number };
type VisualizationExport = {
  title: string;
  fileBase: string;
  csv: string;
};

type DashboardClientProps = {
  data?: GoldenDashboardData;
  renderedAtIso?: string;
  dataVersion?: string;
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
const convertHeatmapToPercentOfXAxisTotal = (cells: HeatmapCell[]): HeatmapCell[] => {
  const totalsByX = new Map<string, number>();

  cells.forEach(cell => {
    const key = String(cell.x);
    totalsByX.set(key, (totalsByX.get(key) || 0) + cell.value);
  });

  return convertHeatmapToPercent(cells, cell => totalsByX.get(String(cell.x)) || 0);
};
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

type QualityScope = 'risk' | 'adoption' | 'vendor';
type CompanyScope = 'all' | 'cniOnly';
type FilterIndexRow = GoldenDashboardData['filterIndex']['perReport'][number];

const FILTER_ROW = {
  year: 0,
  monthIndex: 1,
  cniSectorIndex: 2,
  isicSectorIndex: 3,
  marketSegmentIndex: 4,
  mentionMask: 5,
  adoptionMask: 6,
  riskMask: 7,
  vendorMask: 8,
  adoptionSignalMask: 9,
  riskSignalMask: 10,
  vendorSignalMask: 11,
  adoptionSubstantivenessMask: 12,
  riskSubstantivenessMask: 13,
  vendorSubstantivenessMask: 14,
} as const;

const hasBit = (mask: number, index: number) => index >= 0 && (mask & (1 << index)) !== 0;

const forEachMaskLabel = (
  mask: number,
  labels: string[],
  callback: (label: string) => void
) => {
  labels.forEach((label, index) => {
    if (hasBit(mask, index)) callback(label);
  });
};

const initClientYearSeries = (years: number[], keys: string[]) =>
  years.map(year => {
    const row: Record<string, number> = { year };
    keys.forEach(key => {
      row[key] = 0;
    });
    return row;
  });

const initClientMonthSeries = (months: string[], keys: string[]) =>
  months.map(month => {
    const row: Record<string, string | number> = { month };
    keys.forEach(key => {
      row[key] = 0;
    });
    return row;
  });

const addYearSeriesCount = (
  series: Record<string, number>[],
  year: number,
  key: string
) => {
  const row = series.find(entry => entry.year === year);
  if (!row) return;
  row[key] = (row[key] || 0) + 1;
};

const addMonthSeriesCount = (
  series: Record<string, string | number>[],
  month: string,
  key: string
) => {
  const row = series.find(entry => entry.month === month);
  if (!row) return;
  row[key] = (Number(row[key]) || 0) + 1;
};

const addHeatmapCount = (
  map: Map<string, number>,
  parts: (string | number | undefined)[]
) => {
  if (parts.some(part => part === undefined || part === '')) return;
  const key = parts.join('|||');
  map.set(key, (map.get(key) || 0) + 1);
};

const mapToYearHeatmap = (map: Map<string, number>) =>
  Array.from(map.entries()).map(([key, value]) => {
    const [year, x, y] = key.split('|||');
    return { year: Number(year), x, y, value };
  });

const mapToMentionYearHeatmap = (map: Map<string, number>) =>
  Array.from(map.entries()).map(([key, value]) => {
    const [x, y] = key.split('|||');
    return { x: Number(x), y, value };
  });

const getSignalMaskForScope = (row: FilterIndexRow, scope: QualityScope) => {
  if (scope === 'adoption') return row[FILTER_ROW.adoptionSignalMask];
  if (scope === 'vendor') return row[FILTER_ROW.vendorSignalMask];
  return row[FILTER_ROW.riskSignalMask];
};

const getSubstantivenessMaskForScope = (row: FilterIndexRow, scope: QualityScope) => {
  if (scope === 'adoption') return row[FILTER_ROW.adoptionSubstantivenessMask];
  if (scope === 'vendor') return row[FILTER_ROW.vendorSubstantivenessMask];
  return row[FILTER_ROW.riskSubstantivenessMask];
};

const matchesMarketSegmentFilter = (marketSegment: string, filter: string) => {
  if (filter === 'all') return true;
  if (filter === 'Main Market') {
    return marketSegment === 'FTSE 100' || marketSegment === 'FTSE 250' || marketSegment === 'Main Market';
  }
  if (filter === 'Main Market (FTSE 100 only)') return marketSegment === 'FTSE 100';
  if (filter === 'Main Market (FTSE 350 only)') {
    return marketSegment === 'FTSE 100' || marketSegment === 'FTSE 250';
  }
  return marketSegment === filter;
};

const isCniMappedSectorLabel = (sector: string) => {
  const normalized = sector.trim();
  return normalized !== '' && normalized !== 'Unknown' && normalized !== 'Other';
};

const buildFilteredDatasetFromIndex = ({
  baseData,
  data,
  datasetKey,
  scope,
  signalStrengthFilter,
  substantivenessFilter,
  marketSegmentFilter,
  companyScope,
}: {
  baseData: GoldenDataset;
  data: GoldenDashboardData;
  datasetKey: DatasetKey;
  scope: QualityScope;
  signalStrengthFilter: string;
  substantivenessFilter: string;
  marketSegmentFilter: string;
  companyScope: CompanyScope;
}): GoldenDataset => {
  const rows = data.filterIndex[datasetKey];
  const filterIndexMonths =
    datasetKey === 'perReport' ? data.filterIndex.perReportMonths : data.filterIndex.perChunkMonths;
  const signalLevelIndex = data.labels.riskSignalLevels.indexOf(signalStrengthFilter);
  const substantivenessIndex = data.labels.substantivenessBands.indexOf(substantivenessFilter);

  const mentionTrend = initClientYearSeries(baseData.years, data.labels.mentionTypes);
  const adoptionTrend = initClientYearSeries(baseData.years, data.labels.adoptionTypes);
  const riskTrend = initClientYearSeries(baseData.years, data.labels.riskLabels);
  const vendorTrend = initClientYearSeries(baseData.years, data.labels.vendorTags);
  const mentionTrendMonthly = initClientMonthSeries(baseData.months, data.labels.mentionTypes);
  const riskTrendMonthly = initClientMonthSeries(baseData.months, data.labels.riskLabels);
  const adoptionTrendMonthly = initClientMonthSeries(baseData.months, data.labels.adoptionTypes);
  const vendorTrendMonthly = initClientMonthSeries(baseData.months, data.labels.vendorTags);

  const riskBySectorYearCounts = new Map<string, number>();
  const riskByIsicSectorYearCounts = new Map<string, number>();
  const adoptionBySectorYearCounts = new Map<string, number>();
  const adoptionByIsicSectorYearCounts = new Map<string, number>();
  const vendorBySectorYearCounts = new Map<string, number>();
  const vendorByIsicSectorYearCounts = new Map<string, number>();
  const riskMentionBySectorYearCounts = new Map<string, number>();
  const riskMentionByIsicSectorYearCounts = new Map<string, number>();

  let totalRows = 0;
  let aiSignalRows = 0;
  let adoptionRows = 0;
  let riskRows = 0;
  let vendorRows = 0;

  rows.forEach(row => {
    const sector = data.sectors[row[FILTER_ROW.cniSectorIndex]];
    const isicSector = data.isicSectors[row[FILTER_ROW.isicSectorIndex]];
    const marketSegment = data.filterIndex.marketSegments[row[FILTER_ROW.marketSegmentIndex]] || '';
    if (!sector || !isicSector) return;
    if (companyScope === 'cniOnly' && !isCniMappedSectorLabel(sector)) return;
    if (!matchesMarketSegmentFilter(marketSegment, marketSegmentFilter)) return;
    if (signalStrengthFilter !== 'all' && !hasBit(getSignalMaskForScope(row, scope), signalLevelIndex)) return;
    if (substantivenessFilter !== 'all' && !hasBit(getSubstantivenessMaskForScope(row, scope), substantivenessIndex)) return;

    totalRows += 1;
    const year = row[FILTER_ROW.year];
    const month = filterIndexMonths[row[FILTER_ROW.monthIndex]];
    const mentionMask = row[FILTER_ROW.mentionMask];
    const adoptionMask = row[FILTER_ROW.adoptionMask];
    const riskMask = row[FILTER_ROW.riskMask];
    const vendorMask = row[FILTER_ROW.vendorMask];
    const hasAiSignal = mentionMask > 0;
    const hasAdoption = hasBit(mentionMask, data.labels.mentionTypes.indexOf('adoption'));
    const hasRisk =
      hasBit(mentionMask, data.labels.mentionTypes.indexOf('risk')) ||
      riskMask > 0;
    const hasVendor = hasBit(mentionMask, data.labels.mentionTypes.indexOf('vendor'));

    if (hasAiSignal) aiSignalRows += 1;
    if (hasAdoption) adoptionRows += 1;
    if (hasRisk) riskRows += 1;
    if (hasVendor) vendorRows += 1;

    forEachMaskLabel(mentionMask, data.labels.mentionTypes, label => {
      addYearSeriesCount(mentionTrend, year, label);
      if (month) addMonthSeriesCount(mentionTrendMonthly, month, label);
    });
    forEachMaskLabel(adoptionMask, data.labels.adoptionTypes, label => {
      addYearSeriesCount(adoptionTrend, year, label);
      addHeatmapCount(adoptionBySectorYearCounts, [year, label, sector]);
      addHeatmapCount(adoptionByIsicSectorYearCounts, [year, label, isicSector]);
      if (month) addMonthSeriesCount(adoptionTrendMonthly, month, label);
    });
    forEachMaskLabel(riskMask, data.labels.riskLabels, label => {
      addYearSeriesCount(riskTrend, year, label);
      addHeatmapCount(riskBySectorYearCounts, [year, label, sector]);
      addHeatmapCount(riskByIsicSectorYearCounts, [year, label, isicSector]);
      if (month) addMonthSeriesCount(riskTrendMonthly, month, label);
    });
    forEachMaskLabel(vendorMask, data.labels.vendorTags, label => {
      addYearSeriesCount(vendorTrend, year, label);
      addHeatmapCount(vendorBySectorYearCounts, [year, label, sector]);
      addHeatmapCount(vendorByIsicSectorYearCounts, [year, label, isicSector]);
      if (month) addMonthSeriesCount(vendorTrendMonthly, month, label);
    });
    if (hasRisk) {
      addHeatmapCount(riskMentionBySectorYearCounts, [year, sector]);
      addHeatmapCount(riskMentionByIsicSectorYearCounts, [year, isicSector]);
    }
  });

  return {
    ...baseData,
    summary: {
      ...baseData.summary,
      totalReports: totalRows,
      aiSignalReports: aiSignalRows,
      adoptionReports: adoptionRows,
      riskReports: riskRows,
      vendorReports: vendorRows,
    },
    mentionTrend,
    mentionTrendMonthly,
    adoptionTrend,
    riskTrend,
    vendorTrend,
    riskTrendMonthly,
    adoptionTrendMonthly,
    vendorTrendMonthly,
    riskBySectorYear: mapToYearHeatmap(riskBySectorYearCounts),
    riskByIsicSectorYear: mapToYearHeatmap(riskByIsicSectorYearCounts),
    adoptionBySectorYear: mapToYearHeatmap(adoptionBySectorYearCounts),
    adoptionByIsicSectorYear: mapToYearHeatmap(adoptionByIsicSectorYearCounts),
    vendorBySectorYear: mapToYearHeatmap(vendorBySectorYearCounts),
    vendorByIsicSectorYear: mapToYearHeatmap(vendorByIsicSectorYearCounts),
    riskMentionBySectorYear: mapToMentionYearHeatmap(riskMentionBySectorYearCounts),
    riskMentionByIsicSectorYear: mapToMentionYearHeatmap(riskMentionByIsicSectorYearCounts),
  };
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

function DashboardLoadingState({
  message,
  action,
}: {
  message: string;
  action?: ReactNode;
}) {
  return (
    <div className="relative min-h-screen overflow-hidden bg-[linear-gradient(180deg,#ffffff_0%,#f7f9fc_52%,#f4f5f7_100%)] text-primary">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute inset-x-0 top-0 h-[24rem] bg-[radial-gradient(circle_at_top,rgba(199,216,238,0.48)_0%,rgba(199,216,238,0.22)_24%,rgba(199,216,238,0)_68%)]" />
      </div>
      <main className="relative mx-auto flex min-h-screen max-w-5xl items-center px-6 py-16">
        <div className="w-full rounded-2xl border border-border bg-white p-8 shadow-[0_1px_2px_rgba(15,23,42,0.05)] sm:p-10">
          <div className="mb-6 h-2 w-24 rounded-full bg-accent/80" />
          <h1 className="text-3xl font-bold tracking-tight text-primary sm:text-4xl">
            AI Risk Observatory Data Explorer
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted">
            {message}
          </p>
          {action ? <div className="mt-6">{action}</div> : null}
        </div>
      </main>
    </div>
  );
}

export default function DashboardClient({
  data: initialData,
  renderedAtIso: initialRenderedAtIso,
  dataVersion,
}: DashboardClientProps) {
  const [data, setData] = useState<GoldenDashboardData | null>(initialData ?? null);
  const [renderedAtIso, setRenderedAtIso] = useState<string | null>(initialRenderedAtIso ?? null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [requestKey, setRequestKey] = useState(0);

  useEffect(() => {
    if (initialData && initialRenderedAtIso) return;

    const controller = new AbortController();
    let isActive = true;

    const loadDashboardData = async () => {
      try {
        setLoadError(null);
        const dashboardDataUrl = dataVersion
          ? `/api/dashboard-data?v=${encodeURIComponent(dataVersion)}`
          : '/api/dashboard-data';
        const response = await fetch(dashboardDataUrl, {
          cache: dataVersion ? 'force-cache' : 'no-store',
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const nextData = (await response.json()) as GoldenDashboardData;
        if (!isActive) return;

        setData(nextData);
        setRenderedAtIso(response.headers.get('x-rendered-at') ?? new Date().toISOString());
      } catch (error) {
        if (!isActive || controller.signal.aborted) return;
        setLoadError(error instanceof Error ? error.message : 'Unable to load dashboard data.');
      }
    };

    void loadDashboardData();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, [initialData, initialRenderedAtIso, requestKey, dataVersion]);

  if (!data || !renderedAtIso) {
    if (loadError) {
      return (
        <DashboardLoadingState
          message={`The dashboard dataset could not be loaded. ${loadError}`}
          action={(
            <button
              type="button"
              onClick={() => setRequestKey(prev => prev + 1)}
              className="inline-flex h-10 items-center justify-center rounded border border-slate-950 bg-slate-950 px-4 text-sm font-semibold text-white transition hover:bg-slate-800"
            >
              Retry loading data
            </button>
          )}
        />
      );
    }

    return (
      <DashboardLoadingState message="Loading the dashboard dataset. The first request can take a moment while the data is transferred." />
    );
  }

  return <DashboardContent data={data} renderedAtIso={renderedAtIso} />;
}

function DashboardContent({
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
  const [riskBreakdownMode, setRiskBreakdownMode] = useState<RiskBreakdownMode>('categories');
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
  });
  const [signalQualityMode, setSignalQualityMode] = useState<SignalQualityMode>('explicitness');
  const [metricMode, setMetricMode] = useState<MetricMode>('pct_reports');
  const [signalQualityMetricMode, setSignalQualityMetricMode] = useState<SignalQualityMetricMode>('count');
  const [marketSegmentFilter, setMarketSegmentFilter] = useState<string>('all');
  const [showCniCompaniesOnly, setShowCniCompaniesOnly] = useState(false);
  const [expandedIsicGroups, setExpandedIsicGroups] = useState<string[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);
  const [isSignalQualityOpen, setIsSignalQualityOpen] = useState(false);
  const [signalStrengthFilter, setSignalStrengthFilter] = useState<string>('all');
  const [boilerplateFilter, setBoilerplateFilter] = useState<string>('all');
  const [chartDisplayType, setChartDisplayType] = useState<'bar' | 'grouped' | 'line'>('bar');
  const [showChartLegend, setShowChartLegend] = useState(true);
  const [shareButtonLabel, setShareButtonLabel] = useState('Share');
  const [copiedReferenceKey, setCopiedReferenceKey] = useState<string | null>(null);
  const [isExportingVisualization, setIsExportingVisualization] = useState(false);
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(null);
  const visualizationExportRef = useRef<HTMLDivElement | null>(null);
  const defaultYearRange = {
    start: 0,
    end: Math.max(data.years.length - 1, 0),
  };

  const resetDashboardSettings = () => {
    setDatasetKey('perReport');
    setTrendTimeAxis('year');
    setRiskBreakdownMode('categories');
    setRiskFilter('all');
    setAdoptionFilter('all');
    setRiskSectorView('cni');
    setAdoptionSectorView('cni');
    setVendorSectorView('cni');
    setVendorFilter('all');
    setSignalQualityMode('explicitness');
    setMetricMode('pct_reports');
    setSignalQualityMetricMode('count');
    setMarketSegmentFilter('all');
    setShowCniCompaniesOnly(true);
    setExpandedIsicGroups([]);
    setChartDisplayType('bar');
    setShowChartLegend(true);
    setSignalStrengthFilter('all');
    setBoilerplateFilter('all');
    setYearRangeIndices(defaultYearRange);
    setIsSignalQualityOpen(false);
  };

  useEffect(() => {
    const applyPresetFromSearch = () => {
      const preset = new URLSearchParams(window.location.search).get('preset');

      if (!preset || !['ai-risk-line', 'llm-adoption-line', 'cyber-risk-line'].includes(preset)) {
        return false;
      }

      const dashboardPreset = preset as DashboardPreset;
      setDatasetKey('perReport');
      setTrendTimeAxis('year');
      setMetricMode('pct_reports');
      setVisualizationMode('chart');
      setChartDisplayType('line');
      setShowChartLegend(true);
      setMarketSegmentFilter('all');
      setShowCniCompaniesOnly(false);

      if (dashboardPreset === 'llm-adoption-line') {
        setActiveView(2);
        setAdoptionFilter('llm');
      } else {
        setActiveView(1);
        setAdoptionFilter('all');

        if (dashboardPreset === 'ai-risk-line') {
          setRiskBreakdownMode('phase1');
          setRiskFilter('all');
        } else {
          setRiskBreakdownMode('categories');
          setRiskFilter('cybersecurity');
        }
      }

      // Strip preset param from URL so refresh gives the normal dashboard
      const cleanHash = window.location.hash || '#risk';
      window.history.replaceState(null, '', `/data${cleanHash}`);
      return true;
    };

    const syncViewFromHash = () => {
      const nextView = getViewIdFromHash(window.location.hash);
      setActiveView(prev => (prev === nextView ? prev : nextView));
      setVisualizationMode(nextView === 4 ? 'heatmap' : 'chart');
    };

    setDashboardBaseUrl(`${window.location.origin}/data`);
    if (!applyPresetFromSearch()) {
      syncViewFromHash();
    }
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

  const currentViewUrl = `${dashboardBaseUrl}${
    isSignalQualityOpen
      ? '#signal-quality'
      : activeView === DEFAULT_VIEW_ID
        ? ''
        : `#${getViewHash(activeView)}`
  }`;

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const isTrendChartView = visualizationMode === 'chart' && !isSignalQualityOpen;
  const effectiveCompanyScope = isTrendChartView && showCniCompaniesOnly ? 'cniOnly' : 'all';
  const baseResolvedDatasets =
    effectiveCompanyScope === 'cniOnly'
      ? (
          marketSegmentFilter === 'all'
            ? (data.byCompanyScope?.cniOnly ?? data.datasets)
            : (
                data.byMarketSegmentAndCompanyScope?.[marketSegmentFilter]?.cniOnly
                ?? data.byMarketSegment[marketSegmentFilter]
                ?? data.datasets
              )
        )
      : (
          marketSegmentFilter === 'all'
            ? data.datasets
            : (data.byMarketSegment[marketSegmentFilter] ?? data.datasets)
        );
  const viewScope: QualityScope = activeView === 2 ? 'adoption' : activeView === 3 ? 'vendor' : 'risk';
  const hasAdvancedQualityFilter =
    !isSignalQualityOpen && (signalStrengthFilter !== 'all' || boilerplateFilter !== 'all');
  const resolvedDatasets = useMemo(() => {
    if (!hasAdvancedQualityFilter) return baseResolvedDatasets;

    return {
      perReport: buildFilteredDatasetFromIndex({
        baseData: baseResolvedDatasets.perReport,
        data,
        datasetKey: 'perReport',
        scope: viewScope,
        signalStrengthFilter,
        substantivenessFilter: boilerplateFilter,
        marketSegmentFilter,
        companyScope: effectiveCompanyScope,
      }),
      perChunk: buildFilteredDatasetFromIndex({
        baseData: baseResolvedDatasets.perChunk,
        data,
        datasetKey: 'perChunk',
        scope: viewScope,
        signalStrengthFilter,
        substantivenessFilter: boilerplateFilter,
        marketSegmentFilter,
        companyScope: effectiveCompanyScope,
      }),
    };
  }, [
    baseResolvedDatasets,
    boilerplateFilter,
    data,
    effectiveCompanyScope,
    hasAdvancedQualityFilter,
    marketSegmentFilter,
    signalStrengthFilter,
    viewScope,
  ]);
  const activeData = resolvedDatasets[datasetKey];
  const activeReportData = resolvedDatasets.perReport;
  const reportBaselineData = baseResolvedDatasets.perReport;
  const canShowReportShare =
    !isSignalQualityOpen && datasetKey === 'perReport';
  const effectiveMetricMode: MetricMode = canShowReportShare ? metricMode : 'count';
  const isAllReportsShareMode = effectiveMetricMode === 'pct_reports';
  const isAiMentionShareMode = effectiveMetricMode === 'pct_ai_reports';
  const isPercentageMetricMode = isAllReportsShareMode || isAiMentionShareMode;
  const percentageDenominatorLongLabel = isAiMentionShareMode
    ? 'AI-reporting annual reports'
    : 'annual reports';
  const percentageDenominatorShortLabel = isAiMentionShareMode
    ? 'AI-reporting reports'
    : 'reports';
  const availableYears = activeData.years;
  const maxYearIndex = Math.max(availableYears.length - 1, 0);
  const fullDatasetYearSpan = reportBaselineData.years.length > 0
    ? `${reportBaselineData.years[0]} to ${reportBaselineData.years[reportBaselineData.years.length - 1]}`
    : 'the full available period';
  const datasetScopeLabel = marketSegmentFilter !== 'all' && effectiveCompanyScope === 'cniOnly'
    ? `for UK listed companies (selection: CNI sectors) in ${marketSegmentFilter}`
    : marketSegmentFilter !== 'all'
      ? `for ${marketSegmentFilter} companies`
      : effectiveCompanyScope === 'cniOnly'
        ? 'for UK listed companies (selection: CNI sectors)'
        : '';
  const datasetScopeFragment = datasetScopeLabel ? ` ${datasetScopeLabel}` : '';

  const datasetSummaryByView = useMemo(() => {
    const totalReports = reportBaselineData.blindSpotTrend.reduce(
      (sum, row) => sum + (Number(row.total_reports) || 0),
      0
    );
    const riskMentionReports = activeReportData.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.risk) || 0),
      0
    );
    const excerptRiskMentions = resolvedDatasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.risk) || 0),
      0
    );
    const adoptionMentionReports = activeReportData.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.adoption) || 0),
      0
    );
    const excerptAdoptionMentions = resolvedDatasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.adoption) || 0),
      0
    );
    const vendorMentionReports = activeReportData.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.vendor) || 0),
      0
    );
    const excerptVendorMentions = resolvedDatasets.perChunk.mentionTrend.reduce(
      (sum, row) => sum + (Number(row.vendor) || 0),
      0
    );
    const riskSignalTotal = reportBaselineData.riskSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );
    const adoptionSignalTotal = reportBaselineData.adoptionSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );
    const vendorSignalTotal = reportBaselineData.vendorSignalHeatmap.reduce(
      (sum, row) => sum + (Number(row.value) || 0),
      0
    );

    return {
      1: `Across ${formatNumber(totalReports)} annual reports${datasetScopeFragment} from ${fullDatasetYearSpan}, ${formatNumber(riskMentionReports)} mention AI risk across ${formatNumber(excerptRiskMentions)} tagged AI mentions.`,
      2: `Across ${formatNumber(totalReports)} annual reports${datasetScopeFragment} from ${fullDatasetYearSpan}, ${formatNumber(adoptionMentionReports)} mention AI adoption across ${formatNumber(excerptAdoptionMentions)} tagged AI mentions.`,
      3: `Across ${formatNumber(totalReports)} annual reports${datasetScopeFragment} from ${fullDatasetYearSpan}, ${formatNumber(vendorMentionReports)} name AI vendors across ${formatNumber(excerptVendorMentions)} tagged AI mentions.`,
      4: `Across ${formatNumber(totalReports)} annual reports${datasetScopeFragment} from ${fullDatasetYearSpan}, the dataset includes ${formatNumber(riskSignalTotal)} risk, ${formatNumber(adoptionSignalTotal)} adoption, and ${formatNumber(vendorSignalTotal)} vendor quality assessments.`,
    } as Record<number, string>;
  }, [activeReportData.mentionTrend, reportBaselineData, resolvedDatasets.perChunk, fullDatasetYearSpan, datasetScopeFragment]);

  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const adoptionLegendKeys = useMemo(() => [...adoptionStackKeys].reverse(), [adoptionStackKeys]);
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

  const mentionTrendInRange = useMemo(
    () =>
      activeData.mentionTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [activeData.mentionTrend, selectedStartYear, selectedEndYear]
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
      activeReportData.riskBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.riskBySectorYear, selectedStartYear, selectedEndYear]
  );

  const riskByIsicSectorYearInRange = useMemo(
    () =>
      activeReportData.riskByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.riskByIsicSectorYear, selectedStartYear, selectedEndYear]
  );

  const riskMentionBySectorYearInRange = useMemo(
    () =>
      activeReportData.riskMentionBySectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeReportData.riskMentionBySectorYear, selectedStartYear, selectedEndYear]
  );

  const riskMentionByIsicSectorYearInRange = useMemo(
    () =>
      activeReportData.riskMentionByIsicSectorYear.filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeReportData.riskMentionByIsicSectorYear, selectedStartYear, selectedEndYear]
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
      activeReportData.adoptionBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.adoptionBySectorYear, selectedStartYear, selectedEndYear]
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
      activeReportData.adoptionByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.adoptionByIsicSectorYear, selectedStartYear, selectedEndYear]
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
      activeReportData.vendorBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.vendorBySectorYear, selectedStartYear, selectedEndYear]
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
      activeReportData.vendorByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeReportData.vendorByIsicSectorYear, selectedStartYear, selectedEndYear]
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
    if (riskBreakdownMode === 'phase1') {
      return mentionTrendInRange.map(row => ({
        year: row.year,
        risk: row.risk || 0,
      }));
    }
    if (riskFilter === 'all') return riskTrendInRange;
    return riskTrendInRange.map(row => ({
      year: row.year,
      [riskFilter]: row[riskFilter] || 0,
    }));
  }, [mentionTrendInRange, riskBreakdownMode, riskTrendInRange, riskFilter]);

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

  const monthlyMentionTrendInRange = useMemo(() => {
    const startMonth = `${selectedStartYear}-01`;
    const endMonth = `${selectedEndYear}-12`;
    return activeData.mentionTrendMonthly.filter(row => {
      const m = row.month as string;
      return m >= startMonth && m <= endMonth;
    });
  }, [activeData.mentionTrendMonthly, selectedStartYear, selectedEndYear]);

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

  const aiMentionReportTotalsByYear = useMemo(
    () =>
      new Map(
        reportBaselineData.blindSpotTrend.map(row => {
          const totalReports = Number(row.total_reports) || 0;
          const aiMentionReports = Number(row.ai_mention) || Math.max(totalReports - (Number(row.no_ai_mention) || 0), 0);
          return [Number(row.year), aiMentionReports];
        })
      ),
    [reportBaselineData.blindSpotTrend]
  );

  const aiMentionReportTotalsByMonth = useMemo(
    () =>
      new Map(
        reportBaselineData.blindSpotTrendMonthly.map(row => {
          const totalReports = Number(row.total_reports) || 0;
          const aiMentionReports = Math.max(totalReports - (Number(row.no_ai_mention) || 0), 0);
          return [String(row.month), aiMentionReports];
        })
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
    if (riskBreakdownMode === 'phase1') {
      return monthlyMentionTrendInRange.map(row => ({
        month: row.month,
        risk: row.risk || 0,
      }));
    }
    if (riskFilter === 'all') return monthlyRiskTrendInRange;
    return monthlyRiskTrendInRange.map(row => ({
      month: row.month,
      [riskFilter]: row[riskFilter] || 0,
    }));
  }, [monthlyMentionTrendInRange, monthlyRiskTrendInRange, riskBreakdownMode, riskFilter]);

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

  const riskChartStackKeys = useMemo(
    () => riskBreakdownMode === 'phase1'
      ? ['risk']
      : riskFilter === 'all'
        ? riskStackKeys
        : [riskFilter],
    [riskBreakdownMode, riskFilter, riskStackKeys]
  );
  const riskChartColors = riskBreakdownMode === 'phase1' ? phase1RiskColors : riskColors;

  const displayRiskTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyRiskTrend : filteredRiskTrend;
    if (!isPercentageMetricMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      riskChartStackKeys,
      trendTimeAxis === 'month'
        ? (isAiMentionShareMode ? aiMentionReportTotalsByMonth : reportTotalsByMonth)
        : (isAiMentionShareMode ? aiMentionReportTotalsByYear : reportTotalsByYear)
    );
  }, [
    trendTimeAxis,
    filteredMonthlyRiskTrend,
    filteredRiskTrend,
    isPercentageMetricMode,
    isAiMentionShareMode,
    riskChartStackKeys,
    aiMentionReportTotalsByMonth,
    aiMentionReportTotalsByYear,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);

  const displayAdoptionTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyAdoptionTrend : filteredAdoptionTrend;
    if (!isPercentageMetricMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      adoptionFilter === 'all' ? adoptionStackKeys : [adoptionFilter],
      trendTimeAxis === 'month'
        ? (isAiMentionShareMode ? aiMentionReportTotalsByMonth : reportTotalsByMonth)
        : (isAiMentionShareMode ? aiMentionReportTotalsByYear : reportTotalsByYear)
    );
  }, [
    trendTimeAxis,
    filteredMonthlyAdoptionTrend,
    filteredAdoptionTrend,
    isPercentageMetricMode,
    isAiMentionShareMode,
    adoptionFilter,
    adoptionStackKeys,
    aiMentionReportTotalsByMonth,
    aiMentionReportTotalsByYear,
    reportTotalsByMonth,
    reportTotalsByYear,
  ]);

  const displayVendorTrend = useMemo(() => {
    const source = trendTimeAxis === 'month' ? filteredMonthlyVendorTrend : filteredVendorTrend;
    if (!isPercentageMetricMode) return source;
    return convertTrendRowsToPercent(
      source,
      trendTimeAxis === 'month' ? 'month' : 'year',
      effectiveVendorFilter === 'all' ? vendorStackKeys : [effectiveVendorFilter],
      trendTimeAxis === 'month'
        ? (isAiMentionShareMode ? aiMentionReportTotalsByMonth : reportTotalsByMonth)
        : (isAiMentionShareMode ? aiMentionReportTotalsByYear : reportTotalsByYear)
    );
  }, [
    trendTimeAxis,
    filteredMonthlyVendorTrend,
    filteredVendorTrend,
    isPercentageMetricMode,
    isAiMentionShareMode,
    effectiveVendorFilter,
    vendorStackKeys,
    aiMentionReportTotalsByMonth,
    aiMentionReportTotalsByYear,
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
  const adoptionSubstantivenessHeatmapInRange = useMemo(
    () =>
      (activeData.adoptionSubstantivenessHeatmap ?? []).filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.adoptionSubstantivenessHeatmap, selectedStartYear, selectedEndYear]
  );
  const vendorSubstantivenessHeatmapInRange = useMemo(
    () =>
      (activeData.vendorSubstantivenessHeatmap ?? []).filter(
        cell => cell.x >= selectedStartYear && cell.x <= selectedEndYear
      ),
    [activeData.vendorSubstantivenessHeatmap, selectedStartYear, selectedEndYear]
  );
  const signalQualityScopeLabel = activeView === 2 ? 'adoption' : activeView === 3 ? 'vendor' : 'risk';

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
  const riskHeatmapPhase1YearDataInRange =
    riskSectorView === 'cni' ? riskMentionBySectorYearInRange : riskMentionByIsicSectorYearInRange;

  const riskHeatmapData = useMemo(() => {
    if (riskBreakdownMode === 'phase1') return riskHeatmapPhase1YearDataInRange;
    if (riskFilter === 'all') return riskHeatmapTaxonomyDataInRange;
    return riskHeatmapTaxonomyYearDataInRange
      .filter(cell => cell.x === riskFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [
    riskBreakdownMode,
    riskFilter,
    riskHeatmapPhase1YearDataInRange,
    riskHeatmapTaxonomyDataInRange,
    riskHeatmapTaxonomyYearDataInRange,
  ]);

  const riskHeatmapShowsCategoryColumns = riskBreakdownMode === 'categories' && riskFilter === 'all';
  const riskHeatmapXLabels = riskHeatmapShowsCategoryColumns ? data.labels.riskLabels : filteredYears;
  const riskHeatmapBaseColor = riskHeatmapShowsCategoryColumns
    ? '#e63946'
    : riskBreakdownMode === 'phase1'
      ? phase1RiskColors.risk
      : (riskColors[riskFilter] || '#e63946');
  const visibleRiskHeatmapData = useMemo(() => {
    if (riskSectorView === 'cni') {
      const displayData = riskHeatmapShowsCategoryColumns
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
    const displayData = riskHeatmapShowsCategoryColumns
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
    riskHeatmapData,
    riskHeatmapShowsCategoryColumns,
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
  const selectedSignalQualityHeatmap = useMemo(() => {
    const isPercentOfYearTotal = signalQualityMetricMode === 'pct_year_total';
    const valueFormatter = isPercentOfYearTotal
      ? (value: number) => formatPercent(value)
      : (value: number) => `${value}`;
    const totalsMode = isPercentOfYearTotal ? 'average' as const : 'sum' as const;
    const totalsLabel = isPercentOfYearTotal ? 'Avg' : 'Total';
    const totalValueFormatter = isPercentOfYearTotal
      ? (value: number) => formatPercent(value)
      : undefined;

    if (signalQualityMode === 'substantiveness') {
      const substantivenessHeatmapByScope = {
        risk: substantivenessHeatmapInRange,
        adoption: adoptionSubstantivenessHeatmapInRange,
        vendor: vendorSubstantivenessHeatmapInRange,
      };
      const scopeColors = { risk: '#f59e0b', adoption: '#3b82f6', vendor: '#64748b' };
      const scopeLabel = signalQualityScopeLabel === 'adoption' ? 'Adoption' : signalQualityScopeLabel === 'vendor' ? 'Vendor' : 'Risk';

      const rawData = (substantivenessHeatmapByScope[signalQualityScopeLabel] ?? []);
      const heatmapData = isPercentOfYearTotal
        ? convertHeatmapToPercentOfXAxisTotal(rawData)
        : rawData;

      return {
        data: heatmapData,
        yLabels: data.labels.substantivenessBands,
        baseColor: scopeColors[signalQualityScopeLabel],
        title: `AI ${scopeLabel} Substantiveness Distribution`,
        subtitle:
          isPercentOfYearTotal
            ? `Heatmap of report-level ${scopeLabel.toLowerCase()}-disclosure quality by substantiveness band (rows: Substantive, Moderate, Boilerplate) and publication year (columns). Each cell shows the percentage share of that year's substantiveness classifications that fell into the band; colour intensity encodes relative frequency.`
            : `Heatmap of report-level ${scopeLabel.toLowerCase()}-disclosure quality by substantiveness band (rows: Substantive, Moderate, Boilerplate) and publication year (columns). Each cell counts the number of reports whose AI-${scopeLabel.toLowerCase()} language was classified into that quality tier in a given year; colour intensity encodes relative frequency.`,
        tooltip:
          isPercentOfYearTotal
            ? 'Substantiveness measures depth and specificity of disclosure at report level. Percentage mode normalises each year to 100%, making it easier to compare how disclosure quality shifts over time.'
            : 'Substantiveness measures depth and specificity of disclosure at report level. Substantive disclosures include concrete mechanisms and detail, while boilerplate disclosures remain generic.',
        yAxisLabel: 'Quality Band',
        compact: false,
        valueFormatter,
        totalsMode,
        totalsLabel,
        totalValueFormatter,
      };
    }

    const baseConfigByScope = {
      risk: {
        rawData: riskSignalHeatmapInRange,
        baseColor: '#e63946',
        title: 'AI Risk Signal Strength',
        subtitle:
          isPercentOfYearTotal
            ? "Heatmap of risk-classification signal shares by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell shows the percentage share of that year's risk signal scores that fell into the level; colour intensity encodes relative frequency."
            : "Heatmap of risk-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell counts how many label-level risk classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.",
        tooltip:
          isPercentOfYearTotal
            ? 'Risk signal strength scores how directly the text supports a risk classification. 3 = explicit statement; 2 = strong implicit evidence; 1 = weak implicit evidence. Percentage mode normalises each year to 100% for easier comparison across years.'
            : 'Risk signal strength scores how directly the text supports a risk classification. 3 = explicit statement; 2 = strong implicit evidence; 1 = weak implicit evidence. Each cell counts label-level outcomes, not unique reports.',
      },
      adoption: {
        rawData: adoptionSignalHeatmapInRange,
        baseColor: '#3b82f6',
        title: 'AI Adoption Signal Strength',
        subtitle:
          isPercentOfYearTotal
            ? "Heatmap of adoption-classification signal shares by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell shows the percentage share of that year's adoption signal scores that fell into the level; colour intensity encodes relative frequency."
            : "Heatmap of adoption-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell counts how many label-level adoption classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.",
        tooltip:
          isPercentOfYearTotal
            ? 'Applies the same signal-strength rubric to AI adoption mentions. Percentage mode normalises each year to 100%, highlighting changes in the mix of explicit versus implicit disclosures over time.'
            : 'Applies the same signal-strength rubric to AI adoption mentions. Higher rows indicate clearer, more directly supported adoption disclosures, while lower rows reflect softer inferential language.',
      },
      vendor: {
        rawData: vendorSignalHeatmapInRange,
        baseColor: '#64748b',
        title: 'AI Vendor Signal Strength',
        subtitle:
          isPercentOfYearTotal
            ? "Heatmap of vendor-classification signal shares by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell shows the percentage share of that year's vendor signal scores that fell into the level; colour intensity encodes relative frequency."
            : "Heatmap of vendor-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and publication year (columns). Each cell counts how many label-level vendor classifications fell into that strength tier in a given year; colour intensity encodes relative frequency.",
        tooltip:
          isPercentOfYearTotal
            ? 'Measures how directly a vendor relationship is stated in the text. Percentage mode normalises each year to 100%, making it easier to compare shifts in disclosure clarity over time.'
            : 'Measures how directly a vendor relationship is stated in the text. Low explicitness can indicate more opaque supplier disclosure; higher explicit counts suggest clearer provider attribution.',
      },
    } as const;

    const selectedConfig = baseConfigByScope[signalQualityScopeLabel];
    const heatmapData = isPercentOfYearTotal
      ? convertHeatmapToPercentOfXAxisTotal(selectedConfig.rawData)
      : selectedConfig.rawData;

    return {
      data: heatmapData,
      yLabels: data.labels.riskSignalLevels,
      baseColor: selectedConfig.baseColor,
      title: selectedConfig.title,
      subtitle: selectedConfig.subtitle,
      tooltip: selectedConfig.tooltip,
      yAxisLabel: 'Signal Level',
      compact: true,
      valueFormatter,
      totalsMode,
      totalsLabel,
      totalValueFormatter,
    };
  }, [
    signalQualityMode,
    signalQualityMetricMode,
    signalQualityScopeLabel,
    riskSignalHeatmapInRange,
    adoptionSignalHeatmapInRange,
    vendorSignalHeatmapInRange,
    substantivenessHeatmapInRange,
    adoptionSubstantivenessHeatmapInRange,
    vendorSubstantivenessHeatmapInRange,
    data.labels.riskSignalLevels,
    data.labels.substantivenessBands,
  ]);
  const riskSelectedYearSpan = filteredYears.length > 0
    ? `${selectedStartYear}–${selectedEndYear}`
    : 'N/A';
  const stackedChartYAxisFormatter = isPercentageMetricMode
    ? (value: number) => `${Math.round(value)}%`
    : undefined;
  const stackedChartTooltipFormatter = (value: number) =>
    isPercentageMetricMode ? formatPercent(value) : formatNumber(value);
  const riskHeatmapValueFormatter = (value: number) => formatPercent(value);
  const dashboardUpdatedLabel = 'Dataset updated 3 Apr 2026';
  const currentVisualizationExport = useMemo<VisualizationExport>(() => {
    if (activeView === 1) {
      if (visualizationMode === 'chart') {
        return {
          title: riskBreakdownMode === 'phase1' ? 'AI Risk Mentioned Over Time' : 'AI Risk Categories Mentioned Over Time',
          fileBase: riskBreakdownMode === 'phase1' ? 'ai-risk-mentioned-over-time' : 'ai-risk-categories-mentioned-over-time',
          csv: toCsv(displayRiskTrend),
        };
      }

      return {
        title: riskBreakdownMode === 'phase1'
          ? 'Percentage of Companies Mentioning AI as a Risk by Sector and Year'
          : riskFilter === 'all'
            ? 'Risk Distribution by Sector'
            : `${formatLabel(riskFilter)} Risk Mentions by Sector and Year`,
        fileBase: riskBreakdownMode === 'phase1'
          ? 'ai-risk-mentioned-by-sector-and-year'
          : riskFilter === 'all'
            ? 'risk-distribution-by-sector'
            : `${slugify(formatLabel(riskFilter))}-risk-mentions-by-sector-and-year`,
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

    if (isSignalQualityOpen) {
      return {
        title: selectedSignalQualityHeatmap.title,
        fileBase: slugify(selectedSignalQualityHeatmap.title),
        csv: heatmapCellsToCsv(selectedSignalQualityHeatmap.data),
      };
    }

    return {
      title: 'AI Risk Mentioned Over Time',
      fileBase: 'ai-risk-mentioned-over-time',
      csv: toCsv(displayRiskTrend),
    };
  }, [
    activeView,
    isSignalQualityOpen,
    visualizationMode,
    displayRiskTrend,
    riskBreakdownMode,
    riskFilter,
    visibleRiskHeatmapData,
    displayAdoptionTrend,
    adoptionFilter,
    visibleAdoptionHeatmapData,
    displayVendorTrend,
    effectiveVendorFilter,
    visibleVendorHeatmapData,
    selectedSignalQualityHeatmap,
  ]);

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

  const handleMetricModeChange = (nextMode: MetricMode) => {
    if (nextMode !== 'count') {
      setDatasetKey('perReport');
    }
    setMetricMode(nextMode);
  };

  const metricModeToggle = (
    <select
      id="metric-mode"
      value={effectiveMetricMode}
      onChange={event => handleMetricModeChange(event.target.value as MetricMode)}
      className="w-full aisi-select"
    >
      <option value="pct_reports">% of all reports</option>
      <option value="pct_ai_reports">% of AI-reporting reports</option>
      <option value="count">{datasetKey === 'perChunk' ? 'Number of AI mentions (total)' : 'Number of reports (total)'}</option>
    </select>
  );

  const signalQualityMetricModeToggle = (
    <div className="inline-flex w-full items-center overflow-hidden rounded border border-border bg-white p-1">
      <button
        type="button"
        onClick={() => setSignalQualityMetricMode('count')}
        className={`flex-1 ${segmentedButtonClass} ${
          signalQualityMetricMode === 'count'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        Raw Count
      </button>
      <button
        type="button"
        onClick={() => setSignalQualityMetricMode('pct_year_total')}
        className={`flex-1 ${segmentedButtonClass} ${
          signalQualityMetricMode === 'pct_year_total'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        }`}
      >
        % of Year Total
      </button>
    </div>
  );

  const showVisualizationToggle = !isSignalQualityOpen;
  const canUseNonStackedChart = visualizationMode === 'chart' && !isSignalQualityOpen && trendTimeAxis === 'year';
  const activeChartDisplayType = canUseNonStackedChart ? chartDisplayType : 'bar';

  const visualizationModeToggle = (
    <div className="inline-flex shrink-0 items-center overflow-hidden rounded border border-border bg-white p-0.5">
      <button
        type="button"
        aria-pressed={visualizationMode === 'chart'}
        onClick={() => setVisualizationMode('chart')}
        className={`inline-flex items-center gap-2 ${segmentedButtonTallClass} ${
          visualizationMode === 'chart'
            ? 'bg-primary text-white'
            : 'text-primary hover:bg-secondary'
        }`}
      >
        <TrendChartIcon />
        Trend Chart
      </button>
      <button
        type="button"
        aria-pressed={visualizationMode === 'heatmap'}
        onClick={() => setVisualizationMode('heatmap')}
        className={`inline-flex items-center gap-2 ${segmentedButtonTallClass} ${
          visualizationMode === 'heatmap'
            ? 'bg-primary text-white'
            : 'text-primary hover:bg-secondary'
        }`}
      >
        <SectorHeatmapIcon />
        Sector Heatmap
      </button>
    </div>
  );

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
        Stacked
      </button>
      <button
        type="button"
        onClick={() => {
          if (!canUseNonStackedChart) return;
          setChartDisplayType('grouped');
        }}
        disabled={!canUseNonStackedChart}
        className={`${segmentedButtonClass} ${
          activeChartDisplayType === 'grouped'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        } ${!canUseNonStackedChart ? 'cursor-not-allowed opacity-40 hover:bg-white' : ''}`}
      >
        Grouped
      </button>
      <button
        type="button"
        onClick={() => {
          if (!canUseNonStackedChart) return;
          setChartDisplayType('line');
        }}
        disabled={!canUseNonStackedChart}
        className={`${segmentedButtonClass} ${
          activeChartDisplayType === 'line'
            ? 'bg-primary text-white'
            : 'text-muted-foreground hover:bg-secondary'
        } ${!canUseNonStackedChart ? 'cursor-not-allowed opacity-40 hover:bg-white' : ''}`}
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

  const riskSubcategoriesToggle = (
    <button
      type="button"
      role="switch"
      aria-checked={riskBreakdownMode === 'categories'}
      onClick={() => setRiskBreakdownMode(prev => prev === 'categories' ? 'phase1' : 'categories')}
      className="inline-flex items-center rounded-full transition"
      title={riskBreakdownMode === 'categories' ? 'Show as single AI risk signal' : 'Break into sub-categories'}
    >
      <span
        className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
          riskBreakdownMode === 'categories' ? 'bg-primary' : 'bg-slate-300'
        }`}
        aria-hidden="true"
      >
        <span
          className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
            riskBreakdownMode === 'categories' ? 'translate-x-3.5' : 'translate-x-0.5'
          }`}
        />
      </span>
    </button>
  );

  const cniCompaniesOnlyToggle = (
    <button
      type="button"
      role="switch"
      aria-checked={showCniCompaniesOnly}
      onClick={() => setShowCniCompaniesOnly(prev => !prev)}
      className="inline-flex items-center rounded-full transition"
      title={showCniCompaniesOnly ? 'Show all companies' : 'Show only CNI companies'}
    >
      <span
        className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
          showCniCompaniesOnly ? 'bg-primary' : 'bg-slate-300'
        }`}
        aria-hidden="true"
      >
        <span
          className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
            showCniCompaniesOnly ? 'translate-x-3.5' : 'translate-x-0.5'
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

  const visualizationHeaderExtra = (
    <div className="flex flex-wrap items-center justify-end gap-2">
      {settingsPanelToggle}
    </div>
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
  const renderTaxonomyLead = (subject: 'risk' | 'adoption' | 'vendor') => (
    <p className="mb-3 text-sm leading-relaxed text-slate-500">
      The taxonomy used in our classification of {
        subject === 'adoption'
          ? 'mentioned AI adoption'
          : subject === 'risk'
            ? 'mentioned risk from AI'
            : 'mentioned AI vendors'
      }:
    </p>
  );

  const mentionTagFilterTooltip = 'Filters the view to AI mentions with the selected tag. In Per Report mode, a report is included if at least one AI mention in that filing has that tag.';

  const switchDashboardView = (viewId: number) => {
    setActiveView(viewId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  const openSignalQuality = () => {
    setIsSignalQualityOpen(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderSettingsPanel = () => (
    <div className="overflow-hidden rounded-lg border border-border bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)] min-[960px]:flex min-[960px]:h-full min-[960px]:flex-col">
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
      <div className="min-[960px]:flex-1 min-[960px]:overflow-y-auto">

        {isSignalQualityOpen ? (
          <div className="px-5 py-5 space-y-3">
            {renderSettingsSectionHeading('View', 'Switch between signal strength (how explicit each mention is) and substantiveness (how detailed and concrete the disclosure is).')}
            {signalQualityModeToggle}
            {signalQualityMode === 'explicitness' && (
              <p className="text-[11px] leading-relaxed text-muted-foreground pt-1">
                Signal-strength data follows the current dashboard tab: Risk, Adoption, or Vendors.
              </p>
            )}
            {signalQualityMode === 'substantiveness' && (
              <p className="text-[11px] leading-relaxed text-muted-foreground pt-1">
                Substantiveness data follows the current dashboard tab: Risk, Adoption, or Vendors.
              </p>
            )}
            <div className="pt-2 space-y-3">
              {renderSettingsSectionHeading(
                'Heatmap Metric',
                signalQualityMode === 'substantiveness'
                  ? 'Raw count shows the number of disclosure classifications in each year-band cell. % of year total normalises each year to 100% so you can compare the mix of substantiveness bands over time.'
                  : 'Raw count shows the number of signal scores in each year-level cell. % of year total normalises each year to 100% so you can compare the mix of explicit versus implicit classifications over time.'
              )}
              {signalQualityMetricModeToggle}
            </div>
          </div>
        ) : (<>

        {/* ── Filters ── */}
        <CollapsibleSection title="Filters" variant="settings" defaultOpen noDivider={isSignalQualityOpen}>
          <div className="space-y-5">
            {activeView === 1 && riskBreakdownMode === 'categories' && (
              <div className="space-y-3">
                {renderSettingsSectionHeading('Focus On One Risk Type', mentionTagFilterTooltip)}
                <div className="flex items-center gap-2">
                  <select
                    id="risk-filter"
                    value={riskFilter}
                    onChange={e => setRiskFilter(e.target.value)}
                    className="min-w-0 flex-1 aisi-select"
                  >
                    <option value="all">All Risk Types</option>
                    {riskStackKeys.map(label => (
                      <option key={label} value={label}>{formatLabel(label)}</option>
                    ))}
                  </select>
                  {riskFilter !== 'all' && (
                    <button onClick={() => setRiskFilter('all')} className={inlineClearButtonClass}>Clear</button>
                  )}
                </div>
              </div>
            )}
            {activeView === 2 && (
              <div className="space-y-3">
                {renderSettingsSectionHeading('Focus On One AI Type', mentionTagFilterTooltip)}
                <div className="flex items-center gap-2">
                  <select
                    id="adoption-filter"
                    value={adoptionFilter}
                    onChange={e => setAdoptionFilter(e.target.value)}
                    className="min-w-0 flex-1 aisi-select"
                  >
                    <option value="all">All AI Types</option>
                    {data.labels.adoptionTypes.map(label => (
                      <option key={label} value={label}>{formatLabel(label)}</option>
                    ))}
                  </select>
                  {adoptionFilter !== 'all' && (
                    <button onClick={() => setAdoptionFilter('all')} className={inlineClearButtonClass}>Clear</button>
                  )}
                </div>
              </div>
            )}
            {activeView === 3 && (
              <div className="space-y-3">
                {renderSettingsSectionHeading('Focus On One Vendor', mentionTagFilterTooltip)}
                <div className="flex items-center gap-2">
                  <select
                    id="vendor-filter"
                    value={effectiveVendorFilter}
                    onChange={e => setVendorFilter(e.target.value)}
                    className="min-w-0 flex-1 aisi-select"
                  >
                    <option value="all">All Vendors</option>
                    {vendorStackKeys.map(label => (
                      <option key={label} value={label}>{formatLabel(label)}</option>
                    ))}
                  </select>
                  {effectiveVendorFilter !== 'all' && (
                    <button onClick={() => setVendorFilter('all')} className={inlineClearButtonClass}>Clear</button>
                  )}
                </div>
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
            {isTrendChartView && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Company Scope',
                  "Restrict the trend chart to companies mapped to a named UK Critical National Infrastructure sector. This excludes firms that don't have a direct mapping to a CNI sector."
                )}
                <div className="flex items-center gap-2">
                  {cniCompaniesOnlyToggle}
                  <p className="text-sm font-semibold leading-tight text-primary">Show only CNI companies</p>
                </div>
              </div>
            )}
          </div>
        </CollapsibleSection>

        {/* ── Display ── */}
        <CollapsibleSection title="Display" variant="settings" noDivider={isSignalQualityOpen}>
          <div className="space-y-5">
            {isTrendChartView && (
              <div className="flex items-center gap-2">
                {legendVisibilityToggle}
                <p className="text-sm font-semibold leading-tight text-primary">Show Legend</p>
              </div>
            )}
            {activeView === 1 && (
              <div className="flex items-center gap-2">
                {riskSubcategoriesToggle}
                <p className="text-sm font-semibold leading-tight text-primary">Show subcategories</p>
                <InfoTooltip content={
                  <>
                    <p>Splits the AI risk signal into specific taxonomy categories (e.g. cybersecurity, governance, model risk).</p>
                    <p className="mt-2">Because a single report can carry multiple risk labels, category percentages may not sum to 100%.</p>
                  </>
                } />
              </div>
            )}
            {isTrendChartView && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Chart Style',
                  'Bar chart stacks all categories. Line chart shows each category as a separate trend line. Line charts are only available in the yearly view.'
                )}
                {chartTypeToggle}
              </div>
            )}
            {isTrendChartView && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Y-Axis Metric',
                  'Choose what the Y-axis shows. % of all reports shows the share of all annual filings in each period that contain the selected label. % of AI-reporting reports shows the share among only the filings that mention AI at all. Number of reports (total) shows the raw number of matching annual reports. Number of AI mentions (total) shows the raw number of matching AI mentions. Percentage modes always use Per Report aggregation.'
                )}
                {metricModeToggle}
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
          </div>
        </CollapsibleSection>

        {/* ── Advanced ── */}
        <CollapsibleSection title="Advanced" variant="settings" noDivider={isSignalQualityOpen}>
          <div className="space-y-5">
            {isTrendChartView && (
              <div className="space-y-3">
                {renderSettingsSectionHeading('Group by')}
                {trendTimeToggle}
              </div>
            )}
            {!isSignalQualityOpen && effectiveMetricMode === 'count' && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Aggregate Data By',
                  'Aggregate by report (one annual filing = one data point) or by AI mention (each tagged AI mention counts separately). Per Report is the default and avoids double-counting a single filing.'
                )}
                {datasetToggle}
              </div>
            )}
            {!isSignalQualityOpen && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Signal Strength',
                  'Filter to a specific signal strength level. Explicit = directly stated in the text; Strong Implicit = clearly implied; Weak Implicit = faint or inferential evidence.'
                )}
                <div className="flex items-center gap-2">
                  <select
                    value={signalStrengthFilter}
                    onChange={e => setSignalStrengthFilter(e.target.value)}
                    className="min-w-0 flex-1 aisi-select"
                  >
                    <option value="all">All Levels</option>
                    {data.labels.riskSignalLevels.map(level => (
                      <option key={level} value={level}>{formatLabel(level)}</option>
                    ))}
                  </select>
                  {signalStrengthFilter !== 'all' && (
                    <button onClick={() => setSignalStrengthFilter('all')} className={inlineClearButtonClass}>Clear</button>
                  )}
                </div>
              </div>
            )}
            {!isSignalQualityOpen && (
              <div className="space-y-3">
                {renderSettingsSectionHeading(
                  'Substantiveness',
                  'Filter to reports with a specific disclosure quality. Substantive = concrete and detailed; Moderate = some specificity; Boilerplate = generic language only.'
                )}
                <div className="flex items-center gap-2">
                  <select
                    value={boilerplateFilter}
                    onChange={e => setBoilerplateFilter(e.target.value)}
                    className="min-w-0 flex-1 aisi-select"
                  >
                    <option value="all">All Levels</option>
                    {data.labels.substantivenessBands.map(band => (
                      <option key={band} value={band}>{formatLabel(band)}</option>
                    ))}
                  </select>
                  {boilerplateFilter !== 'all' && (
                    <button onClick={() => setBoilerplateFilter('all')} className={inlineClearButtonClass}>Clear</button>
                  )}
                </div>
              </div>
            )}
            <div className="space-y-3">
              {renderSettingsSectionHeading(
                'Signal Quality View',
                'A dedicated heatmap showing how explicit and substantive AI disclosures are across sectors and years. Use it to focus on the most credible signals and filter out boilerplate.'
              )}
              <p className="text-xs leading-relaxed text-muted-foreground">
                View a heatmap of signal quality and substantiveness levels per each year for {signalQualityScopeLabel}.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                {signalQualityViewButton}
              </div>
            </div>
          </div>
        </CollapsibleSection>

      </>)}
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
  const plainCitation = `AI Risk Observatory. "${currentVisualizationExport.title}" dashboard view, UK public-company annual report dataset. Accessed ${accessedOnLabel}. ${citationTargetUrl}`;
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
          className="absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:bg-slate-100 hover:text-primary"
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
        <pre className="overflow-x-auto whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 pr-11 font-mono text-xs leading-relaxed text-slate-700">
        {value}
        </pre>
      </div>
    </div>
  );

  const sharedCitationItem: RiskInfoPanelItem = {
    value: 'cite',
    label: 'Citations',
    title: '',
    content: (
      <div className="space-y-5 text-sm leading-relaxed text-slate-600">
        <p>
          AI Risk Observatory&apos;s data is free to use, distribute, and reproduce provided the source and authors are credited under the{' '}
          <a
            href="https://creativecommons.org/licenses/by/4.0/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline underline-offset-2 hover:opacity-75"
          >
            Creative Commons Attribution license
          </a>.
        </p>
        {renderCopyableReferenceBlock('citation-plain', 'Citation', plainCitation)}
        {renderCopyableReferenceBlock('citation-bibtex', 'BibTeX Citation', bibtexCitation)}
        {renderCopyableReferenceBlock('citation-link', 'Direct Link', citationTargetUrl)}
      </div>
    ),
  };

  const sharedFaqItem: RiskInfoPanelItem = {
    value: 'faq',
    label: 'FAQ',
    title: 'Frequently Asked Questions',
    content: (
      <div className="text-sm">
        <CollapsibleSection variant="faq" title="Does the adoption prompt only count adoption by the reporting company?" open={openFaqIndex === 0} onToggle={() => setOpenFaqIndex(openFaqIndex === 0 ? null : 0)}>
          The adoption prompt is designed to capture AI adoption by the reporting company. However, it does not
          impose a fully explicit entity-scope gate, so in ambiguous cases the classifier may also pick up adjacent
          forms of adoption described in the excerpt rather than only strict first-party company adoption.
        </CollapsibleSection>
        <CollapsibleSection variant="faq" title="How do we handle ambiguity?" open={openFaqIndex === 1} onToggle={() => setOpenFaqIndex(openFaqIndex === 1 ? null : 1)}>
          We handle ambiguity in two ways. First, Phase 2 labels use signal scores to show how directly the text
          supports a classification: higher signal means the type is explicit and well-supported, while lower signal
          means the label is more inferential or weakly evidenced. The{' '}
          <button
            type="button"
            onClick={openSignalQuality}
            className="text-primary underline underline-offset-2 hover:opacity-75"
          >
            Signal Quality view
          </button>
          {' '}surfaces these signal levels so you can filter to clearer disclosures only.
          Second, we score substantiveness from boilerplate to substantive, which separates generic AI language from
          disclosures that provide concrete implementation detail, metrics, or explanation. See{' '}
          <a href="/about#phase-2" className="text-primary underline underline-offset-2 hover:opacity-75">
            Phase 2: Detailed Classification
          </a>{' '}
          for the full rubric.
        </CollapsibleSection>
        <CollapsibleSection variant="faq" title="What is the most conservative way to read the data?" open={openFaqIndex === 2} onToggle={() => setOpenFaqIndex(openFaqIndex === 2 ? null : 2)}>
          Use the <strong className="text-slate-800">% of all reports</strong> metric with the{' '}
          <strong className="text-slate-800">Per Report</strong> aggregation. Both avoid double-counting and give
          the most conservative reading of prevalence — each annual filing counts as one data point regardless of
          how many AI mentions it contains. The <strong className="text-slate-800">% of AI-reporting reports</strong>{' '}
          option is useful for comparing the mix of labels within the AI-reporting subset, but it is not a whole-dataset prevalence measure.
        </CollapsibleSection>
        <CollapsibleSection variant="faq" title="What does 'publication year' mean?" open={openFaqIndex === 3} onToggle={() => setOpenFaqIndex(openFaqIndex === 3 ? null : 3)}>
          Years on the x-axis reflect when the report was filed or published, not the end of the company&apos;s
          fiscal year. A report covering fiscal year 2021 but published in April 2022 will appear under 2022. This
          is consistent across the entire dataset.
        </CollapsibleSection>
        <CollapsibleSection variant="faq" title="Why are some sectors missing from the heatmap?" open={openFaqIndex === 4} onToggle={() => setOpenFaqIndex(openFaqIndex === 4 ? null : 4)}>
          The heatmap only shows sectors with at least one report in the selected year range. If a CNI sector has
          no matching reports after filters are applied, it is omitted from the view. Switching to ISIC grouping or
          broadening the year range may reveal additional sectors.
        </CollapsibleSection>
        <CollapsibleSection variant="faq" title="Why do category counts add up to more than 100%?" open={openFaqIndex === 5} onToggle={() => setOpenFaqIndex(openFaqIndex === 5 ? null : 5)}>
          A single report or AI mention can be tagged with more than one category. A disclosure about a vendor deploying AI in a way that creates
          operational risk will be tagged for both Vendor and Risk. This means counts across categories are not
          mutually exclusive — a single report or AI mention can contribute to several bars or cells simultaneously. See{' '}
          <a href="/about#phase-1" className="text-primary underline underline-offset-2 hover:opacity-75">
            Phase 1: Mention Classification
          </a>{' '}
          in the methodology for how labels are assigned.
        </CollapsibleSection>
      </div>
    ),
  };

  const sharedDownloadItem: RiskInfoPanelItem = {
    value: 'download',
    label: 'Download',
    title: 'Download and Reuse',
    content: (
      <div className="space-y-3 text-sm leading-relaxed text-slate-600">
        <p>
          Download the complete dataset for offline analysis. The release is available under{' '}
          <a
            href="https://creativecommons.org/licenses/by/4.0/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline underline-offset-2 hover:opacity-75"
          >
            CC BY 4.0
          </a>
          , which permits reuse, redistribution, and adaptation with appropriate attribution. Raw annual report
          documents are not included in the release.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href="https://github.com/84rt/AI-Risk-Observatory/releases/download/dataset-v1.0/airo-dataset-v1.0.zip"
            className={inlinePanelButtonClass}
          >
            Download Dataset
          </a>
        </div>
      </div>
    ),
  };

  const riskInfoPanelItems: RiskInfoPanelItem[] = [
    {
      value: 'definitions',
      label: 'Categories',
      title: 'Risk Categories',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            A single report or AI mention can be tagged with more than one category when the disclosure covers multiple AI risk dimensions.
          </p>
          {renderTaxonomyLead('risk')}
          <div className="overflow-hidden rounded border border-slate-200">
            <table className="w-full border-collapse text-sm leading-relaxed text-slate-600">
              <tbody>
                {riskCategoryDefinitions.map(item => (
                  <tr key={item.label} className="border-t border-slate-200 odd:bg-white even:bg-slate-50 first:border-t-0">
                    <th scope="row" className="w-48 py-2 px-3 text-left align-top font-medium text-slate-800">
                      {item.label}
                    </th>
                    <td className="py-2 px-3 align-top">
                      {item.definition}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
    sharedFaqItem,
  ];

  const adoptionInfoPanelItems: RiskInfoPanelItem[] = [
    {
      value: 'definitions',
      label: 'Categories',
      title: 'Adoption Types',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            A single report or AI mention can be tagged with more than one adoption type when the disclosure covers multiple AI adoption dimensions.
          </p>
          {renderTaxonomyLead('adoption')}
          <div className="overflow-hidden rounded border border-slate-200">
            <table className="w-full border-collapse text-sm leading-relaxed text-slate-600">
              <tbody>
                {adoptionTypeDefinitions.map(item => (
                  <tr key={item.label} className="border-t border-slate-200 odd:bg-white even:bg-slate-50 first:border-t-0">
                    <th scope="row" className="w-48 py-2 px-3 text-left align-top font-medium text-slate-800">
                      {item.label}
                    </th>
                    <td className="py-2 px-3 align-top">
                      {item.definition}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
    sharedFaqItem,
  ];

  const vendorInfoPanelItems: RiskInfoPanelItem[] = [
    {
      value: 'definitions',
      label: 'Categories',
      title: 'Vendor Tags',
      content: (
        <>
          <p className="mb-3 text-sm leading-relaxed text-slate-500">
            A single report or AI mention can be tagged with more than one vendor tag when the disclosure covers multiple AI provider relationships.
          </p>
          {renderTaxonomyLead('vendor')}
          <div className="overflow-hidden rounded border border-slate-200">
            <table className="w-full border-collapse text-sm leading-relaxed text-slate-600">
              <tbody>
                {vendorTagDefinitions.map(item => (
                  <tr key={item.label} className="border-t border-slate-200 odd:bg-white even:bg-slate-50 first:border-t-0">
                    <th scope="row" className="w-48 py-2 px-3 text-left align-top font-medium text-slate-800">
                      {item.label}
                    </th>
                    <td className="py-2 px-3 align-top">
                      {item.definition}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ),
    },
    sharedCitationItem,
    sharedDownloadItem,
    sharedFaqItem,
  ];

  const signalQualityInfoPanelItems: RiskInfoPanelItem[] = [
    {
      value: 'definitions',
      label: 'Categories',
      title: 'Quality Metrics',
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
    sharedCitationItem,
    sharedDownloadItem,
    sharedFaqItem,
  ];

  const activeInfoPanelViewKey = isSignalQualityOpen ? 4 : activeView;
  const activeInfoPanelItems =
    isSignalQualityOpen
      ? signalQualityInfoPanelItems
      : activeView === 1
      ? riskInfoPanelItems
      : activeView === 2
        ? adoptionInfoPanelItems
        : activeView === 3
          ? vendorInfoPanelItems
          : signalQualityInfoPanelItems;

  const selectedInfoPanelKey = infoPanelSelections[activeInfoPanelViewKey] ?? 'cite';
  const selectedInfoPanel =
    activeInfoPanelItems.find(item => item.value === selectedInfoPanelKey) ?? activeInfoPanelItems[0];

  const infoPanelSection = (
    <section>
      <div className="grid gap-6 lg:grid-cols-[180px_minmax(0,1fr)] lg:gap-8">
        <div className="lg:pr-2">
          <div className="flex flex-wrap gap-2 lg:flex-col lg:gap-1.5">
            {activeInfoPanelItems.map(item => (
              <button
                key={item.value}
                type="button"
                onClick={() =>
                  setInfoPanelSelections(prev => ({
                    ...prev,
                    [activeInfoPanelViewKey]: item.value,
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
            <span className="aisi-tag">{isSignalQualityOpen ? 'Signal Quality' : view.title}</span>
            {selectedInfoPanel.title && (
              <h3 className="mt-3 text-lg font-semibold text-primary">{selectedInfoPanel.title}</h3>
            )}
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
        Per AI Mention
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
  const signalQualityViewButton = (
    <button
      type="button"
      onClick={openSignalQuality}
      className={`${tabButtonClass} ${
        isSignalQualityOpen
          ? 'border-primary bg-primary text-white'
          : 'border-border bg-secondary text-primary hover:bg-white'
      }`}
    >
      Open Signal Quality View
    </button>
  );

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

  const renderVisualizationArea = (visualization: ReactNode) => (
    <>
      <div className={isSettingsOpen ? 'relative min-[960px]:pr-[300px]' : ''}>
        <div className="min-w-0">
          {visualization}
        </div>
        {isSettingsOpen && (
          <aside className="hidden min-[960px]:absolute min-[960px]:top-0 min-[960px]:right-0 min-[960px]:block min-[960px]:w-[280px] min-[960px]:h-full">
            {renderSettingsPanel()}
          </aside>
        )}
      </div>
      {visualizationActions}
      {isSettingsOpen && <div className="min-[960px]:hidden">{renderSettingsPanel()}</div>}
      <div className="border-t border-border pt-8">
        {infoPanelSection}
      </div>
    </>
  );

  return (
    <div className="relative min-h-screen overflow-hidden bg-[linear-gradient(180deg,#ffffff_0%,#f7f9fc_52%,#f4f5f7_100%)] text-primary">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute inset-x-0 top-0 h-[26rem] bg-[radial-gradient(circle_at_top,rgba(199,216,238,0.48)_0%,rgba(199,216,238,0.22)_24%,rgba(199,216,238,0)_68%)]" />
      </div>
      <main className="relative mx-auto max-w-[1320px] px-6 py-10 sm:py-14">
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
            <div className="mt-3 max-w-full overflow-x-auto pb-4 [scrollbar-gutter:stable]">
              <p className="inline-block min-w-max whitespace-nowrap text-sm leading-relaxed text-muted sm:text-base">
                {datasetSummaryByView[activeView] ?? view.description}
              </p>
            </div>
          </div>

          <div className={isSettingsOpen ? 'mt-8 overflow-x-auto min-[960px]:pr-[300px]' : 'mt-8 overflow-x-auto'}>
            <div className="flex min-w-max items-start justify-between gap-3">
              <div className="flex shrink-0 items-center gap-2.5">
                {VIEWS.map(item => (
                  <div key={item.id} className="flex items-center gap-2.5">
                    <button
                      onClick={() => switchDashboardView(item.id)}
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
              {showVisualizationToggle && visualizationModeToggle}
            </div>
          </div>
        </div>

        <div className="mt-3">

        {isSignalQualityOpen ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3 rounded-lg border border-border bg-secondary px-4 py-3">
              <button
                type="button"
                onClick={() => setIsSignalQualityOpen(false)}
                className="inline-flex shrink-0 items-center gap-1.5 text-sm font-semibold text-primary transition hover:text-accent"
              >
                <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" aria-hidden="true">
                  <path d="M10 3L5 8L10 13" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Back to {VIEWS.find(v => v.id === activeView)?.title ?? 'main view'}
              </button>
              <span className="text-muted-foreground" aria-hidden="true">·</span>
              <p className="text-xs leading-relaxed text-muted-foreground">
                This is an advanced validation metric that allowes us to determene the strength of our findings.
              </p>
            </div>
            {renderVisualizationArea(
              <GenericHeatmap
                exportRef={visualizationExportRef}
                exportMode={isExportingVisualization}
                exportWatermark={exportWatermark}
                data={selectedSignalQualityHeatmap.data}
                xLabels={filteredYears}
                yLabels={selectedSignalQualityHeatmap.yLabels}
                baseColor={selectedSignalQualityHeatmap.baseColor}
                valueFormatter={selectedSignalQualityHeatmap.valueFormatter}
                yLabelFormatter={formatLabel}
                showTotals={true}
                totalsMode={selectedSignalQualityHeatmap.totalsMode}
                totalsLabel={selectedSignalQualityHeatmap.totalsLabel}
                totalValueFormatter={selectedSignalQualityHeatmap.totalValueFormatter}
                showBlindSpots={false}
                title={selectedSignalQualityHeatmap.title}
                subtitle={selectedSignalQualityHeatmap.subtitle}
                tooltip={selectedSignalQualityHeatmap.tooltip}
                xAxisLabel="Year"
                yAxisLabel={selectedSignalQualityHeatmap.yAxisLabel}
                compact={selectedSignalQualityHeatmap.compact}
                headerExtra={visualizationHeaderExtra}
                footerExtra={visualizationFooter}
              />
            )}
          </div>
        ) : (<>

        {activeView === 1 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              renderVisualizationArea(
                <StackedBarChart
                  key={`risk-trend-${trendTimeAxis}`}
                  exportRef={visualizationExportRef}
                  exportMode={isExportingVisualization}
                  exportWatermark={exportWatermark}
                  data={displayRiskTrend}
                  xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
                  stackKeys={riskChartStackKeys}
                  colors={riskChartColors}
                  yAxisTickFormatter={stackedChartYAxisFormatter}
                  tooltipValueFormatter={stackedChartTooltipFormatter}
                  yAxisDomain={isPercentageMetricMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={visualizationHeaderExtra}
                  legendKeys={riskBreakdownMode === 'phase1' ? ['risk'] : [...riskStackKeys].reverse()}
                  activeLegendKey={riskBreakdownMode === 'phase1' ? null : riskFilter === 'all' ? null : riskFilter}
                  onLegendItemClick={
                    riskBreakdownMode === 'phase1'
                      ? undefined
                      : (key) => setRiskFilter(prev => (prev === key ? 'all' : key))
                  }
                  footerExtra={visualizationFooter}
                  title={riskBreakdownMode === 'phase1' ? 'AI Risk Mentioned Over Time' : 'AI Risk Categories Mentioned Over Time'}
                  subtitle={
                    riskBreakdownMode === 'phase1'
                      ? isPercentageMetricMode
                        ? `Bar chart showing the percentage of ${percentageDenominatorLongLabel} with a phase 1 AI-risk signal (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis).`
                        : `Bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'AI mentions'} with a phase 1 AI-risk signal (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis).`
                      : isPercentageMetricMode
                      ? `Stacked bar chart showing the percentage of ${percentageDenominatorLongLabel} mentioning each AI risk category (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). Each coloured segment is the share of ${percentageDenominatorShortLabel} in that period tagged with the category.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'AI mentions'} mentioning each AI risk category (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). Each colour represents one risk category; bars are additive because a single ${datasetKey === 'perReport' ? 'report' : 'AI mention'} can be tagged with multiple categories. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    riskBreakdownMode === 'phase1'
                      ? <>
                          <p>This uses the phase 1 classifier signal for AI risk, before the phase 2 risk taxonomy split.</p>
                          <p className="mt-2">Switch to Categories in the settings panel to see the risk taxonomy breakdown.</p>
                        </>
                      : <>
                          <p>
                            {isPercentageMetricMode
                              ? `Each segment shows the share of ${percentageDenominatorShortLabel} in that period tagged with the risk category.`
                              : 'Each bar is stacked by risk category: the total height is the sum of all risk-category mentions that year, and each colour represents one category.'}
                          </p>
                          <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'AI mention'} can be tagged with multiple risk categories and therefore contribute to several coloured segments within the same year&apos;s bar; segments are not mutually exclusive.</p>
                          <p className="mt-2">Year-on-year growth may also reflect shifts in disclosure requirements or reporting culture rather than changes in actual risk levels — see the About page for more detail.</p>
                          <p className="mt-2">Click a legend item to isolate a single category.</p>
                        </>
                  }
                />
              )
            ) : (
              renderVisualizationArea(
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
                  title={
                    riskBreakdownMode === 'phase1'
                      ? 'Percentage of Companies Mentioning AI as a Risk by Sector and Year'
                      : riskFilter === 'all'
                        ? 'Risk Distribution by Sector'
                        : `${formatLabel(riskFilter)} Risk Mentions by Sector and Year`
                  }
                  subtitle={
                    riskBreakdownMode === 'phase1'
                      ? `Heatmap of phase 1 AI-risk report-share percentages by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and publication year (columns). Each cell shows the percentage of reports in that sector-year with an AI-risk mention signal.`
                      : riskFilter === 'all'
                        ? `Heatmap of report-share percentages by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and risk category (columns). Each cell shows the percentage of reports in that sector, across the selected years, that mention the risk type at least once.`
                        : `Heatmap of ${formatLabel(riskFilter)} report-share percentages by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and publication year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(riskFilter)}.`
                  }
                  tooltip={
                    riskBreakdownMode === 'phase1'
                      ? <>
                          <p>Each cell is the share of reports in that sector-year with a phase 1 AI-risk signal.</p>
                          <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                          <p className="mt-2">Switch to Categories in the settings panel to split this into phase 2 risk taxonomy labels.</p>
                        </>
                      : riskFilter === 'all'
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
                  xAxisLabel={riskHeatmapShowsCategoryColumns ? 'Risk Type' : 'Year'}
                  yAxisLabel={riskHeatmapAxisSectorLabel}
                  rowGroups={riskSectorView === 'isic' ? isicRowGroups : undefined}
                  expandedRowGroups={riskSectorView === 'isic' ? expandedIsicGroups : undefined}
                  onToggleRowGroup={riskSectorView === 'isic' ? toggleIsicGroup : undefined}
                  labelColumnWidth={riskSectorView === 'isic' ? 330 : undefined}
                  rowHeight={riskSectorView === 'isic' ? 58 : undefined}
                  yLabelClassName={riskSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
                  yLabelSubtext={riskSectorView === 'cni' ? data.companiesPerSector : undefined}
                  headerExtra={visualizationHeaderExtra}
                  footerExtra={visualizationFooter}
                />
              )
            )}
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              renderVisualizationArea(
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
                  yAxisDomain={isPercentageMetricMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={visualizationHeaderExtra}
                  legendKeys={adoptionLegendKeys}
                  activeLegendKey={adoptionFilter === 'all' ? null : adoptionFilter}
                  onLegendItemClick={(key) => setAdoptionFilter(prev => (prev === key ? 'all' : key))}
                  footerExtra={visualizationFooter}
                  title="AI Adoption Mentioned Over Time"
                  subtitle={
                    isPercentageMetricMode
                      ? `Stacked bar chart showing the percentage of ${percentageDenominatorLongLabel} referencing each AI adoption maturity level — Traditional AI (non-LLM), LLM, and Agentic — (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). A single report may be tagged with multiple adoption types, so stacked totals can exceed 100%.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'AI mentions'} referencing each AI adoption maturity level — Traditional AI (non-LLM), LLM, and Agentic — (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'AI mention'} may be tagged with multiple adoption types. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    <>
                      <p>{isPercentageMetricMode ? `Each segment shows the share of ${percentageDenominatorShortLabel} in that period tagged with the adoption type.` : 'Each bar is stacked by adoption type: Traditional AI (non-LLM), LLM, and Agentic.'}</p>
                      <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'AI mention'} can be tagged with multiple adoption types and can therefore contribute to more than one segment within the same year.</p>
                      <p className="mt-2">Click a legend item to isolate one adoption type.</p>
                    </>
                  }
                />
              )
            ) : (
              renderVisualizationArea(
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
                      : `Heatmap of ${formatLabel(adoptionFilter)} report-share percentages by ${adoptionSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and publication year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(adoptionFilter)}.`
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
                  yLabelSubtext={adoptionSectorView === 'cni' ? data.companiesPerSector : undefined}
                  headerExtra={visualizationHeaderExtra}
                  footerExtra={visualizationFooter}
                />
              )
            )}
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
            {visualizationMode === 'chart' ? (
              renderVisualizationArea(
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
                  yAxisDomain={isPercentageMetricMode ? [0, 100] : undefined}
                  allowLineChart={trendTimeAxis === 'year'}
                  showChartTypeToggle={false}
                  chartType={activeChartDisplayType}
                  onChartTypeChange={setChartDisplayType}
                  showLegend={showChartLegend}
                  legendPosition="right"
                  headerExtra={visualizationHeaderExtra}
                  legendKeys={vendorStackKeys}
                  activeLegendKey={effectiveVendorFilter === 'all' ? null : effectiveVendorFilter}
                  onLegendItemClick={(key) => setVendorFilter(prev => (prev === key ? 'all' : key))}
                  footerExtra={visualizationFooter}
                  title="AI Vendors Mentioned Over Time"
                  subtitle={
                    isPercentageMetricMode
                      ? `Stacked bar chart showing the percentage of ${percentageDenominatorLongLabel} referencing each AI vendor or provider tag (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). A single report may reference multiple vendors, so stacked totals can exceed 100%.`
                      : `Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'AI mentions'} referencing each AI vendor or provider tag (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'publication years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'AI mention'} may reference multiple vendors. The y-axis scale adjusts dynamically to the data shown.`
                  }
                  tooltip={
                    <>
                      <p>{isPercentageMetricMode ? `Each segment shows the share of ${percentageDenominatorShortLabel} in that period tagged with the vendor reference.` : 'Each bar is stacked by vendor tag (OpenAI, Microsoft, Google, Amazon / AWS, Meta, Anthropic, Internal, Other).'}</p>
                      <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'AI mention'} can include multiple vendor tags, so one item may contribute to more than one segment.</p>
                      <p className="mt-2"><span className="font-medium">Internal</span> means in-house AI development.</p>
                      <p className="mt-2">Click a legend item to isolate one vendor tag.</p>
                    </>
                  }
                />
              )
            ) : (
              renderVisualizationArea(
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
                      : `Heatmap of ${formatLabel(effectiveVendorFilter)} report-share percentages by ${vendorSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and publication year (columns). Each cell shows the percentage of reports in that sector-year mentioning ${formatLabel(effectiveVendorFilter)}.`
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
                  yLabelSubtext={vendorSectorView === 'cni' ? data.companiesPerSector : undefined}
                  headerExtra={visualizationHeaderExtra}
                  footerExtra={visualizationFooter}
                />
              )
            )}
          </div>
        )}

        </>)}
        </div>
      </main>
    </div>
  );
}
