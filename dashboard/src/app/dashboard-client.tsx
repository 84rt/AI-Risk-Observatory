'use client';

import { useMemo, useState } from 'react';
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

const VIEWS: View[] = [
  {
    id: 1,
    title: 'Risk',
    heading: 'Reports that mentioned risk from AI',
    description: 'AI risk categories over time and across sectors.',
  },
  {
    id: 2,
    title: 'Adoption',
    heading: 'Reports that mentioned adoption of AI',
    description: 'AI adoption type (non-LLM, LLM, agentic) across sectors and over time.',
  },
  {
    id: 3,
    title: 'Vendors',
    heading: 'Reports that mentioned vendors of AI technology',
    description: 'Which technology vendors companies name in their reports, and how that varies by sector.',
  },
  {
    id: 4,
    title: 'Signal Quality',
    heading: 'Metrics of findings quality and strength',
    description: 'How explicit and substantive each disclosure is — from concrete detail to boilerplate language.',
  },
  {
    id: 5,
    title: 'Blind Spots',
    heading: 'Reports that did not mention AI',
    description: 'Where disclosures are absent: reports that do not mention AI at all, and reports that do not mention AI risk.',
  },
];

const adoptionColors: Record<string, string> = {
  non_llm: '#f59e0b',     // amber-500
  llm: '#0ea5e9',         // sky-500
  agentic: '#7c3aed',     // violet-700
};

const vendorColors: Record<string, string> = {
  openai: '#7c3aed',      // violet-600 (purple)
  microsoft: '#3b82f6',   // blue-500
  google: '#22c55e',      // green-500
  internal: '#f97316',    // orange-500
  other: '#94a3b8',       // slate-400
  undisclosed: '#e2e8f0', // slate-200
};

const riskColors: Record<string, string> = {
  cybersecurity:            '#ef4444', // red-500
  operational_technical:    '#f97316', // orange-500
  regulatory_compliance:    '#f59e0b', // amber-500
  reputational_ethical:     '#14b8a6', // teal-500
  information_integrity:    '#0ea5e9', // sky-500
  third_party_supply_chain: '#22c55e', // green-500
  strategic_competitive:    '#84cc16', // lime-500
  workforce_impacts:        '#0f766e', // teal-700
  environmental_impact:     '#10b981', // emerald-500
  national_security:        '#7c3aed', // violet-700
};

const blindSpotColors: Record<string, string> = {
  no_ai_mention:      '#ef4444', // red-500
  no_ai_risk_mention: '#f59e0b', // amber-500
};

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
type SignalQualityFilter = 'all' | 'risk_signal' | 'adoption_signal' | 'vendor_signal' | 'substantiveness';
type BlindSpotFilter = 'all' | 'no_ai_mention' | 'no_ai_risk_mention';
type MetricMode = 'count' | 'pct_reports';
type ChartRow = Record<string, string | number | null | undefined>;
type HeatmapCell = { x: string | number; y: string | number; value: number };

const toNumber = (value: string | number | null | undefined) => Number(value) || 0;
const toPercent = (value: number, total: number) => (total > 0 ? (value / total) * 100 : 0);
const formatPercent = (value: number) => `${value.toFixed(1)}%`;
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
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('perReport');
  const [trendTimeAxis, setTrendTimeAxis] = useState<TrendTimeAxis>('year');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [adoptionFilter, setAdoptionFilter] = useState<AdoptionFilter>('all');
  const [riskSectorView, setRiskSectorView] = useState<RiskSectorView>('cni');
  const [adoptionSectorView, setAdoptionSectorView] = useState<RiskSectorView>('cni');
  const [vendorSectorView, setVendorSectorView] = useState<RiskSectorView>('cni');
  const [vendorFilter, setVendorFilter] = useState<string>('all');
  const [signalQualityFilter, setSignalQualityFilter] = useState<SignalQualityFilter>('all');
  const [blindSpotFilter, setBlindSpotFilter] = useState<BlindSpotFilter>('all');
  const [metricMode, setMetricMode] = useState<MetricMode>('count');
  const [marketSegmentFilter, setMarketSegmentFilter] = useState<string>('all');
  const [expandedIsicGroups, setExpandedIsicGroups] = useState<string[]>([]);

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
  const riskStackKeys = useMemo(() => data.labels.riskLabels, [data.labels.riskLabels]);
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
    ? '#64748b'
    : (riskColors[riskFilter] || '#64748b');
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
    ? '#64748b'
    : (adoptionColors[adoptionFilter] || '#64748b');
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
  const blindSpotHeatmapData = blindSpotFilter === 'no_ai_mention'
    ? noAiBySectorYearInRange
    : blindSpotFilter === 'no_ai_risk_mention'
      ? noAiRiskBySectorYearInRange
      : null;
  const displayBlindSpotHeatmapData = useMemo(() => {
    if (!blindSpotHeatmapData) return blindSpotHeatmapData;
    return convertHeatmapToPercent(
      blindSpotHeatmapData,
      cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
    );
  }, [blindSpotHeatmapData, reportTotalsBySectorYearMap]);
  const displayNoAiBySectorYearInRange = useMemo(() => {
    return convertHeatmapToPercent(
      noAiBySectorYearInRange,
      cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
    );
  }, [noAiBySectorYearInRange, reportTotalsBySectorYearMap]);
  const displayNoAiRiskBySectorYearInRange = useMemo(() => {
    return convertHeatmapToPercent(
      noAiRiskBySectorYearInRange,
      cell => reportTotalsBySectorYearMap.get(`${cell.x}|||${cell.y}`) || 0
    );
  }, [noAiRiskBySectorYearInRange, reportTotalsBySectorYearMap]);
  const blindSpotHeatmapTitle = blindSpotFilter === 'no_ai_mention'
    ? 'No AI Mention by Sector and Year'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'No AI Risk Mention by Sector and Year'
      : '';
  const blindSpotHeatmapSubtitle = blindSpotFilter === 'no_ai_mention'
    ? 'Heatmap of annual reports containing no AI mention, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with zero AI mention signal; colour intensity encodes relative frequency.'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'Heatmap of annual reports containing no AI risk disclosure, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with no AI risk mention; colour intensity encodes relative frequency.'
      : '';
  const blindSpotHeatmapTooltip = blindSpotFilter === 'no_ai_mention'
    ? 'Each cell is the share of reports in that sector-year that do not mention AI at all. The Avg column and row show simple averages across the displayed years and sectors.'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'Each cell is the share of reports in that sector-year with no AI risk mention. The Avg column and row show simple averages across the displayed years and sectors.'
      : '';
  const blindSpotHeatmapColor = '#64748b';
  const showRiskSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'risk_signal';
  const showAdoptionSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'adoption_signal';
  const showVendorSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'vendor_signal';
  const showSubstantivenessPanel = signalQualityFilter === 'all' || signalQualityFilter === 'substantiveness';
  const signalPanelGridClass = 'grid gap-8';
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

  const makeSectorToggle = (
    current: RiskSectorView,
    setter: (v: RiskSectorView) => void
  ) => (
    <div className="inline-flex h-9 items-center rounded-md border border-slate-200 bg-white/90 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => setter('cni')}
        className={`h-8 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
          current === 'cni'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        CNI
      </button>
      <button
        type="button"
        onClick={() => setter('isic')}
        className={`h-8 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
          current === 'isic'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        ISIC
      </button>
    </div>
  );

  const trendTimeToggle = (
    <div className="inline-flex h-9 items-center rounded-md border border-slate-200 bg-white/90 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => setTrendTimeAxis('year')}
        className={`h-8 rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          trendTimeAxis === 'year'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        Year
      </button>
      <button
        type="button"
        onClick={() => setTrendTimeAxis('month')}
        className={`h-8 rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          trendTimeAxis === 'month'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        Month
      </button>
    </div>
  );

  const metricModeToggle = canShowReportShare ? (
    <div className="inline-flex h-9 items-center rounded-md border border-slate-200 bg-white/90 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => setMetricMode('count')}
        className={`h-8 rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          effectiveMetricMode === 'count'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        Count
      </button>
      <button
        type="button"
        onClick={() => setMetricMode('pct_reports')}
        className={`h-8 rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          effectiveMetricMode === 'pct_reports'
            ? 'bg-amber-500 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        % of Reports
      </button>
    </div>
  ) : null;

  const combinedToggles = (
    <div className="flex items-center gap-2">
      {trendTimeToggle}
      {metricModeToggle}
    </div>
  );

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      {/* Sticky control bar */}
      <div className="sticky top-0 z-20 border-b border-slate-200 bg-[#f6f3ef]/70 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-6">
          {/* Row 1: View tabs */}
          <div className="flex items-center gap-1 py-2">
            {VIEWS.map(item => (
              <button
                key={item.id}
                onClick={() => {
                  setActiveView(item.id);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`relative h-9 rounded-md px-3.5 text-sm font-medium transition-all ${
                  activeView === item.id
                    ? 'bg-white text-slate-900 shadow-sm ring-1 ring-slate-200/80'
                    : 'text-slate-500 hover:bg-white/60 hover:text-slate-900'
                }`}
              >
                {item.title}
                {activeView === item.id && (
                  <span className="absolute inset-x-2 -bottom-1 h-0.5 rounded-full bg-amber-500" />
                )}
              </button>
            ))}
          </div>

          {/* Row 2: Year range + dataset toggle + view-specific controls */}
          <div className="flex flex-wrap items-center gap-2 border-t border-slate-200/60 py-2">
            <div className="h-9 w-[240px] shrink-0 flex flex-col justify-center rounded-md border border-slate-200 bg-white/90 px-3 shadow-sm sm:w-[320px] lg:w-[340px]">
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
                <div className="absolute left-0 right-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-slate-200" />
                <div
                  className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-amber-500"
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
              <div className="relative h-3 text-[9px] font-semibold uppercase tracking-[0.1em] text-slate-500">
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
                          index >= startIndex && index <= endIndex ? 'text-slate-900' : 'text-slate-500'
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

            {/* Dataset toggle */}
            {activeView !== 5 && (
              <select
                value={datasetKey}
                onChange={event => {
                  const nextDatasetKey = event.target.value as DatasetKey;
                  setDatasetKey(nextDatasetKey);
                  const nextYears = data.datasets[nextDatasetKey].years;
                  setYearRangeIndices({ start: 0, end: Math.max(nextYears.length - 1, 0) });
                }}
                className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
              >
                <option value="perReport">Per Report</option>
                <option value="perChunk">Per Excerpt</option>
              </select>
            )}

            {/* Market segment filter */}
            <select
              value={marketSegmentFilter}
              onChange={event => {
                setMarketSegmentFilter(event.target.value);
                setYearRangeIndices({ start: 0, end: Math.max(data.years.length - 1, 0) });
              }}
              className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
            >
              <option value="all">All Companies</option>
              {data.marketSegments.map(segment => (
                <option key={segment} value={segment}>{segment}</option>
              ))}
            </select>

            {/* Risk-specific controls */}
            {activeView === 1 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="risk-filter"
                  value={riskFilter}
                  onChange={e => setRiskFilter(e.target.value)}
                  className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Risk Types</option>
                  {data.labels.riskLabels.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {riskFilter !== 'all' && (
                  <button
                    onClick={() => setRiskFilter('all')}
                    className="h-9 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </>
            )}

            {/* Adoption-specific controls */}
            {activeView === 2 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="adoption-filter"
                  value={adoptionFilter}
                  onChange={e => setAdoptionFilter(e.target.value)}
                  className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Adoption Types</option>
                  {data.labels.adoptionTypes.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {adoptionFilter !== 'all' && (
                  <button
                    onClick={() => setAdoptionFilter('all')}
                    className="h-9 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </>
            )}

            {/* Vendor-specific controls */}
            {activeView === 3 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="vendor-filter"
                  value={effectiveVendorFilter}
                  onChange={e => setVendorFilter(e.target.value)}
                  className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Vendors</option>
                  {vendorStackKeys.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {effectiveVendorFilter !== 'all' && (
                  <button
                    onClick={() => setVendorFilter('all')}
                    className="h-9 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </>
            )}

            {/* Signal-quality specific controls */}
            {activeView === 4 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="signal-quality-filter"
                  value={signalQualityFilter}
                  onChange={e => setSignalQualityFilter(e.target.value as SignalQualityFilter)}
                  className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Quality Panels</option>
                  <option value="risk_signal">Risk Signal Strength</option>
                  <option value="adoption_signal">Adoption Signal Strength</option>
                  <option value="vendor_signal">Vendor Signal Strength</option>
                  <option value="substantiveness">Risk Substantiveness</option>
                </select>
                {signalQualityFilter !== 'all' && (
                  <button
                    onClick={() => setSignalQualityFilter('all')}
                    className="h-9 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </>
            )}

            {/* Blind-spot specific controls */}
            {activeView === 5 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="blind-spot-filter"
                  value={blindSpotFilter}
                  onChange={e => setBlindSpotFilter(e.target.value as BlindSpotFilter)}
                  className="h-9 rounded-md border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Blind Spots</option>
                  <option value="no_ai_mention">No AI Mention</option>
                  <option value="no_ai_risk_mention">No AI Risk Mention</option>
                </select>
                {blindSpotFilter !== 'all' && (
                  <button
                    onClick={() => setBlindSpotFilter('all')}
                    className="h-9 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-6 py-12">
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-slate-900">
            <span className="mr-2 inline-block h-5 w-1 rounded-full bg-amber-500 align-middle" />
            {view.heading}
          </h2>
          {activeView === 1 ? (
            <div className="mt-2 max-w-5xl space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
              <p>
                <span className="font-semibold text-slate-900">{formatNumber(riskOverviewStats.riskMentionReports)}</span>{' '}
                of the{' '}
                <span className="font-semibold text-slate-900">{formatNumber(riskOverviewStats.totalReports)}</span>{' '}
                annual reports examined between{' '}
                <span className="font-semibold text-slate-900">{riskSelectedYearSpan}</span>{' '}
                include at least one AI risk disclosure, across{' '}
                <span className="font-semibold text-slate-900">{formatNumber(riskOverviewStats.excerptRiskMentions)}</span>{' '}
                individual passages.
              </p>
              <p>
                <span className="font-medium text-slate-800">Per Report</span> counts each company filing once —
                useful for measuring how broadly a risk is disclosed across the market.{' '}
                <span className="font-medium text-slate-800">Per Excerpt</span> counts every individual passage where
                a risk from AI appears — useful for visualising the depth of emphasis companies place on that risk.
              </p>
              <div>
                You can use the controls above to narrow the date range, focus on a specific risk category, or switch
                the sector taxonomy between{' '}
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
          ) : activeView === 2 ? (
            <div className="mt-2 max-w-5xl space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
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
                Use the controls above to narrow the year range and focus on a single adoption type (
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
          ) : activeView === 3 ? (
            <div className="mt-2 max-w-5xl space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
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
                Use the control above to focus on one vendor tag (
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
          ) : activeView === 4 ? (
            <div className="mt-2 max-w-5xl space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
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
                Use the control above to focus on a single quality panel, or keep all panels visible to compare
                risk/adoption/vendor explicitness against risk substantiveness side-by-side.
              </p>
            </div>
          ) : activeView === 5 ? (
            <div className="mt-2 max-w-5xl space-y-3 text-sm leading-relaxed text-slate-600 sm:text-base">
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
                Use the filter above to focus on one blind-spot type. The trend chart and heatmap will then isolate that
                specific absence pattern by year and sector.
              </p>
            </div>
          ) : (
            <p className="mt-2 max-w-3xl text-slate-600">{view.description}</p>
          )}
        </div>

        {activeView === 1 && (
          <div className="space-y-8">
            {/* Risk Trend Over Time */}
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
              showChartTypeToggle
              legendPosition="right"
              legendKeys={riskStackKeys}
              activeLegendKey={riskFilter === 'all' ? null : riskFilter}
              onLegendItemClick={(key) => setRiskFilter(prev => (prev === key ? 'all' : key))}
              title="AI Risk Mentioned Over Time"
              headerExtra={combinedToggles}
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

            {/* Risk by Sector Heatmap */}
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
                      <p className="mt-2">Select one risk type above to switch the columns from risk categories to years.</p>
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
              headerExtra={makeSectorToggle(riskSectorView, setRiskSectorView)}
            />

            <details className="group rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-semibold text-slate-900 marker:hidden [&::-webkit-details-marker]:hidden">
                Risk Category Definitions
                <svg className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </summary>
              <div className="border-t border-slate-100 px-6 pb-5 pt-4">
                <p className="mb-3 text-sm leading-relaxed text-slate-500">
                  Categories were assigned by an LLM-assisted classifier trained on an AI risk taxonomy. A single report or passage can be tagged with more than one category.
                </p>
                <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
                  <li><span className="font-medium text-slate-800">Cybersecurity:</span> Risks relating to data breaches, AI-enabled attacks, or vulnerabilities introduced by deploying AI systems.</li>
                  <li><span className="font-medium text-slate-800">Operational / Technical:</span> Risks of system failures, integration problems, or performance degradation arising from AI implementation.</li>
                  <li><span className="font-medium text-slate-800">Regulatory / Compliance:</span> Risks from evolving compliance obligations, legal liability, or uncertainty in the regulatory landscape for AI.</li>
                  <li><span className="font-medium text-slate-800">Reputational / Ethical:</span> Risks of brand damage, public concern over algorithmic bias, or broader ethical considerations in AI deployment.</li>
                  <li><span className="font-medium text-slate-800">Information Integrity:</span> Risks from AI-generated misinformation, model hallucinations, or degraded data quality affecting decision-making.</li>
                  <li><span className="font-medium text-slate-800">Third-Party Supply Chain:</span> Risks arising from dependence on external AI vendors, APIs, or suppliers whose reliability or conduct is outside the company&apos;s direct control.</li>
                  <li><span className="font-medium text-slate-800">Strategic / Competitive:</span> Risks of competitive displacement, market disruption, or falling behind peers in AI adoption and innovation.</li>
                  <li><span className="font-medium text-slate-800">Workforce Impacts:</span> Risks relating to job displacement, emerging skills gaps, or changes in labour relations driven by AI automation.</li>
                  <li><span className="font-medium text-slate-800">Environmental Impact:</span> Risks associated with the energy consumption, carbon footprint, or resource demands of AI infrastructure.</li>
                  <li><span className="font-medium text-slate-800">National Security:</span> Risks to critical systems, geopolitical exposure, or security-of-state concerns linked to AI deployment or dependency.</li>
                </ul>
              </div>
            </details>
          </div>
        )}

        {activeView === 2 && (
          <div className="space-y-8">
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
              showChartTypeToggle
              legendPosition="right"
              legendKeys={adoptionStackKeys}
              activeLegendKey={adoptionFilter === 'all' ? null : adoptionFilter}
              onLegendItemClick={(key) => setAdoptionFilter(prev => (prev === key ? 'all' : key))}
              title="AI Adoption Mentioned Over Time"
              headerExtra={combinedToggles}
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
                      <p className="mt-2">Select one adoption type above to switch the columns from adoption types to years.</p>
                    </>
                  : <>
                      <p>This filtered view shows the share of reports in each sector-year mentioning one adoption type.</p>
                      <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                      <p className="mt-2">Clearing the filter returns the categorical sector view.</p>
                    </>
              }
              xAxisLabel={adoptionFilter === 'all' ? 'Adoption Type' : 'Year'}
              yAxisLabel={adoptionSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
              headerExtra={makeSectorToggle(adoptionSectorView, setAdoptionSectorView)}
              rowGroups={adoptionSectorView === 'isic' ? isicRowGroups : undefined}
              expandedRowGroups={adoptionSectorView === 'isic' ? expandedIsicGroups : undefined}
              onToggleRowGroup={adoptionSectorView === 'isic' ? toggleIsicGroup : undefined}
              labelColumnWidth={adoptionSectorView === 'isic' ? 330 : undefined}
              rowHeight={adoptionSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={adoptionSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
            />

            <details className="group rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-semibold text-slate-900 marker:hidden [&::-webkit-details-marker]:hidden">
                Adoption Type Definitions
                <svg className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </summary>
              <div className="border-t border-slate-100 px-6 pb-5 pt-4">
                <p className="mb-3 text-sm leading-relaxed text-slate-500">
                  Adoption labels describe the maturity of AI deployment referenced in each disclosure.
                </p>
                <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
                  <li><span className="font-medium text-slate-800">Non-LLM:</span> Traditional AI or machine-learning systems that are not based on large language models.</li>
                  <li><span className="font-medium text-slate-800">LLM:</span> Large language model usage, including generative AI assistants and model-based workflows.</li>
                  <li><span className="font-medium text-slate-800">Agentic:</span> More autonomous AI systems, often coordinating multi-step tasks with limited human intervention.</li>
                </ul>
              </div>
            </details>
          </div>
        )}

        {activeView === 3 && (
          <div className="space-y-8">
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
              showChartTypeToggle
              legendPosition="right"
              legendKeys={vendorStackKeys}
              activeLegendKey={effectiveVendorFilter === 'all' ? null : effectiveVendorFilter}
              onLegendItemClick={(key) => setVendorFilter(prev => (prev === key ? 'all' : key))}
              title="AI Vendors Mentioned Over Time"
              headerExtra={combinedToggles}
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
                      <p className="mt-2">Select one vendor above to switch columns from vendor tags to years.</p>
                    </>
                  : <>
                      <p>This filtered view shows the share of reports in each sector-year mentioning one vendor tag.</p>
                      <p className="mt-2">The Avg column and Avg row show simple averages across the displayed years and sectors.</p>
                      <p className="mt-2">Clear the filter to return to the all-vendor categorical view.</p>
                    </>
              }
              xAxisLabel={effectiveVendorFilter === 'all' ? 'Vendor' : 'Year'}
              yAxisLabel={vendorSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
              headerExtra={makeSectorToggle(vendorSectorView, setVendorSectorView)}
              rowGroups={vendorSectorView === 'isic' ? isicRowGroups : undefined}
              expandedRowGroups={vendorSectorView === 'isic' ? expandedIsicGroups : undefined}
              onToggleRowGroup={vendorSectorView === 'isic' ? toggleIsicGroup : undefined}
              labelColumnWidth={vendorSectorView === 'isic' ? 330 : undefined}
              rowHeight={vendorSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={vendorSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
            />

            <details className="group rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-semibold text-slate-900 marker:hidden [&::-webkit-details-marker]:hidden">
                Vendor Tag Definitions
                <svg className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </summary>
              <div className="border-t border-slate-100 px-6 pb-5 pt-4">
                <p className="mb-3 text-sm leading-relaxed text-slate-500">
                  Vendor tags indicate which provider is named (or implied) in the disclosure text.
                </p>
                <ul className="space-y-2 text-sm leading-relaxed text-slate-600">
                  <li><span className="font-medium text-slate-800">OpenAI / Microsoft / Google:</span> The provider is explicitly named in the text.</li>
                  <li><span className="font-medium text-slate-800">Internal:</span> The company describes AI as built or operated in-house.</li>
                  <li><span className="font-medium text-slate-800">Other:</span> A named provider outside the primary tracked vendor set.</li>
                </ul>
              </div>
            </details>
          </div>
        )}

        {activeView === 4 && (
          <div className="space-y-8">
            {(showRiskSignalPanel || showAdoptionSignalPanel || showVendorSignalPanel) && (
            <div className={signalPanelGridClass}>
              {showRiskSignalPanel && (
              <GenericHeatmap
                data={riskSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#64748b"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="AI Risk Signal Strength"
                subtitle="Heatmap of risk-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level risk classifications fell into that strength tier in a given year; colour intensity encodes relative frequency."
                tooltip="Risk signal strength scores how directly the text supports a risk classification. 3 = explicit statement; 2 = strong implicit evidence; 1 = weak implicit evidence. Each cell counts label-level outcomes, not unique reports."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
              )}
              {showAdoptionSignalPanel && (
              <GenericHeatmap
                data={adoptionSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#64748b"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="AI Adoption Signal Strength"
                subtitle="Heatmap of adoption-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level adoption classifications fell into that strength tier in a given year; colour intensity encodes relative frequency."
                tooltip="Applies the same signal-strength rubric to AI adoption mentions. Higher rows indicate clearer, more directly supported adoption disclosures, while lower rows reflect softer inferential language."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
              )}
              {showVendorSignalPanel && (
              <GenericHeatmap
                data={vendorSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#64748b"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="AI Vendor Signal Strength"
                subtitle="Heatmap of vendor-classification signal counts by signal-strength level (rows: Explicit, Strong Implicit, Weak Implicit) and fiscal year (columns). Each cell counts how many label-level vendor classifications fell into that strength tier in a given year; colour intensity encodes relative frequency."
                tooltip="Measures how directly a vendor relationship is stated in the text. Low explicitness can indicate more opaque supplier disclosure; higher explicit counts suggest clearer provider attribution."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
              )}
            </div>
            )}
            {showSubstantivenessPanel && (
            <GenericHeatmap
              data={substantivenessHeatmapInRange}
              xLabels={filteredYears}
              yLabels={data.labels.substantivenessBands}
              baseColor="#64748b"
              valueFormatter={value => `${value}`}
              yLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={false}
              title="AI Risk Substantiveness Distribution"
              subtitle="Heatmap of report-level risk-disclosure quality by substantiveness band (rows: Substantive, Moderate, Boilerplate) and fiscal year (columns). Each cell counts the number of reports whose AI-risk language was classified into that quality tier in a given year; colour intensity encodes relative frequency."
              tooltip="Substantiveness measures depth and specificity of risk disclosure at report level. Substantive disclosures include concrete mechanisms and mitigation/action detail, while boilerplate disclosures remain generic."
              xAxisLabel="Year"
              yAxisLabel="Quality Band"
            />
            )}
            <details className="group rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-semibold text-slate-900 marker:hidden [&::-webkit-details-marker]:hidden">
                Quality Metric Definitions
                <svg className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </summary>
              <div className="border-t border-slate-100 px-6 pb-5 pt-4 text-sm text-slate-600">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <p className="font-medium text-slate-800">Signal Strength (Risk / Adoption / Vendor):</p>
                    <p className="mt-1 leading-relaxed">
                      Signal strength captures how explicitly a label is supported by the text.
                      Each cell counts label-level classifications, and the strongest available signal per label is retained.
                    </p>
                    <ul className="mt-2 space-y-0.5 text-xs">
                      <li><span className="font-medium">3 Explicit:</span> direct, named, concrete statement</li>
                      <li><span className="font-medium">2 Strong implicit:</span> clear but inferential evidence</li>
                      <li><span className="font-medium">1 Weak implicit:</span> plausible but lightly supported evidence</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium text-slate-800">Substantiveness:</p>
                    <p className="mt-1 leading-relaxed">
                      Substantiveness evaluates the quality of AI-risk disclosure at report level.
                    </p>
                    <ul className="mt-2 space-y-0.5 text-xs">
                      <li><span className="font-medium">Substantive:</span> concrete mechanism + tangible mitigation/action</li>
                      <li><span className="font-medium">Moderate:</span> specific risk area, limited detail</li>
                      <li><span className="font-medium">Boilerplate:</span> generic risk language without concrete detail</li>
                    </ul>
                  </div>
                </div>
              </div>
            </details>
          </div>
        )}

        {activeView === 5 && (
          <div className="space-y-8">
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
              showChartTypeToggle
              legendPosition="right"
              legendKeys={['no_ai_mention', 'no_ai_risk_mention']}
              activeLegendKey={blindSpotFilter === 'all' ? null : blindSpotFilter}
              onLegendItemClick={(key) =>
                setBlindSpotFilter(prev => (prev === key ? 'all' : (key as BlindSpotFilter)))
              }
              title={trendTimeAxis === 'month' ? 'Blind Spots by Month' : 'Blind Spots by Year'}
              headerExtra={combinedToggles}
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
                    ? 'This report-level view compares the share of reports exhibiting two absence patterns: no AI mention at all vs. no AI risk mention. Use the legend or top filter to isolate one series.'
                    : 'This report-level view compares two absence patterns: no AI mention at all vs. no AI risk mention. Use the legend or top filter to isolate one series.'
                  : `${formatLabel(blindSpotFilter)} is currently isolated. Clear the filter to compare both blind-spot types together.`
              }
            />

            {blindSpotFilter === 'all' ? (
              <div className="grid gap-8 xl:grid-cols-2">
                <GenericHeatmap
                  data={displayNoAiBySectorYearInRange}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor="#64748b"
                  valueFormatter={blindSpotHeatmapValueFormatter}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={false}
                  title="No AI Mention by Sector and Year"
                  subtitle="Heatmap of annual reports containing no AI mention, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with zero AI mention signal; colour intensity encodes relative frequency."
                  tooltip="Each cell is the share of reports in that sector-year that do not mention AI at all. The Avg column and row show simple averages across the displayed years and sectors."
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                />
                <GenericHeatmap
                  data={displayNoAiRiskBySectorYearInRange}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor="#64748b"
                  valueFormatter={blindSpotHeatmapValueFormatter}
                  showTotals={true}
                  totalsMode="average"
                  totalsLabel="Avg"
                  showBlindSpots={false}
                  title="No AI Risk Mention by Sector and Year"
                  subtitle="Heatmap of annual reports containing no AI risk disclosure, by CNI sector (rows) and fiscal year (columns). Each cell shows the percentage of reports in that sector-year with no AI risk mention; colour intensity encodes relative frequency."
                  tooltip="Each cell is the share of reports in that sector-year with no AI risk mention. The Avg column and row show simple averages across the displayed years and sectors."
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                />
              </div>
            ) : (
              displayBlindSpotHeatmapData && (
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
                />
              )
            )}

            <details className="group rounded-2xl border border-slate-200/80 bg-white/90 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]">
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-semibold text-slate-900 marker:hidden [&::-webkit-details-marker]:hidden">
                Blind Spot Definitions
                <svg className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </summary>
              <div className="border-t border-slate-100 px-6 pb-5 pt-4 text-sm leading-relaxed text-slate-600">
                <p>
                  Blind spot metrics are calculated from report-level coverage (one annual report per company-year baseline),
                  including reports with zero extracted AI passages.
                </p>
                <ul className="mt-3 space-y-1 text-xs">
                  <li><span className="font-medium text-slate-800">No AI Mention:</span> no AI disclosure signal appears in the report.</li>
                  <li><span className="font-medium text-slate-800">No AI Risk Mention:</span> AI may be mentioned, but no AI-risk disclosure is present.</li>
                </ul>
              </div>
            </details>
          </div>
        )}

      </main>
    </div>
  );
}
