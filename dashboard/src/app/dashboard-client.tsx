'use client';

import { useMemo, useState } from 'react';
import { GenericHeatmap, StackedBarChart, InfoTooltip } from '@/components/overview-charts';
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
    heading: 'AI Risk Mentioned',
    description: 'AI risk categories over time and across sectors.',
  },
  {
    id: 2,
    title: 'Adoption',
    heading: 'Adoption Type Mentioned',
    description: 'AI adoption type (non-LLM, LLM, agentic) across sectors and over time.',
  },
  {
    id: 3,
    title: 'Vendors',
    heading: 'Vendors',
    description: 'Which technology vendors companies name in their reports, and how that varies by sector.',
  },
  {
    id: 4,
    title: 'Signal Quality',
    heading: 'Signal Quality',
    description: 'How explicit and substantive each disclosure is — from concrete detail to boilerplate language.',
  },
  {
    id: 5,
    title: 'Blind Spots',
    heading: 'Blind Spots',
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

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const activeData = data.datasets[datasetKey];
  const reportBaselineData = data.datasets.perReport;
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
      activeData.riskBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.riskBySectorYear, selectedStartYear, selectedEndYear]
  );

  const riskByIsicSectorYearInRange = useMemo(
    () =>
      activeData.riskByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.riskByIsicSectorYear, selectedStartYear, selectedEndYear]
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
      activeData.adoptionBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.adoptionBySectorYear, selectedStartYear, selectedEndYear]
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
      activeData.adoptionByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.adoptionByIsicSectorYear, selectedStartYear, selectedEndYear]
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
      activeData.vendorBySectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.vendorBySectorYear, selectedStartYear, selectedEndYear]
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
      activeData.vendorByIsicSectorYear.filter(
        cell => cell.year >= selectedStartYear && cell.year <= selectedEndYear
      ),
    [activeData.vendorByIsicSectorYear, selectedStartYear, selectedEndYear]
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

    const excerptRiskMentions = data.datasets.perChunk.mentionTrend.reduce((sum, row) => {
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
    data.datasets.perChunk.mentionTrend,
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

    const excerptAdoptionMentions = data.datasets.perChunk.mentionTrend.reduce((sum, row) => {
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
    data.datasets.perChunk.mentionTrend,
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

    const excerptVendorMentions = data.datasets.perChunk.mentionTrend.reduce((sum, row) => {
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
    data.datasets.perChunk.mentionTrend,
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

  const riskHeatmapYLabels = riskSectorView === 'cni' ? data.sectors : data.isicSectors;
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
  const adoptionHeatmapYLabels = adoptionSectorView === 'cni' ? data.sectors : data.isicSectors;
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
  const vendorHeatmapYLabels = vendorSectorView === 'cni' ? data.sectors : data.isicSectors;
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
  const filteredBlindSpotTrend = useMemo(() => {
    if (blindSpotFilter === 'all') return blindSpotTrendInRange;
    return blindSpotTrendInRange.map(row => ({
      year: row.year,
      [blindSpotFilter]: row[blindSpotFilter] || 0,
    }));
  }, [blindSpotFilter, blindSpotTrendInRange]);
  const blindSpotHeatmapData = blindSpotFilter === 'no_ai_mention'
    ? noAiBySectorYearInRange
    : blindSpotFilter === 'no_ai_risk_mention'
      ? noAiRiskBySectorYearInRange
      : null;
  const blindSpotHeatmapTitle = blindSpotFilter === 'no_ai_mention'
    ? 'No AI Mention by Sector and Year'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'No AI Risk Mention by Sector and Year'
      : '';
  const blindSpotHeatmapSubtitle = blindSpotFilter === 'no_ai_mention'
    ? 'Heatmap of annual reports containing no AI mention, by CNI sector (rows) and fiscal year (columns). Each cell shows the count of reports in that sector-year with zero AI mention signal; colour intensity encodes relative frequency.'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'Heatmap of annual reports containing no AI risk disclosure, by CNI sector (rows) and fiscal year (columns). Each cell shows the count of reports in that sector-year with no AI risk mention; colour intensity encodes relative frequency.'
      : '';
  const blindSpotHeatmapTooltip = blindSpotFilter === 'no_ai_mention'
    ? 'Each cell is the number of reports in that sector-year that do not mention AI at all.'
    : blindSpotFilter === 'no_ai_risk_mention'
      ? 'Each cell is the number of reports in that sector-year with no AI risk mention.'
      : '';
  const blindSpotHeatmapColor = '#64748b';
  const showRiskSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'risk_signal';
  const showAdoptionSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'adoption_signal';
  const showVendorSignalPanel = signalQualityFilter === 'all' || signalQualityFilter === 'vendor_signal';
  const showSubstantivenessPanel = signalQualityFilter === 'all' || signalQualityFilter === 'substantiveness';
  const visibleSignalPanelCount = [
    showRiskSignalPanel,
    showAdoptionSignalPanel,
    showVendorSignalPanel,
  ].filter(Boolean).length;
  const signalPanelGridClass =
    visibleSignalPanelCount <= 1
      ? 'grid gap-8'
      : visibleSignalPanelCount === 2
        ? 'grid gap-8 md:grid-cols-2'
        : 'grid gap-8 lg:grid-cols-3';
  const riskSelectedYearSpan = filteredYears.length > 0
    ? `${selectedStartYear}–${selectedEndYear}`
    : 'N/A';

  const makeSectorToggle = (
    current: RiskSectorView,
    setter: (v: RiskSectorView) => void
  ) => (
    <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white/90 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => setter('cni')}
        className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
          current === 'cni'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        CNI
      </button>
      <button
        type="button"
        onClick={() => setter('isic')}
        className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
          current === 'isic'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        ISIC
      </button>
    </div>
  );

  const trendTimeToggle = (
    <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white/90 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => setTrendTimeAxis('year')}
        className={`rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          trendTimeAxis === 'year'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        Year
      </button>
      <button
        type="button"
        onClick={() => setTrendTimeAxis('month')}
        className={`rounded-md px-2.5 py-1 text-xs font-semibold transition ${
          trendTimeAxis === 'month'
            ? 'bg-slate-900 text-white'
            : 'text-slate-600 hover:bg-slate-100'
        }`}
      >
        Month
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      {/* Sticky control bar */}
      <div className="sticky top-0 z-20 border-b border-slate-200 bg-[#f6f3ef]/70 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-6">
          {/* Row 1: View tabs */}
          <div className="flex items-center gap-2 py-2">
            {VIEWS.map(item => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-all ${
                  activeView === item.id
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-slate-500 hover:bg-white hover:text-slate-900'
                }`}
              >
                {item.title}
              </button>
            ))}
          </div>

          {/* Row 2: Year range + dataset toggle + view-specific controls */}
          <div className="flex flex-wrap items-center gap-2 border-t border-slate-200/60 py-2">
            <div className="w-[240px] rounded-lg border border-slate-200 bg-white/90 px-3 py-1 shadow-sm sm:w-[320px] lg:w-[340px]">
              <div
                className="relative h-4"
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
                  className="year-range-slider pointer-events-none absolute inset-0 h-4 w-full appearance-none bg-transparent"
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
                  className="year-range-slider pointer-events-none absolute inset-0 h-4 w-full appearance-none bg-transparent"
                  aria-label="End year"
                  disabled={maxYearIndex === 0}
                />
              </div>
              <div className="relative mt-0.5 h-4 text-[9px] font-semibold uppercase tracking-[0.1em] text-slate-500">
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
            <select
              value={datasetKey}
              onChange={event => {
                const nextDatasetKey = event.target.value as DatasetKey;
                setDatasetKey(nextDatasetKey);
                const nextYears = data.datasets[nextDatasetKey].years;
                setYearRangeIndices({ start: 0, end: Math.max(nextYears.length - 1, 0) });
              }}
              className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
            >
              <option value="perReport">Per Report</option>
              <option value="perChunk">Per Excerpt</option>
            </select>

            {/* Risk-specific controls */}
            {activeView === 1 && (
              <>
                <div className="h-6 w-px bg-slate-200 mx-1" />
                <select
                  id="risk-filter"
                  value={riskFilter}
                  onChange={e => setRiskFilter(e.target.value)}
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Risk Types</option>
                  {data.labels.riskLabels.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {riskFilter !== 'all' && (
                  <button
                    onClick={() => setRiskFilter('all')}
                    className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
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
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Adoption Types</option>
                  {data.labels.adoptionTypes.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {adoptionFilter !== 'all' && (
                  <button
                    onClick={() => setAdoptionFilter('all')}
                    className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
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
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Vendors</option>
                  {vendorStackKeys.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {effectiveVendorFilter !== 'all' && (
                  <button
                    onClick={() => setVendorFilter('all')}
                    className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
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
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
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
                    className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
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
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Blind Spots</option>
                  <option value="no_ai_mention">No AI Mention</option>
                  <option value="no_ai_risk_mention">No AI Risk Mention</option>
                </select>
                {blindSpotFilter !== 'all' && (
                  <button
                    onClick={() => setBlindSpotFilter('all')}
                    className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-600 hover:bg-slate-50"
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
          <h2 className="text-2xl font-semibold text-slate-900">{view.heading}</h2>
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
              data={trendTimeAxis === 'month' ? filteredMonthlyRiskTrend : filteredRiskTrend}
              xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
              stackKeys={riskFilter === 'all' ? riskStackKeys : [riskFilter]}
              colors={riskColors}
              allowLineChart
              legendPosition="right"
              legendKeys={riskStackKeys}
              activeLegendKey={riskFilter === 'all' ? null : riskFilter}
              onLegendItemClick={(key) => setRiskFilter(prev => (prev === key ? 'all' : key))}
              title="Risk Trend Over Time"
              headerExtra={trendTimeToggle}
              subtitle={`Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} mentioning each AI risk category (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). Each colour represents one risk category; bars are additive because a single ${datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple categories. The y-axis scale adjusts dynamically to the data shown.`}
              tooltip={
                <>
                  <p>Each bar is stacked by risk category: the total height is the sum of all risk-category mentions that year, and each colour represents one category.</p>
                  <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple risk categories and therefore contribute to several coloured segments within the same year&apos;s bar; segments are not mutually exclusive.</p>
                  <p className="mt-2">Year-on-year growth may also reflect shifts in disclosure requirements or reporting culture rather than changes in actual risk levels — see the About page for more detail.</p>
                  <p className="mt-2">Click a legend item to isolate a single category.</p>
                </>
              }
            />

            {/* Risk by Sector Heatmap */}
            <GenericHeatmap
              data={riskHeatmapData}
              xLabels={riskHeatmapXLabels}
              yLabels={riskHeatmapYLabels}
              baseColor={riskHeatmapBaseColor}
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'excerpts'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title={riskFilter === 'all' ? 'Risk Distribution by Sector' : `${formatLabel(riskFilter)} Risk Mentions by Sector and Year`}
              subtitle={
                riskFilter === 'all'
                  ? `Heatmap of ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and risk category (columns). Colour intensity encodes the number of ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} containing each risk type within each sector; darker cells indicate higher counts relative to the dataset maximum.`
                  : `Heatmap of ${formatLabel(riskFilter)} ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Colour intensity encodes relative frequency; darker cells indicate a higher concentration of ${formatLabel(riskFilter)} mentions in that sector-year.`
              }
              tooltip={
                riskFilter === 'all'
                  ? <>
                      <p>Colour intensity is scaled relative to the full dataset, not absolute counts — two cells with the same shade may differ in raw numbers, and a dark cell means relatively more {datasetKey === 'perReport' ? 'reports' : 'passages'} than lighter cells in this view.</p>
                      <p className="mt-2">To track how a specific risk has evolved across sectors over time, select it from the &apos;Focus on Risk Type&apos; control above; this switches the columns from risk categories to individual years.</p>
                      <p className="mt-2">For analysis of regulatory disclosure requirements that may shape reporting patterns, see the About page.</p>
                    </>
                  : <>
                      <p>Colour intensity is scaled relative to the full dataset — two cells with the same shade may differ in raw numbers.</p>
                      <p className="mt-2">Clearing the risk-type filter returns the view to all categories, with columns showing each risk category again.</p>
                      <p className="mt-2">For analysis of regulatory disclosure requirements that may shape reporting patterns, see the About page.</p>
                    </>
              }
              xAxisLabel={riskFilter === 'all' ? 'Risk Type' : 'Year'}
              yAxisLabel={riskHeatmapAxisSectorLabel}
              labelColumnWidth={riskSectorView === 'isic' ? 290 : undefined}
              rowHeight={riskSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={riskSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
              headerExtra={makeSectorToggle(riskSectorView, setRiskSectorView)}
            />

            <details className="group rounded-2xl border border-slate-200 bg-white/90 shadow-sm">
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
              data={trendTimeAxis === 'month' ? filteredMonthlyAdoptionTrend : filteredAdoptionTrend}
              xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
              stackKeys={adoptionFilter === 'all' ? adoptionStackKeys : [adoptionFilter]}
              colors={adoptionColors}
              allowLineChart
              legendPosition="right"
              legendKeys={adoptionStackKeys}
              activeLegendKey={adoptionFilter === 'all' ? null : adoptionFilter}
              onLegendItemClick={(key) => setAdoptionFilter(prev => (prev === key ? 'all' : key))}
              title="Adoption Type Over Time"
              headerExtra={trendTimeToggle}
              subtitle={`Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} referencing each AI adoption maturity level — Non-LLM, LLM, and Agentic — (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'passage'} may be tagged with multiple adoption types. The y-axis scale adjusts dynamically to the data shown.`}
              tooltip={
                <>
                  <p>Each bar is stacked by adoption type: Non-LLM, LLM, and Agentic.</p>
                  <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple adoption types and can therefore contribute to more than one segment within the same year.</p>
                  <p className="mt-2">Click a legend item to isolate one adoption type.</p>
                </>
              }
            />
            <GenericHeatmap
              data={adoptionHeatmapData}
              xLabels={adoptionHeatmapXLabels}
              yLabels={adoptionHeatmapYLabels}
              baseColor={adoptionHeatmapBaseColor}
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'excerpts'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title={adoptionFilter === 'all' ? 'Adoption Intensity by Sector' : `${formatLabel(adoptionFilter)} Mentions by Sector and Year`}
              subtitle={
                adoptionFilter === 'all'
                  ? `Heatmap of ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${adoptionSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and adoption type (columns). Colour intensity encodes the number of ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} mentioning each adoption maturity level within each sector; darker cells indicate higher counts relative to the dataset maximum.`
                  : `Heatmap of ${formatLabel(adoptionFilter)} ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${adoptionSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Colour intensity encodes relative frequency; darker cells indicate a higher concentration of ${formatLabel(adoptionFilter)} mentions in that sector-year.`
              }
              tooltip={
                adoptionFilter === 'all'
                  ? <>
                      <p>Colour intensity is scaled relative to the full dataset, not absolute counts — darker cells indicate relatively more adoption mentions in this view.</p>
                      <p className="mt-2">Hatched cells indicate no observed adoption mentions for that sector/type pairing.</p>
                      <p className="mt-2">Select one adoption type above to switch the columns from adoption types to years.</p>
                    </>
                  : <>
                      <p>This filtered view shows how one adoption type evolves across sectors over time.</p>
                      <p className="mt-2">Clearing the filter returns the categorical view with all adoption types.</p>
                    </>
              }
              xAxisLabel={adoptionFilter === 'all' ? 'Adoption Type' : 'Year'}
              yAxisLabel={adoptionSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
              headerExtra={makeSectorToggle(adoptionSectorView, setAdoptionSectorView)}
              labelColumnWidth={adoptionSectorView === 'isic' ? 290 : undefined}
              rowHeight={adoptionSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={adoptionSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
            />

            <details className="group rounded-2xl border border-slate-200 bg-white/90 shadow-sm">
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
              data={trendTimeAxis === 'month' ? filteredMonthlyVendorTrend : filteredVendorTrend}
              xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
              stackKeys={effectiveVendorFilter === 'all' ? vendorStackKeys : [effectiveVendorFilter]}
              colors={vendorColors}
              allowLineChart
              legendPosition="right"
              legendKeys={vendorStackKeys}
              activeLegendKey={effectiveVendorFilter === 'all' ? null : effectiveVendorFilter}
              onLegendItemClick={(key) => setVendorFilter(prev => (prev === key ? 'all' : key))}
              title="Vendor References Over Time"
              headerExtra={trendTimeToggle}
              subtitle={`Stacked bar chart showing the number of ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} referencing each AI vendor or provider tag (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). A single ${datasetKey === 'perReport' ? 'report' : 'passage'} may reference multiple vendors. The y-axis scale adjusts dynamically to the data shown.`}
              tooltip={
                <>
                  <p>Each bar is stacked by vendor tag (OpenAI, Microsoft, Google, Internal, Other).</p>
                  <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can include multiple vendor tags, so one item may contribute to more than one segment.</p>
                  <p className="mt-2"><span className="font-medium">Internal</span> means in-house AI development.</p>
                  <p className="mt-2">Click a legend item to isolate one vendor tag.</p>
                </>
              }
            />
            <GenericHeatmap
              data={vendorHeatmapData}
              xLabels={vendorHeatmapXLabels}
              yLabels={vendorHeatmapYLabels}
              baseColor={vendorHeatmapBaseColor}
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'excerpts'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title={effectiveVendorFilter === 'all' ? 'Vendor Concentration by Sector' : `${formatLabel(effectiveVendorFilter)} Mentions by Sector and Year`}
              subtitle={
                effectiveVendorFilter === 'all'
                  ? `Heatmap of ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${vendorSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and vendor tag (columns). Colour intensity encodes the number of ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} naming each vendor within each sector; darker cells indicate higher counts relative to the dataset maximum.`
                  : `Heatmap of ${formatLabel(effectiveVendorFilter)} ${datasetKey === 'perReport' ? 'report' : 'passage'} counts by ${vendorSectorView === 'cni' ? 'CNI' : 'ISIC'} sector (rows) and fiscal year (columns). Colour intensity encodes relative frequency; darker cells indicate a higher concentration of ${formatLabel(effectiveVendorFilter)} mentions in that sector-year.`
              }
              tooltip={
                effectiveVendorFilter === 'all'
                  ? <>
                      <p>Colour intensity is scaled relative to the full dataset, not absolute counts — darker cells indicate relatively higher mention concentration in this view.</p>
                      <p className="mt-2">Hatched cells indicate no observed mentions for that sector/vendor pairing.</p>
                      <p className="mt-2">Select one vendor above to switch columns from vendor tags to years.</p>
                    </>
                  : <>
                      <p>This filtered view shows how one vendor tag evolves across sectors over time.</p>
                      <p className="mt-2">Clear the filter to return to the all-vendor categorical view.</p>
                    </>
              }
              xAxisLabel={effectiveVendorFilter === 'all' ? 'Vendor' : 'Year'}
              yAxisLabel={vendorSectorView === 'cni' ? 'CNI Sector' : 'ISIC Sector'}
              headerExtra={makeSectorToggle(vendorSectorView, setVendorSectorView)}
              labelColumnWidth={vendorSectorView === 'isic' ? 290 : undefined}
              rowHeight={vendorSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={vendorSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
            />

            <details className="group rounded-2xl border border-slate-200 bg-white/90 shadow-sm">
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
                title="Risk Signal Strength"
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
                title="Adoption Signal Strength"
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
                title="Vendor Signal Strength"
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
              title="Risk Substantiveness Distribution"
              subtitle="Heatmap of report-level risk-disclosure quality by substantiveness band (rows: Substantive, Moderate, Boilerplate) and fiscal year (columns). Each cell counts the number of reports whose AI-risk language was classified into that quality tier in a given year; colour intensity encodes relative frequency."
              tooltip="Substantiveness measures depth and specificity of risk disclosure at report level. Substantive disclosures include concrete mechanisms and mitigation/action detail, while boilerplate disclosures remain generic."
              xAxisLabel="Year"
              yAxisLabel="Quality Band"
            />
            )}
            <details className="group rounded-2xl border border-slate-200 bg-white/90 shadow-sm">
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
              data={trendTimeAxis === 'month' ? filteredMonthlyBlindSpotTrend : filteredBlindSpotTrend}
              xAxisKey={trendTimeAxis === 'month' ? 'month' : 'year'}
              stackKeys={blindSpotFilter === 'all' ? ['no_ai_mention', 'no_ai_risk_mention'] : [blindSpotFilter]}
              colors={blindSpotColors}
              allowLineChart
              legendPosition="right"
              legendKeys={['no_ai_mention', 'no_ai_risk_mention']}
              activeLegendKey={blindSpotFilter === 'all' ? null : blindSpotFilter}
              onLegendItemClick={(key) =>
                setBlindSpotFilter(prev => (prev === key ? 'all' : (key as BlindSpotFilter)))
              }
              title={trendTimeAxis === 'month' ? 'Blind Spots by Month' : 'Blind Spots by Year'}
              headerExtra={trendTimeToggle}
              subtitle={
                blindSpotFilter === 'all'
                  ? `Stacked bar chart showing the number of annual reports with no AI mention (red) and no AI risk mention (amber) on the y-axis, plotted across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} on the x-axis. The two series are not mutually exclusive: a report with no AI mention is also counted under no AI risk mention. The y-axis scale adjusts dynamically to the data shown.`
                  : `Bar chart showing the number of annual reports classified as ${formatLabel(blindSpotFilter)} (y-axis) across ${trendTimeAxis === 'month' ? 'report release months' : 'fiscal years'} (x-axis). The y-axis scale adjusts dynamically to the data shown.`
              }
              tooltip={
                blindSpotFilter === 'all'
                  ? 'This report-level view compares two absence patterns: no AI mention at all vs. no AI risk mention. Use the legend or top filter to isolate one series.'
                  : `${formatLabel(blindSpotFilter)} is currently isolated. Clear the filter to compare both blind-spot types together.`
              }
            />

            {blindSpotFilter === 'all' ? (
              <div className="grid gap-8 xl:grid-cols-2">
                <GenericHeatmap
                  data={noAiBySectorYearInRange}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor="#64748b"
                  valueFormatter={value => `${value}`}
                  showTotals={true}
                  showBlindSpots={false}
                  title="No AI Mention by Sector and Year"
                  subtitle="Heatmap of annual reports containing no AI mention, by CNI sector (rows) and fiscal year (columns). Each cell shows the count of reports in that sector-year with zero AI mention signal; colour intensity encodes relative frequency."
                  tooltip="Each cell is the number of reports in that sector-year that do not mention AI at all."
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                />
                <GenericHeatmap
                  data={noAiRiskBySectorYearInRange}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor="#64748b"
                  valueFormatter={value => `${value}`}
                  showTotals={true}
                  showBlindSpots={false}
                  title="No AI Risk Mention by Sector and Year"
                  subtitle="Heatmap of annual reports containing no AI risk disclosure, by CNI sector (rows) and fiscal year (columns). Each cell shows the count of reports in that sector-year with no AI risk mention; colour intensity encodes relative frequency."
                  tooltip="Each cell is the number of reports in that sector-year with no AI risk mention."
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                />
              </div>
            ) : (
              blindSpotHeatmapData && (
                <GenericHeatmap
                  data={blindSpotHeatmapData}
                  xLabels={blindSpotYearsInRange}
                  yLabels={data.sectors}
                  baseColor={blindSpotHeatmapColor}
                  valueFormatter={value => `${value}`}
                  showTotals={true}
                  showBlindSpots={false}
                  title={blindSpotHeatmapTitle}
                  subtitle={blindSpotHeatmapSubtitle}
                  tooltip={blindSpotHeatmapTooltip}
                  xAxisLabel="Year"
                  yAxisLabel="Sector"
                />
              )
            )}

            <details className="group rounded-2xl border border-slate-200 bg-white/90 shadow-sm">
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
