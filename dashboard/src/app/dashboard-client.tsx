'use client';

import { useMemo, useState } from 'react';
import { GenericHeatmap, StackedBarChart, InfoTooltip } from '@/components/overview-charts';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
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

const mentionColors: Record<string, string> = {
  adoption: '#0ea5e9',
  risk: '#f97316',
  vendor: '#14b8a6',
  general_ambiguous: '#64748b',
  harm: '#ef4444',
};

const adoptionColors: Record<string, string> = {
  non_llm: '#0f766e',
  llm: '#38bdf8',
  agentic: '#f59e0b',
};

const vendorColors: Record<string, string> = {
  openai: '#0ea5e9',
  microsoft: '#1d4ed8',
  google: '#f97316',
  internal: '#0f766e',
  other: '#64748b',
  undisclosed: '#cbd5e1',
};

const riskColors: Record<string, string> = {
  cybersecurity: '#ef4444',
  operational_technical: '#f97316',
  regulatory_compliance: '#f59e0b',
  reputational_ethical: '#14b8a6',
  information_integrity: '#0ea5e9',
  third_party_supply_chain: '#22c55e',
  strategic_competitive: '#84cc16',
  workforce_impacts: '#0f766e',
  environmental_impact: '#10b981',
  national_security: '#7c3aed',
};

const blindSpotColors: Record<string, string> = {
  no_ai_mention: '#f87171',
  no_ai_risk_mention: '#be789a',
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
  };
  if (overrides[val]) return overrides[val];
  return val
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

type DatasetKey = 'perReport' | 'perChunk';
type RiskSectorView = 'cni' | 'isic';

export default function DashboardClient({ data }: { data: GoldenDashboardData }) {
  const [activeView, setActiveView] = useState(1);
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('perReport');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [adoptionFilter, setAdoptionFilter] = useState<AdoptionFilter>('all');
  const [riskSectorView, setRiskSectorView] = useState<RiskSectorView>('cni');
  const [vendorFilter, setVendorFilter] = useState<string>('all');

  const view = VIEWS.find(item => item.id === activeView) ?? VIEWS[0];
  const activeData = data.datasets[datasetKey];
  const reportBaselineData = data.datasets.perReport;
  const availableYears = activeData.years;
  const maxYearIndex = Math.max(availableYears.length - 1, 0);

  const mentionStackKeys = useMemo(() => data.labels.mentionTypes, [data.labels.mentionTypes]);
  const adoptionStackKeys = useMemo(() => data.labels.adoptionTypes, [data.labels.adoptionTypes]);
  const vendorStackKeys = useMemo(() => data.labels.vendorTags, [data.labels.vendorTags]);
  const riskStackKeys = useMemo(() => data.labels.riskLabels, [data.labels.riskLabels]);

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

  const mentionTrendInRange = useMemo(
    () =>
      activeData.mentionTrend.filter(row => {
        const year = Number(row.year);
        return year >= selectedStartYear && year <= selectedEndYear;
      }),
    [activeData.mentionTrend, selectedStartYear, selectedEndYear]
  );

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

  const heroRiskTrend = useMemo(
    () =>
      activeData.riskTrend.map(row => {
        const total = riskStackKeys.reduce((sum, key) => sum + (Number(row[key]) || 0), 0);
        return { year: row.year, risk: total };
      }),
    [activeData.riskTrend, riskStackKeys]
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
    if (vendorFilter === 'all') return vendorTrendInRange;
    return vendorTrendInRange.map(row => ({
      year: row.year,
      [vendorFilter]: row[vendorFilter] || 0,
    }));
  }, [vendorTrendInRange, vendorFilter]);

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
    ? '#f97316'
    : (riskColors[riskFilter] || '#f97316');
  const adoptionHeatmapData = useMemo(() => {
    if (adoptionFilter === 'all') return adoptionBySectorInRange;
    return adoptionBySectorYearInRange
      .filter(cell => cell.x === adoptionFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [adoptionFilter, adoptionBySectorInRange, adoptionBySectorYearInRange]);
  const adoptionHeatmapXLabels = adoptionFilter === 'all' ? data.labels.adoptionTypes : filteredYears;
  const adoptionHeatmapBaseColor = adoptionFilter === 'all'
    ? '#0ea5e9'
    : (adoptionColors[adoptionFilter] || '#0ea5e9');
  const vendorHeatmapData = useMemo(() => {
    if (vendorFilter === 'all') return vendorBySectorInRange;
    return vendorBySectorYearInRange
      .filter(cell => cell.x === vendorFilter)
      .map(cell => ({ x: cell.year, y: cell.y, value: cell.value }));
  }, [vendorFilter, vendorBySectorInRange, vendorBySectorYearInRange]);
  const vendorHeatmapXLabels = vendorFilter === 'all' ? data.labels.vendorTags : filteredYears;
  const vendorHeatmapBaseColor = vendorFilter === 'all'
    ? '#14b8a6'
    : (vendorColors[vendorFilter] || '#14b8a6');
  const riskSelectedYearSpan = filteredYears.length > 0
    ? `${selectedStartYear}–${selectedEndYear}`
    : 'N/A';

  return (
    <div className="min-h-screen bg-[#f6f3ef] text-slate-900">
      {/* Hero section */}
      <header className="relative overflow-hidden border-b border-slate-200/70">
        <div className="absolute inset-0">
          <div className="absolute -top-24 left-10 h-64 w-64 rounded-full bg-amber-200/70 blur-3xl" />
          <div className="absolute top-10 right-0 h-72 w-72 rounded-full bg-sky-200/70 blur-3xl" />
          <div className="absolute bottom-0 left-1/3 h-48 w-48 rounded-full bg-emerald-200/60 blur-3xl" />
        </div>
        <div className="relative mx-auto max-w-7xl px-6 py-12">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="animate-rise">
              <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
                AI Risk Observatory
              </h1>
              <p className="mt-3 max-w-2xl text-base text-slate-600 sm:text-lg">
                Tracking how UK Critical National Infrastructure companies disclose AI-related risks, adoption, and vendor dependencies in their annual reports.
              </p>
              <p className="mt-2 max-w-2xl text-sm text-slate-500">
                This dashboard visualises findings from an NLP pipeline that analyses annual reports of UK Critical National Infrastructure companies for AI-related disclosures.{' '}
                <a href="/about" className="underline decoration-slate-400 hover:text-slate-700">Learn more about our methodology</a>.
              </p>
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {activeData.summary.totalCompanies} Companies
                </span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {activeData.summary.totalReports} Annual Reports
                </span>
                <span className="rounded-full bg-white/80 px-3 py-1 font-semibold">
                  {selectedStartYear}–{selectedEndYear}
                </span>
              </div>
            </div>
            <div className="animate-rise animate-rise-delay-1 relative w-full max-w-md lg:w-[380px]">
              <div className="h-[130px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={heroRiskTrend} margin={{ top: 4, right: 12, bottom: 0, left: -12 }}>
                    <defs>
                      <linearGradient id="heroEdgeFade" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#000" />
                        <stop offset="8%" stopColor="#fff" />
                        <stop offset="92%" stopColor="#fff" />
                        <stop offset="100%" stopColor="#000" />
                      </linearGradient>
                      <mask id="heroEdgeFadeMask" maskUnits="objectBoundingBox" maskContentUnits="objectBoundingBox">
                        <rect x="0" y="0" width="1" height="1" fill="url(#heroEdgeFade)" />
                      </mask>
                      <linearGradient id="heroRiskLineGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#fb923c" />
                        <stop offset="100%" stopColor="#f97316" />
                      </linearGradient>
                      <linearGradient id="heroRiskGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f97316" stopOpacity={0.25} />
                        <stop offset="100%" stopColor="#f97316" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="year" tick={{ fontSize: 10, fill: '#a8a29e' }} axisLine={false} tickLine={false} dy={4} />
                    <YAxis tick={{ fontSize: 9, fill: '#a09890' }} axisLine={false} tickLine={false} width={36} tickCount={4} />
                    <Tooltip
                      contentStyle={{ fontSize: 11, borderRadius: 10, border: 'none', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)', boxShadow: '0 4px 24px rgba(0,0,0,.1)' }}
                      labelFormatter={label => `${label}`}
                      formatter={(value: number) => [value, 'Risk mentions']}
                      cursor={{ stroke: '#d6d3d1', strokeWidth: 1, strokeDasharray: '3 3' }}
                    />
                    <Area
                      type="monotone"
                      dataKey="risk"
                      stroke="url(#heroRiskLineGrad)"
                      strokeWidth={2.5}
                      fill="url(#heroRiskGrad)"
                      mask="url(#heroEdgeFadeMask)"
                      dot={false}
                      activeDot={{ r: 4, strokeWidth: 2, fill: '#fff', stroke: '#f97316' }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-1 text-[11px] text-slate-400">AI risk mentions in UK public company annual reports</p>
            </div>
          </div>
        </div>
      </header>

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
                <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white/90 p-0.5 shadow-sm">
                  <button
                    type="button"
                    onClick={() => setRiskSectorView('cni')}
                    className={`rounded-md px-3 py-1.5 text-sm font-semibold transition ${
                      riskSectorView === 'cni'
                        ? 'bg-slate-900 text-white'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    CNI
                  </button>
                  <button
                    type="button"
                    onClick={() => setRiskSectorView('isic')}
                    className={`rounded-md px-3 py-1.5 text-sm font-semibold transition ${
                      riskSectorView === 'isic'
                        ? 'bg-slate-900 text-white'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    ISIC
                  </button>
                </div>
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
                  value={vendorFilter}
                  onChange={e => setVendorFilter(e.target.value)}
                  className="h-9 rounded-lg border border-slate-200 bg-white/90 px-3 text-sm font-medium text-slate-700 shadow-sm"
                >
                  <option value="all">All Vendors</option>
                  {data.labels.vendorTags.map(label => (
                    <option key={label} value={label}>{formatLabel(label)}</option>
                  ))}
                </select>
                {vendorFilter !== 'all' && (
                  <button
                    onClick={() => setVendorFilter('all')}
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
                <InfoTooltip content="Sector classifications for companies that do not fall clearly within a CNI sector have been approximated using an LLM-assisted mapping process. Full details are available on the Methodology page." />
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
                contain at least one named AI vendor reference, spanning{' '}
                <span className="font-semibold text-slate-900">{formatNumber(vendorOverviewStats.excerptVendorMentions)}</span>{' '}
                individual passages.
              </p>
              <p>
                <span className="font-medium text-slate-800">Per Report</span> shows how many filings mention a given
                vendor at least once; <span className="font-medium text-slate-800">Per Excerpt</span> shows the full
                volume of vendor-tagged passages across all repoirts and therefore the depth of mention.
              </p>
              <p>
                Use the control above to focus on one vendor tag (
                <span className="font-medium text-slate-800">OpenAI</span>,{' '}
                <span className="font-medium text-slate-800">Microsoft</span>,{' '}
                <span className="font-medium text-slate-800">Google</span>,{' '}
                <span className="font-medium text-slate-800">Internal</span>,{' '}
                <span className="font-medium text-slate-800">Undisclosed</span>, or{' '}
                <span className="font-medium text-slate-800">Other</span>). Selecting one vendor switches the heatmap
                from categorical columns (vendor tags) to yearly columns.
              </p>
              <p>
                Read the trend chart for time patterns and the heatmap for sector concentration. Together they show both
                which vendors are cited and where those dependencies appear most strongly.
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
              data={filteredRiskTrend}
              xAxisKey="year"
              stackKeys={riskFilter === 'all' ? riskStackKeys : [riskFilter]}
              colors={riskColors}
              allowLineChart
              legendPosition="right"
              legendKeys={riskStackKeys}
              activeLegendKey={riskFilter === 'all' ? null : riskFilter}
              onLegendItemClick={(key) => setRiskFilter(prev => (prev === key ? 'all' : key))}
              title="Risk Trend Over Time"
              subtitle={`How often each AI risk category appeared in ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'}, year by year. One ${datasetKey === 'perReport' ? 'report' : 'passage'} can be counted in more than one category.`}
              tooltip={
                <>
                  <p>Each bar is stacked by risk category: the total height is the sum of all risk-category mentions that year, and each colour represents one category.</p>
                  <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can be tagged with multiple risk categories and therefore contribute to several coloured segments within the same year's bar — segments are not mutually exclusive.</p>
                  <p className="mt-2">Year-on-year growth may also reflect shifts in disclosure requirements or reporting culture rather than changes in actual risk levels — see the Methodology page for more detail.</p>
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
                  ? `How many ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} in each ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sector mentioned each risk category.`
                  : `How frequently ${formatLabel(riskFilter)} was mentioned across ${riskSectorView === 'cni' ? 'CNI' : 'ISIC'} sectors, broken down by year.`
              }
              tooltip={
                riskFilter === 'all'
                  ? <>
                      <p>Colour intensity is scaled relative to the full dataset, not absolute counts — two cells with the same shade may differ in raw numbers, and a dark cell means relatively more {datasetKey === 'perReport' ? 'reports' : 'passages'} than lighter cells in this view.</p>
                      <p className="mt-2">To track how a specific risk has evolved across sectors over time, select it from the 'Focus on Risk Type' control above; this switches the columns from risk categories to individual years.</p>
                      <p className="mt-2">For analysis of regulatory disclosure requirements that may shape reporting patterns, see the Methodology page.</p>
                    </>
                  : <>
                      <p>Colour intensity is scaled relative to the full dataset — two cells with the same shade may differ in raw numbers.</p>
                      <p className="mt-2">Clearing the risk-type filter returns the view to all categories, with columns showing each risk category again.</p>
                      <p className="mt-2">For analysis of regulatory disclosure requirements that may shape reporting patterns, see the Methodology page.</p>
                    </>
              }
              xAxisLabel={riskFilter === 'all' ? 'Risk Type' : 'Year'}
              yAxisLabel={riskHeatmapAxisSectorLabel}
              labelColumnWidth={riskSectorView === 'isic' ? 290 : undefined}
              rowHeight={riskSectorView === 'isic' ? 58 : undefined}
              yLabelClassName={riskSectorView === 'isic' ? 'text-xs leading-snug' : undefined}
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
                  <li><span className="font-medium text-slate-800">Third-Party Supply Chain:</span> Risks arising from dependence on external AI vendors, APIs, or suppliers whose reliability or conduct is outside the company's direct control.</li>
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
              data={filteredAdoptionTrend}
              xAxisKey="year"
              stackKeys={adoptionFilter === 'all' ? adoptionStackKeys : [adoptionFilter]}
              colors={adoptionColors}
              allowLineChart
              legendPosition="right"
              legendKeys={adoptionStackKeys}
              activeLegendKey={adoptionFilter === 'all' ? null : adoptionFilter}
              onLegendItemClick={(key) => setAdoptionFilter(prev => (prev === key ? 'all' : key))}
              title="Adoption Type Over Time"
              subtitle={`How frequently AI adoption appears in ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'}, grouped by maturity level (Non-LLM, LLM, Agentic).`}
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
              yLabels={data.sectors}
              baseColor={adoptionHeatmapBaseColor}
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'excerpts'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title={adoptionFilter === 'all' ? 'Adoption Intensity by Sector' : `${formatLabel(adoptionFilter)} Mentions by Sector and Year`}
              subtitle={
                adoptionFilter === 'all'
                  ? `How many ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} in each CNI sector mention each adoption type.`
                  : `How frequently ${formatLabel(adoptionFilter)} was mentioned across CNI sectors, broken down by year.`
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
              yAxisLabel="Sector"
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
              data={filteredVendorTrend}
              xAxisKey="year"
              stackKeys={vendorFilter === 'all' ? vendorStackKeys : [vendorFilter]}
              colors={vendorColors}
              allowLineChart
              legendPosition="right"
              legendKeys={vendorStackKeys}
              activeLegendKey={vendorFilter === 'all' ? null : vendorFilter}
              onLegendItemClick={(key) => setVendorFilter(prev => (prev === key ? 'all' : key))}
              title="Vendor References Over Time"
              subtitle={`How frequently each vendor tag appears in ${datasetKey === 'perReport' ? 'annual reports' : 'text passages'} over time.`}
              tooltip={
                <>
                  <p>Each bar is stacked by vendor tag (OpenAI, Microsoft, Google, Internal, Other, Undisclosed).</p>
                  <p className="mt-2">A single {datasetKey === 'perReport' ? 'report' : 'passage'} can include multiple vendor tags, so one item may contribute to more than one segment.</p>
                  <p className="mt-2"><span className="font-medium">Internal</span> means in-house AI development; <span className="font-medium">Undisclosed</span> means AI is referenced but no provider is named.</p>
                  <p className="mt-2">Click a legend item to isolate one vendor tag.</p>
                </>
              }
            />
            <GenericHeatmap
              data={vendorHeatmapData}
              xLabels={vendorHeatmapXLabels}
              yLabels={data.sectors}
              baseColor={vendorHeatmapBaseColor}
              valueFormatter={value => `${value} ${datasetKey === 'perReport' ? 'reports' : 'excerpts'}`}
              xLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={true}
              title={vendorFilter === 'all' ? 'Vendor Concentration by Sector' : `${formatLabel(vendorFilter)} Mentions by Sector and Year`}
              subtitle={
                vendorFilter === 'all'
                  ? `How many ${datasetKey === 'perReport' ? 'annual reports' : 'passages'} in each CNI sector mention each vendor tag.`
                  : `How frequently ${formatLabel(vendorFilter)} was mentioned across CNI sectors, broken down by year.`
              }
              tooltip={
                vendorFilter === 'all'
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
              xAxisLabel={vendorFilter === 'all' ? 'Vendor' : 'Year'}
              yAxisLabel="Sector"
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
                  <li><span className="font-medium text-slate-800">Undisclosed:</span> AI is referenced but the provider is not identified.</li>
                  <li><span className="font-medium text-slate-800">Other:</span> A named provider outside the primary tracked vendor set.</li>
                </ul>
              </div>
            </details>
          </div>
        )}

        {activeView === 4 && (
          <div className="space-y-8">
            <div className="grid gap-8 lg:grid-cols-3">
              <GenericHeatmap
                data={riskSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#f97316"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Risk Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                tooltip="Measures how explicitly each risk mention is evidenced. For each label in each excerpt the pipeline picks the strongest signal from any contributing source. 3-Explicit: the report directly names the risk in concrete terms. 2-Strong Implicit: the link is clear but inferential. 1-Weak Implicit: plausible but lightly supported. Higher rows indicate stronger, more verifiable disclosures."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
              <GenericHeatmap
                data={adoptionSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#0ea5e9"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Adoption Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                tooltip="Same signal-strength scoring as the risk heatmap, applied to AI adoption mentions. Explicit signals (row 3) indicate direct, named acknowledgement of AI system deployment. Weak implicit signals (row 1) may represent cautious or ambiguous language. Trends toward higher rows suggest improving disclosure quality over time."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
              <GenericHeatmap
                data={vendorSignalHeatmapInRange}
                xLabels={filteredYears}
                yLabels={data.labels.riskSignalLevels}
                baseColor="#14b8a6"
                valueFormatter={value => `${value}`}
                yLabelFormatter={formatLabel}
                showTotals={true}
                showBlindSpots={false}
                title="Vendor Signal Strength"
                subtitle="3 (Explicit) = direct statement, 2 (Strong Implicit) = clear inference, 1 (Weak Implicit) = lightly supported."
                tooltip="Signal-strength scoring for vendor references. Explicit (row 3) means a specific vendor or platform is directly named. Strong implicit (row 2) means the vendor relationship is clearly implied. Weak implicit (row 1) means a vendor connection is inferred from context. Low explicitness across years may indicate opaque AI supply chains."
                xAxisLabel="Year"
                yAxisLabel="Signal Level"
                compact={true}
              />
            </div>
            <GenericHeatmap
              data={substantivenessHeatmapInRange}
              xLabels={filteredYears}
              yLabels={data.labels.substantivenessBands}
              baseColor="#14b8a6"
              valueFormatter={value => `${value}`}
              yLabelFormatter={formatLabel}
              showTotals={true}
              showBlindSpots={false}
              title="Risk Substantiveness Distribution"
              subtitle="Disclosure quality for AI-risk language. Substantive = concrete mechanism + tangible mitigation. Moderate = specific risk area, limited detail. Boilerplate = generic language without concrete detail."
              tooltip="Rates the quality of AI-risk language in each annual report. Substantive disclosures describe a concrete risk mechanism and a tangible mitigation or action taken. Moderate disclosures identify a specific risk area but provide limited detail. Boilerplate disclosures use generic risk language with no concrete specifics. A shift toward Substantive over time indicates improving disclosure practice."
              xAxisLabel="Year"
              yAxisLabel="Quality Band"
            />
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-sm text-slate-600 shadow-sm">
              <p className="font-semibold text-slate-900">Quality Metric Definitions</p>
              <div className="mt-2 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="font-medium text-slate-800">Signal Strength (Risk / Adoption / Vendor):</p>
                  <p className="mt-1 leading-relaxed">
                    How explicitly each mention evidences its classification.
                    Each cell counts individual label mentions across all excerpts.
                    Per label per excerpt, the strongest signal from any source wins.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">3 Explicit:</span> direct, named, concrete statement</li>
                    <li><span className="font-medium">2 Strong implicit:</span> clear but inferential link</li>
                    <li><span className="font-medium">1 Weak implicit:</span> plausible but lightly supported</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-slate-800">Substantiveness:</p>
                  <p className="mt-1 leading-relaxed">
                    Disclosure quality for AI-risk language per report.
                  </p>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li><span className="font-medium">Substantive:</span> concrete mechanism + tangible mitigation/action</li>
                    <li><span className="font-medium">Moderate:</span> specific risk area, limited detail</li>
                    <li><span className="font-medium">Boilerplate:</span> generic risk language without concrete detail</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeView === 5 && (
          <div className="space-y-8">
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Reports in Range</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{formatNumber(blindSpotTotalsInRange.totalReports)}</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">No AI Mention</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{formatNumber(blindSpotTotalsInRange.noAiMention)}</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">No AI Risk Mention</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{formatNumber(blindSpotTotalsInRange.noAiRiskMention)}</p>
              </div>
            </div>

            <StackedBarChart
              data={blindSpotTrendInRange}
              xAxisKey="year"
              stackKeys={['no_ai_mention', 'no_ai_risk_mention']}
              colors={blindSpotColors}
              allowLineChart
              title="Blind Spots by Year"
              subtitle="Yearly counts of annual reports with no AI mention at all, and with no AI risk mention."
              tooltip="This view is based on report-level coverage. 'No AI mention' means no AI disclosure signal in that report. 'No AI risk mention' means the report does not disclose AI risk, even if AI may be mentioned elsewhere."
            />

            <div className="grid gap-8 xl:grid-cols-2">
              <GenericHeatmap
                data={noAiBySectorYearInRange}
                xLabels={blindSpotYearsInRange}
                yLabels={data.sectors}
                baseColor="#f87171"
                valueFormatter={value => `${value}`}
                showTotals={true}
                showBlindSpots={false}
                title="No AI Mention by Sector and Year"
                subtitle="Counts of annual reports with zero AI mention signal."
                tooltip="Each cell is the number of reports in that sector-year that do not mention AI at all."
                xAxisLabel="Year"
                yAxisLabel="Sector"
              />
              <GenericHeatmap
                data={noAiRiskBySectorYearInRange}
                xLabels={blindSpotYearsInRange}
                yLabels={data.sectors}
                baseColor="#be789a"
                valueFormatter={value => `${value}`}
                showTotals={true}
                showBlindSpots={false}
                title="No AI Risk Mention by Sector and Year"
                subtitle="Counts of annual reports that do not disclose AI risk."
                tooltip="Each cell is the number of reports in that sector-year with no AI risk mention."
                xAxisLabel="Year"
                yAxisLabel="Sector"
              />
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 text-xs leading-relaxed text-slate-600 shadow-sm">
              Blind spot metrics are calculated from report-level coverage (one annual report per company-year baseline), including zero-mention reports.
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
