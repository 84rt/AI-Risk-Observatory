import fs from 'fs';
import path from 'path';

const DATA_PATH = path.join(process.cwd(), 'data', 'dashboard-data.json');
const OUTPUT_PATH = path.join(process.cwd(), 'analysis', 'dashboard-findings.md');

if (!fs.existsSync(DATA_PATH)) {
  console.error(`Missing dashboard data: ${DATA_PATH}`);
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));
const dataset = data.datasets.perReport;
const now = new Date();
const generatedOn = now.toISOString().slice(0, 10);
const currentYear = now.getFullYear();
const years = [...data.years].sort((a, b) => a - b);
const latestYear = years[years.length - 1];
const lastFullYear = latestYear === currentYear ? latestYear - 1 : latestYear;
const previousFullYear = years.includes(lastFullYear - 1) ? lastFullYear - 1 : years[years.length - 2];

const MIN_ISIC_REPORTS = 20;
const MIN_SECTOR_REPORTS_FOR_HEADLINES = 20;

const fmtNumber = value => new Intl.NumberFormat('en-GB').format(value);
const fmtPct = value => `${value.toFixed(1)}%`;
const fmtSignedPp = value => `${value >= 0 ? '+' : ''}${value.toFixed(1)} pp`;
const fmtSignedCount = value => `${value >= 0 ? '+' : ''}${fmtNumber(value)}`;
const fmtRatio = value => value.toFixed(2);
const fmtPp = value => `${value.toFixed(1)} pp`;

const pct = (numerator, denominator) => (denominator > 0 ? (numerator / denominator) * 100 : 0);

const LABEL_OVERRIDES = {
  llm: 'LLM',
  non_llm: 'Traditional AI (non-LLM)',
  agentic: 'Agentic',
  strategic_competitive: 'Strategic / Competitive',
  cybersecurity: 'Cybersecurity',
  operational_technical: 'Operational / Technical',
  workforce_impacts: 'Workforce Impacts',
  regulatory_compliance: 'Regulatory / Compliance',
  reputational_ethical: 'Reputational / Ethical',
  third_party_supply_chain: 'Third-Party Supply Chain',
  information_integrity: 'Information Integrity',
  environmental_impact: 'Environmental Impact',
  national_security: 'National Security',
  openai: 'OpenAI',
  microsoft: 'Microsoft',
  google: 'Google',
  amazon: 'Amazon / AWS',
  meta: 'Meta',
  anthropic: 'Anthropic',
  internal: 'Internal',
  other: 'Other',
  undisclosed: 'Undisclosed',
};

const labelize = value => {
  if (LABEL_OVERRIDES[value]) return LABEL_OVERRIDES[value];
  return value
    .split('_')
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
};

const keyBy = (rows, keyFn) => {
  const map = new Map();
  rows.forEach(row => map.set(keyFn(row), row));
  return map;
};

const sumRow = row =>
  Object.entries(row)
    .filter(([key]) => key !== 'year')
    .reduce((sum, [, value]) => sum + Number(value || 0), 0);

const annualRowsFor = sourceDataset => {
  const blindSpotByYear = keyBy(sourceDataset.blindSpotTrend, row => row.year);
  const mentionByYear = keyBy(sourceDataset.mentionTrend, row => row.year);

  return sourceDataset.years.map(year => {
    const blindSpot = blindSpotByYear.get(year);
    const mention = mentionByYear.get(year);
    const totalReports = Number(blindSpot?.total_reports || 0);
    const aiMentionReports = Number(blindSpot?.ai_mention || 0);
    const adoptionReports = Number(mention?.adoption || 0);
    const riskReports = Number(mention?.risk || 0);
    const vendorReports = Number(mention?.vendor || 0);
    const generalAmbiguousReports = Number(mention?.general_ambiguous || 0);

    return {
      year,
      totalReports,
      aiMentionReports,
      adoptionReports,
      riskReports,
      vendorReports,
      generalAmbiguousReports,
      aiMentionPct: pct(aiMentionReports, totalReports),
      adoptionPct: pct(adoptionReports, totalReports),
      riskPct: pct(riskReports, totalReports),
      vendorPct: pct(vendorReports, totalReports),
      generalAmbiguousPct: pct(generalAmbiguousReports, totalReports),
      adoptionRiskGapPp: pct(adoptionReports - riskReports, totalReports),
      aiRiskGapPp: pct(aiMentionReports - riskReports, totalReports),
    };
  });
};

const annualRows = annualRowsFor(dataset);
const annualByYear = keyBy(annualRows, row => row.year);
const currentFullYearRow = annualByYear.get(lastFullYear);
const previousFullYearRow = annualByYear.get(previousFullYear);

const describeYoY = (currentRow, previousRow) => ({
  aiMentionPp: currentRow.aiMentionPct - previousRow.aiMentionPct,
  adoptionPp: currentRow.adoptionPct - previousRow.adoptionPct,
  riskPp: currentRow.riskPct - previousRow.riskPct,
  vendorPp: currentRow.vendorPct - previousRow.vendorPct,
  adoptionRiskGapPp: currentRow.adoptionRiskGapPp - previousRow.adoptionRiskGapPp,
});

const yoy = describeYoY(currentFullYearRow, previousFullYearRow);

const metricTrendRows = annualRows.map((row, index) => {
  const prev = index > 0 ? annualRows[index - 1] : null;
  return {
    year: row.year,
    totalReports: row.totalReports,
    aiMentionReports: row.aiMentionReports,
    aiMentionPct: row.aiMentionPct,
    aiMentionYoY: prev ? row.aiMentionPct - prev.aiMentionPct : null,
    adoptionReports: row.adoptionReports,
    adoptionPct: row.adoptionPct,
    adoptionYoY: prev ? row.adoptionPct - prev.adoptionPct : null,
    riskReports: row.riskReports,
    riskPct: row.riskPct,
    riskYoY: prev ? row.riskPct - prev.riskPct : null,
    vendorReports: row.vendorReports,
    vendorPct: row.vendorPct,
    vendorYoY: prev ? row.vendorPct - prev.vendorPct : null,
    generalAmbiguousReports: row.generalAmbiguousReports,
    generalAmbiguousPct: row.generalAmbiguousPct,
    adoptionRiskGapPp: row.adoptionRiskGapPp,
  };
});

const categoryRowsForYear = (trendRows, year, denominator) => {
  const row = trendRows.find(entry => entry.year === year);
  if (!row) return [];
  return Object.entries(row)
    .filter(([key]) => key !== 'year')
    .map(([key, value]) => ({
      key,
      count: Number(value || 0),
      pctOfReports: pct(Number(value || 0), denominator),
    }));
};

const categoryDeltaRows = (trendRows, currentYearValue, previousYearValue, reportDenominatorCurrent, reportDenominatorPrevious) => {
  const currentRows = keyBy(categoryRowsForYear(trendRows, currentYearValue, reportDenominatorCurrent), row => row.key);
  const previousRows = keyBy(categoryRowsForYear(trendRows, previousYearValue, reportDenominatorPrevious), row => row.key);

  return Array.from(currentRows.values())
    .map(row => {
      const prev = previousRows.get(row.key) || { count: 0, pctOfReports: 0 };
      return {
        key: row.key,
        currentCount: row.count,
        previousCount: prev.count,
        countChange: row.count - prev.count,
        currentPct: row.pctOfReports,
        previousPct: prev.pctOfReports,
        ppChange: row.pctOfReports - prev.pctOfReports,
      };
    })
    .sort((a, b) => b.ppChange - a.ppChange);
};

const riskCategoryRows = categoryDeltaRows(
  dataset.riskTrend,
  lastFullYear,
  previousFullYear,
  currentFullYearRow.totalReports,
  previousFullYearRow.totalReports
);

const adoptionCategoryRows = categoryDeltaRows(
  dataset.adoptionTrend,
  lastFullYear,
  previousFullYear,
  currentFullYearRow.totalReports,
  previousFullYearRow.totalReports
);

const vendorCategoryRows = categoryDeltaRows(
  dataset.vendorTrend,
  lastFullYear,
  previousFullYear,
  currentFullYearRow.totalReports,
  previousFullYearRow.totalReports
);

const totalAssignmentsForYear = (trendRows, year) => {
  const row = trendRows.find(entry => entry.year === year);
  return row ? sumRow(row) : 0;
};

const concentrationStats = rows => {
  const total = rows.reduce((sum, row) => sum + row.currentCount, 0);
  const topThreeShare = total > 0
    ? (rows
      .slice()
      .sort((a, b) => b.currentCount - a.currentCount)
      .slice(0, 3)
      .reduce((sum, row) => sum + row.currentCount, 0) / total) * 100
    : 0;
  const hhi = total > 0
    ? rows.reduce((sum, row) => {
      const share = row.currentCount / total;
      return sum + (share * 100) ** 2;
    }, 0)
    : 0;
  return { total, topThreeShare, hhi };
};

const riskConcentration = concentrationStats(riskCategoryRows);
const adoptionConcentration = concentrationStats(adoptionCategoryRows);
const vendorConcentration = concentrationStats(vendorCategoryRows);
const vendorOpacityShare =
  vendorConcentration.total > 0
    ? (vendorCategoryRows
      .filter(row => ['other', 'undisclosed'].includes(row.key))
      .reduce((sum, row) => sum + row.currentCount, 0) / vendorConcentration.total) * 100
    : 0;
const namedVendorRows = vendorCategoryRows.filter(
  row => !['other', 'undisclosed', 'internal'].includes(row.key)
);
const namedVendorConcentration = concentrationStats(namedVendorRows);

const currentRiskAssignments = totalAssignmentsForYear(dataset.riskTrend, lastFullYear);
const previousRiskAssignments = totalAssignmentsForYear(dataset.riskTrend, previousFullYear);
const currentAdoptionAssignments = totalAssignmentsForYear(dataset.adoptionTrend, lastFullYear);
const previousAdoptionAssignments = totalAssignmentsForYear(dataset.adoptionTrend, previousFullYear);
const currentVendorAssignments = totalAssignmentsForYear(dataset.vendorTrend, lastFullYear);
const previousVendorAssignments = totalAssignmentsForYear(dataset.vendorTrend, previousFullYear);

const currentRiskLabelsPerRiskReport = currentRiskAssignments / currentFullYearRow.riskReports;
const previousRiskLabelsPerRiskReport = previousRiskAssignments / previousFullYearRow.riskReports;
const currentAdoptionLabelsPerAdoptionReport = currentAdoptionAssignments / currentFullYearRow.adoptionReports;
const previousAdoptionLabelsPerAdoptionReport = previousAdoptionAssignments / previousFullYearRow.adoptionReports;
const currentVendorLabelsPerVendorReport = currentVendorAssignments / currentFullYearRow.vendorReports;
const previousVendorLabelsPerVendorReport = previousVendorAssignments / previousFullYearRow.vendorReports;

const annualRowsForSubset = sourceDataset => {
  const rows = annualRowsFor(sourceDataset);
  const row = rows.find(entry => entry.year === lastFullYear);
  return row || null;
};

const marketSegmentRows = Object.entries(data.byMarketSegment)
  .map(([segment, source]) => {
    const lifetime = annualRowsFor(source.perReport).reduce((acc, row) => ({
      totalReports: acc.totalReports + row.totalReports,
      aiMentionReports: acc.aiMentionReports + row.aiMentionReports,
      adoptionReports: acc.adoptionReports + row.adoptionReports,
      riskReports: acc.riskReports + row.riskReports,
      vendorReports: acc.vendorReports + row.vendorReports,
    }), { totalReports: 0, aiMentionReports: 0, adoptionReports: 0, riskReports: 0, vendorReports: 0 });
    const current = annualRowsForSubset(source.perReport);
    return {
      segment,
      lifetimeReports: lifetime.totalReports,
      lifetimeAiPct: pct(lifetime.aiMentionReports, lifetime.totalReports),
      lifetimeAdoptionPct: pct(lifetime.adoptionReports, lifetime.totalReports),
      lifetimeRiskPct: pct(lifetime.riskReports, lifetime.totalReports),
      lifetimeVendorPct: pct(lifetime.vendorReports, lifetime.totalReports),
      currentReports: current?.totalReports || 0,
      currentAiPct: current?.aiMentionPct || 0,
      currentAdoptionPct: current?.adoptionPct || 0,
      currentRiskPct: current?.riskPct || 0,
      currentVendorPct: current?.vendorPct || 0,
    };
  })
  .sort((a, b) => b.currentRiskPct - a.currentRiskPct);

const companyScopeRows = Object.entries(data.byCompanyScope)
  .map(([scope, source]) => {
    const current = annualRowsForSubset(source.perReport);
    return {
      scope,
      lifetimeReports: source.perReport.summary.totalReports,
      lifetimeCompanies: source.perReport.summary.totalCompanies,
      lifetimeAiPct: pct(source.perReport.summary.aiSignalReports, source.perReport.summary.totalReports),
      lifetimeAdoptionPct: pct(source.perReport.summary.adoptionReports, source.perReport.summary.totalReports),
      lifetimeRiskPct: pct(source.perReport.summary.riskReports, source.perReport.summary.totalReports),
      lifetimeVendorPct: pct(source.perReport.summary.vendorReports, source.perReport.summary.totalReports),
      currentReports: current?.totalReports || 0,
      currentAiPct: current?.aiMentionPct || 0,
      currentAdoptionPct: current?.adoptionPct || 0,
      currentRiskPct: current?.riskPct || 0,
      currentVendorPct: current?.vendorPct || 0,
    };
  });

const reportCountBySectorYear = keyBy(dataset.reportCountBySectorYear, row => `${row.x}|||${row.y}`);
const noAiBySectorYear = keyBy(dataset.noAiBySectorYear, row => `${row.x}|||${row.y}`);
const noAiRiskBySectorYear = keyBy(dataset.noAiRiskBySectorYear, row => `${row.x}|||${row.y}`);
const riskMentionBySectorYear = keyBy(dataset.riskMentionBySectorYear, row => `${row.x}|||${row.y}`);

const sectorRows = data.sectors.map(sector => {
  const companyCount = Number(data.companiesPerSector?.[sector] || 0);
  const totalCurrent = Number(reportCountBySectorYear.get(`${lastFullYear}|||${sector}`)?.value || 0);
  const totalPrevious = Number(reportCountBySectorYear.get(`${previousFullYear}|||${sector}`)?.value || 0);
  const noAiCurrent = Number(noAiBySectorYear.get(`${lastFullYear}|||${sector}`)?.value || 0);
  const noAiPrevious = Number(noAiBySectorYear.get(`${previousFullYear}|||${sector}`)?.value || 0);
  const noRiskCurrent = Number(noAiRiskBySectorYear.get(`${lastFullYear}|||${sector}`)?.value || 0);
  const noRiskPrevious = Number(noAiRiskBySectorYear.get(`${previousFullYear}|||${sector}`)?.value || 0);
  const riskCurrent = Number(riskMentionBySectorYear.get(`${lastFullYear}|||${sector}`)?.value || 0);
  const riskPrevious = Number(riskMentionBySectorYear.get(`${previousFullYear}|||${sector}`)?.value || 0);
  const aiCurrent = totalCurrent - noAiCurrent;
  const aiPrevious = totalPrevious - noAiPrevious;

  const aiPctCurrent = pct(aiCurrent, totalCurrent);
  const aiPctPrevious = pct(aiPrevious, totalPrevious);
  const riskPctCurrent = pct(riskCurrent, totalCurrent);
  const riskPctPrevious = pct(riskPrevious, totalPrevious);
  const blindSpotPctCurrent = pct(noRiskCurrent, totalCurrent);

  return {
    sector,
    companyCount,
    totalCurrent,
    totalPrevious,
    aiCurrent,
    riskCurrent,
    noRiskCurrent,
    aiPctCurrent,
    riskPctCurrent,
    blindSpotPctCurrent,
    aiPpChange: aiPctCurrent - aiPctPrevious,
    riskPpChange: riskPctCurrent - riskPctPrevious,
  };
});

const sectorRowsForHeadlines = sectorRows.filter(
  row => row.totalCurrent >= MIN_SECTOR_REPORTS_FOR_HEADLINES && row.totalPrevious >= MIN_SECTOR_REPORTS_FOR_HEADLINES
);
const sectorLeaders = sectorRows
  .filter(row => row.totalCurrent >= MIN_SECTOR_REPORTS_FOR_HEADLINES)
  .slice()
  .sort((a, b) => b.riskPctCurrent - a.riskPctCurrent);
const sectorRisers = sectorRowsForHeadlines
  .slice()
  .sort((a, b) => b.riskPpChange - a.riskPpChange);
const sectorBlindSpots = sectorRows
  .filter(row => row.totalCurrent >= MIN_SECTOR_REPORTS_FOR_HEADLINES)
  .slice()
  .sort((a, b) => b.blindSpotPctCurrent - a.blindSpotPctCurrent);

const reportCountByIsicSectorYear = keyBy(dataset.reportCountByIsicSectorYear, row => `${row.x}|||${row.y}`);
const riskMentionByIsicSectorYear = keyBy(dataset.riskMentionByIsicSectorYear, row => `${row.x}|||${row.y}`);

const isicRows = data.isicSectors.map(isicSector => {
  const totalCurrent = Number(reportCountByIsicSectorYear.get(`${lastFullYear}|||${isicSector}`)?.value || 0);
  const totalPrevious = Number(reportCountByIsicSectorYear.get(`${previousFullYear}|||${isicSector}`)?.value || 0);
  const riskCurrent = Number(riskMentionByIsicSectorYear.get(`${lastFullYear}|||${isicSector}`)?.value || 0);
  const riskPrevious = Number(riskMentionByIsicSectorYear.get(`${previousFullYear}|||${isicSector}`)?.value || 0);
  const riskPctCurrent = pct(riskCurrent, totalCurrent);
  const riskPctPrevious = pct(riskPrevious, totalPrevious);

  return {
    isicSector,
    totalCurrent,
    riskCurrent,
    riskPctCurrent,
    riskPpChange: riskPctCurrent - riskPctPrevious,
  };
});

const isicRowsSampled = isicRows.filter(row => row.totalCurrent >= MIN_ISIC_REPORTS);
const isicLeaders = isicRowsSampled.slice().sort((a, b) => b.riskPctCurrent - a.riskPctCurrent);
const isicRisers = isicRowsSampled.slice().sort((a, b) => b.riskPpChange - a.riskPpChange);

const heatmapRowsForYear = (rows, year) =>
  rows
    .filter(row => row.x === year)
    .map(row => ({ key: row.y, value: Number(row.value || 0) }));

const mixRows = (rows, year) => {
  const items = heatmapRowsForYear(rows, year);
  const total = items.reduce((sum, item) => sum + item.value, 0);
  return {
    total,
    items: items.map(item => ({
      key: item.key,
      value: item.value,
      pct: pct(item.value, total),
    })),
  };
};

const riskSignalPrevious = mixRows(dataset.riskSignalHeatmap, previousFullYear);
const riskSignalCurrent = mixRows(dataset.riskSignalHeatmap, lastFullYear);
const substantivenessPrevious = mixRows(dataset.substantivenessHeatmap, previousFullYear);
const substantivenessCurrent = mixRows(dataset.substantivenessHeatmap, lastFullYear);

const signalQualityRows = riskSignalCurrent.items.map(item => {
  const prev = riskSignalPrevious.items.find(entry => entry.key === item.key) || { pct: 0, value: 0 };
  return {
    key: item.key,
    currentValue: item.value,
    currentPct: item.pct,
    previousValue: prev.value,
    previousPct: prev.pct,
    ppChange: item.pct - prev.pct,
  };
});

const substantivenessRows = substantivenessCurrent.items.map(item => {
  const prev = substantivenessPrevious.items.find(entry => entry.key === item.key) || { pct: 0, value: 0 };
  return {
    key: item.key,
    currentValue: item.value,
    currentPct: item.pct,
    previousValue: prev.value,
    previousPct: prev.pct,
    ppChange: item.pct - prev.pct,
  };
});

const substantivenessByYearBand = keyBy(dataset.substantivenessHeatmap, row => `${row.x}|||${row.y}`);
const getSubstantivenessCount = (year, band) =>
  Number(substantivenessByYearBand.get(`${year}|||${band}`)?.value || 0);

const qualityTrendRows = annualRows
  .filter(row => row.year >= 2020)
  .map(row => {
    const substantiveRiskReports = getSubstantivenessCount(row.year, 'substantive');
    const moderateRiskReports = getSubstantivenessCount(row.year, 'moderate');
    const boilerplateRiskReports = getSubstantivenessCount(row.year, 'boilerplate');
    return {
      year: row.year,
      substantiveRiskReports,
      moderateRiskReports,
      boilerplateRiskReports,
      substantiveRiskPctOfReports: pct(substantiveRiskReports, row.totalReports),
      substantiveRiskPctOfRiskReports: pct(substantiveRiskReports, row.riskReports),
      qualityGapPp: row.riskPct - pct(substantiveRiskReports, row.totalReports),
      adoptionRiskRatio: row.riskReports > 0 ? row.adoptionReports / row.riskReports : null,
      generalAmbiguousReports: row.generalAmbiguousReports,
      generalAmbiguousPct: row.generalAmbiguousPct,
    };
  });

const qualityTrendByYear = keyBy(qualityTrendRows, row => row.year);
const row2020 = annualByYear.get(2020);
const row2021 = annualByYear.get(2021);
const row2025 = annualByYear.get(2025);
const row2026 = annualByYear.get(2026);
const quality2020 = qualityTrendByYear.get(2020);
const quality2021 = qualityTrendByYear.get(2021);
const quality2025 = qualityTrendByYear.get(2025);
const quality2026 = qualityTrendByYear.get(2026);

const longRunComparisonRows = [
  {
    metric: 'Any AI mention rate',
    value2021: row2021.aiMentionPct,
    value2025: row2025.aiMentionPct,
    ppChange: row2025.aiMentionPct - row2021.aiMentionPct,
  },
  {
    metric: 'AI adoption rate',
    value2021: row2021.adoptionPct,
    value2025: row2025.adoptionPct,
    ppChange: row2025.adoptionPct - row2021.adoptionPct,
  },
  {
    metric: 'AI risk rate',
    value2021: row2021.riskPct,
    value2025: row2025.riskPct,
    ppChange: row2025.riskPct - row2021.riskPct,
  },
  {
    metric: 'General / ambiguous rate',
    value2021: row2021.generalAmbiguousPct,
    value2025: row2025.generalAmbiguousPct,
    ppChange: row2025.generalAmbiguousPct - row2021.generalAmbiguousPct,
  },
  {
    metric: 'AI vendor rate',
    value2021: row2021.vendorPct,
    value2025: row2025.vendorPct,
    ppChange: row2025.vendorPct - row2021.vendorPct,
  },
  {
    metric: 'Substantive risk rate (of all reports)',
    value2021: quality2021.substantiveRiskPctOfReports,
    value2025: quality2025.substantiveRiskPctOfReports,
    ppChange: quality2025.substantiveRiskPctOfReports - quality2021.substantiveRiskPctOfReports,
  },
  {
    metric: 'Substantive share of risk reports',
    value2021: quality2021.substantiveRiskPctOfRiskReports,
    value2025: quality2025.substantiveRiskPctOfRiskReports,
    ppChange: quality2025.substantiveRiskPctOfRiskReports - quality2021.substantiveRiskPctOfRiskReports,
  },
  {
    metric: 'Quality gap: risk minus substantive risk',
    value2021: quality2021.qualityGapPp,
    value2025: quality2025.qualityGapPp,
    ppChange: quality2025.qualityGapPp - quality2021.qualityGapPp,
    format: 'pp',
  },
  {
    metric: 'Adoption-to-risk ratio',
    value2021: quality2021.adoptionRiskRatio,
    value2025: quality2025.adoptionRiskRatio,
    ppChange: quality2025.adoptionRiskRatio - quality2021.adoptionRiskRatio,
    format: 'ratio',
  },
];

const startSeriesComparisonRows = [
  {
    metric: 'Any AI mention rate',
    value2020: row2020.aiMentionPct,
    value2025: row2025.aiMentionPct,
    ppChange: row2025.aiMentionPct - row2020.aiMentionPct,
  },
  {
    metric: 'AI adoption rate',
    value2020: row2020.adoptionPct,
    value2025: row2025.adoptionPct,
    ppChange: row2025.adoptionPct - row2020.adoptionPct,
  },
  {
    metric: 'AI risk rate',
    value2020: row2020.riskPct,
    value2025: row2025.riskPct,
    ppChange: row2025.riskPct - row2020.riskPct,
  },
  {
    metric: 'General / ambiguous rate',
    value2020: row2020.generalAmbiguousPct,
    value2025: row2025.generalAmbiguousPct,
    ppChange: row2025.generalAmbiguousPct - row2020.generalAmbiguousPct,
  },
  {
    metric: 'Substantive risk rate (of all reports)',
    value2020: quality2020.substantiveRiskPctOfReports,
    value2025: quality2025.substantiveRiskPctOfReports,
    ppChange: quality2025.substantiveRiskPctOfReports - quality2020.substantiveRiskPctOfReports,
  },
  {
    metric: 'Adoption-to-risk ratio',
    value2020: quality2020.adoptionRiskRatio,
    value2025: quality2025.adoptionRiskRatio,
    ppChange: quality2025.adoptionRiskRatio - quality2020.adoptionRiskRatio,
    format: 'ratio',
  },
];

const partial2026DirectionalRows = [
  {
    metric: 'Reports in sample',
    value2025: row2025.totalReports,
    value2026: row2026.totalReports,
    format: 'count',
  },
  {
    metric: 'Any AI mention rate',
    value2025: row2025.aiMentionPct,
    value2026: row2026.aiMentionPct,
    format: 'pct',
  },
  {
    metric: 'AI adoption rate',
    value2025: row2025.adoptionPct,
    value2026: row2026.adoptionPct,
    format: 'pct',
  },
  {
    metric: 'AI risk rate',
    value2025: row2025.riskPct,
    value2026: row2026.riskPct,
    format: 'pct',
  },
  {
    metric: 'Substantive risk rate (of all reports)',
    value2025: quality2025.substantiveRiskPctOfReports,
    value2026: quality2026.substantiveRiskPctOfReports,
    format: 'pct',
  },
  {
    metric: 'Adoption-to-risk ratio',
    value2025: quality2025.adoptionRiskRatio,
    value2026: quality2026.adoptionRiskRatio,
    format: 'ratio',
  },
];

const researchPriorityMetrics = [
  'Any AI mention rate by year',
  'AI risk mention rate by year',
  'AI adoption mention rate by year',
  'Adoption-to-risk ratio by year',
  'General / ambiguous rate by year',
  'Substantive risk rate by year, both as a share of all reports and as a share of risk-reporting reports',
  'Quality gap: AI risk mention rate minus substantive risk rate',
  'Sector AI-risk rate and sector AI-risk blind-spot rate',
  'Market-segment gap: FTSE 100, FTSE 350, Main Market, and AIM',
  'Risk-category composition shift over time',
  'Vendor opacity rate: other + undisclosed as a share of all vendor references',
  'Named-vendor concentration among explicitly named provider references',
];

const nextAnalyses = [
  {
    analysis: 'Company transition analysis',
    question: 'How many firms move from no AI disclosure to adoption, then from adoption to risk, and how many remain stuck in general / ambiguous language?',
    dataNeed: 'Requires company-year panel data or regeneration from raw report rows; not recoverable from the current dashboard artifact alone.',
  },
  {
    analysis: 'Persistence analysis',
    question: 'Once a company starts mentioning AI risk or reaches substantive disclosure, does it keep doing so in later years?',
    dataNeed: 'Requires company-year panel data.',
  },
  {
    analysis: 'Quality-adjusted sector analysis',
    question: 'Which sectors produce substantive risk disclosure rather than merely mentioning AI risk?',
    dataNeed: 'Partly supported now at aggregate level; best done with sector-level substantive report counts.',
  },
  {
    analysis: 'Over-index / under-index analysis',
    question: 'Which sectors and market segments disclose AI risk above or below the overall baseline once normalized?',
    dataNeed: 'Supported now from the current artifact.',
  },
  {
    analysis: 'Pre/post inflection analysis',
    question: 'Does the 2023 -> 2024 break look like a slope change or a level shift, consistent with a ChatGPT / anticipatory Provision 29 shock?',
    dataNeed: 'Supported now from annual series.',
  },
  {
    analysis: 'Boilerplate / staleness tracking',
    question: 'Are firms repeating the same AI-risk language year after year, or materially updating it?',
    dataNeed: 'Requires company-level text history; this is one of the highest-value next analyses for the paper.',
  },
  {
    analysis: 'Adoption-quality analysis',
    question: 'Are adoption disclosures becoming more operationally specific, or merely more common?',
    dataNeed: 'Requires extending substantiveness scoring to adoption chunks.',
  },
];

const recommendedHeadlineOutputs = [
  'A 2021-2025 core metrics table with percentage-point changes.',
  'A quality-gap table showing risk mention rate versus substantive risk rate.',
  'A CNI sector blind-spot table.',
  'A market-segment comparison centered on FTSE 100 versus AIM.',
  'A company transition analysis showing movement from no disclosure to adoption to risk.',
];

const topRiskRise = riskCategoryRows[0];
const topAdoptionRise = adoptionCategoryRows[0];
const topVendorBucket = vendorCategoryRows.slice().sort((a, b) => b.currentCount - a.currentCount)[0];
const namedVendorLeader = vendorCategoryRows
  .filter(row => !['other', 'undisclosed', 'internal'].includes(row.key))
  .slice()
  .sort((a, b) => b.currentCount - a.currentCount)[0];
const topMarketSegment = marketSegmentRows[0];
const bottomMarketSegment = marketSegmentRows[marketSegmentRows.length - 1];
const topSectorLeader = sectorLeaders[0];
const topSectorRiser = sectorRisers[0];
const topBlindSpot = sectorBlindSpots[0];
const topIsicLeader = isicLeaders[0];

const lifetimeSectorCompanyRows = Object.entries(data.companiesPerSector)
  .map(([sector, companyCount]) => ({
    sector,
    companyCount: Number(companyCount),
    share: pct(Number(companyCount), data.datasets.perReport.summary.totalCompanies),
  }))
  .sort((a, b) => b.companyCount - a.companyCount);

const headlineBullets = [
  `The current artifact covers ${fmtNumber(data.datasets.perReport.summary.totalReports)} reports across ${fmtNumber(data.datasets.perReport.summary.totalCompanies)} companies from ${years[0]}-${latestYear}. ${fmtPct(pct(data.reportClassificationBreakdown.phase1SignalReports, data.reportClassificationBreakdown.totalReports))} of all reports contain at least one non-\`none\` AI signal.`,
  `In ${lastFullYear}, ${fmtPct(currentFullYearRow.aiMentionPct)} of reports mentioned AI at all, up ${fmtSignedPp(yoy.aiMentionPp)} from ${previousFullYear}; AI-risk disclosure rose even faster to ${fmtPct(currentFullYearRow.riskPct)} (${fmtSignedPp(yoy.riskPp)}).`,
  `The adoption-versus-risk disclosure gap narrowed from ${fmtPp(previousFullYearRow.adoptionRiskGapPp)} in ${previousFullYear} to ${fmtPp(currentFullYearRow.adoptionRiskGapPp)} in ${lastFullYear}, suggesting risk disclosure is catching up with general AI adoption language.`,
  `${labelize(topRiskRise.key)} was the fastest-rising risk category in ${lastFullYear}, increasing ${fmtSignedPp(topRiskRise.ppChange)} to ${fmtPct(topRiskRise.currentPct)} of all reports. Cybersecurity and Operational / Technical risk were close behind.`,
  `LLM disclosure was the fastest-rising adoption category, reaching ${fmtPct(topAdoptionRise.currentPct)} of all reports in ${lastFullYear} (${fmtSignedPp(topAdoptionRise.ppChange)} YoY). Agentic references also rose to ${fmtPct(adoptionCategoryRows.find(row => row.key === 'agentic')?.currentPct || 0)}.`,
  `${topMarketSegment.segment} had the highest AI-risk rate in ${lastFullYear} at ${fmtPct(topMarketSegment.currentRiskPct)}, while ${bottomMarketSegment.segment} remained far lower at ${fmtPct(bottomMarketSegment.currentRiskPct)}.`,
  `Among CNI sectors with at least ${MIN_SECTOR_REPORTS_FOR_HEADLINES} reports in ${lastFullYear}, ${topSectorRiser.sector} saw the biggest rise in AI-risk disclosure (${fmtSignedPp(topSectorRiser.riskPpChange)}), while ${topBlindSpot.sector} had the largest remaining AI-risk blind spot (${fmtPct(topBlindSpot.blindSpotPctCurrent)} of reports still without an AI-risk mention).`,
  `Vendor references remain fragmented. The largest vendor bucket in ${lastFullYear} was \`${topVendorBucket.key}\` at ${fmtPct((topVendorBucket.currentCount / vendorConcentration.total) * 100)} of vendor assignments; the leading named vendor was ${labelize(namedVendorLeader.key)} at ${fmtPct((namedVendorLeader.currentCount / vendorConcentration.total) * 100)}.`,
  `In ${lastFullYear}, opaque vendor references (\`other\` + \`undisclosed\`) accounted for ${fmtPct(vendorOpacityShare)} of all vendor assignments. Among explicitly named vendors, the top three accounted for ${fmtPct(namedVendorConcentration.topThreeShare)} of named-vendor assignments.`,
  `Risk disclosures became denser over time: average risk labels per risk-reporting company rose from ${fmtRatio(previousRiskLabelsPerRiskReport)} in ${previousFullYear} to ${fmtRatio(currentRiskLabelsPerRiskReport)} in ${lastFullYear}.`,
  `At ISIC level, the strongest large-sample AI-risk disclosure rate in ${lastFullYear} was in ${topIsicLeader.isicSector} (${fmtPct(topIsicLeader.riskPctCurrent)}; n=${fmtNumber(topIsicLeader.totalCurrent)} reports, using a minimum-sample filter of ${MIN_ISIC_REPORTS}).`,
];

const table = (rows, columns) => {
  const escape = value => String(value).replaceAll('|', '\\|');
  const header = `| ${columns.map(column => column.header).join(' | ')} |`;
  const divider = `| ${columns.map(column => (column.align === 'right' ? '---:' : ':---')).join(' | ')} |`;
  const body = rows.map(row => `| ${columns.map(column => escape(column.value(row))).join(' | ')} |`).join('\n');
  return [header, divider, body].filter(Boolean).join('\n');
};

const findingsMd = `# Dashboard Findings

Generated on ${generatedOn} from \`data/dashboard-data.json\`.

## Scope and caveats

- This report uses the existing precomputed dashboard artifact only. It does not add any new metrics to the dashboard.
- ${latestYear} is present in the artifact, but because the current date is ${generatedOn}, ${latestYear} should be treated as a partial year. The main YoY comparisons below therefore use ${previousFullYear} -> ${lastFullYear}.
- Tables based on \`riskTrend\`, \`adoptionTrend\`, and \`vendorTrend\` are label-assignment counts, not unique-report counts. A single report can contribute to multiple labels.
- Sector-level unique adoption and vendor rates are not derivable from the current artifact because the available sector arrays for adoption and vendor are label-level counts rather than unique report counts.

## Recommended comparison windows

- Main research window: \`2021 -> 2025\`. This matches the paper draft's primary analysis period and avoids over-weighting the partial 2020 and partial 2026 edges of the series.
- Supporting long-run window: \`2020 -> 2025\`. This is useful as a start-of-series anchor and for communicating scale of change since the earliest observable baseline.
- Directional snapshot only: \`2026\`. Because the current date is ${generatedOn}, 2026 should be treated as a partial-year directional signal, not as a directly comparable full-year endpoint.

## Priority metrics for the paper

${researchPriorityMetrics.map(item => `- ${item}`).join('\n')}

## Headline findings

${headlineBullets.map(item => `- ${item}`).join('\n')}

## Coverage summary

${table(
  [
    {
      reports: data.datasets.perReport.summary.totalReports,
      companies: data.datasets.perReport.summary.totalCompanies,
      signalReports: data.reportClassificationBreakdown.phase1SignalReports,
      signalPct: pct(data.reportClassificationBreakdown.phase1SignalReports, data.reportClassificationBreakdown.totalReports),
      averagePhase1Labels: data.reportClassificationBreakdown.averageLabelsPerSignalReport,
    },
  ],
  [
    { header: 'Reports', align: 'right', value: row => fmtNumber(row.reports) },
    { header: 'Companies', align: 'right', value: row => fmtNumber(row.companies) },
    { header: 'AI signal reports', align: 'right', value: row => fmtNumber(row.signalReports) },
    { header: 'AI signal rate', align: 'right', value: row => fmtPct(row.signalPct) },
    { header: 'Avg phase-1 labels / signal report', align: 'right', value: row => fmtRatio(row.averagePhase1Labels) },
  ]
)}

### Company distribution by CNI sector

${table(
  lifetimeSectorCompanyRows,
  [
    { header: 'Sector', value: row => row.sector },
    { header: 'Companies', align: 'right', value: row => fmtNumber(row.companyCount) },
    { header: 'Share of all companies', align: 'right', value: row => fmtPct(row.share) },
  ]
)}

## Annual report-level trend summary

${table(
  metricTrendRows,
  [
    { header: 'Year', align: 'right', value: row => row.year },
    { header: 'Reports', align: 'right', value: row => fmtNumber(row.totalReports) },
    { header: 'AI mention %', align: 'right', value: row => fmtPct(row.aiMentionPct) },
    { header: 'Adoption %', align: 'right', value: row => fmtPct(row.adoptionPct) },
    { header: 'Risk %', align: 'right', value: row => fmtPct(row.riskPct) },
    { header: 'General / ambiguous %', align: 'right', value: row => fmtPct(row.generalAmbiguousPct) },
    { header: 'Vendor %', align: 'right', value: row => fmtPct(row.vendorPct) },
    { header: 'Adoption-risk gap', align: 'right', value: row => fmtPp(row.adoptionRiskGapPp) },
  ]
)}

### Core 2021 -> 2025 research metrics

${table(
  longRunComparisonRows,
  [
    { header: 'Metric', value: row => row.metric },
    {
      header: '2021',
      align: 'right',
      value: row => row.format === 'ratio' ? fmtRatio(row.value2021) : row.format === 'pp' ? fmtPp(row.value2021) : fmtPct(row.value2021),
    },
    {
      header: '2025',
      align: 'right',
      value: row => row.format === 'ratio' ? fmtRatio(row.value2025) : row.format === 'pp' ? fmtPp(row.value2025) : fmtPct(row.value2025),
    },
    {
      header: 'Change',
      align: 'right',
      value: row => row.format === 'ratio' ? `${row.ppChange >= 0 ? '+' : ''}${fmtRatio(row.ppChange)}` : fmtSignedPp(row.ppChange),
    },
  ]
)}

### Supporting 2020 -> 2025 start-of-series comparison

${table(
  startSeriesComparisonRows,
  [
    { header: 'Metric', value: row => row.metric },
    {
      header: '2020',
      align: 'right',
      value: row => row.format === 'ratio' ? fmtRatio(row.value2020) : row.format === 'pp' ? fmtPp(row.value2020) : fmtPct(row.value2020),
    },
    {
      header: '2025',
      align: 'right',
      value: row => row.format === 'ratio' ? fmtRatio(row.value2025) : row.format === 'pp' ? fmtPp(row.value2025) : fmtPct(row.value2025),
    },
    {
      header: 'Change',
      align: 'right',
      value: row => row.format === 'ratio' ? `${row.ppChange >= 0 ? '+' : ''}${fmtRatio(row.ppChange)}` : fmtSignedPp(row.ppChange),
    },
  ]
)}

### Partial 2026 directional snapshot

2026 is partial and should not be compared to full-year 2025 as if both were complete annual cohorts. This table is included only to show direction of travel.

${table(
  partial2026DirectionalRows,
  [
    { header: 'Metric', value: row => row.metric },
    {
      header: '2025',
      align: 'right',
      value: row => row.format === 'count'
        ? fmtNumber(row.value2025)
        : row.format === 'ratio'
          ? fmtRatio(row.value2025)
          : fmtPct(row.value2025),
    },
    {
      header: '2026 partial',
      align: 'right',
      value: row => row.format === 'count'
        ? fmtNumber(row.value2026)
        : row.format === 'ratio'
          ? fmtRatio(row.value2026)
          : fmtPct(row.value2026),
    },
  ]
)}

## Quality-gap analysis

This is the most directly policy-relevant metric family in the current dataset: it separates the growth in AI-risk mentions from the much smaller share of reports that contain genuinely substantive AI-risk disclosure.

${table(
  qualityTrendRows,
  [
    { header: 'Year', align: 'right', value: row => row.year },
    { header: 'Risk reports', align: 'right', value: row => fmtNumber(annualByYear.get(row.year).riskReports) },
    { header: 'Risk rate', align: 'right', value: row => fmtPct(annualByYear.get(row.year).riskPct) },
    { header: 'Substantive risk reports', align: 'right', value: row => fmtNumber(row.substantiveRiskReports) },
    { header: 'Substantive rate', align: 'right', value: row => fmtPct(row.substantiveRiskPctOfReports) },
    { header: 'Substantive share of risk reports', align: 'right', value: row => fmtPct(row.substantiveRiskPctOfRiskReports) },
    { header: 'Quality gap', align: 'right', value: row => fmtPp(row.qualityGapPp) },
  ]
)}

### ${previousFullYear} -> ${lastFullYear} headline rate changes

${table(
  [
    {
      metric: 'Any AI mention',
      previous: previousFullYearRow.aiMentionPct,
      current: currentFullYearRow.aiMentionPct,
      change: yoy.aiMentionPp,
      countChange: currentFullYearRow.aiMentionReports - previousFullYearRow.aiMentionReports,
    },
    {
      metric: 'AI adoption mention',
      previous: previousFullYearRow.adoptionPct,
      current: currentFullYearRow.adoptionPct,
      change: yoy.adoptionPp,
      countChange: currentFullYearRow.adoptionReports - previousFullYearRow.adoptionReports,
    },
    {
      metric: 'AI risk mention',
      previous: previousFullYearRow.riskPct,
      current: currentFullYearRow.riskPct,
      change: yoy.riskPp,
      countChange: currentFullYearRow.riskReports - previousFullYearRow.riskReports,
    },
    {
      metric: 'AI vendor mention',
      previous: previousFullYearRow.vendorPct,
      current: currentFullYearRow.vendorPct,
      change: yoy.vendorPp,
      countChange: currentFullYearRow.vendorReports - previousFullYearRow.vendorReports,
    },
    {
      metric: 'Adoption-risk gap',
      previous: previousFullYearRow.adoptionRiskGapPp,
      current: currentFullYearRow.adoptionRiskGapPp,
      change: yoy.adoptionRiskGapPp,
      countChange: 0,
    },
  ],
  [
    { header: 'Metric', value: row => row.metric },
    { header: `${previousFullYear}`, align: 'right', value: row => row.metric === 'Adoption-risk gap' ? fmtPp(row.previous) : fmtPct(row.previous) },
    { header: `${lastFullYear}`, align: 'right', value: row => row.metric === 'Adoption-risk gap' ? fmtPp(row.current) : fmtPct(row.current) },
    { header: 'Change', align: 'right', value: row => fmtSignedPp(row.change) },
    { header: 'Count change', align: 'right', value: row => row.metric === 'Adoption-risk gap' ? 'n/a' : fmtSignedCount(row.countChange) },
  ]
)}

## Risk taxonomy findings

- Risk-category counts are label assignments, so totals exceed the number of unique risk-reporting companies.
- In ${lastFullYear}, the top three risk categories accounted for ${fmtPct(riskConcentration.topThreeShare)} of all risk-label assignments. The HHI for risk-category concentration was ${fmtNumber(Math.round(riskConcentration.hhi))}.
- Average risk labels per risk-reporting report rose from ${fmtRatio(previousRiskLabelsPerRiskReport)} in ${previousFullYear} to ${fmtRatio(currentRiskLabelsPerRiskReport)} in ${lastFullYear}.

${table(
  riskCategoryRows,
  [
    { header: 'Risk category', value: row => labelize(row.key) },
    { header: `${lastFullYear} count`, align: 'right', value: row => fmtNumber(row.currentCount) },
    { header: `${lastFullYear} % of reports`, align: 'right', value: row => fmtPct(row.currentPct) },
    { header: `${previousFullYear} % of reports`, align: 'right', value: row => fmtPct(row.previousPct) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.ppChange) },
  ]
)}

## Adoption findings

- Adoption-category counts are label assignments, not unique reports.
- In ${lastFullYear}, the top three adoption categories accounted for ${fmtPct(adoptionConcentration.topThreeShare)} of all adoption-label assignments.
- Average adoption labels per adoption-reporting report rose from ${fmtRatio(previousAdoptionLabelsPerAdoptionReport)} in ${previousFullYear} to ${fmtRatio(currentAdoptionLabelsPerAdoptionReport)} in ${lastFullYear}.

${table(
  adoptionCategoryRows,
  [
    { header: 'Adoption category', value: row => labelize(row.key) },
    { header: `${lastFullYear} count`, align: 'right', value: row => fmtNumber(row.currentCount) },
    { header: `${lastFullYear} % of reports`, align: 'right', value: row => fmtPct(row.currentPct) },
    { header: `${previousFullYear} % of reports`, align: 'right', value: row => fmtPct(row.previousPct) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.ppChange) },
  ]
)}

## Vendor findings

- Vendor-category counts are label assignments, not unique reports.
- In ${lastFullYear}, the top three vendor buckets accounted for ${fmtPct(vendorConcentration.topThreeShare)} of all vendor-label assignments. The HHI for vendor concentration was ${fmtNumber(Math.round(vendorConcentration.hhi))}.
- Average vendor labels per vendor-reporting report rose from ${fmtRatio(previousVendorLabelsPerVendorReport)} in ${previousFullYear} to ${fmtRatio(currentVendorLabelsPerVendorReport)} in ${lastFullYear}.

${table(
  vendorCategoryRows,
  [
    { header: 'Vendor bucket', value: row => labelize(row.key) },
    { header: `${lastFullYear} count`, align: 'right', value: row => fmtNumber(row.currentCount) },
    { header: `${lastFullYear} % of reports`, align: 'right', value: row => fmtPct(row.currentPct) },
    { header: `${previousFullYear} % of reports`, align: 'right', value: row => fmtPct(row.previousPct) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.ppChange) },
  ]
)}

## Market segment comparison

${table(
  marketSegmentRows,
  [
    { header: 'Market segment', value: row => row.segment },
    { header: 'Lifetime reports', align: 'right', value: row => fmtNumber(row.lifetimeReports) },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.currentReports) },
    { header: `${lastFullYear} AI mention %`, align: 'right', value: row => fmtPct(row.currentAiPct) },
    { header: `${lastFullYear} adoption %`, align: 'right', value: row => fmtPct(row.currentAdoptionPct) },
    { header: `${lastFullYear} risk %`, align: 'right', value: row => fmtPct(row.currentRiskPct) },
    { header: `${lastFullYear} vendor %`, align: 'right', value: row => fmtPct(row.currentVendorPct) },
  ]
)}

## CNI-only versus all companies

${table(
  companyScopeRows,
  [
    { header: 'Scope', value: row => row.scope },
    { header: 'Lifetime reports', align: 'right', value: row => fmtNumber(row.lifetimeReports) },
    { header: 'Lifetime companies', align: 'right', value: row => fmtNumber(row.lifetimeCompanies) },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.currentReports) },
    { header: `${lastFullYear} AI mention %`, align: 'right', value: row => fmtPct(row.currentAiPct) },
    { header: `${lastFullYear} adoption %`, align: 'right', value: row => fmtPct(row.currentAdoptionPct) },
    { header: `${lastFullYear} risk %`, align: 'right', value: row => fmtPct(row.currentRiskPct) },
    { header: `${lastFullYear} vendor %`, align: 'right', value: row => fmtPct(row.currentVendorPct) },
  ]
)}

## CNI sector summary (${lastFullYear})

${table(
  sectorRows.slice().sort((a, b) => b.riskPctCurrent - a.riskPctCurrent),
  [
    { header: 'Sector', value: row => row.sector },
    { header: 'Companies', align: 'right', value: row => fmtNumber(row.companyCount) },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalCurrent) },
    { header: 'AI mention %', align: 'right', value: row => fmtPct(row.aiPctCurrent) },
    { header: 'AI risk %', align: 'right', value: row => fmtPct(row.riskPctCurrent) },
    { header: 'No AI-risk %', align: 'right', value: row => fmtPct(row.blindSpotPctCurrent) },
    { header: 'AI mention YoY', align: 'right', value: row => fmtSignedPp(row.aiPpChange) },
    { header: 'AI risk YoY', align: 'right', value: row => fmtSignedPp(row.riskPpChange) },
  ]
)}

### Largest CNI sector rises in AI-risk disclosure (${previousFullYear} -> ${lastFullYear})

${table(
  sectorRisers,
  [
    { header: 'Sector', value: row => row.sector },
    { header: `${previousFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalPrevious) },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalCurrent) },
    { header: `${lastFullYear} AI risk %`, align: 'right', value: row => fmtPct(row.riskPctCurrent) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.riskPpChange) },
  ]
)}

### Largest CNI-sector AI-risk blind spots (${lastFullYear})

${table(
  sectorBlindSpots,
  [
    { header: 'Sector', value: row => row.sector },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalCurrent) },
    { header: 'No AI-risk reports', align: 'right', value: row => fmtNumber(row.noRiskCurrent) },
    { header: 'No AI-risk %', align: 'right', value: row => fmtPct(row.blindSpotPctCurrent) },
  ]
)}

## ISIC industries with strongest AI-risk disclosure

Minimum sample filter: ${MIN_ISIC_REPORTS} reports in ${lastFullYear}.

### Highest AI-risk disclosure rates

${table(
  isicLeaders.slice(0, 15),
  [
    { header: 'ISIC industry', value: row => row.isicSector },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalCurrent) },
    { header: `${lastFullYear} risk reports`, align: 'right', value: row => fmtNumber(row.riskCurrent) },
    { header: `${lastFullYear} AI risk %`, align: 'right', value: row => fmtPct(row.riskPctCurrent) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.riskPpChange) },
  ]
)}

### Fastest-rising AI-risk disclosure rates

${table(
  isicRisers.slice(0, 15),
  [
    { header: 'ISIC industry', value: row => row.isicSector },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.totalCurrent) },
    { header: `${lastFullYear} AI risk %`, align: 'right', value: row => fmtPct(row.riskPctCurrent) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.riskPpChange) },
  ]
)}

## Signal quality

Risk signal strength is based on label-level assignments, not unique reports. Risk substantiveness is reported at report level.

### Risk signal mix (${previousFullYear} vs ${lastFullYear})

${table(
  signalQualityRows,
  [
    { header: 'Signal level', value: row => row.key },
    { header: `${previousFullYear} share`, align: 'right', value: row => fmtPct(row.previousPct) },
    { header: `${lastFullYear} share`, align: 'right', value: row => fmtPct(row.currentPct) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.ppChange) },
    { header: `${lastFullYear} assignments`, align: 'right', value: row => fmtNumber(row.currentValue) },
  ]
)}

### Risk substantiveness mix (${previousFullYear} vs ${lastFullYear})

${table(
  substantivenessRows,
  [
    { header: 'Band', value: row => row.key },
    { header: `${previousFullYear} share`, align: 'right', value: row => fmtPct(row.previousPct) },
    { header: `${lastFullYear} share`, align: 'right', value: row => fmtPct(row.currentPct) },
    { header: 'YoY change', align: 'right', value: row => fmtSignedPp(row.ppChange) },
    { header: `${lastFullYear} reports`, align: 'right', value: row => fmtNumber(row.currentValue) },
  ]
)}

## Notes for follow-up analysis

- The artifact already supports strong report-level findings for annual trends, CNI sectors, market segments, and ISIC risk rates.
- The cleanest candidate metrics for later dashboard work are the ones that are already robust here: report-level rates, YoY percentage-point changes, blind-spot rates, segment gaps, and quality-adjusted risk rates.
- If we later want sector-level unique adoption or vendor rates, the artifact will need unique report counts by sector for those dimensions rather than label-assignment counts.

## Additional analyses to run next

${nextAnalyses.map(item => `- ${item.analysis}: ${item.question} ${item.dataNeed}`).join('\n')}

## Recommended headline outputs for the paper

${recommendedHeadlineOutputs.map(item => `- ${item}`).join('\n')}
`;

fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
fs.writeFileSync(OUTPUT_PATH, `${findingsMd.trim()}\n`, 'utf8');

console.log(`Wrote ${OUTPUT_PATH}`);
