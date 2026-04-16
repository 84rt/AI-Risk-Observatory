import fs from 'fs';
import path from 'path';

const DASHBOARD_DATA_PATH = path.join(process.cwd(), 'data', 'dashboard-data.json');
const REPORT_FIGURES_PATH = path.join(process.cwd(), '..', 'report', 'report-figures.json');

const MAX_REPORT_YEAR = 2025;
const FIGURE_5_START_YEAR = 2021;

const ADOPTION_COLORS = {
  non_llm: '#64748b',
  llm: '#3b82f6',
  agentic: '#f59e0b',
};

const VENDOR_COLORS = {
  openai: '#e63946',
  microsoft: '#3b82f6',
  google: '#16a34a',
  amazon: '#f59e0b',
  meta: '#1e3a8a',
  anthropic: '#d97706',
  internal: '#0b0c0c',
  other: '#64748b',
  undisclosed: '#e2e8f0',
};

const MENTION_COLORS = {
  adoption: '#64748b',
  risk: '#e63946',
  vendor: '#3b82f6',
  general_ambiguous: '#9ca3af',
};

const RISK_COLORS = {
  strategic_competitive: '#1d4ed8',
  cybersecurity: '#3b82f6',
  operational_technical: '#93c5fd',
  regulatory_compliance: '#dbeafe',
  reputational_ethical: '#fecdd3',
  third_party_supply_chain: '#fca5a5',
  information_integrity: '#ef4444',
  workforce_impacts: '#f87171',
  environmental_impact: '#7f1d1d',
  national_security: '#b91c1c',
};

const MENTION_LABELS = {
  adoption: 'Adoption',
  risk: 'Risk',
  vendor: 'Vendor',
  general_ambiguous: 'General / Ambiguous',
};

const ADOPTION_LABELS = {
  non_llm: 'Traditional AI (non-LLM)',
  llm: 'LLM / Generative AI',
  agentic: 'Agentic AI',
};

const RISK_LABELS = {
  strategic_competitive: 'Strategic / Competitive',
  cybersecurity: 'Cybersecurity',
  operational_technical: 'Operational / Technical',
  regulatory_compliance: 'Regulatory / Compliance',
  reputational_ethical: 'Reputational / Ethical',
  third_party_supply_chain: 'Third-Party / Supply Chain',
  information_integrity: 'Information Integrity',
  workforce_impacts: 'Workforce Impacts',
  environmental_impact: 'Environmental Impact',
  national_security: 'National Security',
};

const VENDOR_LABELS = {
  openai: 'OpenAI',
  microsoft: 'Microsoft',
  google: 'Google',
  amazon: 'Amazon / AWS',
  meta: 'Meta',
  anthropic: 'Anthropic',
  internal: 'Internal / proprietary',
  other: 'Other (named, unlisted)',
  undisclosed: 'Undisclosed',
};

const MARKET_SEGMENT_ORDER = [
  'Main Market (FTSE 100 only)',
  'Main Market (FTSE 350 only)',
  'Main Market',
  'AIM',
];

const MARKET_SEGMENT_LABELS = {
  'Main Market (FTSE 100 only)': 'FTSE 100',
  'Main Market (FTSE 350 only)': 'FTSE 350',
  'Main Market': 'Main Market',
  AIM: 'AIM',
};

const requiredDatasetKeys = [
  'mentionTrend',
  'adoptionTrend',
  'riskTrend',
  'vendorTrend',
  'blindSpotTrend',
  'riskMentionBySectorYear',
  'noAiRiskBySectorYear',
  'reportCountBySectorYear',
  'substantivenessHeatmap',
];

const readJson = filePath => {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing required JSON file: ${filePath}`);
  }
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
};

const round1 = value => Math.round((value + Number.EPSILON) * 10) / 10;

const pctRaw = (count, total) => {
  if (!Number.isFinite(count) || !Number.isFinite(total) || total <= 0) return 0;
  return (count / total) * 100;
};

const pct = (count, total) => {
  return round1(pctRaw(count, total));
};

const getYearRow = (rows, year, yearKey = 'year') => {
  const row = rows.find(entry => entry[yearKey] === year);
  if (!row) {
    throw new Error(`Missing ${yearKey}=${year} row in dataset.`);
  }
  return row;
};

const getSectorYearValue = (rows, year, sector) => {
  const row = rows.find(entry => entry.x === year && entry.y === sector);
  return row ? row.value : 0;
};

const ensureDashboardShape = dashboardData => {
  if (!dashboardData?.datasets?.perReport) {
    throw new Error('dashboard-data.json is missing datasets.perReport.');
  }

  const perReport = dashboardData.datasets.perReport;
  for (const key of requiredDatasetKeys) {
    if (!perReport[key]) {
      throw new Error(`dashboard-data.json is missing datasets.perReport.${key}.`);
    }
  }

  if (!dashboardData.byCompanyScope?.cniOnly?.perReport) {
    throw new Error('dashboard-data.json is missing byCompanyScope.cniOnly.perReport.');
  }

  for (const segment of MARKET_SEGMENT_ORDER) {
    if (!dashboardData.byMarketSegment?.[segment]?.perReport) {
      throw new Error(`dashboard-data.json is missing byMarketSegment["${segment}"].perReport.`);
    }
  }
};

const buildFigure1 = dashboardData => {
  const dataset = dashboardData.datasets.perReport;
  const years = dashboardData.years.filter(year => year <= MAX_REPORT_YEAR);

  const data = years.map(year => {
    const mentionRow = getYearRow(dataset.mentionTrend, year);
    const blindSpotRow = getYearRow(dataset.blindSpotTrend, year);
    const totalReports = blindSpotRow.total_reports;

    return {
      year,
      total_reports: totalReports,
      adoption_count: mentionRow.adoption,
      risk_count: mentionRow.risk,
      vendor_count: mentionRow.vendor,
      general_ambiguous_count: mentionRow.general_ambiguous,
      adoption_rate_pct: pct(mentionRow.adoption, totalReports),
      risk_rate_pct: pct(mentionRow.risk, totalReports),
      vendor_rate_pct: pct(mentionRow.vendor, totalReports),
      general_ambiguous_rate_pct: pct(mentionRow.general_ambiguous, totalReports),
      adoption_minus_risk_gap_pp: round1(
        pctRaw(mentionRow.adoption, totalReports) - pctRaw(mentionRow.risk, totalReports)
      ),
    };
  });

  return {
    id: 'figure1',
    title: 'Annual disclosure rates by signal type',
    chartType: 'line',
    domain: {
      years: [years[0], years.at(-1)],
      measure: 'share of all reports in year (%)',
    },
    series: [
      { key: 'adoption_rate_pct', label: MENTION_LABELS.adoption, color: MENTION_COLORS.adoption },
      { key: 'risk_rate_pct', label: MENTION_LABELS.risk, color: MENTION_COLORS.risk },
      { key: 'general_ambiguous_rate_pct', label: MENTION_LABELS.general_ambiguous, color: MENTION_COLORS.general_ambiguous },
      { key: 'vendor_rate_pct', label: MENTION_LABELS.vendor, color: MENTION_COLORS.vendor },
    ],
    shading: {
      label: 'Adoption-risk gap',
      lowerKey: 'risk_rate_pct',
      upperKey: 'adoption_rate_pct',
    },
    data,
    notes: [
      'Base dataset: datasets.perReport.',
      'Percentages are report-level counts divided by total reports in the publication year.',
      '2026 partial data is excluded.',
    ],
  };
};

const buildFigure2 = dashboardData => {
  const dataset = dashboardData.datasets.perReport;
  const years = dashboardData.years.filter(year => year <= MAX_REPORT_YEAR);

  return {
    id: 'figure2',
    title: 'Adoption type report counts by year',
    chartType: 'stacked-area',
    domain: {
      years: [years[0], years.at(-1)],
      measure: 'report count',
    },
    series: [
      { key: 'non_llm', label: ADOPTION_LABELS.non_llm, color: ADOPTION_COLORS.non_llm },
      { key: 'llm', label: ADOPTION_LABELS.llm, color: ADOPTION_COLORS.llm },
      { key: 'agentic', label: ADOPTION_LABELS.agentic, color: ADOPTION_COLORS.agentic },
    ],
    data: years.map(year => {
      const row = getYearRow(dataset.adoptionTrend, year);
      return {
        year,
        non_llm: row.non_llm,
        llm: row.llm,
        agentic: row.agentic,
      };
    }),
    notes: [
      'Base dataset: datasets.perReport.',
      'Adoption labels are not mutually exclusive, so stacked totals may exceed the number of unique reports.',
      '2026 partial data is excluded.',
    ],
  };
};

const buildFigure3 = dashboardData => {
  const dataset = dashboardData.datasets.perReport;
  const row2024 = getYearRow(dataset.riskTrend, 2024);
  const row2025 = getYearRow(dataset.riskTrend, 2025);
  const total2024 = getYearRow(dataset.blindSpotTrend, 2024).total_reports;
  const total2025 = getYearRow(dataset.blindSpotTrend, 2025).total_reports;

  const data = dashboardData.labels.riskLabels.map(key => {
    const count2024 = row2024[key] ?? 0;
    const count2025 = row2025[key] ?? 0;
    const rate2024 = pct(count2024, total2024);
    const rate2025 = pct(count2025, total2025);
    const rawRate2024 = pctRaw(count2024, total2024);
    const rawRate2025 = pctRaw(count2025, total2025);

    return {
      key,
      label: RISK_LABELS[key] ?? key,
      color: RISK_COLORS[key] ?? '#94a3b8',
      count_2024: count2024,
      count_2025: count2025,
      rate_2024_pct: rate2024,
      rate_2025_pct: rate2025,
      delta_pp: round1(rawRate2025 - rawRate2024),
    };
  }).sort((a, b) => b.delta_pp - a.delta_pp);

  return {
    id: 'figure3',
    title: 'Risk-category YoY rate change',
    chartType: 'horizontal-bar',
    comparison: {
      fromYear: 2024,
      toYear: 2025,
      measure: 'percentage-point change in share of all reports',
    },
    data,
    notes: [
      'Base dataset: datasets.perReport.',
      'Rates are risk-category report counts divided by total reports in the year.',
      'Risk labels are not mutually exclusive.',
    ],
  };
};

const buildFigure4 = dashboardData => {
  const row2025 = getYearRow(dashboardData.datasets.perReport.vendorTrend, 2025);
  const vendorKeys = dashboardData.labels.vendorTags;
  const totalAssignments = vendorKeys.reduce((sum, key) => sum + (row2025[key] ?? 0), 0);

  const trackedNamedKeys = ['openai', 'microsoft', 'google', 'amazon', 'meta', 'anthropic'];
  const opaqueKeys = ['other', 'undisclosed'];

  const data = vendorKeys.map(key => ({
    key,
    label: VENDOR_LABELS[key] ?? key,
    color: VENDOR_COLORS[key] ?? '#94a3b8',
    group: trackedNamedKeys.includes(key)
      ? 'named_tracked'
      : opaqueKeys.includes(key)
        ? 'opaque_or_untracked'
        : key === 'internal'
          ? 'internal'
          : 'other',
    assignments_2025: row2025[key] ?? 0,
    share_pct: pct(row2025[key] ?? 0, totalAssignments),
  })).sort((a, b) => b.assignments_2025 - a.assignments_2025);

  const groupedTotals = {
    named_tracked_assignments: trackedNamedKeys.reduce((sum, key) => sum + (row2025[key] ?? 0), 0),
    opaque_or_untracked_assignments: opaqueKeys.reduce((sum, key) => sum + (row2025[key] ?? 0), 0),
    internal_assignments: row2025.internal ?? 0,
  };

  return {
    id: 'figure4',
    title: 'Vendor reference distribution in 2025',
    chartType: 'treemap',
    measure: 'vendor-tag assignments',
    total_assignments_2025: totalAssignments,
    groupedTotals: {
      ...groupedTotals,
      named_tracked_share_pct: pct(groupedTotals.named_tracked_assignments, totalAssignments),
      opaque_or_untracked_share_pct: pct(groupedTotals.opaque_or_untracked_assignments, totalAssignments),
      internal_share_pct: pct(groupedTotals.internal_assignments, totalAssignments),
    },
    data,
    notes: [
      'Base dataset: datasets.perReport.',
      'Values are vendor-tag assignments in 2025, not unique reports.',
      'A single report may contribute multiple vendor tags.',
    ],
  };
};

const buildFigure5 = dashboardData => {
  const dataset = dashboardData.datasets.perReport;
  const years = dashboardData.years.filter(year => year >= FIGURE_5_START_YEAR && year <= MAX_REPORT_YEAR);

  const data = years.map(year => {
    const blindSpotRow = getYearRow(dataset.blindSpotTrend, year);
    const substantiveCount = dataset.substantivenessHeatmap.find(entry => entry.x === year && entry.y === 'substantive')?.value ?? 0;
    const riskMentionRate = pct(blindSpotRow.ai_risk_mention, blindSpotRow.total_reports);
    const substantiveRate = pct(substantiveCount, blindSpotRow.total_reports);
    const rawRiskMentionRate = pctRaw(blindSpotRow.ai_risk_mention, blindSpotRow.total_reports);
    const rawSubstantiveRate = pctRaw(substantiveCount, blindSpotRow.total_reports);

    return {
      year,
      total_reports: blindSpotRow.total_reports,
      risk_reports: blindSpotRow.ai_risk_mention,
      substantive_risk_reports: substantiveCount,
      risk_mention_rate_pct: riskMentionRate,
      substantive_risk_rate_pct: substantiveRate,
      quality_gap_pp: round1(rawRiskMentionRate - rawSubstantiveRate),
    };
  });

  return {
    id: 'figure5',
    title: 'Risk mention rate vs substantive risk rate',
    chartType: 'dual-axis-line-area',
    domain: {
      years: [years[0], years.at(-1)],
      measure: 'share of all reports in year (%)',
    },
    series: [
      { key: 'risk_mention_rate_pct', label: 'Risk mention rate', color: MENTION_COLORS.risk },
      { key: 'substantive_risk_rate_pct', label: 'Substantive risk rate', color: MENTION_COLORS.vendor },
    ],
    shading: {
      key: 'quality_gap_pp',
      label: 'Quality gap',
      color: '#fecdd3',
    },
    data,
    notes: [
      'Base dataset: datasets.perReport.',
      'Substantive risk rate uses the "substantive" band from substantivenessHeatmap.',
      '2026 partial data is excluded.',
    ],
  };
};

const buildFigure6 = dashboardData => {
  const dataset = dashboardData.byCompanyScope.cniOnly.perReport;
  const sectors = dashboardData.sectors.filter(sector => sector !== 'Other');

  const data = sectors.map(sector => {
    const totalReports = getSectorYearValue(dataset.reportCountBySectorYear, 2025, sector);
    const riskReports = getSectorYearValue(dataset.riskMentionBySectorYear, 2025, sector);
    const noAiRiskReports = getSectorYearValue(dataset.noAiRiskBySectorYear, 2025, sector);

    return {
      sector,
      total_reports: totalReports,
      risk_reports: riskReports,
      no_ai_risk_reports: noAiRiskReports,
      risk_rate_pct: pct(riskReports, totalReports),
      no_ai_risk_rate_pct: pct(noAiRiskReports, totalReports),
    };
  }).sort((a, b) => b.risk_rate_pct - a.risk_rate_pct);

  return {
    id: 'figure6',
    title: 'CNI sector AI-risk disclosure rate vs blind spot',
    chartType: 'grouped-horizontal-bar',
    year: 2025,
    series: [
      { key: 'risk_rate_pct', label: 'AI-risk disclosure rate', color: MENTION_COLORS.risk },
      { key: 'no_ai_risk_rate_pct', label: 'No AI-risk mention rate', color: '#cbd5e1' },
    ],
    data,
    notes: [
      'Base dataset: byCompanyScope.cniOnly.perReport.',
      'Sector "Other" is excluded.',
      'Rates are report counts divided by total reports for the sector-year.',
    ],
  };
};

const buildFigure7 = dashboardData => {
  const dataset = dashboardData.byCompanyScope.cniOnly.perReport;
  const sectors = dashboardData.sectors.filter(sector => sector !== 'Other');
  const years = dashboardData.years.filter(year => year >= 2021 && year <= MAX_REPORT_YEAR);

  const cells = [];
  for (const sector of sectors) {
    for (const year of years) {
      const currentReports = getSectorYearValue(dataset.reportCountBySectorYear, year, sector);
      const previousReports = getSectorYearValue(dataset.reportCountBySectorYear, year - 1, sector);
      const currentRisk = getSectorYearValue(dataset.riskMentionBySectorYear, year, sector);
      const previousRisk = getSectorYearValue(dataset.riskMentionBySectorYear, year - 1, sector);
      const currentRate = pct(currentRisk, currentReports);
      const previousRate = pct(previousRisk, previousReports);
      const rawCurrentRate = pctRaw(currentRisk, currentReports);
      const rawPreviousRate = pctRaw(previousRisk, previousReports);

      cells.push({
        sector,
        year,
        current_risk_rate_pct: currentRate,
        previous_risk_rate_pct: previousRate,
        delta_pp: round1(rawCurrentRate - rawPreviousRate),
      });
    }
  }

  return {
    id: 'figure7',
    title: 'CNI sector AI-risk YoY rate change heatmap',
    chartType: 'heatmap',
    domain: {
      years: [years[0], years.at(-1)],
      measure: 'percentage-point change vs previous year',
    },
    sectors,
    years,
    cells,
    notes: [
      'Base dataset: byCompanyScope.cniOnly.perReport.',
      'Sector "Other" is excluded.',
      'Each cell is the current year risk rate minus the previous year risk rate.',
    ],
  };
};

const buildFigure8 = dashboardData => {
  const data = MARKET_SEGMENT_ORDER.map(segmentKey => {
    const dataset = dashboardData.byMarketSegment[segmentKey].perReport;
    const mentionRow = getYearRow(dataset.mentionTrend, 2025);
    const blindSpotRow = getYearRow(dataset.blindSpotTrend, 2025);

    return {
      segment_key: segmentKey,
      segment_label: MARKET_SEGMENT_LABELS[segmentKey] ?? segmentKey,
      total_reports: blindSpotRow.total_reports,
      ai_mention_reports: blindSpotRow.ai_mention,
      adoption_reports: mentionRow.adoption,
      risk_reports: mentionRow.risk,
      ai_mention_rate_pct: pct(blindSpotRow.ai_mention, blindSpotRow.total_reports),
      adoption_rate_pct: pct(mentionRow.adoption, blindSpotRow.total_reports),
      risk_rate_pct: pct(mentionRow.risk, blindSpotRow.total_reports),
    };
  });

  return {
    id: 'figure8',
    title: '2025 AI disclosure rates by market segment',
    chartType: 'grouped-bar',
    year: 2025,
    series: [
      { key: 'ai_mention_rate_pct', label: 'AI mention rate', color: MENTION_COLORS.general_ambiguous },
      { key: 'adoption_rate_pct', label: 'Adoption rate', color: MENTION_COLORS.adoption },
      { key: 'risk_rate_pct', label: 'Risk rate', color: MENTION_COLORS.risk },
    ],
    data,
    notes: [
      'Base dataset: byMarketSegment[*].perReport.',
      'Rates are report counts divided by total reports in the segment for 2025.',
      'FTSE 100 and FTSE 350 are the dashboard slices named "Main Market (FTSE 100 only)" and "Main Market (FTSE 350 only)".',
    ],
  };
};

const buildReportFigures = dashboardData => {
  ensureDashboardShape(dashboardData);

  return {
    generatedAt: new Date().toISOString(),
    source: {
      dashboardDataPath: DASHBOARD_DATA_PATH,
      maxReportYear: MAX_REPORT_YEAR,
      baseDataset: 'perReport',
      cniSectorScope: 'byCompanyScope.cniOnly.perReport',
    },
    figures: {
      figure1: buildFigure1(dashboardData),
      figure2: buildFigure2(dashboardData),
      figure3: buildFigure3(dashboardData),
      figure4: buildFigure4(dashboardData),
      figure5: buildFigure5(dashboardData),
      figure6: buildFigure6(dashboardData),
      figure7: buildFigure7(dashboardData),
      figure8: buildFigure8(dashboardData),
    },
  };
};

const dashboardData = readJson(DASHBOARD_DATA_PATH);
const reportFigures = buildReportFigures(dashboardData);

fs.writeFileSync(REPORT_FIGURES_PATH, `${JSON.stringify(reportFigures, null, 2)}\n`, 'utf8');

const stats = fs.statSync(REPORT_FIGURES_PATH);
console.log(`Wrote ${REPORT_FIGURES_PATH} (${stats.size} bytes)`);
