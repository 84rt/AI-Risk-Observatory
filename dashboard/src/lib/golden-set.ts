import fs from 'fs';
import path from 'path';

type GoldenAnnotation = {
  annotation_id: string;
  company_name: string;
  report_year: number;
  mention_types: string[];
  adoption_types: string[];
  adoption_confidence: Record<string, number>;
  risk_taxonomy: string[];
  risk_confidence: Record<string, number>;
  risk_substantiveness: number | null;
  vendor_tags: string[];
};

export type GoldenDashboardData = {
  years: number[];
  sectors: string[];
  labels: {
    mentionTypes: string[];
    adoptionTypes: string[];
    riskLabels: string[];
    vendorTags: string[];
    confidenceBands: string[];
    substantivenessBands: string[];
  };
  summary: {
    totalReports: number;
    totalCompanies: number;
    aiSignalReports: number;
    adoptionReports: number;
    riskReports: number;
    vendorReports: number;
  };
  mentionTrend: Record<string, number>[];
  adoptionTrend: Record<string, number>[];
  riskTrend: Record<string, number>[];
  vendorTrend: Record<string, number>[];
  riskBySector: { x: string; y: string; value: number }[];
  confidenceHeatmap: { x: number; y: string; value: number }[];
  substantivenessHeatmap: { x: number; y: string; value: number }[];
};

const GOLDEN_SET_PATH = path.join(
  process.cwd(),
  '..',
  'data',
  'golden_set',
  'human',
  'annotations.jsonl'
);

const COMPANIES_PATH = path.join(
  process.cwd(),
  '..',
  'data',
  'reference',
  'golden_set_companies.csv'
);

const mentionTypes = [
  'adoption',
  'risk',
  'vendor',
  'general_ambiguous',
  'harm',
  'none',
];

const adoptionTypes = ['non_llm', 'llm', 'agentic', 'none'];

const riskLabels = [
  'cybersecurity',
  'operational_technical',
  'regulatory',
  'reputational_ethical',
  'information_integrity',
  'third_party_supply_chain',
  'strategic_market',
  'workforce',
  'environmental',
  'none',
];

const vendorTags = ['openai', 'microsoft', 'google', 'internal', 'other', 'undisclosed'];

const confidenceBands = ['High', 'Medium', 'Low'];
const substantivenessBands = ['High', 'Medium', 'Low'];

const toKey = (value: string) => value.trim().toLowerCase();

const parseCompanySectors = () => {
  const content = fs.readFileSync(COMPANIES_PATH, 'utf8').trim();
  const lines = content.split(/\r?\n/);
  const header = lines.shift();
  if (!header) {
    return { sectorMap: new Map<string, string>(), sectors: [] as string[] };
  }
  const headers = header.split(',');
  const nameIndex = headers.indexOf('company_name');
  const sectorIndex = headers.indexOf('sector');
  const sectorMap = new Map<string, string>();
  const sectors: string[] = [];
  const sectorSet = new Set<string>();

  lines.forEach(line => {
    const cells = line.split(',');
    const name = cells[nameIndex]?.trim();
    const sector = cells[sectorIndex]?.trim();
    if (!name || !sector) return;
    sectorMap.set(toKey(name), sector);
    if (!sectorSet.has(sector)) {
      sectorSet.add(sector);
      sectors.push(sector);
    }
  });

  return { sectorMap, sectors };
};

const parseAnnotations = () => {
  const content = fs.readFileSync(GOLDEN_SET_PATH, 'utf8').trim();
  if (!content) return [] as GoldenAnnotation[];
  return content.split(/\r?\n/).map(line => JSON.parse(line) as GoldenAnnotation);
};

const initYearSeries = (years: number[], keys: string[]) => {
  return years.map(year => {
    const row: Record<string, number> = { year };
    keys.forEach(key => {
      row[key] = 0;
    });
    return row;
  });
};

const addCount = (
  series: Record<string, number>[],
  year: number,
  key: string
) => {
  const row = series.find(entry => entry.year === year);
  if (!row) return;
  row[key] = (row[key] || 0) + 1;
};

const bucketize = (value: number) => {
  if (value >= 0.67) return 'High';
  if (value >= 0.34) return 'Medium';
  return 'Low';
};

const CONFIDENCE_THRESHOLD = 0.2;

type ReportData = {
  company_name: string;
  report_year: number;
  sector: string;
  mentionTypes: Set<string>;
  adoptionTypes: Set<string>;
  riskLabels: Set<string>;
  vendorTags: Set<string>;
  adoptionConfidences: Map<string, number[]>;
  riskConfidences: Map<string, number[]>;
  substantivenessValues: number[];
};

const aggregateToReports = (
  annotations: GoldenAnnotation[],
  sectorMap: Map<string, string>
): ReportData[] => {
  const reportMap = new Map<string, ReportData>();

  annotations.forEach(item => {
    const reportKey = `${item.company_name}|||${item.report_year}`;

    if (!reportMap.has(reportKey)) {
      reportMap.set(reportKey, {
        company_name: item.company_name,
        report_year: item.report_year,
        sector: sectorMap.get(toKey(item.company_name)) || 'Unknown',
        mentionTypes: new Set(),
        adoptionTypes: new Set(),
        riskLabels: new Set(),
        vendorTags: new Set(),
        adoptionConfidences: new Map(),
        riskConfidences: new Map(),
        substantivenessValues: [],
      });
    }

    const report = reportMap.get(reportKey)!;

    // Mention types - no confidence, just presence
    (item.mention_types || []).forEach(type => report.mentionTypes.add(type));

    // Adoption types - check confidence threshold
    (item.adoption_types || []).forEach(type => {
      const confidence = item.adoption_confidence?.[type] ?? 0;
      if (confidence >= CONFIDENCE_THRESHOLD) {
        report.adoptionTypes.add(type);
      }
      // Track all confidences for averaging
      if (!report.adoptionConfidences.has(type)) {
        report.adoptionConfidences.set(type, []);
      }
      if (confidence > 0) {
        report.adoptionConfidences.get(type)!.push(confidence);
      }
    });

    // Risk labels - check confidence threshold
    (item.risk_taxonomy || []).forEach(label => {
      const confidence = item.risk_confidence?.[label] ?? 0;
      if (confidence >= CONFIDENCE_THRESHOLD) {
        report.riskLabels.add(label);
      }
      // Track all confidences for averaging
      if (!report.riskConfidences.has(label)) {
        report.riskConfidences.set(label, []);
      }
      if (confidence > 0) {
        report.riskConfidences.get(label)!.push(confidence);
      }
    });

    // Vendor tags - no confidence, just presence
    (item.vendor_tags || []).forEach(tag => report.vendorTags.add(tag));

    // Substantiveness
    if (item.risk_substantiveness !== null && item.risk_substantiveness !== undefined) {
      report.substantivenessValues.push(item.risk_substantiveness);
    }
  });

  return Array.from(reportMap.values());
};

const averageConfidence = (values: number[]): number => {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
};

export const loadGoldenSetDashboardData = (): GoldenDashboardData => {
  const { sectorMap, sectors } = parseCompanySectors();
  const annotations = parseAnnotations();
  const years = Array.from(
    new Set(annotations.map(item => item.report_year))
  ).sort();

  // Aggregate chunks to report level
  const reports = aggregateToReports(annotations, sectorMap);

  const mentionTrend = initYearSeries(years, mentionTypes);
  const adoptionTrend = initYearSeries(years, adoptionTypes);
  const riskTrend = initYearSeries(years, riskLabels);
  const vendorTrend = initYearSeries(years, vendorTags);

  const riskBySectorCounts = new Map<string, number>();
  const confidenceCounts = new Map<string, number>();
  const substantivenessCounts = new Map<string, number>();

  const companies = new Set<string>();
  let aiSignalReports = 0;
  let adoptionReports = 0;
  let riskReports = 0;
  let vendorReports = 0;

  reports.forEach(report => {
    companies.add(report.company_name);
    const year = report.report_year;

    // Check if report has AI signal (any mention type other than just 'none')
    const hasSignal =
      report.mentionTypes.size > 0 &&
      !(report.mentionTypes.size === 1 && report.mentionTypes.has('none'));
    if (hasSignal) aiSignalReports += 1;

    if (report.mentionTypes.has('adoption')) adoptionReports += 1;
    if (report.mentionTypes.has('risk')) riskReports += 1;
    if (report.mentionTypes.has('vendor')) vendorReports += 1;

    // Count reports per mention type
    report.mentionTypes.forEach(type => addCount(mentionTrend, year, type));

    // Count reports per adoption type (only those meeting threshold)
    report.adoptionTypes.forEach(type => addCount(adoptionTrend, year, type));

    // Count reports per risk label (only those meeting threshold)
    report.riskLabels.forEach(label => addCount(riskTrend, year, label));

    // Count reports per vendor tag
    report.vendorTags.forEach(tag => addCount(vendorTrend, year, tag));

    // Risk by sector (x = risk label, y = sector)
    report.riskLabels.forEach(label => {
      const key = `${label}|||${report.sector}`;
      riskBySectorCounts.set(key, (riskBySectorCounts.get(key) || 0) + 1);
    });

    // Confidence heatmap - use average confidence across all tags in the report
    const allConfidences: number[] = [];
    report.adoptionConfidences.forEach(values => {
      const avg = averageConfidence(values);
      if (avg > 0) allConfidences.push(avg);
    });
    report.riskConfidences.forEach(values => {
      const avg = averageConfidence(values);
      if (avg > 0) allConfidences.push(avg);
    });

    if (allConfidences.length > 0) {
      const avgConfidence = averageConfidence(allConfidences);
      const band = bucketize(avgConfidence);
      const key = `${year}|||${band}`;
      confidenceCounts.set(key, (confidenceCounts.get(key) || 0) + 1);
    }

    // Substantiveness heatmap - use average across chunks in the report
    if (report.substantivenessValues.length > 0) {
      const avgSubstantiveness = averageConfidence(report.substantivenessValues);
      const band = bucketize(avgSubstantiveness);
      const key = `${year}|||${band}`;
      substantivenessCounts.set(key, (substantivenessCounts.get(key) || 0) + 1);
    }
  });

  const riskBySector = Array.from(riskBySectorCounts.entries()).map(([key, value]) => {
    const [label, sector] = key.split('|||');
    return { x: label, y: sector, value };
  });

  const confidenceHeatmap: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    confidenceBands.forEach(band => {
      confidenceHeatmap.push({
        x: year,
        y: band,
        value: confidenceCounts.get(`${year}|||${band}`) || 0,
      });
    });
  });

  const substantivenessHeatmap: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    substantivenessBands.forEach(band => {
      substantivenessHeatmap.push({
        x: year,
        y: band,
        value: substantivenessCounts.get(`${year}|||${band}`) || 0,
      });
    });
  });

  return {
    years,
    sectors: sectors.length ? sectors : ['Unknown'],
    labels: {
      mentionTypes,
      adoptionTypes,
      riskLabels,
      vendorTags,
      confidenceBands,
      substantivenessBands,
    },
    summary: {
      totalReports: reports.length,
      totalCompanies: companies.size,
      aiSignalReports,
      adoptionReports,
      riskReports,
      vendorReports,
    },
    mentionTrend,
    adoptionTrend,
    riskTrend,
    vendorTrend,
    riskBySector,
    confidenceHeatmap,
    substantivenessHeatmap,
  };
};
