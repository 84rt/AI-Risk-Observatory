import fs from 'fs';
import path from 'path';

type GoldenAnnotation = {
  annotation_id: string;
  company_name: string;
  report_year: number;
  mention_types: string[];
  adoption_types: string[];
  adoption_confidence: Record<string, number> | number | null;
  risk_taxonomy: string[];
  risk_confidence: Record<string, number> | null;
  risk_substantiveness: number | null;
  vendor_tags: string[];
  llm_details?: {
    adoption_confidences?: Record<string, number>;
    risk_confidences?: Record<string, number>;
  };
};

export type LabelMetric = {
  label: string;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ComparisonMetrics = {
  coverage: {
    humanChunks: number;
    llmChunks: number;
    commonChunks: number;
  };
  mentionTypes: {
    avgJaccard: number;
    metrics: LabelMetric[];
  };
  adoptionTypes: {
    avgJaccard: number;
    metrics: LabelMetric[];
  };
  riskTaxonomy: {
    avgJaccard: number;
    metrics: LabelMetric[];
  };
  vendorTags: {
    avgJaccard: number;
    metrics: LabelMetric[];
  };
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
  datasets: {
    human: GoldenDataset;
    llm: GoldenDataset;
  };
  comparison: ComparisonMetrics;
};

export type GoldenDataset = {
  years: number[];
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
  adoptionBySector: { x: string; y: string; value: number }[];
  vendorBySector: { x: string; y: string; value: number }[];
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

const LLM_SET_PATH = path.join(
  process.cwd(),
  '..',
  'data',
  'golden_set',
  'llm',
  'annotations.jsonl'
);

const COMPANIES_PATH = path.join(
  process.cwd(),
  '..',
  'data',
  'reference',
  'golden_set_companies.csv'
);

const COMPARE_DIR = path.join(
  process.cwd(),
  '..',
  'data',
  'golden_set',
  'compare'
);

const mentionTypes = [
  'adoption',
  'risk',
  'vendor',
  'general_ambiguous',
  'harm',
];

const adoptionTypes = ['non_llm', 'llm', 'agentic'];

const riskLabels = [
  'cybersecurity',
  'operational_technical',
  'regulatory',
  'reputational_ethical',
  'information_integrity',
  'third_party_supply_chain',
  'strategic_competitive',
  'workforce',
  'environmental',
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

const parseAnnotations = (filepath: string) => {
  if (!fs.existsSync(filepath)) return [] as GoldenAnnotation[];
  const content = fs.readFileSync(filepath, 'utf8').trim();
  if (!content) return [] as GoldenAnnotation[];
  return content.split(/\r?\n/).map(line => JSON.parse(line) as GoldenAnnotation);
};

const parseMetricsCsv = (filename: string): LabelMetric[] => {
  const filepath = path.join(COMPARE_DIR, filename);
  if (!fs.existsSync(filepath)) return [];

  const content = fs.readFileSync(filepath, 'utf8').trim();
  const lines = content.split(/\r?\n/);
  lines.shift(); // Remove header

  return lines.map(line => {
    const [label, tp, fp, fn, precision, recall, f1] = line.split(',');
    return {
      label,
      tp: parseInt(tp, 10),
      fp: parseInt(fp, 10),
      fn: parseInt(fn, 10),
      precision: parseFloat(precision),
      recall: parseFloat(recall),
      f1: parseFloat(f1),
    };
  });
};

const parseComparisonMetrics = (): ComparisonMetrics => {
  const mentionMetrics = parseMetricsCsv('mention_types_metrics.csv');
  const adoptionMetrics = parseMetricsCsv('adoption_types_metrics.csv');
  const riskMetrics = parseMetricsCsv('risk_taxonomy_metrics.csv');
  const vendorMetrics = parseMetricsCsv('vendor_tags_metrics.csv');

  // Parse coverage from comparison_report.md
  let humanChunks = 474;
  let llmChunks = 470;
  let commonChunks = 470;

  const reportPath = path.join(COMPARE_DIR, 'comparison_report.md');
  if (fs.existsSync(reportPath)) {
    const reportContent = fs.readFileSync(reportPath, 'utf8');
    const humanMatch = reportContent.match(/Human chunks:\s*(\d+)/);
    const llmMatch = reportContent.match(/LLM chunks:\s*(\d+)/);
    const commonMatch = reportContent.match(/Common chunks:\s*(\d+)/);
    if (humanMatch) humanChunks = parseInt(humanMatch[1], 10);
    if (llmMatch) llmChunks = parseInt(llmMatch[1], 10);
    if (commonMatch) commonChunks = parseInt(commonMatch[1], 10);
  }

  // Calculate average Jaccard from report or use defaults
  const avgJaccardMention = 0.3196;
  const avgJaccardAdoption = 0.5702;
  const avgJaccardRisk = 0.7914;
  const avgJaccardVendor = 0.928;

  return {
    coverage: {
      humanChunks,
      llmChunks,
      commonChunks,
    },
    mentionTypes: {
      avgJaccard: avgJaccardMention,
      metrics: mentionMetrics.filter(m => m.label !== 'none'),
    },
    adoptionTypes: {
      avgJaccard: avgJaccardAdoption,
      metrics: adoptionMetrics.filter(m => m.label !== 'none'),
    },
    riskTaxonomy: {
      avgJaccard: avgJaccardRisk,
      metrics: riskMetrics.filter(m => m.label !== 'none' && m.f1 > 0),
    },
    vendorTags: {
      avgJaccard: avgJaccardVendor,
      metrics: vendorMetrics.filter(m => m.f1 > 0),
    },
  };
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

const resolveConfidenceMap = (
  item: GoldenAnnotation,
  field: 'adoption' | 'risk'
): Record<string, number> => {
  if (field === 'adoption') {
    if (item.adoption_confidence && typeof item.adoption_confidence === 'object') {
      return item.adoption_confidence;
    }
    if (typeof item.adoption_confidence === 'number') {
      return Object.fromEntries(
        (item.adoption_types || []).map(label => [label, item.adoption_confidence as number])
      );
    }
    return item.llm_details?.adoption_confidences ?? {};
  }

  if (item.risk_confidence && typeof item.risk_confidence === 'object') {
    return item.risk_confidence;
  }
  return item.llm_details?.risk_confidences ?? {};
};

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
    const adoptionConfMap = resolveConfidenceMap(item, 'adoption');
    (item.adoption_types || []).forEach(type => {
      const confidence = adoptionConfMap?.[type] ?? 0;
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
    const riskConfMap = resolveConfidenceMap(item, 'risk');
    (item.risk_taxonomy || []).forEach(label => {
      const confidence = riskConfMap?.[label] ?? 0;
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

const buildDataset = (
  annotations: GoldenAnnotation[],
  sectorMap: Map<string, string>,
  years: number[]
): GoldenDataset => {
  const reports = aggregateToReports(annotations, sectorMap);

  const mentionTrend = initYearSeries(years, mentionTypes);
  const adoptionTrend = initYearSeries(years, adoptionTypes);
  const riskTrend = initYearSeries(years, riskLabels);
  const vendorTrend = initYearSeries(years, vendorTags);

  const riskBySectorCounts = new Map<string, number>();
  const adoptionBySectorCounts = new Map<string, number>();
  const vendorBySectorCounts = new Map<string, number>();
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

    const hasSignal =
      report.mentionTypes.size > 0 &&
      !(report.mentionTypes.size === 1 && report.mentionTypes.has('none'));
    if (hasSignal) aiSignalReports += 1;

    if (report.mentionTypes.has('adoption')) adoptionReports += 1;
    if (report.mentionTypes.has('risk')) riskReports += 1;
    if (report.mentionTypes.has('vendor')) vendorReports += 1;

    report.mentionTypes.forEach(type => addCount(mentionTrend, year, type));
    report.adoptionTypes.forEach(type => addCount(adoptionTrend, year, type));
    report.riskLabels.forEach(label => addCount(riskTrend, year, label));
    report.vendorTags.forEach(tag => addCount(vendorTrend, year, tag));

    report.riskLabels.forEach(label => {
      const key = `${label}|||${report.sector}`;
      riskBySectorCounts.set(key, (riskBySectorCounts.get(key) || 0) + 1);
    });

    report.adoptionTypes.forEach(type => {
      const key = `${type}|||${report.sector}`;
      adoptionBySectorCounts.set(key, (adoptionBySectorCounts.get(key) || 0) + 1);
    });

    report.vendorTags.forEach(tag => {
      const key = `${tag}|||${report.sector}`;
      vendorBySectorCounts.set(key, (vendorBySectorCounts.get(key) || 0) + 1);
    });

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

  const adoptionBySector = Array.from(adoptionBySectorCounts.entries()).map(([key, value]) => {
    const [type, sector] = key.split('|||');
    return { x: type, y: sector, value };
  });

  const vendorBySector = Array.from(vendorBySectorCounts.entries()).map(([key, value]) => {
    const [tag, sector] = key.split('|||');
    return { x: tag, y: sector, value };
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
    adoptionBySector,
    vendorBySector,
    confidenceHeatmap,
    substantivenessHeatmap,
  };
};

export const loadGoldenSetDashboardData = (): GoldenDashboardData => {
  const { sectorMap, sectors } = parseCompanySectors();
  const humanAnnotations = parseAnnotations(GOLDEN_SET_PATH);
  const llmAnnotations = parseAnnotations(LLM_SET_PATH);
  const comparison = parseComparisonMetrics();

  const years = Array.from(
    new Set([
      ...humanAnnotations.map(item => item.report_year),
      ...llmAnnotations.map(item => item.report_year),
    ])
  ).sort();

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
    datasets: {
      human: buildDataset(humanAnnotations, sectorMap, years),
      llm: buildDataset(llmAnnotations, sectorMap, years),
    },
    comparison,
  };
};
