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
    totalChunks: number;
    totalCompanies: number;
    aiSignalChunks: number;
    adoptionChunks: number;
    riskChunks: number;
    vendorChunks: number;
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

export const loadGoldenSetDashboardData = (): GoldenDashboardData => {
  const { sectorMap, sectors } = parseCompanySectors();
  const annotations = parseAnnotations();
  const years = Array.from(
    new Set(annotations.map(item => item.report_year))
  ).sort();

  const mentionTrend = initYearSeries(years, mentionTypes);
  const adoptionTrend = initYearSeries(years, adoptionTypes);
  const riskTrend = initYearSeries(years, riskLabels);
  const vendorTrend = initYearSeries(years, vendorTags);

  const riskBySectorCounts = new Map<string, number>();
  const confidenceCounts = new Map<string, number>();
  const substantivenessCounts = new Map<string, number>();

  const companies = new Set<string>();
  let aiSignalChunks = 0;
  let adoptionChunks = 0;
  let riskChunks = 0;
  let vendorChunks = 0;

  annotations.forEach(item => {
    companies.add(item.company_name);
    const year = item.report_year;

    const mentionSet = new Set(item.mention_types || []);
    if (mentionSet.size > 0 && !(mentionSet.size === 1 && mentionSet.has('none'))) {
      aiSignalChunks += 1;
    }

    if (mentionSet.has('adoption')) adoptionChunks += 1;
    if (mentionSet.has('risk')) riskChunks += 1;
    if (mentionSet.has('vendor')) vendorChunks += 1;

    mentionSet.forEach(type => addCount(mentionTrend, year, type));

    const adoptionSet = new Set(item.adoption_types || []);
    adoptionSet.forEach(type => addCount(adoptionTrend, year, type));

    const riskSet = new Set(item.risk_taxonomy || []);
    riskSet.forEach(type => addCount(riskTrend, year, type));

    const vendorSet = new Set(item.vendor_tags || []);
    vendorSet.forEach(tag => addCount(vendorTrend, year, tag));

    riskSet.forEach(label => {
      const sector = sectorMap.get(toKey(item.company_name)) || 'Unknown';
      const key = `${sector}|||${label}`;
      riskBySectorCounts.set(key, (riskBySectorCounts.get(key) || 0) + 1);
    });

    const confidenceValues = [
      ...Object.values(item.adoption_confidence || {}),
      ...Object.values(item.risk_confidence || {}),
    ];
    if (confidenceValues.length > 0) {
      const band = bucketize(Math.max(...confidenceValues));
      const key = `${year}|||${band}`;
      confidenceCounts.set(key, (confidenceCounts.get(key) || 0) + 1);
    }

    if (item.risk_substantiveness !== null && item.risk_substantiveness !== undefined) {
      const band = bucketize(item.risk_substantiveness);
      const key = `${year}|||${band}`;
      substantivenessCounts.set(key, (substantivenessCounts.get(key) || 0) + 1);
    }
  });

  const riskBySector = Array.from(riskBySectorCounts.entries()).map(([key, value]) => {
    const [sector, label] = key.split('|||');
    return { x: sector, y: label, value };
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
      totalChunks: annotations.length,
      totalCompanies: companies.size,
      aiSignalChunks,
      adoptionChunks,
      riskChunks,
      vendorChunks,
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
