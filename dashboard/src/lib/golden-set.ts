import fs from 'fs';
import path from 'path';
import { resolveIsicSectionFromCode } from './isic.ts';

type GoldenAnnotation = {
  annotation_id: string;
  document_id: string;
  company_name: string;
  report_year: number;
  mention_types: string[];
  adoption_types: string[];
  adoption_confidence: Record<string, number> | number | null;
  risk_taxonomy: string[];
  risk_confidence: Record<string, number> | null;
  risk_substantiveness: string | number | null;
  risk_signals?: Array<{ type?: string; signal?: number | string }> | Record<string, number> | null;
  vendor_tags: string[];
  vendor_confidence?: Record<string, number> | null;
  chunk_id?: string;
  chunk_text?: string;
  report_sections?: string[];
};

type ReportUniverseRow = {
  report_id: string;
  company_name: string;
  report_year: number;
  release_month: string;
  sector: string;
  isicSector: string;
  marketSegment: string;
};

export type ReportClassificationFlowNode = {
  id: string;
  name: string;
  stage: 'root' | 'gate' | 'phase1' | 'phase2';
  fill: string;
  reportCount: number;
};

export type ReportClassificationFlow = {
  totalReports: number;
  extractedAiReports: number;
  noExtractedAiReports: number;
  phase1SignalReports: number;
  phase1NoneOnlyReports: number;
  nodes: ReportClassificationFlowNode[];
  links: {
    source: number;
    target: number;
    value: number;
  }[];
};

export type ReportClassificationBreakdownLeaf = {
  id: string;
  label: string;
  count: number;
  pctOfParent: number;
};

export type ReportClassificationBreakdownBranch = {
  id: string;
  label: string;
  count: number;
  pctOfParent: number;
  children?: ReportClassificationBreakdownLeaf[];
};

export type ReportClassificationBreakdown = {
  totalReports: number;
  noAiChunkExtracted: number;
  aiChunkExtracted: number;
  phase1NoneOnly: number;
  phase1SignalReports: number;
  phase1TotalAssignments: number;
  averageLabelsPerSignalReport: number;
  branches: ReportClassificationBreakdownBranch[];
};

export type GoldenDashboardData = {
  years: number[];
  sectors: string[];
  isicSectors: string[];
  isicSectorParents: Record<string, string>;
  marketSegments: string[];
  reportClassificationFlow: ReportClassificationFlow;
  reportClassificationBreakdown: ReportClassificationBreakdown;
  exampleChunks: ExampleChunk[];
  labels: {
    mentionTypes: string[];
    adoptionTypes: string[];
    riskLabels: string[];
    vendorTags: string[];
    substantivenessBands: string[];
    riskSignalLevels: string[];
  };
  datasets: {
    perReport: GoldenDataset;
    perChunk: GoldenDataset;
  };
  byMarketSegment: Record<string, { perReport: GoldenDataset; perChunk: GoldenDataset }>;
};

export type GoldenDataset = {
  years: number[];
  months: string[];
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
  riskTrendMonthly: Record<string, string | number>[];
  adoptionTrendMonthly: Record<string, string | number>[];
  vendorTrendMonthly: Record<string, string | number>[];
  blindSpotTrendMonthly: Record<string, string | number>[];
  riskBySector: { x: string; y: string; value: number }[];
  riskBySectorYear: { year: number; x: string; y: string; value: number }[];
  riskByIsicSector: { x: string; y: string; value: number }[];
  riskByIsicSectorYear: { year: number; x: string; y: string; value: number }[];
  adoptionBySector: { x: string; y: string; value: number }[];
  adoptionBySectorYear: { year: number; x: string; y: string; value: number }[];
  adoptionByIsicSector: { x: string; y: string; value: number }[];
  adoptionByIsicSectorYear: { year: number; x: string; y: string; value: number }[];
  vendorBySector: { x: string; y: string; value: number }[];
  vendorBySectorYear: { year: number; x: string; y: string; value: number }[];
  vendorByIsicSector: { x: string; y: string; value: number }[];
  vendorByIsicSectorYear: { year: number; x: string; y: string; value: number }[];
  riskSignalHeatmap: { x: number; y: string; value: number }[];
  adoptionSignalHeatmap: { x: number; y: string; value: number }[];
  vendorSignalHeatmap: { x: number; y: string; value: number }[];
  substantivenessHeatmap: { x: number; y: string; value: number }[];
  blindSpotTrend: Record<string, number>[];
  noAiBySectorYear: { x: number; y: string; value: number }[];
  noAiRiskBySectorYear: { x: number; y: string; value: number }[];
  reportCountBySectorYear: { x: number; y: string; value: number }[];
  reportCountByIsicSectorYear: { x: number; y: string; value: number }[];
};

export type ExampleChunk = {
  chunkId: string;
  companyName: string;
  reportYear: number;
  chunkText: string;
  reportSections: string[];
  mentionTypes: string[];
  riskLabels: string[];
  adoptionTypes: string[];
  vendorTags: string[];
};

export const ANNOTATIONS_PATH = path.join(
  process.cwd(),
  'data',
  'annotations.jsonl'
);

export const PRECOMPUTED_DASHBOARD_DATA_PATH = path.join(
  process.cwd(),
  'data',
  'dashboard-data.json'
);

const COMPANIES_PATH = path.join(
  process.cwd(),
  'data',
  'golden_set_companies.csv'
);

const DOCUMENT_MONTHS_PATH = path.join(
  process.cwd(),
  'data',
  'document_months.json'
);

let cachedDashboardData: GoldenDashboardData | null = null;

const shouldPreferPrecomputedDashboardData = () => {
  if (process.env.DASHBOARD_FORCE_RAW === '1') return false;
  if (!fs.existsSync(PRECOMPUTED_DASHBOARD_DATA_PATH)) return false;
  if (!fs.existsSync(ANNOTATIONS_PATH)) return true;
  return process.env.NODE_ENV === 'production';
};

const loadPrecomputedDashboardData = (): GoldenDashboardData | null => {
  if (!fs.existsSync(PRECOMPUTED_DASHBOARD_DATA_PATH)) return null;
  const content = fs.readFileSync(PRECOMPUTED_DASHBOARD_DATA_PATH, 'utf8').trim();
  if (!content) return null;
  return JSON.parse(content) as GoldenDashboardData;
};

const loadDocumentMonths = (): Map<string, string> => {
  if (!fs.existsSync(DOCUMENT_MONTHS_PATH)) return new Map();
  const raw = JSON.parse(fs.readFileSync(DOCUMENT_MONTHS_PATH, 'utf8')) as Record<string, string>;
  return new Map(Object.entries(raw));
};

const mentionTypes = [
  'adoption',
  'risk',
  'vendor',
  'general_ambiguous',
  'harm',
];

const adoptionTypes = ['non_llm', 'llm', 'agentic'];

const riskLabels = [
  'strategic_competitive',
  'cybersecurity',
  'operational_technical',
  'regulatory_compliance',
  'reputational_ethical',
  'third_party_supply_chain',
  'information_integrity',
  'workforce_impacts',
  'environmental_impact',
  'national_security',
];

const vendorTags = [
  'openai',
  'microsoft',
  'google',
  'amazon',
  'meta',
  'anthropic',
  'internal',
  'other',
  'undisclosed',
];

const substantivenessBands = ['substantive', 'moderate', 'boilerplate'];
const riskSignalLevels = ['3-explicit', '2-strong_implicit', '1-weak_implicit'];

const RISK_LABEL_ALIASES: Record<string, string> = {
  strategic_market: 'strategic_competitive',
  regulatory: 'regulatory_compliance',
  workforce: 'workforce_impacts',
  environmental: 'environmental_impact',
};

const toKey = (value: string) => value.trim().toLowerCase();
const parseYearFromDate = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const match = trimmed.match(/^(\d{4})/);
  if (!match) return null;
  const year = Number(match[1]);
  return Number.isFinite(year) ? year : null;
};

const parseMonthFromDate = (value: string) => {
  const trimmed = value.trim();
  return /^\d{4}-\d{2}/.test(trimmed) ? trimmed.slice(0, 7) : '';
};

const parseCsvRow = (line: string): string[] => {
  const cells: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      cells.push(current.trim());
      current = '';
      continue;
    }

    current += char;
  }

  cells.push(current.trim());
  return cells;
};

const parseCompanySectors = (): {
  cniSectorMap: Map<string, string>;
  isicSectorMap: Map<string, string>;
  isicSectorParentMap: Map<string, string>;
  marketSegmentMap: Map<string, string>;
  cniSectors: string[];
  isicSectors: string[];
  companySectors: { company_name: string; cniSector: string; isicSector: string; marketSegment: string }[];
  reportUniverseRows: ReportUniverseRow[];
} => {
  if (!fs.existsSync(COMPANIES_PATH)) {
    return {
      cniSectorMap: new Map<string, string>(),
      isicSectorMap: new Map<string, string>(),
      isicSectorParentMap: new Map<string, string>(),
      marketSegmentMap: new Map<string, string>(),
      cniSectors: [] as string[],
      isicSectors: [] as string[],
      companySectors: [] as { company_name: string; cniSector: string; isicSector: string; marketSegment: string }[],
      reportUniverseRows: [] as ReportUniverseRow[],
    };
  }

  const content = fs.readFileSync(COMPANIES_PATH, 'utf8').trim();
  const lines = content.split(/\r?\n/);
  const header = lines.shift();
  if (!header) {
    return {
      cniSectorMap: new Map<string, string>(),
      isicSectorMap: new Map<string, string>(),
      isicSectorParentMap: new Map<string, string>(),
      marketSegmentMap: new Map<string, string>(),
      cniSectors: [] as string[],
      isicSectors: [] as string[],
      companySectors: [] as { company_name: string; cniSector: string; isicSector: string; marketSegment: string }[],
      reportUniverseRows: [] as ReportUniverseRow[],
    };
  }
  const headers = parseCsvRow(header);
  const nameIndex = headers.indexOf('company_name');
  const filingIdIndex = headers.indexOf('filing_id');
  const filingDateIndex = headers.indexOf('filing_date');
  const releaseDatetimeIndex = headers.indexOf('release_datetime');
  const isicCodeIndex = headers.indexOf('isic_code');
  const cniSectorIndex = headers.indexOf('sector') >= 0
    ? headers.indexOf('sector')
    : headers.indexOf('cni_sector');
  const isicSectorIndex = headers.indexOf('isic_section_name') >= 0
    ? headers.indexOf('isic_section_name')
    : headers.indexOf('isic_sector_name') >= 0
      ? headers.indexOf('isic_sector_name')
      : headers.indexOf('isic_sector');
  const marketSegmentIndex = headers.indexOf('market_segment');

  const cniSectorMap = new Map<string, string>();
  const isicSectorMap = new Map<string, string>();
  const isicSectorParentMap = new Map<string, string>();
  const marketSegmentMap = new Map<string, string>();
  const cniSectors: string[] = [];
  const isicSectors: string[] = [];
  const companySectors: { company_name: string; cniSector: string; isicSector: string; marketSegment: string }[] = [];
  const reportUniverseRows: ReportUniverseRow[] = [];
  const cniSectorSet = new Set<string>();
  const isicSectorSet = new Set<string>();

  if (nameIndex < 0 || cniSectorIndex < 0) {
    return {
      cniSectorMap,
      isicSectorMap,
      isicSectorParentMap,
      marketSegmentMap,
      cniSectors,
      isicSectors,
      companySectors,
      reportUniverseRows,
    };
  }

  lines.forEach(line => {
    const cells = parseCsvRow(line);
    const name = cells[nameIndex]?.trim();
    const filingId = filingIdIndex >= 0 ? cells[filingIdIndex]?.trim() || '' : '';
    const filingDate = filingDateIndex >= 0 ? cells[filingDateIndex]?.trim() || '' : '';
    const releaseDatetime = releaseDatetimeIndex >= 0 ? cells[releaseDatetimeIndex]?.trim() || '' : '';
    const isicCode = isicCodeIndex >= 0 ? cells[isicCodeIndex]?.trim() || '' : '';
    const cniSector = cells[cniSectorIndex]?.trim() || 'Unknown';
    const isicSector = isicSectorIndex >= 0
      ? (cells[isicSectorIndex]?.trim() || 'Unknown')
      : 'Unknown';
    const isicParentSector = resolveIsicSectionFromCode(isicCode) || 'Unknown';
    const marketSegment = marketSegmentIndex >= 0
      ? (cells[marketSegmentIndex]?.trim() || 'Other')
      : 'Other';
    if (!name) return;
    cniSectorMap.set(toKey(name), cniSector);
    isicSectorMap.set(toKey(name), isicSector);
    if (isicSector && isicSector !== 'Unknown') {
      isicSectorParentMap.set(isicSector, isicParentSector);
    }
    marketSegmentMap.set(toKey(name), marketSegment);
    companySectors.push({ company_name: name, cniSector, isicSector, marketSegment });
    const reportDate = filingDate || releaseDatetime;
    const reportYear = parseYearFromDate(reportDate);
    if (filingId && reportYear !== null && reportYear >= 2020) {
      reportUniverseRows.push({
        report_id: filingId,
        company_name: name,
        report_year: reportYear,
        release_month: parseMonthFromDate(reportDate),
        sector: cniSector,
        isicSector,
        marketSegment,
      });
    }
    if (!cniSectorSet.has(cniSector)) {
      cniSectorSet.add(cniSector);
      cniSectors.push(cniSector);
    }
    if (!isicSectorSet.has(isicSector)) {
      isicSectorSet.add(isicSector);
      isicSectors.push(isicSector);
    }
  });

  return {
    cniSectorMap,
    isicSectorMap,
    isicSectorParentMap,
    marketSegmentMap,
    cniSectors,
    isicSectors,
    companySectors,
    reportUniverseRows,
  };
};

const parseAnnotations = (filepath: string) => {
  if (!fs.existsSync(filepath)) return [] as GoldenAnnotation[];
  const content = fs.readFileSync(filepath, 'utf8').trim();
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
    return {};
  }

  if (item.risk_confidence && typeof item.risk_confidence === 'object') {
    return item.risk_confidence;
  }
  return {};
};

const normalizeRiskLabel = (label: string): string => {
  const cleaned = (label || '').trim();
  if (!cleaned) return cleaned;
  return RISK_LABEL_ALIASES[cleaned] || cleaned;
};

const normalizeRiskSignalValue = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  if (value < 1) {
    if (value >= 0.67) return 3;
    if (value >= 0.34) return 2;
    if (value > 0) return 1;
    return 0;
  }
  return Math.max(1, Math.min(3, Math.round(value)));
};

const normalizeRiskSubstantiveness = (value: unknown): string | null => {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (value >= 0.67) return 'substantive';
    if (value >= 0.34) return 'moderate';
    return 'boilerplate';
  }
  if (typeof value !== 'string') return null;
  const key = value.trim().toLowerCase();
  if (key === 'substantive' || key === 'moderate' || key === 'boilerplate') return key;
  return null;
};

const formatFlowLabel = (value: string) => {
  const overrides: Record<string, string> = {
    // Stage 1 & 2
    processed_reports: 'Total Reports Examined',
    ai_chunk_extracted: 'Reports with AI mentions',
    no_ai_chunk_extracted: 'Reports with no AI mentions',
    
    // Phase 1 Categories
    adoption: 'Reports with AI Adoption mentions',
    risk: 'Reports with Risk from AI mentions',
    vendor: 'Reports with AI Vendor mentions',
    general_ambiguous: 'Reports with General or ambiguous AI mentions',
    harm: 'Mentions of AI Harm',

    // Phase 2 Tags - Adoption
    non_llm: 'Traditional AI / ML',
    llm: 'LLMs / Generative AI',
    agentic: 'Agentic / Autonomous AI',

    // Phase 2 Tags - Risk (Common Aliases)
    strategic_competitive: 'Strategic / Competitive Risk',
    operational_technical: 'Operational / Technical Risk',
    regulatory_compliance: 'Regulatory / Compliance Risk',
    reputational_ethical: 'Reputational / Ethical Risk',
    third_party_supply_chain: 'Supply Chain / Vendor Risk',
    information_integrity: 'Information Integrity Risk',
    workforce_impacts: 'Workforce / Jobs Impact',
    environmental_impact: 'Environmental Impact',
    national_security: 'National Security Risk',
    cybersecurity: 'Cybersecurity Risk',

    // Phase 2 Tags - Vendor
    openai: 'OpenAI Mentions',
    microsoft: 'Microsoft Mentions',
    google: 'Google Mentions',
    amazon: 'Amazon / AWS Mentions',
    meta: 'Meta Mentions',
    anthropic: 'Anthropic Mentions',
    internal: 'In-house AI Development',
    other: 'Other AI Vendors',
    undisclosed: 'Undisclosed AI Providers',
  };

  if (overrides[value]) return overrides[value];

  return value
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const extractRiskSignalMap = (item: GoldenAnnotation): Record<string, number> => {
  const map: Record<string, number> = {};
  const addEntry = (type: unknown, signal: unknown) => {
    if (typeof type !== 'string') return;
    const label = normalizeRiskLabel(type);
    if (!label) return;
    const n = Number(signal);
    if (!Number.isFinite(n)) return;
    const normalized = normalizeRiskSignalValue(n);
    if (normalized <= 0) return;
    map[label] = Math.max(map[label] ?? 0, normalized);
  };

  const candidates = [
    item.risk_signals,
    item.risk_confidence,
  ];

  candidates.forEach(candidate => {
    if (!candidate) return;
    if (Array.isArray(candidate)) {
      candidate.forEach(entry => addEntry(entry?.type, entry?.signal));
      return;
    }
    if (typeof candidate === 'object') {
      Object.entries(candidate).forEach(([k, v]) => addEntry(k, v));
    }
  });

  return map;
};

type ReportData = {
  report_id: string;
  company_name: string;
  report_year: number;
  release_month: string;
  sector: string;
  isicSector: string;
  marketSegment: string;
  mentionTypes: Set<string>;
  adoptionTypes: Set<string>;
  riskLabels: Set<string>;
  vendorTags: Set<string>;
  adoptionConfidences: Map<string, number[]>;
  riskConfidences: Map<string, number[]>;
  adoptionSignalValues: number[];
  riskSignalValues: number[];
  vendorSignalValues: number[];
  riskSubstantivenessValues: string[];
};

const makeEmptyReportData = (
  reportId: string,
  companyName: string,
  reportYear: number,
  sector: string,
  isicSector: string,
  releaseMonth: string = '',
  marketSegment: string = 'Other'
): ReportData => ({
  report_id: reportId,
  company_name: companyName,
  report_year: reportYear,
  release_month: releaseMonth,
  sector,
  isicSector,
  marketSegment,
  mentionTypes: new Set(),
  adoptionTypes: new Set(),
  riskLabels: new Set(),
  vendorTags: new Set(),
  adoptionConfidences: new Map(),
  riskConfidences: new Map(),
  adoptionSignalValues: [],
  riskSignalValues: [],
  vendorSignalValues: [],
  riskSubstantivenessValues: [],
});

const aggregateToReports = (
  annotations: GoldenAnnotation[],
  cniSectorMap: Map<string, string>,
  isicSectorMap: Map<string, string>,
  marketSegmentMap: Map<string, string>,
  reportUniverseRows: ReportUniverseRow[],
  documentMonths: Map<string, string>
): ReportData[] => {
  const reportMap = new Map<string, ReportData>();

  reportUniverseRows.forEach(report => {
    if (reportMap.has(report.report_id)) return;
    reportMap.set(
      report.report_id,
      makeEmptyReportData(
        report.report_id,
        report.company_name,
        report.report_year,
        report.sector,
        report.isicSector,
        report.release_month,
        report.marketSegment
      )
    );
  });

  annotations.forEach(item => {
    if (item.report_year < 2020) return;

    const reportKey = String(item.document_id).trim() || `${item.company_name}|||${item.report_year}`;

    if (!reportMap.has(reportKey)) {
      reportMap.set(
        reportKey,
        makeEmptyReportData(
          reportKey,
          item.company_name,
          item.report_year,
          cniSectorMap.get(toKey(item.company_name)) || 'Unknown',
          isicSectorMap.get(toKey(item.company_name)) || 'Unknown',
          documentMonths.get(item.document_id) || '',
          marketSegmentMap.get(toKey(item.company_name)) || 'Other'
        )
      );
    }

    const report = reportMap.get(reportKey)!;

    // Mention types - no confidence, just presence
    (item.mention_types || []).forEach(type => report.mentionTypes.add(type));

    // Adoption types - check confidence threshold, collect signal values
    const adoptionConfMap = resolveConfidenceMap(item, 'adoption');
    const adoptionSignalMap: Record<string, number> = {};
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
        const normalized = normalizeRiskSignalValue(confidence);
        if (normalized > 0) {
          adoptionSignalMap[type] = Math.max(adoptionSignalMap[type] ?? 0, normalized);
        }
      }
    });
    report.adoptionSignalValues.push(...Object.values(adoptionSignalMap));

    // Risk labels - check confidence threshold
    const riskConfMap = resolveConfidenceMap(item, 'risk');
    const riskSignalMap = extractRiskSignalMap(item);
    (item.risk_taxonomy || []).forEach(rawLabel => {
      const label = normalizeRiskLabel(rawLabel);
      if (!label || label === 'none') return;
      const confidence = riskConfMap?.[rawLabel] ?? riskConfMap?.[label] ?? 0;
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
    Object.entries(riskSignalMap).forEach(([label, signal]) => {
      if (label !== 'none') report.riskLabels.add(label);
      if (signal > 0) report.riskSignalValues.push(signal);
    });

    // Vendor tags + signals
    (item.vendor_tags || []).forEach(tag => report.vendorTags.add(tag));
    if (item.vendor_confidence && typeof item.vendor_confidence === 'object') {
      const vendorSignalMap: Record<string, number> = {};
      Object.entries(item.vendor_confidence).forEach(([tag, v]) => {
        const n = Number(v);
        if (Number.isFinite(n) && n > 0) {
          const normalized = normalizeRiskSignalValue(n);
          if (normalized > 0) {
            vendorSignalMap[tag] = Math.max(vendorSignalMap[tag] ?? 0, normalized);
          }
        }
      });
      report.vendorSignalValues.push(...Object.values(vendorSignalMap));
    }

    // Substantiveness
    const riskSubstantiveness = normalizeRiskSubstantiveness(item.risk_substantiveness);
    if (riskSubstantiveness) report.riskSubstantivenessValues.push(riskSubstantiveness);
  });

  return Array.from(reportMap.values());
};

const toPct = (count: number, total: number) => (total > 0 ? (count / total) * 100 : 0);

const buildReportClassificationBreakdown = (
  reports: ReportData[]
): ReportClassificationBreakdown => {
  const totalReports = reports.length;
  const extractedChunkReports = reports.filter(report => report.mentionTypes.size > 0);
  const aiChunkExtracted = extractedChunkReports.length;
  const noAiChunkExtracted = totalReports - aiChunkExtracted;
  const phase1NoneOnly = extractedChunkReports.filter(
    report => report.mentionTypes.size === 1 && report.mentionTypes.has('none')
  ).length;
  const signalReports = extractedChunkReports.filter(
    report => !(report.mentionTypes.size === 1 && report.mentionTypes.has('none'))
  );
  const phase1SignalReports = signalReports.length;

  const countByMentionType = (label: string) =>
    signalReports.filter(report => report.mentionTypes.has(label)).length;
  const countBySetValue = (reportsInBranch: ReportData[], accessor: (report: ReportData) => Set<string>, value: string) =>
    reportsInBranch.filter(report => accessor(report).has(value)).length;

  const adoptionCount = countByMentionType('adoption');
  const generalAmbiguousCount = countByMentionType('general_ambiguous');
  const riskCount = countByMentionType('risk');
  const vendorCount = countByMentionType('vendor');
  const harmCount = countByMentionType('harm');

  const adoptionReports = signalReports.filter(report => report.mentionTypes.has('adoption'));
  const riskReports = signalReports.filter(report => report.mentionTypes.has('risk'));
  const vendorReports = signalReports.filter(report => report.mentionTypes.has('vendor'));

  const branches: ReportClassificationBreakdownBranch[] = [
    {
      id: 'adoption',
      label: 'Adoption',
      count: adoptionCount,
      pctOfParent: toPct(adoptionCount, phase1SignalReports),
      children: [
        { id: 'non_llm', label: 'non_llm', count: countBySetValue(adoptionReports, report => report.adoptionTypes, 'non_llm'), pctOfParent: toPct(countBySetValue(adoptionReports, report => report.adoptionTypes, 'non_llm'), adoptionCount) },
        { id: 'llm', label: 'llm', count: countBySetValue(adoptionReports, report => report.adoptionTypes, 'llm'), pctOfParent: toPct(countBySetValue(adoptionReports, report => report.adoptionTypes, 'llm'), adoptionCount) },
        { id: 'agentic', label: 'agentic', count: countBySetValue(adoptionReports, report => report.adoptionTypes, 'agentic'), pctOfParent: toPct(countBySetValue(adoptionReports, report => report.adoptionTypes, 'agentic'), adoptionCount) },
      ],
    },
    {
      id: 'general_ambiguous',
      label: 'General / ambiguous',
      count: generalAmbiguousCount,
      pctOfParent: toPct(generalAmbiguousCount, phase1SignalReports),
    },
    {
      id: 'risk',
      label: 'Risk',
      count: riskCount,
      pctOfParent: toPct(riskCount, phase1SignalReports),
      children: [
        { id: 'strategic_competitive', label: 'strategic/competitive', count: countBySetValue(riskReports, report => report.riskLabels, 'strategic_competitive'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'strategic_competitive'), riskCount) },
        { id: 'cybersecurity', label: 'cybersecurity', count: countBySetValue(riskReports, report => report.riskLabels, 'cybersecurity'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'cybersecurity'), riskCount) },
        { id: 'operational_technical', label: 'operational/technical', count: countBySetValue(riskReports, report => report.riskLabels, 'operational_technical'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'operational_technical'), riskCount) },
        { id: 'regulatory_compliance', label: 'regulatory/compliance', count: countBySetValue(riskReports, report => report.riskLabels, 'regulatory_compliance'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'regulatory_compliance'), riskCount) },
        { id: 'reputational_ethical', label: 'reputational/ethical', count: countBySetValue(riskReports, report => report.riskLabels, 'reputational_ethical'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'reputational_ethical'), riskCount) },
        { id: 'third_party_supply_chain', label: 'third party/supply chain', count: countBySetValue(riskReports, report => report.riskLabels, 'third_party_supply_chain'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'third_party_supply_chain'), riskCount) },
        { id: 'information_integrity', label: 'information integrity', count: countBySetValue(riskReports, report => report.riskLabels, 'information_integrity'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'information_integrity'), riskCount) },
        { id: 'workforce_impacts', label: 'workforce impacts', count: countBySetValue(riskReports, report => report.riskLabels, 'workforce_impacts'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'workforce_impacts'), riskCount) },
        { id: 'environmental_impact', label: 'environmental impact', count: countBySetValue(riskReports, report => report.riskLabels, 'environmental_impact'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'environmental_impact'), riskCount) },
        { id: 'national_security', label: 'national security', count: countBySetValue(riskReports, report => report.riskLabels, 'national_security'), pctOfParent: toPct(countBySetValue(riskReports, report => report.riskLabels, 'national_security'), riskCount) },
      ],
    },
    {
      id: 'vendor',
      label: 'Vendor',
      count: vendorCount,
      pctOfParent: toPct(vendorCount, phase1SignalReports),
      children: [
        { id: 'other', label: 'other', count: countBySetValue(vendorReports, report => report.vendorTags, 'other'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'other'), vendorCount) },
        { id: 'microsoft', label: 'microsoft', count: countBySetValue(vendorReports, report => report.vendorTags, 'microsoft'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'microsoft'), vendorCount) },
        { id: 'internal', label: 'internal', count: countBySetValue(vendorReports, report => report.vendorTags, 'internal'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'internal'), vendorCount) },
        { id: 'openai', label: 'openai', count: countBySetValue(vendorReports, report => report.vendorTags, 'openai'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'openai'), vendorCount) },
        { id: 'undisclosed', label: 'undisclosed', count: countBySetValue(vendorReports, report => report.vendorTags, 'undisclosed'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'undisclosed'), vendorCount) },
        { id: 'google', label: 'google', count: countBySetValue(vendorReports, report => report.vendorTags, 'google'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'google'), vendorCount) },
        { id: 'amazon', label: 'amazon', count: countBySetValue(vendorReports, report => report.vendorTags, 'amazon'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'amazon'), vendorCount) },
        { id: 'meta', label: 'meta', count: countBySetValue(vendorReports, report => report.vendorTags, 'meta'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'meta'), vendorCount) },
        { id: 'anthropic', label: 'anthropic', count: countBySetValue(vendorReports, report => report.vendorTags, 'anthropic'), pctOfParent: toPct(countBySetValue(vendorReports, report => report.vendorTags, 'anthropic'), vendorCount) },
      ],
    },
    {
      id: 'harm',
      label: 'Harm',
      count: harmCount,
      pctOfParent: toPct(harmCount, phase1SignalReports),
    },
  ];

  const phase1TotalAssignments = branches.reduce((sum, branch) => sum + branch.count, 0);

  return {
    totalReports,
    noAiChunkExtracted,
    aiChunkExtracted,
    phase1NoneOnly,
    phase1SignalReports,
    phase1TotalAssignments,
    averageLabelsPerSignalReport:
      phase1SignalReports > 0 ? phase1TotalAssignments / phase1SignalReports : 0,
    branches,
  };
};

const buildReportClassificationFlow = (reports: ReportData[]): ReportClassificationFlow => {
  const nodeDataMap = new Map<string, { name: string; stage: ReportClassificationFlowNode['stage']; fill: string; reportCount: number }>();
  const linkDataMap = new Map<string, number>();

  const ROOT_FILL = '#334155';
  const GATE_FILL = '#f59e0b';
  const NO_AI_FILL = '#94a3b8';

  const CATEGORY_FILLS: Record<string, string> = {
    adoption: '#0ea5e9',
    risk: '#f97316',
    vendor: '#14b8a6',
    general_ambiguous: '#64748b',
    harm: '#ef4444',
    none: '#cbd5e1',
  };

  const addRawLink = (sourceId: string, targetId: string, value = 1) => {
    const key = `${sourceId}|||${targetId}`;
    linkDataMap.set(key, (linkDataMap.get(key) || 0) + value);
  };

  const trackNode = (id: string, name: string, stage: ReportClassificationFlowNode['stage'], fill: string, reportCount = 0) => {
    if (!nodeDataMap.has(id)) {
      nodeDataMap.set(id, { name, stage, fill, reportCount });
    } else if (reportCount > 0) {
      nodeDataMap.get(id)!.reportCount += reportCount;
    }
  };

  // 1. Process Reports
  let extractedCount = 0;
  let noExtractedCount = 0;
  let signalCount = 0;
  let noneOnlyCount = 0;

  const rootId = 'root:processed_reports';
  const gateExtractedId = 'gate:ai_chunk_extracted';
  const gateNoExtractedId = 'gate:no_ai_chunk_extracted';

  reports.forEach(report => {
    const hasExtraction = report.mentionTypes.size > 0;
    if (!hasExtraction) {
      noExtractedCount += 1;
      addRawLink(rootId, gateNoExtractedId, 1);
      return;
    }

    extractedCount += 1;
    addRawLink(rootId, gateExtractedId, 1);
const phase1Labels = Array.from(report.mentionTypes).filter(l => l !== 'none');
const hasSignal = phase1Labels.length > 0;

if (hasSignal) signalCount += 1;
else noneOnlyCount += 1;

// Overlapping Phase 1 flows (only if they have a non-none signal)
phase1Labels.forEach(label => {
  const p1Id = `phase1:${label}`;
  trackNode(p1Id, formatFlowLabel(label), 'phase1', CATEGORY_FILLS[label] || '#cbd5e1', 1);
  addRawLink(gateExtractedId, p1Id, 1);


      if (label === 'adoption') {
        report.adoptionTypes.forEach(t => {
          const p2Id = `phase2:adoption:${t}`;
          trackNode(p2Id, formatFlowLabel(t), 'phase2', CATEGORY_FILLS.adoption, 1);
          addRawLink(p1Id, p2Id, 1);
        });
      } else if (label === 'risk') {
        report.riskLabels.forEach(t => {
          const p2Id = `phase2:risk:${t}`;
          trackNode(p2Id, formatFlowLabel(t), 'phase2', CATEGORY_FILLS.risk, 1);
          addRawLink(p1Id, p2Id, 1);
        });
      } else if (label === 'vendor') {
        report.vendorTags.forEach(t => {
          const p2Id = `phase2:vendor:${t}`;
          trackNode(p2Id, formatFlowLabel(t), 'phase2', CATEGORY_FILLS.vendor, 1);
          addRawLink(p1Id, p2Id, 1);
        });
      }
    });
  });

  // Ensure primary nodes exist
  trackNode(rootId, formatFlowLabel('processed_reports'), 'root', ROOT_FILL, reports.length);
  trackNode(gateExtractedId, formatFlowLabel('ai_chunk_extracted'), 'gate', GATE_FILL, extractedCount);
  trackNode(gateNoExtractedId, formatFlowLabel('no_ai_chunk_extracted'), 'gate', NO_AI_FILL, noExtractedCount);

  // 2. Define Vertical Order
  // Sort Phase 1 categories: Primary (Adoption, Risk, Vendor) by size, 
  // then special cases (None, Harm), and finally General/Ambiguous at the very bottom.
  const phase1Order = Array.from(nodeDataMap.keys())
    .filter(id => id.startsWith('phase1:'))
    .sort((a, b) => {
      const aKey = a.split(':').pop() || '';
      const bKey = b.split(':').pop() || '';
      
      // Bottom priority items
      const bottomPriority = ['harm', 'general_ambiguous'];
      const aIdx = bottomPriority.indexOf(aKey);
      const bIdx = bottomPriority.indexOf(bKey);

      if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
      if (aIdx !== -1) return 1;
      if (bIdx !== -1) return -1;

      // Default: size descending
      return (nodeDataMap.get(b)?.reportCount || 0) - (nodeDataMap.get(a)?.reportCount || 0);
    });
  
  // Group Phase 2 tags by parent category and sort within group by size
  const getPhase2NodesForCategory = (cat: string) => 
    Array.from(nodeDataMap.keys())
      .filter(id => id.startsWith(`phase2:${cat}:`))
      .sort((a, b) => (nodeDataMap.get(b)?.reportCount || 0) - (nodeDataMap.get(a)?.reportCount || 0));

  const sortedNodeIds: string[] = [
    rootId,
    gateExtractedId,
    gateNoExtractedId,
    ...phase1Order,
    ...getPhase2NodesForCategory('adoption'),
    ...getPhase2NodesForCategory('risk'),
    ...getPhase2NodesForCategory('vendor'),
  ];

  // 3. Construct final nodes and remap links
  const nodes: ReportClassificationFlowNode[] = sortedNodeIds.map(id => ({
    id,
    ...nodeDataMap.get(id)!,
  }));

  const nodeIndexById = new Map<string, number>(sortedNodeIds.map((id, index) => [id, index]));

  const links = Array.from(linkDataMap.entries())
    .map(([key, value]) => {
      const [sourceId, targetId] = key.split('|||');
      const source = nodeIndexById.get(sourceId);
      const target = nodeIndexById.get(targetId);
      if (source === undefined || target === undefined) return null;
      return { source, target, value };
    })
    .filter((l): l is { source: number; target: number; value: number } => l !== null);

  return {
    totalReports: reports.length,
    extractedAiReports: extractedCount,
    noExtractedAiReports: noExtractedCount,
    phase1SignalReports: signalCount,
    phase1NoneOnlyReports: noneOnlyCount,
    nodes,
    links,
  };
};

const annotationsAsChunks = (
  annotations: GoldenAnnotation[],
  cniSectorMap: Map<string, string>,
  isicSectorMap: Map<string, string>,
  marketSegmentMap: Map<string, string>,
  documentMonths: Map<string, string>
): ReportData[] => {
  return annotations.filter(item => item.report_year >= 2020).map(item => {
    const adoptionConfMap = resolveConfidenceMap(item, 'adoption');
    const riskConfMap = resolveConfidenceMap(item, 'risk');
    const riskSignalMap = extractRiskSignalMap(item);

    const filteredAdoption = new Set<string>();
    const adoptionSignalMap: Record<string, number> = {};
    (item.adoption_types || []).forEach(type => {
      const confidence = adoptionConfMap?.[type] ?? 0;
      if (confidence >= CONFIDENCE_THRESHOLD) {
        filteredAdoption.add(type);
      }
      if (confidence > 0) {
        const normalized = normalizeRiskSignalValue(confidence);
        if (normalized > 0) {
          adoptionSignalMap[type] = Math.max(adoptionSignalMap[type] ?? 0, normalized);
        }
      }
    });

    const filteredRisk = new Set<string>();
    (item.risk_taxonomy || []).forEach(rawLabel => {
      const label = normalizeRiskLabel(rawLabel);
      if (!label || label === 'none') return;
      const confidence = riskConfMap?.[rawLabel] ?? riskConfMap?.[label] ?? 0;
      if (confidence >= CONFIDENCE_THRESHOLD) {
        filteredRisk.add(label);
      }
    });
    Object.entries(riskSignalMap).forEach(([label]) => {
      if (label !== 'none') filteredRisk.add(label);
    });

    const riskSubstantiveness = normalizeRiskSubstantiveness(item.risk_substantiveness);

    const vendorSignalMap: Record<string, number> = {};
    if (item.vendor_confidence && typeof item.vendor_confidence === 'object') {
      Object.entries(item.vendor_confidence).forEach(([tag, v]) => {
        const n = Number(v);
        if (Number.isFinite(n) && n > 0) {
          const normalized = normalizeRiskSignalValue(n);
          if (normalized > 0) {
            vendorSignalMap[tag] = Math.max(vendorSignalMap[tag] ?? 0, normalized);
          }
        }
      });
    }

    return {
      report_id: String(item.document_id || item.annotation_id),
      company_name: item.company_name,
      report_year: item.report_year,
      release_month: documentMonths.get(item.document_id) || '',
      sector: cniSectorMap.get(toKey(item.company_name)) || 'Unknown',
      isicSector: isicSectorMap.get(toKey(item.company_name)) || 'Unknown',
      marketSegment: marketSegmentMap.get(toKey(item.company_name)) || 'Other',
      mentionTypes: new Set(item.mention_types || []),
      adoptionTypes: filteredAdoption,
      riskLabels: filteredRisk,
      vendorTags: new Set(item.vendor_tags || []),
      adoptionConfidences: new Map(),
      riskConfidences: new Map(),
      adoptionSignalValues: Object.values(adoptionSignalMap),
      riskSignalValues: Object.values(riskSignalMap),
      vendorSignalValues: Object.values(vendorSignalMap),
      riskSubstantivenessValues: riskSubstantiveness ? [riskSubstantiveness] : [],
    };
  });
};

const buildExampleChunks = (annotations: GoldenAnnotation[]): ExampleChunk[] => {
  const EXAMPLE_CHUNK_LIMIT = 3;
  const candidates = annotations.map(item => {
    const chunkId = item.chunk_id?.trim();
    if (!chunkId) return null;

    const mentionTypes = Array.from(new Set((item.mention_types || []).map(type => type?.trim()).filter(Boolean)));
    const riskLabels = Array.from(new Set((item.risk_taxonomy || [])
      .map(normalizeRiskLabel)
      .filter(Boolean)));
    const adoptionTypes = Array.from(new Set((item.adoption_types || []).map(type => type?.trim()).filter(Boolean)));
    const vendorTags = Array.from(new Set((item.vendor_tags || []).map(tag => tag?.trim()).filter(Boolean)));

    return {
      chunkId,
      companyName: item.company_name,
      reportYear: item.report_year,
      chunkText: item.chunk_text || '',
      reportSections: item.report_sections || [],
      mentionTypes,
      riskLabels,
      adoptionTypes,
      vendorTags,
    } as ExampleChunk;
  });

  const filtered = candidates.filter((chunk): chunk is ExampleChunk => {
    if (!chunk) return false;
    const hasText = chunk.chunkText.trim().length > 0;
    const hasPhase1 = chunk.mentionTypes.some(type => type !== 'none');
    const hasPhase2 = chunk.riskLabels.length > 0 || chunk.adoptionTypes.length > 0 || chunk.vendorTags.length > 0;
    return hasText && (hasPhase1 || hasPhase2);
  });

  const sorted = filtered.sort((a, b) => {
    // Prioritize chunks that have a mix of different classification types
    const aDiversity = (a.riskLabels.length > 0 ? 1 : 0) + (a.adoptionTypes.length > 0 ? 1 : 0) + (a.vendorTags.length > 0 ? 1 : 0);
    const bDiversity = (b.riskLabels.length > 0 ? 1 : 0) + (b.adoptionTypes.length > 0 ? 1 : 0) + (b.vendorTags.length > 0 ? 1 : 0);
    
    if (bDiversity !== aDiversity) return bDiversity - aDiversity;
    
    // Then sort by total number of tags
    const aScore = a.riskLabels.length + a.adoptionTypes.length + a.vendorTags.length;
    const bScore = b.riskLabels.length + b.adoptionTypes.length + b.vendorTags.length;
    
    if (bScore !== aScore) return bScore - aScore;
    
    if (b.reportYear !== a.reportYear) return b.reportYear - a.reportYear;
    return a.chunkId.localeCompare(b.chunkId, 'en');
  });

  const selected: ExampleChunk[] = [];
  const seenCompanies = new Set<string>();

  sorted.forEach(chunk => {
    if (selected.length >= EXAMPLE_CHUNK_LIMIT) return;
    const companyKey = toKey(chunk.companyName);
    if (seenCompanies.has(companyKey)) return;
    selected.push(chunk);
    seenCompanies.add(companyKey);
  });

  if (selected.length < EXAMPLE_CHUNK_LIMIT) {
    sorted.forEach(chunk => {
      if (selected.length >= EXAMPLE_CHUNK_LIMIT) return;
      if (selected.some(selectedChunk => selectedChunk.chunkId === chunk.chunkId)) return;
      selected.push(chunk);
    });
  }

  return selected;
};

const initMonthSeries = (months: string[], keys: string[]) => {
  return months.map(month => {
    const row: Record<string, string | number> = { month };
    keys.forEach(key => {
      row[key] = 0;
    });
    return row;
  });
};

const addMonthCount = (
  series: Record<string, string | number>[],
  month: string,
  key: string
) => {
  const row = series.find(entry => entry.month === month);
  if (!row) return;
  row[key] = (Number(row[key]) || 0) + 1;
};

const buildDataset = (
  reports: ReportData[],
  years: number[],
  sectors: string[],
  isicSectors: string[]
): GoldenDataset => {
  // Collect all unique months from reports
  const monthSet = new Set<string>();
  reports.forEach(r => {
    if (r.release_month) monthSet.add(r.release_month);
  });
  const months = Array.from(monthSet).sort();

  const mentionTrend = initYearSeries(years, mentionTypes);
  const adoptionTrend = initYearSeries(years, adoptionTypes);
  const riskTrend = initYearSeries(years, riskLabels);
  const vendorTrend = initYearSeries(years, vendorTags);

  const riskTrendMonthly = initMonthSeries(months, riskLabels);
  const adoptionTrendMonthly = initMonthSeries(months, adoptionTypes);
  const vendorTrendMonthly = initMonthSeries(months, vendorTags);

  const riskBySectorCounts = new Map<string, number>();
  const riskBySectorYearCounts = new Map<string, number>();
  const riskByIsicSectorCounts = new Map<string, number>();
  const riskByIsicSectorYearCounts = new Map<string, number>();
  const adoptionBySectorCounts = new Map<string, number>();
  const adoptionBySectorYearCounts = new Map<string, number>();
  const adoptionByIsicSectorCounts = new Map<string, number>();
  const adoptionByIsicSectorYearCounts = new Map<string, number>();
  const vendorBySectorCounts = new Map<string, number>();
  const vendorBySectorYearCounts = new Map<string, number>();
  const vendorByIsicSectorCounts = new Map<string, number>();
  const vendorByIsicSectorYearCounts = new Map<string, number>();
  const adoptionSignalCounts = new Map<string, number>();
  const riskSignalCounts = new Map<string, number>();
  const vendorSignalCounts = new Map<string, number>();
  const substantivenessCounts = new Map<string, number>();
  const blindSpotYearCounts = new Map<number, { total: number; noAi: number; noAiRisk: number }>();
  const noAiBySectorYearCounts = new Map<string, number>();
  const noAiRiskBySectorYearCounts = new Map<string, number>();
  const reportCountBySectorYearCounts = new Map<string, number>();
  const reportCountByIsicSectorYearCounts = new Map<string, number>();

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
    const hasRiskSignal = report.mentionTypes.has('risk') || report.riskLabels.size > 0;
    if (hasSignal) aiSignalReports += 1;

    if (report.mentionTypes.has('adoption')) adoptionReports += 1;
    if (report.mentionTypes.has('risk')) riskReports += 1;
    if (report.mentionTypes.has('vendor')) vendorReports += 1;

    const yearBlindSpot = blindSpotYearCounts.get(year) || { total: 0, noAi: 0, noAiRisk: 0 };
    yearBlindSpot.total += 1;
    if (!hasSignal) yearBlindSpot.noAi += 1;
    if (!hasRiskSignal) yearBlindSpot.noAiRisk += 1;
    blindSpotYearCounts.set(year, yearBlindSpot);

    const sectorYearKey = `${year}|||${report.sector}`;
    reportCountBySectorYearCounts.set(
      sectorYearKey,
      (reportCountBySectorYearCounts.get(sectorYearKey) || 0) + 1
    );
    const isicSectorYearKey = `${year}|||${report.isicSector}`;
    reportCountByIsicSectorYearCounts.set(
      isicSectorYearKey,
      (reportCountByIsicSectorYearCounts.get(isicSectorYearKey) || 0) + 1
    );
    if (!hasSignal) {
      noAiBySectorYearCounts.set(sectorYearKey, (noAiBySectorYearCounts.get(sectorYearKey) || 0) + 1);
    }
    if (!hasRiskSignal) {
      noAiRiskBySectorYearCounts.set(sectorYearKey, (noAiRiskBySectorYearCounts.get(sectorYearKey) || 0) + 1);
    }

    report.mentionTypes.forEach(type => addCount(mentionTrend, year, type));
    report.adoptionTypes.forEach(type => addCount(adoptionTrend, year, type));
    report.riskLabels.forEach(label => addCount(riskTrend, year, label));
    report.vendorTags.forEach(tag => addCount(vendorTrend, year, tag));

    // Monthly trends
    if (report.release_month) {
      report.riskLabels.forEach(label => addMonthCount(riskTrendMonthly, report.release_month, label));
      report.adoptionTypes.forEach(type => addMonthCount(adoptionTrendMonthly, report.release_month, type));
      report.vendorTags.forEach(tag => addMonthCount(vendorTrendMonthly, report.release_month, tag));
    }

    report.riskLabels.forEach(label => {
      const key = `${label}|||${report.sector}`;
      riskBySectorCounts.set(key, (riskBySectorCounts.get(key) || 0) + 1);
      const yearKey = `${year}|||${label}|||${report.sector}`;
      riskBySectorYearCounts.set(yearKey, (riskBySectorYearCounts.get(yearKey) || 0) + 1);

      const isicKey = `${label}|||${report.isicSector}`;
      riskByIsicSectorCounts.set(isicKey, (riskByIsicSectorCounts.get(isicKey) || 0) + 1);
      const isicYearKey = `${year}|||${label}|||${report.isicSector}`;
      riskByIsicSectorYearCounts.set(
        isicYearKey,
        (riskByIsicSectorYearCounts.get(isicYearKey) || 0) + 1
      );
    });

    report.adoptionTypes.forEach(type => {
      const key = `${type}|||${report.sector}`;
      adoptionBySectorCounts.set(key, (adoptionBySectorCounts.get(key) || 0) + 1);
      const yearKey = `${year}|||${type}|||${report.sector}`;
      adoptionBySectorYearCounts.set(yearKey, (adoptionBySectorYearCounts.get(yearKey) || 0) + 1);

      const isicKey = `${type}|||${report.isicSector}`;
      adoptionByIsicSectorCounts.set(isicKey, (adoptionByIsicSectorCounts.get(isicKey) || 0) + 1);
      const isicYearKey = `${year}|||${type}|||${report.isicSector}`;
      adoptionByIsicSectorYearCounts.set(isicYearKey, (adoptionByIsicSectorYearCounts.get(isicYearKey) || 0) + 1);
    });

    report.vendorTags.forEach(tag => {
      const key = `${tag}|||${report.sector}`;
      vendorBySectorCounts.set(key, (vendorBySectorCounts.get(key) || 0) + 1);
      const yearKey = `${year}|||${tag}|||${report.sector}`;
      vendorBySectorYearCounts.set(yearKey, (vendorBySectorYearCounts.get(yearKey) || 0) + 1);

      const isicKey = `${tag}|||${report.isicSector}`;
      vendorByIsicSectorCounts.set(isicKey, (vendorByIsicSectorCounts.get(isicKey) || 0) + 1);
      const isicYearKey = `${year}|||${tag}|||${report.isicSector}`;
      vendorByIsicSectorYearCounts.set(isicYearKey, (vendorByIsicSectorYearCounts.get(isicYearKey) || 0) + 1);
    });

    report.adoptionSignalValues.forEach(signal => {
      const level =
        signal >= 3
          ? '3-explicit'
          : signal >= 2
            ? '2-strong_implicit'
            : '1-weak_implicit';
      const key = `${year}|||${level}`;
      adoptionSignalCounts.set(key, (adoptionSignalCounts.get(key) || 0) + 1);
    });

    report.riskSignalValues.forEach(signal => {
      const level =
        signal >= 3
          ? '3-explicit'
          : signal >= 2
            ? '2-strong_implicit'
            : '1-weak_implicit';
      const key = `${year}|||${level}`;
      riskSignalCounts.set(key, (riskSignalCounts.get(key) || 0) + 1);
    });

    report.vendorSignalValues.forEach(signal => {
      const level =
        signal >= 3
          ? '3-explicit'
          : signal >= 2
            ? '2-strong_implicit'
            : '1-weak_implicit';
      const key = `${year}|||${level}`;
      vendorSignalCounts.set(key, (vendorSignalCounts.get(key) || 0) + 1);
    });

    if (report.riskSubstantivenessValues.length > 0) {
      const counts: Record<string, number> = {
        substantive: 0,
        moderate: 0,
        boilerplate: 0,
      };
      report.riskSubstantivenessValues.forEach(v => {
        counts[v] = (counts[v] || 0) + 1;
      });
      const band =
        counts.substantive >= counts.moderate && counts.substantive >= counts.boilerplate
          ? 'substantive'
          : counts.moderate >= counts.boilerplate
            ? 'moderate'
            : 'boilerplate';
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

  const riskBySectorYear = Array.from(riskBySectorYearCounts.entries()).map(([key, value]) => {
    const [year, label, sector] = key.split('|||');
    return { year: Number(year), x: label, y: sector, value };
  });

  const adoptionBySectorYear = Array.from(adoptionBySectorYearCounts.entries()).map(([key, value]) => {
    const [year, type, sector] = key.split('|||');
    return { year: Number(year), x: type, y: sector, value };
  });

  const adoptionByIsicSector = Array.from(adoptionByIsicSectorCounts.entries()).map(([key, value]) => {
    const [type, sector] = key.split('|||');
    return { x: type, y: sector, value };
  });

  const adoptionByIsicSectorYear = Array.from(adoptionByIsicSectorYearCounts.entries()).map(([key, value]) => {
    const [year, type, sector] = key.split('|||');
    return { year: Number(year), x: type, y: sector, value };
  });

  const vendorBySectorYear = Array.from(vendorBySectorYearCounts.entries()).map(([key, value]) => {
    const [year, tag, sector] = key.split('|||');
    return { year: Number(year), x: tag, y: sector, value };
  });

  const vendorByIsicSector = Array.from(vendorByIsicSectorCounts.entries()).map(([key, value]) => {
    const [tag, sector] = key.split('|||');
    return { x: tag, y: sector, value };
  });

  const vendorByIsicSectorYear = Array.from(vendorByIsicSectorYearCounts.entries()).map(([key, value]) => {
    const [year, tag, sector] = key.split('|||');
    return { year: Number(year), x: tag, y: sector, value };
  });

  const riskByIsicSector = Array.from(riskByIsicSectorCounts.entries()).map(([key, value]) => {
    const [label, sector] = key.split('|||');
    return { x: label, y: sector, value };
  });

  const riskByIsicSectorYear = Array.from(riskByIsicSectorYearCounts.entries()).map(([key, value]) => {
    const [year, label, sector] = key.split('|||');
    return { year: Number(year), x: label, y: sector, value };
  });

  const adoptionSignalHeatmap: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    riskSignalLevels.forEach(level => {
      adoptionSignalHeatmap.push({
        x: year,
        y: level,
        value: adoptionSignalCounts.get(`${year}|||${level}`) || 0,
      });
    });
  });

  const riskSignalHeatmap: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    riskSignalLevels.forEach(level => {
      riskSignalHeatmap.push({
        x: year,
        y: level,
        value: riskSignalCounts.get(`${year}|||${level}`) || 0,
      });
    });
  });

  const vendorSignalHeatmap: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    riskSignalLevels.forEach(level => {
      vendorSignalHeatmap.push({
        x: year,
        y: level,
        value: vendorSignalCounts.get(`${year}|||${level}`) || 0,
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

  const blindSpotTrend = years.map(year => {
    const counts = blindSpotYearCounts.get(year) || { total: 0, noAi: 0, noAiRisk: 0 };
    return {
      year,
      total_reports: counts.total,
      ai_mention: counts.total - counts.noAi,
      ai_risk_mention: counts.total - counts.noAiRisk,
      no_ai_mention: counts.noAi,
      no_ai_risk_mention: counts.noAiRisk,
    };
  });

  // Monthly blind spot trend
  const blindSpotMonthCounts = new Map<string, { total: number; noAi: number; noAiRisk: number }>();
  reports.forEach(report => {
    if (!report.release_month) return;
    const month = report.release_month;
    const entry = blindSpotMonthCounts.get(month) || { total: 0, noAi: 0, noAiRisk: 0 };
    entry.total += 1;
    const hasSignal = report.mentionTypes.size > 0 && !(report.mentionTypes.size === 1 && report.mentionTypes.has('none'));
    const hasRiskSignal = report.mentionTypes.has('risk') || report.riskLabels.size > 0;
    if (!hasSignal) entry.noAi += 1;
    if (!hasRiskSignal) entry.noAiRisk += 1;
    blindSpotMonthCounts.set(month, entry);
  });
  const blindSpotTrendMonthly = months.map(month => {
    const counts = blindSpotMonthCounts.get(month) || { total: 0, noAi: 0, noAiRisk: 0 };
    return {
      month,
      total_reports: counts.total,
      no_ai_mention: counts.noAi,
      no_ai_risk_mention: counts.noAiRisk,
    };
  });

  const noAiBySectorYear: { x: number; y: string; value: number }[] = [];
  const noAiRiskBySectorYear: { x: number; y: string; value: number }[] = [];
  const reportCountBySectorYear: { x: number; y: string; value: number }[] = [];
  const reportCountByIsicSectorYear: { x: number; y: string; value: number }[] = [];
  years.forEach(year => {
    sectors.forEach(sector => {
      const key = `${year}|||${sector}`;
      noAiBySectorYear.push({
        x: year,
        y: sector,
        value: noAiBySectorYearCounts.get(key) || 0,
      });
      noAiRiskBySectorYear.push({
        x: year,
        y: sector,
        value: noAiRiskBySectorYearCounts.get(key) || 0,
      });
      reportCountBySectorYear.push({
        x: year,
        y: sector,
        value: reportCountBySectorYearCounts.get(key) || 0,
      });
    });
    isicSectors.forEach(sector => {
      const key = `${year}|||${sector}`;
      reportCountByIsicSectorYear.push({
        x: year,
        y: sector,
        value: reportCountByIsicSectorYearCounts.get(key) || 0,
      });
    });
  });

  return {
    years,
    months,
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
    riskTrendMonthly,
    adoptionTrendMonthly,
    vendorTrendMonthly,
    blindSpotTrendMonthly,
    riskBySector,
    riskBySectorYear,
    riskByIsicSector,
    riskByIsicSectorYear,
    adoptionBySector,
    adoptionBySectorYear,
    adoptionByIsicSector,
    adoptionByIsicSectorYear,
    vendorBySector,
    vendorBySectorYear,
    vendorByIsicSector,
    vendorByIsicSectorYear,
    riskSignalHeatmap,
    adoptionSignalHeatmap,
    vendorSignalHeatmap,
    substantivenessHeatmap,
    blindSpotTrend,
    noAiBySectorYear,
    noAiRiskBySectorYear,
    reportCountBySectorYear,
    reportCountByIsicSectorYear,
  };
};

export const buildGoldenSetDashboardDataFromRaw = (): GoldenDashboardData => {
  const {
    cniSectorMap,
    isicSectorMap,
    isicSectorParentMap,
    marketSegmentMap,
    cniSectors,
    isicSectors,
    reportUniverseRows,
  } = parseCompanySectors();
  const annotations = parseAnnotations(ANNOTATIONS_PATH);
  const documentMonths = loadDocumentMonths();

  const years = Array.from(
    new Set([
      ...annotations.map(item => item.report_year),
      ...reportUniverseRows.map(report => report.report_year),
    ])
  ).sort((a, b) => a - b);

  const resolvedCniSectors = cniSectors.length ? cniSectors : ['Unknown'];
  const resolvedIsicSectors = [...isicSectors].sort((a, b) => a.localeCompare(b, 'en'));
  const resolvedMarketSegments = ['FTSE 350', 'Main Market', 'AIM'];

  const perReportData = aggregateToReports(
    annotations,
    cniSectorMap,
    isicSectorMap,
    marketSegmentMap,
    reportUniverseRows,
    documentMonths
  );
  const perChunkData = annotationsAsChunks(annotations, cniSectorMap, isicSectorMap, marketSegmentMap, documentMonths);
  const exampleChunks = buildExampleChunks(annotations);

  const byMarketSegment: Record<string, { perReport: GoldenDataset; perChunk: GoldenDataset }> = {};
  resolvedMarketSegments.forEach(segment => {
    const segReportData = perReportData.filter(r => r.marketSegment === segment);
    const segChunkData = perChunkData.filter(r => r.marketSegment === segment);
    byMarketSegment[segment] = {
      perReport: buildDataset(segReportData, years, resolvedCniSectors, resolvedIsicSectors),
      perChunk: buildDataset(segChunkData, years, resolvedCniSectors, resolvedIsicSectors),
    };
  });

  return {
    years,
    sectors: resolvedCniSectors,
    isicSectors: resolvedIsicSectors,
    isicSectorParents: Object.fromEntries(isicSectorParentMap),
    marketSegments: resolvedMarketSegments,
    reportClassificationFlow: buildReportClassificationFlow(perReportData),
    reportClassificationBreakdown: buildReportClassificationBreakdown(perReportData),
    labels: {
      mentionTypes,
      adoptionTypes,
      riskLabels,
      vendorTags,
      substantivenessBands,
      riskSignalLevels,
    },
    datasets: {
      perReport: buildDataset(perReportData, years, resolvedCniSectors, resolvedIsicSectors),
      perChunk: buildDataset(perChunkData, years, resolvedCniSectors, resolvedIsicSectors),
    },
    exampleChunks,
    byMarketSegment,
  };
};

export const loadGoldenSetDashboardData = (): GoldenDashboardData => {
  if (cachedDashboardData) return cachedDashboardData;

  if (shouldPreferPrecomputedDashboardData()) {
    const precomputedData = loadPrecomputedDashboardData();
    if (precomputedData) {
      cachedDashboardData = precomputedData;
      return precomputedData;
    }
  }

  const rawData = buildGoldenSetDashboardDataFromRaw();
  cachedDashboardData = rawData;
  return rawData;
};
