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
  risk_substantiveness: string | number | null;
  risk_signals?: Array<{ type?: string; signal?: number | string }> | Record<string, number> | null;
  vendor_tags: string[];
  vendor_confidence?: Record<string, number> | null;
};

export type GoldenDashboardData = {
  years: number[];
  sectors: string[];
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
  riskBySectorYear: { year: number; x: string; y: string; value: number }[];
  adoptionBySector: { x: string; y: string; value: number }[];
  adoptionBySectorYear: { year: number; x: string; y: string; value: number }[];
  vendorBySector: { x: string; y: string; value: number }[];
  vendorBySectorYear: { year: number; x: string; y: string; value: number }[];
  riskSignalHeatmap: { x: number; y: string; value: number }[];
  adoptionSignalHeatmap: { x: number; y: string; value: number }[];
  vendorSignalHeatmap: { x: number; y: string; value: number }[];
  substantivenessHeatmap: { x: number; y: string; value: number }[];
};

const ANNOTATIONS_PATH = path.join(
  process.cwd(),
  'data',
  'annotations.jsonl'
);

const COMPANIES_PATH = path.join(
  process.cwd(),
  'data',
  'golden_set_companies.csv'
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
  'regulatory_compliance',
  'reputational_ethical',
  'information_integrity',
  'third_party_supply_chain',
  'strategic_competitive',
  'workforce_impacts',
  'environmental_impact',
  'national_security',
];

const vendorTags = ['openai', 'microsoft', 'google', 'internal', 'other', 'undisclosed'];

const substantivenessBands = ['substantive', 'moderate', 'boilerplate'];
const riskSignalLevels = ['3-explicit', '2-strong_implicit', '1-weak_implicit'];

const RISK_LABEL_ALIASES: Record<string, string> = {
  strategic_market: 'strategic_competitive',
  regulatory: 'regulatory_compliance',
  workforce: 'workforce_impacts',
  environmental: 'environmental_impact',
};

const toKey = (value: string) => value.trim().toLowerCase();

const parseCompanySectors = () => {
  if (!fs.existsSync(COMPANIES_PATH)) {
    return {
      sectorMap: new Map<string, string>(),
      sectors: [] as string[],
      companySectors: [] as { company_name: string; sector: string }[],
    };
  }

  const content = fs.readFileSync(COMPANIES_PATH, 'utf8').trim();
  const lines = content.split(/\r?\n/);
  const header = lines.shift();
  if (!header) {
    return {
      sectorMap: new Map<string, string>(),
      sectors: [] as string[],
      companySectors: [] as { company_name: string; sector: string }[],
    };
  }
  const headers = header.split(',');
  const nameIndex = headers.indexOf('company_name');
  const sectorIndex = headers.indexOf('sector') >= 0
    ? headers.indexOf('sector')
    : headers.indexOf('cni_sector');
  const sectorMap = new Map<string, string>();
  const sectors: string[] = [];
  const companySectors: { company_name: string; sector: string }[] = [];
  const sectorSet = new Set<string>();

  if (nameIndex < 0 || sectorIndex < 0) {
    return { sectorMap, sectors, companySectors };
  }

  lines.forEach(line => {
    const cells = line.split(',');
    const name = cells[nameIndex]?.trim();
    const sector = cells[sectorIndex]?.trim() || 'Unknown';
    if (!name) return;
    sectorMap.set(toKey(name), sector);
    companySectors.push({ company_name: name, sector });
    if (!sectorSet.has(sector)) {
      sectorSet.add(sector);
      sectors.push(sector);
    }
  });

  return { sectorMap, sectors, companySectors };
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
  company_name: string;
  report_year: number;
  sector: string;
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
  companyName: string,
  reportYear: number,
  sector: string
): ReportData => ({
  company_name: companyName,
  report_year: reportYear,
  sector,
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
  sectorMap: Map<string, string>,
  years: number[],
  companySectors: { company_name: string; sector: string }[]
): ReportData[] => {
  const reportMap = new Map<string, ReportData>();

  annotations.forEach(item => {
    const reportKey = `${item.company_name}|||${item.report_year}`;

    if (!reportMap.has(reportKey)) {
      reportMap.set(
        reportKey,
        makeEmptyReportData(
          item.company_name,
          item.report_year,
          sectorMap.get(toKey(item.company_name)) || 'Unknown'
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

  // Ensure report-level dataset includes all expected company-year pairs,
  // including reports with zero AI chunks.
  if (years.length > 0 && companySectors.length > 0) {
    companySectors.forEach(({ company_name, sector }) => {
      years.forEach(year => {
        const key = `${company_name}|||${year}`;
        if (!reportMap.has(key)) {
          reportMap.set(key, makeEmptyReportData(company_name, year, sector || 'Unknown'));
        }
      });
    });
  }

  return Array.from(reportMap.values());
};

const annotationsAsChunks = (
  annotations: GoldenAnnotation[],
  sectorMap: Map<string, string>
): ReportData[] => {
  return annotations.map(item => {
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
      company_name: item.company_name,
      report_year: item.report_year,
      sector: sectorMap.get(toKey(item.company_name)) || 'Unknown',
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

const averageConfidence = (values: number[]): number => {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
};

const buildDataset = (
  reports: ReportData[],
  years: number[]
): GoldenDataset => {
  const mentionTrend = initYearSeries(years, mentionTypes);
  const adoptionTrend = initYearSeries(years, adoptionTypes);
  const riskTrend = initYearSeries(years, riskLabels);
  const vendorTrend = initYearSeries(years, vendorTags);

  const riskBySectorCounts = new Map<string, number>();
  const riskBySectorYearCounts = new Map<string, number>();
  const adoptionBySectorCounts = new Map<string, number>();
  const adoptionBySectorYearCounts = new Map<string, number>();
  const vendorBySectorCounts = new Map<string, number>();
  const vendorBySectorYearCounts = new Map<string, number>();
  const adoptionSignalCounts = new Map<string, number>();
  const riskSignalCounts = new Map<string, number>();
  const vendorSignalCounts = new Map<string, number>();
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
      const yearKey = `${year}|||${label}|||${report.sector}`;
      riskBySectorYearCounts.set(yearKey, (riskBySectorYearCounts.get(yearKey) || 0) + 1);
    });

    report.adoptionTypes.forEach(type => {
      const key = `${type}|||${report.sector}`;
      adoptionBySectorCounts.set(key, (adoptionBySectorCounts.get(key) || 0) + 1);
      const yearKey = `${year}|||${type}|||${report.sector}`;
      adoptionBySectorYearCounts.set(yearKey, (adoptionBySectorYearCounts.get(yearKey) || 0) + 1);
    });

    report.vendorTags.forEach(tag => {
      const key = `${tag}|||${report.sector}`;
      vendorBySectorCounts.set(key, (vendorBySectorCounts.get(key) || 0) + 1);
      const yearKey = `${year}|||${tag}|||${report.sector}`;
      vendorBySectorYearCounts.set(yearKey, (vendorBySectorYearCounts.get(yearKey) || 0) + 1);
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

  const vendorBySectorYear = Array.from(vendorBySectorYearCounts.entries()).map(([key, value]) => {
    const [year, tag, sector] = key.split('|||');
    return { year: Number(year), x: tag, y: sector, value };
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
    riskBySectorYear,
    adoptionBySector,
    adoptionBySectorYear,
    vendorBySector,
    vendorBySectorYear,
    riskSignalHeatmap,
    adoptionSignalHeatmap,
    vendorSignalHeatmap,
    substantivenessHeatmap,
  };
};

export const loadGoldenSetDashboardData = (): GoldenDashboardData => {
  const { sectorMap, sectors, companySectors } = parseCompanySectors();
  const annotations = parseAnnotations(ANNOTATIONS_PATH);

  const years = Array.from(
    new Set(annotations.map(item => item.report_year))
  ).sort((a, b) => a - b);

  const perReportData = aggregateToReports(annotations, sectorMap, years, companySectors);
  const perChunkData = annotationsAsChunks(annotations, sectorMap);

  return {
    years,
    sectors: sectors.length ? sectors : ['Unknown'],
    labels: {
      mentionTypes,
      adoptionTypes,
      riskLabels,
      vendorTags,
      substantivenessBands,
      riskSignalLevels,
    },
    datasets: {
      perReport: buildDataset(perReportData, years),
      perChunk: buildDataset(perChunkData, years),
    },
  };
};
