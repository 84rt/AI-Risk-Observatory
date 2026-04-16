import fs from 'fs';
import path from 'path';

export type ReportFigureSeries = {
  key: string;
  label: string;
  color: string;
};

export type ReportFigureBase = {
  id: string;
  title: string;
  chartType: string;
  notes?: string[];
};

export type ReportFigureLine = ReportFigureBase & {
  chartType: 'line' | 'stacked-area' | 'dual-axis-line-area';
  domain: {
    years: [number, number];
    measure: string;
  };
  series: ReportFigureSeries[];
  shading?: {
    label: string;
    lowerKey?: string;
    upperKey?: string;
    key?: string;
    color?: string;
  };
  data: Array<Record<string, number>>;
};

export type ReportFigureHorizontalBar = ReportFigureBase & {
  chartType: 'horizontal-bar' | 'grouped-horizontal-bar';
  comparison?: {
    fromYear: number;
    toYear: number;
    measure: string;
  };
  year?: number;
  series?: ReportFigureSeries[];
  data: Array<Record<string, string | number>>;
};

export type ReportFigureTreemap = ReportFigureBase & {
  chartType: 'treemap';
  measure: string;
  total_assignments_2025: number;
  data: Array<Record<string, string | number>>;
};

export type ReportFigureHeatmap = ReportFigureBase & {
  chartType: 'heatmap';
  domain: {
    years: [number, number];
    measure: string;
  };
  sectors: string[];
  years: number[];
  cells: Array<{
    sector: string;
    year: number;
    current_risk_rate_pct: number;
    previous_risk_rate_pct: number;
    delta_pp: number;
  }>;
};

export type ReportFigureGroupedBar = ReportFigureBase & {
  chartType: 'grouped-bar';
  year: number;
  series: ReportFigureSeries[];
  data: Array<Record<string, string | number>>;
};

export type ReportFigure =
  | ReportFigureLine
  | ReportFigureHorizontalBar
  | ReportFigureTreemap
  | ReportFigureHeatmap
  | ReportFigureGroupedBar;

export type ReportFiguresDocument = {
  generatedAt: string;
  source: {
    dashboardDataPath: string;
    maxReportYear: number;
    baseDataset: string;
    cniSectorScope: string;
  };
  figures: Record<string, ReportFigure>;
};

const REPORT_FIGURES_PATH = path.join(process.cwd(), '..', 'report', 'report-figures.json');

let cachedReportFigures: ReportFiguresDocument | null = null;

export const loadReportFigures = (): ReportFiguresDocument => {
  if (cachedReportFigures) return cachedReportFigures;
  const raw = fs.readFileSync(REPORT_FIGURES_PATH, 'utf8');
  cachedReportFigures = JSON.parse(raw) as ReportFiguresDocument;
  return cachedReportFigures;
};

export const getReportFigure = (figureId: string): ReportFigure | null => {
  const doc = loadReportFigures();
  return doc.figures[figureId] ?? null;
};

export const getAllReportFigures = (): ReportFigure[] => {
  const doc = loadReportFigures();
  return Object.values(doc.figures);
};
