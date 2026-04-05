export type HeatmapRowGroup = {
  label: string;
  childKeys: string[];
};

export const ISIC_SECTION_ORDER = [
  'Agriculture, forestry and fishing',
  'Mining and quarrying',
  'Manufacturing',
  'Electricity, gas, steam and air conditioning supply',
  'Water supply; sewerage, waste management and remediation activities',
  'Construction',
  'Wholesale and retail trade; repair of motor vehicles and motorcycles',
  'Transportation and storage',
  'Accommodation and food service activities',
  'Information and communication',
  'Financial and insurance activities',
  'Real estate activities',
  'Professional, scientific and technical activities',
  'Administrative and support service activities',
  'Public administration and defence; compulsory social security',
  'Education',
  'Human health and social work activities',
  'Arts, entertainment and recreation',
  'Other service activities',
  'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use',
  'Activities of extraterritorial organizations and bodies',
] as const;

const normalizeIsicLabel = (label: string) => label.replace(/\s+/g, ' ').trim();

export const resolveIsicSectionFromCode = (code: string): string | null => {
  const digits = code.replace(/\D+/g, '');
  if (digits.length < 2) return null;

  const division = Number(digits.slice(0, 2));
  if (!Number.isFinite(division)) return null;

  if (division >= 1 && division <= 3) return 'Agriculture, forestry and fishing';
  if (division >= 5 && division <= 9) return 'Mining and quarrying';
  if (division >= 10 && division <= 33) return 'Manufacturing';
  if (division === 35) return 'Electricity, gas, steam and air conditioning supply';
  if (division >= 36 && division <= 39) return 'Water supply; sewerage, waste management and remediation activities';
  if (division >= 41 && division <= 43) return 'Construction';
  if (division >= 45 && division <= 47) return 'Wholesale and retail trade; repair of motor vehicles and motorcycles';
  if (division >= 49 && division <= 53) return 'Transportation and storage';
  if (division >= 55 && division <= 56) return 'Accommodation and food service activities';
  if (division >= 58 && division <= 63) return 'Information and communication';
  if (division >= 64 && division <= 66) return 'Financial and insurance activities';
  if (division === 68) return 'Real estate activities';
  if (division >= 69 && division <= 75) return 'Professional, scientific and technical activities';
  if (division >= 77 && division <= 82) return 'Administrative and support service activities';
  if (division === 84) return 'Public administration and defence; compulsory social security';
  if (division === 85) return 'Education';
  if (division >= 86 && division <= 88) return 'Human health and social work activities';
  if (division >= 90 && division <= 93) return 'Arts, entertainment and recreation';
  if (division >= 94 && division <= 96) return 'Other service activities';
  if (division >= 97 && division <= 98) {
    return 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use';
  }
  if (division === 99) return 'Activities of extraterritorial organizations and bodies';

  return null;
};

export const buildIsicSectorGroups = (
  labels: string[],
  sectionByLabel: Record<string, string>
): { groups: HeatmapRowGroup[]; ungroupedLabels: string[] } => {
  const normalizedToOriginal = new Map<string, string>();
  labels.forEach(label => {
    const normalized = normalizeIsicLabel(label);
    if (!normalized || normalizedToOriginal.has(normalized)) return;
    normalizedToOriginal.set(normalized, label);
  });

  const grouped = new Map<string, string[]>();
  const ungroupedLabels: string[] = [];

  Array.from(normalizedToOriginal.values()).forEach(label => {
    const parent = sectionByLabel[label];
    if (!parent || parent === 'Unknown') {
      ungroupedLabels.push(label);
      return;
    }

    const existing = grouped.get(parent);
    if (existing) {
      existing.push(label);
      return;
    }
    grouped.set(parent, [label]);
  });

  const groups = Array.from(grouped.entries())
    .map(([label, childKeys]) => ({
      label,
      childKeys: childKeys.sort((a, b) => a.localeCompare(b, 'en')),
    }))
    .sort((a, b) => {
      const aIndex = ISIC_SECTION_ORDER.indexOf(a.label as (typeof ISIC_SECTION_ORDER)[number]);
      const bIndex = ISIC_SECTION_ORDER.indexOf(b.label as (typeof ISIC_SECTION_ORDER)[number]);
      const normalizedA = aIndex === -1 ? Number.MAX_SAFE_INTEGER : aIndex;
      const normalizedB = bIndex === -1 ? Number.MAX_SAFE_INTEGER : bIndex;
      return normalizedA !== normalizedB
        ? normalizedA - normalizedB
        : a.label.localeCompare(b.label, 'en');
    });

  return {
    groups,
    ungroupedLabels: ungroupedLabels.sort((a, b) => a.localeCompare(b, 'en')),
  };
};
