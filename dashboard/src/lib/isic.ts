export type HeatmapRowGroup = {
  label: string;
  childKeys: string[];
};

type IsicParentGroupDefinition = {
  label: string;
  children: string[];
};

const ISIC_PARENT_GROUP_DEFINITIONS: IsicParentGroupDefinition[] = [
  {
    label: 'Mining and quarrying',
    children: [
      'Extraction of crude petroleum',
      'Extraction of natural gas',
      'Mining of iron ores',
      'Mining of other non-ferrous metal ores',
    ],
  },
  {
    label: 'Manufacturing',
    children: [
      'Distilling, rectifying and blending of spirits',
      'Manufacture of air and spacecraft and related machinery',
      'Manufacture of clay building materials',
      'Manufacture of corrugated paper and paperboard and of containers of paper and paperboard',
      'Manufacture of footwear',
      'Manufacture of games and toys',
      'Manufacture of measuring, testing, navigating and control equipment',
      'Manufacture of medical and dental instruments and supplies',
      'Manufacture of other chemical products n.e.c.',
      'Manufacture of other electronic components and boards',
      'Manufacture of other food products n.e.c.',
      'Manufacture of other general-purpose machinery',
      'Manufacture of pharmaceuticals, medicinal chemical and botanical products',
      'Manufacture of plastics products',
      'Manufacture of refined petroleum products; manufacture of fossil fuel products',
      'Manufacture of refractory products',
      'Manufacture of soap and detergents, cleaning and polishing preparations, perfumes and toilet preparations',
      'Manufacture of soft drinks; production of mineral waters and other bottled waters',
      'Manufacture of starches and starch products',
      'Manufacture of tobacco products',
      'Manufacture of wearing apparel, except fur apparel',
      'Preparation and spinning of textile fibres',
    ],
  },
  {
    label: 'Electricity, gas, steam and air conditioning supply',
    children: [
      'Activities of brokers and agents for electric power and natural gas',
      'Electric power generation activities from renewable sources',
      'Electric power transmission and distribution activities',
    ],
  },
  {
    label: 'Water supply; sewerage, waste management and remediation activities',
    children: ['Water collection, treatment and supply'],
  },
  {
    label: 'Construction',
    children: [
      'Construction of residential and non-residential buildings',
      'Other specialized construction activities',
    ],
  },
  {
    label: 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
    children: [
      'Non-specialized retail sale with food, beverages or tobacco predominating',
      'Other non-specialized retail sale',
      'Retail sale of clothing, footwear and leather articles',
      'Retail sale of electrical household appliances, furniture, lighting equipment and other household articles',
      'Retail sale of hardware, building materials, paints and glass',
      'Retail sale of information and communication equipment',
      'Retail sale of other new goods n.e.c.',
      'Wholesale of chemicals, waste and scrap and other products n.e.c.',
      'Wholesale of other machinery and equipment',
    ],
  },
  {
    label: 'Transportation and storage',
    children: [
      'Passenger air transport',
      'Postal activities',
      'Sea and coastal passenger water transport',
      'Urban and suburban passenger land transport',
    ],
  },
  {
    label: 'Accommodation and food service activities',
    children: [
      'Beverage serving activities',
      'Event catering activities',
      'Hotels and similar accommodation activities',
      'Other food service activities',
      'Restaurants and mobile food service activities',
    ],
  },
  {
    label: 'Information and communication',
    children: [
      'Computer consultancy and computer facilities management activities',
      'Other computer programming activities',
      'Other information technology and computer service activities',
      'Other software publishing',
      'Publishing of journals and periodicals',
      'Telecommunication reselling activities and intermediation service activities for telecommunication',
      'Web search portals activities and other information service activities',
      'Wired, wireless, and satellite telecommunication activities',
    ],
  },
  {
    label: 'Financial and insurance activities',
    children: [
      'Activities of holding companies',
      'Activities of non-money market investments funds',
      'Activities of trust, estate and agency accounts',
      'Fund management activities',
      'Life insurance',
      'Non-life insurance',
      'Other financial service activities n.e.c., except insurance and pension funding activities',
      'Other monetary intermediation',
    ],
  },
  {
    label: 'Real estate activities',
    children: [
      'Other real estate activities on a fee or contract basis',
      'Real estate activities with own or leased property',
    ],
  },
  {
    label: 'Professional, scientific and technical activities',
    children: [
      'Advertising activities',
      'All other professional, scientific and technical activities n.e.c.',
      'Architectural and engineering, and related technical consultancy activities',
      'Research and experimental development on natural sciences and engineering',
    ],
  },
  {
    label: 'Administrative and support service activities',
    children: [
      'Activities of employment placement agencies',
      'Combined facilities support activities',
      'Rental and leasing of motor vehicles',
      'Rental and leasing of other machinery, equipment and tangible goods',
      'Temporary employment agency activities and other human resource provisions',
      'Travel agency activities',
    ],
  },
  {
    label: 'Human health and social work activities',
    children: ['Hospital activities'],
  },
];

const normalizeIsicLabel = (label: string) => label.replace(/\s+/g, ' ').trim();

export const buildIsicSectorGroups = (
  labels: string[]
): { groups: HeatmapRowGroup[]; ungroupedLabels: string[] } => {
  const normalizedToOriginal = new Map<string, string>();

  labels.forEach(label => {
    const normalized = normalizeIsicLabel(label);
    if (!normalized || normalizedToOriginal.has(normalized)) return;
    normalizedToOriginal.set(normalized, label);
  });

  const assignedLabels = new Set<string>();
  const groups = ISIC_PARENT_GROUP_DEFINITIONS.map(group => {
    const childKeys = group.children
      .map(normalizeIsicLabel)
      .map(normalized => normalizedToOriginal.get(normalized))
      .filter((label): label is string => Boolean(label))
      .sort((a, b) => a.localeCompare(b, 'en'));

    childKeys.forEach(label => assignedLabels.add(label));

    return {
      label: group.label,
      childKeys,
    };
  }).filter(group => group.childKeys.length > 0);

  const ungroupedLabels = Array.from(normalizedToOriginal.values())
    .filter(label => !assignedLabels.has(label))
    .sort((a, b) => a.localeCompare(b, 'en'));

  return { groups, ungroupedLabels };
};
