// Mock Data Generator for AIRO Dashboard
// Simulating ~1000 firms over 5 years (2020-2024)

export type RiskCategory = 
  | 'operational_reliability'
  | 'security_malicious_use'
  | 'legal_regulatory_compliance'
  | 'workforce_human_capital'
  | 'societal_ethical_reputational'
  | 'frontier_systemic';

export type MockFirmYear = {
  firm_id: string;
  firm_name: string;
  sector: string;
  year: number;
  
  // AI Status
  ai_mentioned: boolean;
  frontier_ai_mentioned: boolean;
  
  // Risk Profile
  dominant_risk: RiskCategory | null;
  risk_mentions_count: number;
  
  // Governance
  governance_maturity: 'none' | 'basic' | 'intermediate' | 'advanced';
  mitigation_gap_score: number; // 0 (good) to 1 (bad)
  
  // Quality
  specificity_level: 'boilerplate' | 'contextual' | 'concrete';
}

const SECTORS = ['Financials', 'Technology', 'Healthcare', 'Industrials', 'Energy', 'Consumer Discretionary'];
const RISKS: RiskCategory[] = [
  'operational_reliability', 
  'security_malicious_use', 
  'legal_regulatory_compliance',
  'workforce_human_capital',
  'societal_ethical_reputational',
  'frontier_systemic'
];

function generateMockData(): MockFirmYear[] {
  const data: MockFirmYear[] = [];
  
  // 1. Define Core "Named" Firms (for realism)
  const coreFirms = [
    { name: 'Barclays PLC', sector: 'Financials' },
    { name: 'HSBC Holdings', sector: 'Financials' },
    { name: 'NatWest Group', sector: 'Financials' },
    { name: 'Lloyds Banking Group', sector: 'Financials' },
    { name: 'Standard Chartered', sector: 'Financials' },
    { name: 'London Stock Exchange', sector: 'Financials' },
    
    { name: 'Relx PLC', sector: 'Technology' },
    { name: 'Sage Group', sector: 'Technology' },
    { name: 'Darktrace', sector: 'Technology' },
    { name: 'Computacenter', sector: 'Technology' },
    { name: 'Softcat', sector: 'Technology' },
    { name: 'MoveFast AI Ltd', sector: 'Technology' },
    
    { name: 'AstraZeneca', sector: 'Healthcare' },
    { name: 'GSK', sector: 'Healthcare' },
    { name: 'Smith & Nephew', sector: 'Healthcare' },
    { name: 'Hikma', sector: 'Healthcare' },
    
    { name: 'BP p.l.c.', sector: 'Energy' },
    { name: 'Shell PLC', sector: 'Energy' },
    { name: 'National Grid', sector: 'Energy' },
    { name: 'SSE', sector: 'Energy' },
    
    { name: 'Rolls-Royce', sector: 'Industrials' },
    { name: 'BAE Systems', sector: 'Industrials' },
    { name: 'Ashtead Group', sector: 'Industrials' },
    { name: 'Bunzl', sector: 'Industrials' },
    
    { name: 'Tesco PLC', sector: 'Consumer Discretionary' },
    { name: 'Sainsbury\'s', sector: 'Consumer Discretionary' },
    { name: 'Next PLC', sector: 'Consumer Discretionary' },
    { name: 'Whitbread', sector: 'Consumer Discretionary' },
  ];

  // 2. Generate Anonymous Firms to reach ~1000 total
  const totalTarget = 1000;
  const anonFirmsCount = totalTarget - coreFirms.length;
  
  const allFirms = [...coreFirms];
  
  for (let i = 0; i < anonFirmsCount; i++) {
    const sector = SECTORS[Math.floor(Math.random() * SECTORS.length)];
    // Generate a semi-realistic name like "Tech Corp 145" or "Industrial Group 88"
    const name = `${sector.split(' ')[0]} Group ${i + 100}`;
    allFirms.push({ name, sector });
  }

  // 3. Generate Data
  allFirms.forEach(firm => {
    for (let year = 2020; year <= 2024; year++) {
      // TREND: AI mentions increase over time
      const baseProb = (year - 2019) * 0.15; // 2020=15%, 2024=75%
      
      const isAiMentioned = Math.random() < baseProb;
      const isFrontier = isAiMentioned && (year >= 2023 ? Math.random() < 0.6 : Math.random() < 0.1);
      
      // Sector bias for risks
      let dominantRisk: RiskCategory = 'operational_reliability';
      const riskRoll = Math.random();
      
      if (firm.sector === 'Financials') {
        dominantRisk = riskRoll > 0.4 ? 'legal_regulatory_compliance' : 
                       riskRoll > 0.2 ? 'security_malicious_use' : 'operational_reliability';
      } else if (firm.sector === 'Healthcare') {
        dominantRisk = riskRoll > 0.3 ? 'legal_regulatory_compliance' : 'societal_ethical_reputational';
      } else if (firm.sector === 'Technology') {
        dominantRisk = isFrontier ? 'frontier_systemic' : 
                       riskRoll > 0.5 ? 'security_malicious_use' : 'workforce_human_capital';
      } else if (firm.sector === 'Energy') {
        dominantRisk = 'operational_reliability';
      } else {
        // Random distribution for others
        dominantRisk = RISKS[Math.floor(Math.random() * RISKS.length)];
      }

      // Governance
      let maturity: 'none' | 'basic' | 'intermediate' | 'advanced' = 'none';
      if (isAiMentioned) {
        if (firm.name === 'MoveFast AI Ltd') {
          maturity = 'none'; // The designated cowboy
        } else {
          // Larger firms (named ones) tend to have better governance
          const isCore = coreFirms.some(c => c.name === firm.name);
          const bonus = isCore ? 0.2 : 0;
          
          const maturityRoll = Math.random() + ((year - 2020) * 0.1) + bonus;
          
          if (maturityRoll > 0.85) maturity = 'advanced';
          else if (maturityRoll > 0.55) maturity = 'intermediate';
          else if (maturityRoll > 0.25) maturity = 'basic';
        }
      }

      // Calculate Gap Score
      const riskCount = isAiMentioned ? Math.floor(Math.random() * 25) + 1 : 0;
      let gapScore = 0;
      if (riskCount > 5) {
        const matScore = maturity === 'none' ? 0 : maturity === 'basic' ? 0.33 : maturity === 'intermediate' ? 0.66 : 1;
        gapScore = (1 - matScore) * (riskCount / 25); 
      }

      // Specificity
      let specificity: 'boilerplate' | 'contextual' | 'concrete' = 'boilerplate';
      if (isAiMentioned) {
        const specRoll = Math.random();
        // Frontier mentions tend to be more specific or totally boilerplate
        if (maturity === 'advanced' || isFrontier) {
           specificity = specRoll > 0.4 ? 'concrete' : 'contextual';
        } else if (maturity === 'intermediate') {
           specificity = specRoll > 0.6 ? 'contextual' : 'boilerplate';
        } else {
           specificity = specRoll > 0.9 ? 'contextual' : 'boilerplate';
        }
      }

      data.push({
        firm_id: firm.name.replace(/\s/g, ''),
        firm_name: firm.name,
        sector: firm.sector,
        year,
        ai_mentioned: isAiMentioned,
        frontier_ai_mentioned: isFrontier,
        dominant_risk: isAiMentioned ? dominantRisk : null,
        risk_mentions_count: riskCount,
        governance_maturity: maturity,
        mitigation_gap_score: gapScore,
        specificity_level: specificity
      });
    }
  });

  return data;
}

export const MOCK_DATA = generateMockData();
