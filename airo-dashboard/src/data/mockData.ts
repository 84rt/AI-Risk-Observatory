// Mock Data Generator for AIRO Dashboard
// Simulating ~100 firms over 5 years (2020-2024)

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
  const firms = [
    { name: 'Barclays PLC', sector: 'Financials' },
    { name: 'HSBC Holdings', sector: 'Financials' },
    { name: 'NatWest Group', sector: 'Financials' },
    { name: 'Relx PLC', sector: 'Technology' },
    { name: 'Sage Group', sector: 'Technology' },
    { name: 'Darktrace', sector: 'Technology' },
    { name: 'AstraZeneca', sector: 'Healthcare' },
    { name: 'GSK', sector: 'Healthcare' },
    { name: 'BP p.l.c.', sector: 'Energy' },
    { name: 'Shell PLC', sector: 'Energy' },
    { name: 'Rolls-Royce', sector: 'Industrials' },
    { name: 'BAE Systems', sector: 'Industrials' },
    { name: 'Tesco PLC', sector: 'Consumer Discretionary' },
    { name: 'Sainsbury\'s', sector: 'Consumer Discretionary' },
    { name: 'MoveFast AI Ltd', sector: 'Technology' }, // The Cowboy
  ];

  // Generate 5 years of data for each firm
  firms.forEach(firm => {
    for (let year = 2020; year <= 2024; year++) {
      // TREND: AI mentions increase over time
      const baseProb = (year - 2019) * 0.15; // 2020=15%, 2024=75%
      
      const isAiMentioned = Math.random() < baseProb;
      const isFrontier = isAiMentioned && (year >= 2023 ? Math.random() < 0.6 : Math.random() < 0.1);
      
      // Sector bias for risks
      let dominantRisk: RiskCategory = 'operational_reliability';
      if (firm.sector === 'Financials') dominantRisk = Math.random() > 0.5 ? 'security_malicious_use' : 'legal_regulatory_compliance';
      if (firm.sector === 'Healthcare') dominantRisk = 'legal_regulatory_compliance';
      if (firm.sector === 'Technology') dominantRisk = isFrontier ? 'frontier_systemic' : 'security_malicious_use';
      if (firm.sector === 'Energy') dominantRisk = 'operational_reliability';

      // Governance usually improves over time, but "MoveFast AI" stays bad
      let maturity: any = 'none';
      if (isAiMentioned) {
        if (firm.name === 'MoveFast AI Ltd') {
          maturity = 'none';
        } else {
          const maturityRoll = Math.random() + ((year - 2020) * 0.1);
          if (maturityRoll > 0.8) maturity = 'advanced';
          else if (maturityRoll > 0.5) maturity = 'intermediate';
          else if (maturityRoll > 0.2) maturity = 'basic';
        }
      }

      // Calculate Gap Score (High Risk count + Low Maturity = High Gap)
      const riskCount = isAiMentioned ? Math.floor(Math.random() * 20) + 1 : 0;
      let gapScore = 0;
      if (riskCount > 5) {
        const matScore = maturity === 'none' ? 0 : maturity === 'basic' ? 0.33 : maturity === 'intermediate' ? 0.66 : 1;
        gapScore = (1 - matScore) * (riskCount / 20); // Normalize roughly 0-1
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
        mitigation_gap_score: gapScore
      });
    }
  });

  return data;
}

export const MOCK_DATA = generateMockData();
