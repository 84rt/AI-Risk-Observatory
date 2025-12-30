const { PrismaClient } = require('@prisma/client')

const prisma = new PrismaClient()

async function main() {
  // 1. Create a Firm (Barclays 2024)
  const barclays2024 = await prisma.firm.upsert({
    where: {
      firm_id_report_year: {
        firm_id: 'GB0031348658',
        report_year: 2024,
      },
    },
    update: {},
    create: {
      firm_id: 'GB0031348658',
      firm_name: 'Barclays PLC',
      sector: 'Financials',
      report_year: 2024,
      ai_mentioned: true,
      ai_risk_mentioned: true,
      frontier_ai_mentioned: true,
      total_ai_mentions: 15,
      total_ai_risk_mentions: 10,
      dominant_tier_1_category: 'security_malicious_use',
      tier_1_distribution: JSON.stringify({
        'operational_reliability': 2,
        'security_malicious_use': 5,
        'legal_regulatory_compliance': 3
      }),
      max_specificity_level: 'concrete',
      max_materiality_signal: 'high',
      has_ai_governance: true,
      max_governance_maturity: 'intermediate',
      specificity_ratio: 0.6,
      mitigation_gap_score: 0.2, // Low gap
    },
  })

  // 2. Add Mentions for Barclays
  // Risk Mention
  await prisma.mention.create({
    data: {
      mention_id: 'BARC-2024-001',
      firm_id: 'GB0031348658',
      firm_name: 'Barclays PLC',
      sector: 'Financials',
      report_year: 2024,
      report_section: 'Principal Risks',
      text_excerpt: 'The increasing sophistication of AI-enabled fraud and cyber-attacks poses a material threat to our customer assets. We have observed a rise in deepfake-enabled social engineering attempts.',
      mention_type: 'risk_statement',
      ai_specificity: 'ai_specific',
      frontier_tech_flag: true,
      tier_1_category: 'security_malicious_use',
      tier_2_driver: 'deepfakes_synthetic_media',
      specificity_level: 'concrete',
      materiality_signal: 'high',
      mitigation_mentioned: true,
      governance_maturity: 'intermediate',
      confidence_score: 0.95,
      reasoning_summary: 'Explicit mention of AI-enabled fraud and deepfakes as a material threat.',
    },
  })

  // Adoption Mention
  await prisma.mention.create({
    data: {
      mention_id: 'BARC-2024-002',
      firm_id: 'GB0031348658',
      firm_name: 'Barclays PLC',
      sector: 'Financials',
      report_year: 2024,
      report_section: 'Strategic Report',
      text_excerpt: 'We are deploying predictive machine learning models to enhance credit decisioning and reduce default rates.',
      mention_type: 'adoption_use_case',
      ai_specificity: 'ai_specific',
      frontier_tech_flag: false,
      specificity_level: 'concrete',
      mitigation_mentioned: false,
      confidence_score: 0.90,
      reasoning_summary: 'Clear description of ML deployment for credit scoring.',
    },
  })
  
  // 3. Create a "Cowboy" Firm (TechStartup 2024) - High Risk, Low Governance
  const techCo = await prisma.firm.upsert({
    where: {
      firm_id_report_year: {
        firm_id: 'GB0011223344',
        report_year: 2024,
      },
    },
    update: {},
    create: {
      firm_id: 'GB0011223344',
      firm_name: 'MoveFast AI Ltd',
      sector: 'Technology',
      report_year: 2024,
      ai_mentioned: true,
      ai_risk_mentioned: true,
      frontier_ai_mentioned: true,
      total_ai_mentions: 50,
      total_ai_risk_mentions: 20,
      dominant_tier_1_category: 'frontier_systemic',
      max_specificity_level: 'concrete',
      max_materiality_signal: 'high',
      has_ai_governance: false,
      max_governance_maturity: 'none',
      specificity_ratio: 0.8,
      mitigation_gap_score: 1.0, // MAX GAP
    },
  })

  console.log({ barclays2024, techCo })
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })















