import sqlite3
import json
import os

def seed():
    # Ensure prisma directory exists
    os.makedirs('prisma', exist_ok=True)
    
    # Connect to the SQLite database inside the prisma/ folder
    # This matches where Prisma looks for "file:./dev.db" relative to schema.prisma
    # Since we run this from airo-dashboard/, and schema is in airo-dashboard/prisma/,
    # file:./dev.db in .env means airo-dashboard/dev.db IF .env is in airo-dashboard/.
    
    # WAIT. Prisma convention:
    # If .env has DATABASE_URL="file:./dev.db"
    # And schema.prisma is in prisma/schema.prisma
    # Then it usually resolves relative to schema.prisma -> prisma/dev.db
    
    # BUT we saw earlier that Prisma migration created "dev.db" in the root of airo-dashboard/ ?
    # Let's check where the file actually is.
    # Previous ls -R prisma showed prisma/dev.db
    # Previous ls showed dev.db in airo-dashboard/
    
    # Let's target the one Prisma IS using.
    # Based on "ls -R prisma" output, "dev.db" exists in "prisma/" folder.
    
    db_path = 'prisma/dev.db'
    print(f"Connecting to {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Insert Firm: Barclays
    print("Inserting Barclays...")
    cursor.execute("""
    INSERT OR REPLACE INTO firms (
        firm_id, firm_name, sector, report_year, 
        ai_mentioned, ai_risk_mentioned, frontier_ai_mentioned,
        total_ai_mentions, total_ai_risk_mentions,
        dominant_tier_1_category, tier_1_distribution,
        max_specificity_level, max_materiality_signal,
        has_ai_governance, max_governance_maturity,
        specificity_ratio, mitigation_gap_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'GB0031348658', 'Barclays PLC', 'Financials', 2024,
        1, 1, 1, 
        15, 10,
        'security_malicious_use', json.dumps({
            'operational_reliability': 2,
            'security_malicious_use': 5,
            'legal_regulatory_compliance': 3
        }),
        'concrete', 'high',
        1, 'intermediate',
        0.6, 0.2
    ))

    # 2. Insert Mentions for Barclays
    print("Inserting Mentions...")
    cursor.execute("""
    INSERT OR REPLACE INTO mentions (
        mention_id, firm_id, firm_name, sector, report_year, report_section,
        text_excerpt, mention_type, ai_specificity, frontier_tech_flag,
        tier_1_category, tier_2_driver, specificity_level, materiality_signal,
        mitigation_mentioned, governance_maturity, confidence_score, reasoning_summary
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'BARC-2024-001', 'GB0031348658', 'Barclays PLC', 'Financials', 2024, 'Principal Risks',
        'The increasing sophistication of AI-enabled fraud and cyber-attacks poses a material threat to our customer assets. We have observed a rise in deepfake-enabled social engineering attempts.',
        'risk_statement', 'ai_specific', 1,
        'security_malicious_use', 'deepfakes_synthetic_media', 'concrete', 'high',
        1, 'intermediate', 0.95, 'Explicit mention of AI-enabled fraud and deepfakes as a material threat.'
    ))

    cursor.execute("""
    INSERT OR REPLACE INTO mentions (
        mention_id, firm_id, firm_name, sector, report_year, report_section,
        text_excerpt, mention_type, ai_specificity, frontier_tech_flag,
        tier_1_category, tier_2_driver, specificity_level, materiality_signal,
        mitigation_mentioned, governance_maturity, confidence_score, reasoning_summary
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'BARC-2024-002', 'GB0031348658', 'Barclays PLC', 'Financials', 2024, 'Strategic Report',
        'We are deploying predictive machine learning models to enhance credit decisioning and reduce default rates.',
        'adoption_use_case', 'ai_specific', 0,
        None, None, 'concrete', None,
        0, None, 0.90, 'Clear description of ML deployment for credit scoring.'
    ))

    # 3. Insert Firm: MoveFast AI (The "Cowboy")
    print("Inserting MoveFast AI...")
    cursor.execute("""
    INSERT OR REPLACE INTO firms (
        firm_id, firm_name, sector, report_year, 
        ai_mentioned, ai_risk_mentioned, frontier_ai_mentioned,
        total_ai_mentions, total_ai_risk_mentions,
        dominant_tier_1_category, tier_1_distribution,
        max_specificity_level, max_materiality_signal,
        has_ai_governance, max_governance_maturity,
        specificity_ratio, mitigation_gap_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'GB0011223344', 'MoveFast AI Ltd', 'Technology', 2024,
        1, 1, 1, 
        50, 20,
        'frontier_systemic', None,
        'concrete', 'high',
        0, 'none',
        0.8, 1.0
    ))

    conn.commit()
    conn.close()
    print("Seed complete.")

if __name__ == "__main__":
    seed()
