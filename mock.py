import pandas as pd
import numpy as np
import random

def generate_mock_data():
    # Configuration
    companies = [
        {"name": "Barclays PLC", "sector": "Financials"},
        {"name": "HSBC Holdings", "sector": "Financials"},
        {"name": "Tesco PLC", "sector": "Consumer Staples"},
        {"name": "Unilever", "sector": "Consumer Goods"},
        {"name": "AstraZeneca", "sector": "Healthcare"},
        {"name": "GSK", "sector": "Healthcare"},
        {"name": "BP p.l.c.", "sector": "Energy"},
        {"name": "Pearson", "sector": "Consumer Discretionary"},
        {"name": "Relx", "sector": "Technology"},
        {"name": "Vodafone Group", "sector": "Telecommunications"},
        {"name": "Darktrace", "sector": "Technology"},
        {"name": "Rolls-Royce", "sector": "Industrials"}
    ]
    
    years = [2021, 2022, 2023, 2024, 2025]
    
    # The Taxonomy we defined
    categories = [
        "Societal & Systemic Harms",
        "Security & Malicious Use",
        "Workforce & Economic Displacement",
        "Critical Business & Reliability",
        "Legal & Regulatory Compliance",
        "Frontier Signal (Loss of Control)"
    ]

    data = []

    for company in companies:
        base_risk_profile = random.uniform(0.5, 1.5) # Some companies are more risk-averse
        
        for year in years:
            # Simulate the "AI Hype Cycle"
            # 2021: Low mentions
            # 2023: Spike in Generative AI interest
            # 2025: Maturity and Regulation
            
            time_multiplier = 1.0
            if year == 2021: time_multiplier = 0.2
            if year == 2022: time_multiplier = 0.4
            if year == 2023: time_multiplier = 1.8 # ChatGPT moment
            if year == 2024: time_multiplier = 2.5
            if year == 2025: time_multiplier = 3.0

            for category in categories:
                # Sector nuances (Making the data look "real")
                sector_weight = 1.0
                
                # Financials care about Cyber and Compliance
                if company['sector'] == "Financials":
                    if category == "Security & Malicious Use": sector_weight = 2.5
                    if category == "Legal & Regulatory Compliance": sector_weight = 2.0
                
                # Tech/Media cares about Copyright (Legal) and Reliability
                if company['sector'] == "Technology" or company['name'] == "Pearson":
                    if category == "Legal & Regulatory Compliance": sector_weight = 3.0
                    if category == "Critical Business & Reliability": sector_weight = 2.0

                # Healthcare cares about Reliability (Life or death)
                if company['sector'] == "Healthcare":
                    if category == "Critical Business & Reliability": sector_weight = 2.5
                
                # Frontier Signal is very rare
                if category == "Frontier Signal (Loss of Control)":
                    sector_weight = 0.05

                # Generate mention count (Poisson distribution for count data)
                # Lambda = Base * Time * Sector * Random Noise
                lambda_val = base_risk_profile * time_multiplier * sector_weight * 2
                count = np.random.poisson(lambda_val)

                # Generate a "Snippet" for the evidence explorer (only if count > 0)
                snippet = ""
                if count > 0:
                    snippet = f"Mock excerpt from {company['name']} {year} AR: We recognize the potential for {category.lower()} to materially impact operations..."

                row = {
                    "Company": company['name'],
                    "Sector": company['sector'],
                    "Year": year,
                    "Risk_Category": category,
                    "Mention_Count": count,
                    "Evidence_Snippet": snippet
                }
                data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = "airo_dashboard_mock_data.csv"
    df.to_csv(filename, index=False)
    print(f"Successfully generated {filename} with {len(df)} rows.")
    print(df.head())

if __name__ == "__main__":
    generate_mock_data()