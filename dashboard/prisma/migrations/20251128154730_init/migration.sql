-- CreateTable
CREATE TABLE "mentions" (
    "mention_id" TEXT NOT NULL PRIMARY KEY,
    "firm_id" TEXT NOT NULL,
    "firm_name" TEXT NOT NULL,
    "sector" TEXT NOT NULL,
    "sector_code" TEXT,
    "report_year" INTEGER NOT NULL,
    "report_section" TEXT,
    "text_excerpt" TEXT NOT NULL,
    "page_number" INTEGER,
    "mention_type" TEXT NOT NULL,
    "ai_specificity" TEXT NOT NULL,
    "frontier_tech_flag" BOOLEAN NOT NULL DEFAULT false,
    "tier_1_category" TEXT,
    "tier_2_driver" TEXT,
    "specificity_level" TEXT NOT NULL,
    "materiality_signal" TEXT,
    "mitigation_mentioned" BOOLEAN NOT NULL DEFAULT false,
    "governance_maturity" TEXT,
    "confidence_score" REAL,
    "reasoning_summary" TEXT,
    "model_version" TEXT,
    "extraction_date" DATETIME,
    "review_status" TEXT NOT NULL DEFAULT 'unreviewed',
    "reviewer_notes" TEXT,
    CONSTRAINT "mentions_firm_id_report_year_fkey" FOREIGN KEY ("firm_id", "report_year") REFERENCES "firms" ("firm_id", "report_year") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "firms" (
    "firm_id" TEXT NOT NULL,
    "firm_name" TEXT NOT NULL,
    "sector" TEXT NOT NULL,
    "sector_code" TEXT,
    "report_year" INTEGER NOT NULL,
    "ai_mentioned" BOOLEAN NOT NULL DEFAULT false,
    "ai_risk_mentioned" BOOLEAN NOT NULL DEFAULT false,
    "frontier_ai_mentioned" BOOLEAN NOT NULL DEFAULT false,
    "total_ai_mentions" INTEGER NOT NULL DEFAULT 0,
    "total_ai_risk_mentions" INTEGER NOT NULL DEFAULT 0,
    "dominant_tier_1_category" TEXT,
    "tier_1_distribution" TEXT,
    "max_specificity_level" TEXT,
    "max_materiality_signal" TEXT,
    "has_ai_governance" BOOLEAN NOT NULL DEFAULT false,
    "max_governance_maturity" TEXT,
    "ai_in_principal_risks" BOOLEAN NOT NULL DEFAULT false,
    "specificity_ratio" REAL,
    "mitigation_gap_score" REAL,
    "last_updated" DATETIME,

    PRIMARY KEY ("firm_id", "report_year")
);
