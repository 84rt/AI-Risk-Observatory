"""Visualization helpers for classifier workbench scripts."""


def visualize_all(results_list: list[tuple[dict, dict]], show_text: bool = True, max_text_len: int = 300):
    """
    Visualize side-by-side comparison of ALL classification fields for each chunk.

    Args:
        results_list: List of (chunk, llm_result) tuples
        show_text: Whether to show chunk text
        max_text_len: Max characters to show for text
    """
    for i, (chunk, llm_result) in enumerate(results_list):
        print("\n" + "█" * 90)
        print(f"  CHUNK {i}: {chunk.get('company_name', 'Unknown')} ({chunk.get('report_year', '?')})")
        print("█" * 90)

        # Chunk metadata
        print(f"\n📋 ID: {chunk.get('chunk_id', 'N/A')[:50]}...")
        print(f"📂 Section: {chunk.get('report_sections', ['N/A'])[0][:60]}")
        print(f"🔑 Keywords: {', '.join(chunk.get('matched_keywords', [])) or 'none'}")

        # Text preview
        if show_text:
            text = chunk.get('chunk_text', '')[:max_text_len]
            if len(chunk.get('chunk_text', '')) > max_text_len:
                text += "..."
            print(f"\n📝 TEXT:\n{'-'*90}")
            print(text)
            print(f"{'-'*90}")

        # Side-by-side comparison table
        print(f"\n{'─'*90}")
        print(f"{'FIELD':<25} │ {'HUMAN':<30} │ {'LLM':<30}")
        print(f"{'─'*90}")

        # Define all fields to compare
        fields = [
            ("mention_types", "mention_types", "mention_types"),
            ("adoption_types", "adoption_types", "adoption_types"),
            ("risk_taxonomy", "risk_taxonomy", "risk_taxonomy"),
            ("vendor_tags", "vendor_tags", "vendor_tags"),
            ("risk_substantiveness", "risk_substantiveness", "risk_substantiveness"),
        ]

        for field_name, human_key, llm_key in fields:
            # Get human value
            h_val = chunk.get(human_key, [])
            if isinstance(h_val, list):
                h_str = ", ".join(str(v) for v in h_val) if h_val else "—"
            elif h_val is None:
                h_str = "—"
            else:
                h_str = str(h_val)

            # Get LLM value
            l_val = llm_result.get(llm_key, [])
            if isinstance(l_val, list):
                # Handle enum values
                l_str = ", ".join(
                    str(v.value) if hasattr(v, 'value') else str(v)
                    for v in l_val
                ) if l_val else "—"
            elif l_val is None:
                l_str = "—"
            else:
                l_str = str(l_val)

            # Compare
            h_set = set(h_val) if isinstance(h_val, list) else {h_val}
            l_set = set(l_val) if isinstance(l_val, list) else {l_val}

            if h_set == l_set:
                match = "✅"
            elif h_set & l_set:
                match = "🟡"
            elif not h_val and not l_val:
                match = "⚪"  # Both empty
            else:
                match = "❌"

            print(f"{field_name:<25} │ {h_str:<30} │ {l_str:<30}   {match}")

        # Confidence + reasoning
        if llm_result.get('confidence'):
            print(f"\n🎯 LLM Confidence: {llm_result['confidence']:.2f}")
        if llm_result.get('reasoning'):
            reasoning = llm_result['reasoning'][:200]
            if len(llm_result.get('reasoning', '')) > 200:
                reasoning += "..."
            print(f"💭 LLM Reasoning: {reasoning}")

        print()


def visualize_summary_table(results_list: list[tuple[dict, dict]]):
    """
    Print a compact summary table of all chunks with match status for each field.
    """
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Human vs LLM Comparison")
    print("=" * 100)
    print(f"{'#':<4} {'Company':<25} {'Mention':<10} {'Adoption':<10} {'Risk':<10} {'Vendor':<10}")
    print("-" * 100)

    stats = {"mention": {"exact": 0, "partial": 0, "diff": 0},
             "adoption": {"exact": 0, "partial": 0, "diff": 0},
             "risk": {"exact": 0, "partial": 0, "diff": 0},
             "vendor": {"exact": 0, "partial": 0, "diff": 0}}

    def match_status(human_val, llm_val) -> str:
        h_set = set(human_val) if isinstance(human_val, list) else set()
        l_set = set(llm_val) if isinstance(llm_val, list) else set()
        # Convert enums to strings for comparison
        l_set = {str(v.value) if hasattr(v, 'value') else str(v) for v in l_set}

        if h_set == l_set:
            return "✅ EXACT"
        elif h_set & l_set:
            return "🟡 PART"
        elif not h_set and not l_set:
            return "⚪ EMPTY"
        else:
            return "❌ DIFF"

    for i, (chunk, llm_result) in enumerate(results_list):
        company = chunk.get('company_name', 'Unknown')[:23]

        m_status = match_status(chunk.get('mention_types', []), llm_result.get('mention_types', []))
        a_status = match_status(chunk.get('adoption_types', []), llm_result.get('adoption_types', []))
        r_status = match_status(chunk.get('risk_taxonomy', []), llm_result.get('risk_taxonomy', []))
        v_status = match_status(chunk.get('vendor_tags', []), llm_result.get('vendor_tags', []))

        # Update stats
        for field, status in [("mention", m_status), ("adoption", a_status),
                               ("risk", r_status), ("vendor", v_status)]:
            if "EXACT" in status or "EMPTY" in status:
                stats[field]["exact"] += 1
            elif "PART" in status:
                stats[field]["partial"] += 1
            else:
                stats[field]["diff"] += 1

        print(f"{i:<4} {company:<25} {m_status:<10} {a_status:<10} {r_status:<10} {v_status:<10}")

    # Print totals
    total = len(results_list)
    print("-" * 100)
    print(f"{'TOTALS':<30}", end="")
    for field in ["mention", "adoption", "risk", "vendor"]:
        exact_pct = stats[field]["exact"] / total * 100 if total > 0 else 0
        print(f"{exact_pct:>5.0f}% ✅  ", end="")
    print()
    print("=" * 100)

    # Legend
    print("\nLegend: ✅=Exact match  🟡=Partial overlap  ❌=Different  ⚪=Both empty")


def aggregate_by_report(results_list: list[tuple[dict, dict]]) -> dict:
    """Aggregate tags per report for human vs LLM."""
    reports: dict = {}
    for chunk, llm_result in results_list:
        key = (
            chunk.get("company_id") or chunk.get("company_name", "Unknown"),
            chunk.get("report_year", "Unknown"),
        )
        if key not in reports:
            reports[key] = {
                "company": chunk.get("company_name", "Unknown"),
                "year": chunk.get("report_year", "Unknown"),
                "human": {
                    "mention_types": set(),
                    "adoption_types": set(),
                    "risk_taxonomy": set(),
                    "vendor_tags": set(),
                },
                "llm": {
                    "mention_types": set(),
                    "adoption_types": set(),
                    "risk_taxonomy": set(),
                    "vendor_tags": set(),
                },
            }

        # Human aggregates
        for field in ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]:
            vals = chunk.get(field, []) or []
            if isinstance(vals, list):
                reports[key]["human"][field].update(vals)

        # LLM aggregates
        for field in ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]:
            vals = llm_result.get(field, []) or []
            if isinstance(vals, list):
                reports[key]["llm"][field].update(
                    [v.value if hasattr(v, "value") else v for v in vals]
                )

    return reports


def visualize_report_summary(results_list: list[tuple[dict, dict]]):
    """Visualize per-report aggregated tags for human vs LLM."""
    reports = aggregate_by_report(results_list)

    print("\n" + "=" * 80)
    print("PER-REPORT AGGREGATED TAGS: Human vs LLM")
    print("=" * 80)

    for (_, year), report in reports.items():
        print("\n" + "■" * 80)
        print(f"REPORT: {report['company']} ({year})")
        print("■" * 80)

        print(f"\n{'FIELD':<20} │ {'HUMAN':<28} │ {'LLM':<28}")
        print(f"{'─'*80}")
        for field in ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]:
            h = report["human"][field]
            l = report["llm"][field]
            h_str = ", ".join(sorted(str(v) for v in h)) if h else "—"
            l_str = ", ".join(sorted(str(v) for v in l)) if l else "—"
            print(f"{field:<20} │ {h_str:<28} │ {l_str:<28}")
