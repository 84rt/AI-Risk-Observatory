'use client';

import { useEffect, useState } from 'react';

type TocItem = {
  id: string;
  label: string;
  level: 1 | 2;
};

const TOC_ITEMS: TocItem[] = [
  { id: 'overview', label: 'Overview', level: 1 },
  { id: 'data', label: '1. Data', level: 1 },
  { id: 'data-scope', label: 'Scope', level: 2 },
  { id: 'data-decisions', label: 'Decisions & Rationale', level: 2 },
  { id: 'data-acknowledgements', label: 'Data Provider Acknowledgment', level: 2 },
  { id: 'preprocessing', label: '2. Pre-processing', level: 1 },
  { id: 'preprocessing-approach', label: 'Chunking Approach', level: 2 },
  { id: 'preprocessing-results', label: 'Chunking Results', level: 2 },
  { id: 'processing', label: '3. Processing', level: 1 },
  { id: 'phase-1', label: 'Phase 1: Mention-Type Classification', level: 2 },
  { id: 'phase-2', label: 'Phase 2: Deep-Taxonomy Classification', level: 2 },
  { id: 'quality-assurance', label: '4. Quality Assurance', level: 1 },
  { id: 'footnotes', label: 'Footnotes', level: 1 },
];

export function MethodologyToc() {
  const [activeId, setActiveId] = useState<string>(TOC_ITEMS[0].id);

  useEffect(() => {
    const ids = TOC_ITEMS.map(item => item.id);

    const observer = new IntersectionObserver(
      entries => {
        // Find the topmost intersecting entry to avoid jumpy behaviour
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: '-8% 0px -80% 0px', threshold: 0 }
    );

    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <nav aria-label="Page contents" className="border-l border-border pl-5">
      <p className="mb-4 text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
        On this page
      </p>
      <ul className="space-y-0.5">
        {TOC_ITEMS.map(item => {
          const isActive = activeId === item.id;
          return (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                className={`flex items-center gap-2 rounded py-1 text-[12px] leading-snug transition-colors ${
                  item.level === 2 ? 'pl-3' : ''
                } ${
                  isActive
                    ? 'font-semibold text-accent'
                    : 'text-muted-foreground hover:text-primary'
                }`}
              >
                {item.label}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
