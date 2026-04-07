'use client';

import { useMemo, useState } from 'react';

type ExampleChunkView = {
  chunkId: string;
  companyName: string;
  reportYear: number;
  chunkText: string;
  reportSections: string[];
  mentionTypes: string[];
  riskLabels: string[];
  adoptionTypes: string[];
  vendorTags: string[];
};

type Phase2Group = {
  id: 'risk' | 'adoption' | 'vendor';
  label: string;
  items: string[];
  className: string;
};

const formatTag = (value: string) => value
  .split('_')
  .filter(Boolean)
  .map(token => `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
  .join(' ');

const tagStyles: Record<Phase2Group['id'], string> = {
  risk: 'aisi-pill pill-red',
  adoption: 'aisi-pill pill-sky',
  vendor: 'aisi-pill pill-teal',
};

export default function ExampleBrowser({ exampleChunks }: { exampleChunks: ExampleChunkView[] }) {
  const normalizedList = useMemo(() => exampleChunks.slice(0, 3), [exampleChunks]);
  const [activeChunkId, setActiveChunkId] = useState(normalizedList[0]?.chunkId || '');
  const activeChunk = normalizedList.find(chunk => chunk.chunkId === activeChunkId) ?? normalizedList[0];

  if (!activeChunk) return null;

  const phase2Groups = ([
    { id: 'risk', label: 'Risk', items: activeChunk.riskLabels, className: tagStyles.risk },
    { id: 'adoption', label: 'Adoption', items: activeChunk.adoptionTypes, className: tagStyles.adoption },
    { id: 'vendor', label: 'Vendor', items: activeChunk.vendorTags, className: tagStyles.vendor },
  ] satisfies Phase2Group[]).filter(group => group.items.length > 0);

  return (
    <section className="border-b border-border bg-white">
      <div className="mx-auto max-w-7xl px-6 py-20">
        <div className="mb-12 max-w-3xl">
          <span className="aisi-tag">Annotation</span>
          <h2 className="aisi-h2 uppercase">Labeled Examples</h2>
          <p className="mt-4 text-lg text-muted">
            Browse a handful of annotated chunks. Pick an example to inspect the excerpt, its per-chunk metadata, and the Phase 2 taxonomy labels attached to it.
          </p>
        </div>

        <div className="border border-border bg-white">
          <div className="grid gap-px border-b border-border bg-border lg:grid-cols-3">
            {normalizedList.map(chunk => {
              const isActive = chunk.chunkId === activeChunkId;
              const chunkPhase2Count = chunk.riskLabels.length + chunk.adoptionTypes.length + chunk.vendorTags.length;

              return (
                <button
                  key={`tab-${chunk.chunkId}`}
                  type="button"
                  onClick={() => setActiveChunkId(chunk.chunkId)}
                  className={`flex min-h-[92px] flex-col gap-4 border-b-2 px-5 py-4 text-left transition ${
                    isActive
                      ? 'border-b-primary bg-secondary text-primary'
                      : 'border-b-transparent bg-white text-muted hover:bg-secondary/60 hover:text-primary'
                  }`}
                >
                  <p className="text-[15px] font-bold leading-tight text-primary">
                    {chunk.companyName}
                  </p>

                  <div className="mt-auto flex items-center justify-between gap-3">
                    <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                      {chunk.reportYear}
                    </p>
                    <span className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.14em] ${
                      isActive
                        ? 'border-primary bg-white text-primary'
                        : 'border-border bg-white text-muted-foreground'
                    }`}>
                      {chunkPhase2Count} tag{chunkPhase2Count === 1 ? '' : 's'}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          <div className="space-y-6 p-8">
            {phase2Groups.length > 0 && (
              <div>
                <div className="grid gap-3 md:grid-cols-3">
                  {phase2Groups.map(group => (
                    <div key={group.id} className="space-y-2">
                      <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                        {group.label}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {group.items.map(item => (
                          <span key={`${group.id}-${item}`} className={group.className}>
                            {formatTag(item)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="overflow-hidden border border-border bg-secondary/60">
              <div className="border-b border-border px-6 py-4">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
                  Full Excerpt Text
                </p>
              </div>
              <div className="max-h-[32rem] overflow-y-auto px-7 py-6">
                <p className="whitespace-pre-line text-[1.05rem] leading-8 text-primary/75">
                  &ldquo;{activeChunk.chunkText}&rdquo;
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
