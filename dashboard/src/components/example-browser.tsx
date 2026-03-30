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

const formatTag = (value: string) => value
  .split('_')
  .filter(Boolean)
  .map(token => `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
  .join(' ');

const tagStyles = {
  risk: 'aisi-pill pill-red',
  adoption: 'aisi-pill pill-sky',
  vendor: 'aisi-pill pill-teal',
};

export default function ExampleBrowser({ exampleChunks }: { exampleChunks: ExampleChunkView[] }) {
  const normalizedList = useMemo(() => exampleChunks.slice(0, 5), [exampleChunks]);
  const [activeChunkId, setActiveChunkId] = useState(normalizedList[0]?.chunkId || '');
  const activeChunk = normalizedList.find(chunk => chunk.chunkId === activeChunkId) ?? normalizedList[0];

  if (!activeChunk) return null;

  return (
    <section className="border-b border-border bg-white">
      <div className="mx-auto max-w-7xl px-6 py-20">
        <div className="mb-12 max-w-3xl">
          <span className="aisi-tag">Annotation</span>
          <h2 className="aisi-h2 uppercase">Labeled Examples</h2>
          <p className="mt-4 text-lg text-muted">
            Browse a handful of annotated chunks. Click a tab to load that chunk&apos;s Phase 1 mention type and Phase 2 taxonomy tags.
          </p>
        </div>
        
        <div className="border border-border bg-white">
          <div className="flex flex-wrap border-b border-border bg-secondary p-1">
            {normalizedList.map(chunk => {
              const isActive = chunk.chunkId === activeChunkId;
              return (
                <button
                  key={`tab-${chunk.chunkId}`}
                  type="button"
                  onClick={() => setActiveChunkId(chunk.chunkId)}
                  className={`px-6 py-3 text-[10px] font-bold uppercase tracking-widest transition ${
                    isActive
                      ? 'bg-white text-primary'
                      : 'text-muted-foreground hover:bg-white/50 hover:text-primary'
                  }`}
                >
                  {chunk.companyName} · {chunk.reportYear}
                </button>
              );
            })}
          </div>

          <div className="p-8 flex flex-col gap-8">
            <div className="flex flex-wrap items-center gap-4 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              <span>Phase 1</span>
              <div className="flex flex-wrap gap-2">
                {(activeChunk.mentionTypes.length ? activeChunk.mentionTypes : ['none']).map(type => (
                  <span key={`phase1-${type}`} className="aisi-pill pill-slate">
                    {formatTag(type)}
                  </span>
                ))}
              </div>
            </div>

            <div className="border-l-4 border-accent bg-secondary p-8 text-lg leading-relaxed text-muted font-medium italic">
              &ldquo;{activeChunk.chunkText}&rdquo;
            </div>

            <div className="grid gap-8 sm:grid-cols-3 text-xs">
              {activeChunk.riskLabels.length > 0 && (
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground mb-3">Phase 2 — Risk</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {activeChunk.riskLabels.map(label => (
                      <span key={`risk-${label}`} className={tagStyles.risk}>
                        {formatTag(label)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {activeChunk.adoptionTypes.length > 0 && (
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground mb-3">Phase 2 — Adoption</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {activeChunk.adoptionTypes.map(type => (
                      <span key={`adoption-${type}`} className={tagStyles.adoption}>
                        {formatTag(type)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {activeChunk.vendorTags.length > 0 && (
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground mb-3">Phase 2 — Vendor</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {activeChunk.vendorTags.map(tag => (
                      <span key={`vendor-${tag}`} className={tagStyles.vendor}>
                        {formatTag(tag)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

