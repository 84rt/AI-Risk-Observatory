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
  risk: 'rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 font-semibold text-amber-700',
  adoption: 'rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 font-semibold text-sky-700',
  vendor: 'rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-700',
};

export default function ExampleBrowser({ exampleChunks }: { exampleChunks: ExampleChunkView[] }) {
  const normalizedList = useMemo(() => exampleChunks.slice(0, 5), [exampleChunks]);
  const [activeChunkId, setActiveChunkId] = useState(normalizedList[0]?.chunkId || '');
  const activeChunk = normalizedList.find(chunk => chunk.chunkId === activeChunkId) ?? normalizedList[0];

  if (!activeChunk) return null;

  return (
    <section className="mx-auto max-w-6xl px-6 pb-12">
      <div className="mb-6 max-w-3xl">
        <h2 className="text-2xl font-semibold text-slate-900">Labeled Examples</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 sm:text-base">
          Browse a handful of annotated chunks. Click a tab to load that chunk&apos;s Phase 1 mention type and Phase 2 taxonomy tags.
        </p>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
        <div className="mb-4 flex flex-wrap gap-3 border-b border-slate-100 pb-4">
          {normalizedList.map(chunk => {
            const isActive = chunk.chunkId === activeChunkId;
            return (
              <button
                key={`tab-${chunk.chunkId}`}
                type="button"
                onClick={() => setActiveChunkId(chunk.chunkId)}
                className={`rounded-lg border px-4 py-2 text-sm font-semibold transition ${
                  isActive
                    ? 'border-amber-300 bg-amber-50 text-slate-900 shadow-inner'
                    : 'border-transparent bg-slate-100/60 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                {chunk.companyName} · {chunk.reportYear}
              </button>
            );
          })}
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.3em] text-slate-500">
            <span>Phase 1</span>
            {(activeChunk.mentionTypes.length ? activeChunk.mentionTypes : ['none']).map(type => (
              <span key={`phase1-${type}`} className="rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-700">
                {formatTag(type)}
              </span>
            ))}
          </div>

          <div className="max-h-[220px] overflow-y-auto rounded-xl border border-slate-200/80 bg-slate-50/70 p-4 text-sm leading-relaxed text-slate-600">
            {activeChunk.chunkText}
          </div>

          <div className="grid gap-4 text-xs">
            {activeChunk.reportSections.length > 0 && (
              <div>
                <p className="text-[11px] font-semibold text-slate-500">Report Sections</p>
                <p className="mt-1 text-sm text-slate-700">
                  {activeChunk.reportSections.join(' · ')}
                </p>
              </div>
            )}
            {activeChunk.riskLabels.length > 0 && (
              <div>
                <p className="text-[11px] font-semibold text-slate-500">Phase 2 — Risk</p>
                <div className="mt-2 flex flex-wrap gap-2">
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
                <p className="text-[11px] font-semibold text-slate-500">Phase 2 — Adoption</p>
                <div className="mt-2 flex flex-wrap gap-2">
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
                <p className="text-[11px] font-semibold text-slate-500">Phase 2 — Vendor</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {activeChunk.vendorTags.map(tag => (
                    <span key={`vendor-${tag}`} className={tagStyles.vendor}>
                      {formatTag(tag)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="text-[11px] font-medium uppercase tracking-[0.3em] text-slate-400">
            {activeChunk.companyName} · {activeChunk.reportYear}
          </div>
        </div>
      </div>
    </section>
  );
}
