'use client';

import { useState, type ReactNode } from 'react';

export function CollapsibleSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-border">
      <button
        type="button"
        onClick={() => setOpen(prev => !prev)}
        className="flex w-full items-center justify-between px-6 py-4 text-left text-sm font-bold uppercase tracking-widest text-primary hover:bg-secondary transition-colors"
        aria-expanded={open}
      >
        <span>{title}</span>
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
          className={`shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        >
          <path
            d="M3 6L8 11L13 6"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      {open && (
        <div className="border-t border-border px-6 py-6">
          {children}
        </div>
      )}
    </div>
  );
}
