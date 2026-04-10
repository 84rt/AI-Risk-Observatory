'use client';

import { useState, type ReactNode } from 'react';

export function CollapsibleSection({
  title,
  children,
  variant = 'default',
  open: controlledOpen,
  onToggle,
}: {
  title: string;
  children: ReactNode;
  variant?: 'default' | 'faq' | 'settings';
  open?: boolean;
  onToggle?: () => void;
}) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = controlledOpen !== undefined;
  const open = isControlled ? controlledOpen : internalOpen;
  const toggle = isControlled ? onToggle! : () => setInternalOpen(prev => !prev);

  if (variant === 'settings') {
    return (
      <div className="border-b border-border last:border-b-0">
        <button
          type="button"
          onClick={toggle}
          className="flex w-full items-center justify-between px-5 py-3 text-left text-[11px] font-bold uppercase tracking-[0.12em] text-muted-foreground hover:text-primary transition-colors"
          aria-expanded={open}
        >
          <span>{title}</span>
          <svg
            width="13"
            height="13"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
            className={`shrink-0 ml-3 transition-transform ${open ? 'rotate-180' : ''}`}
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
          <div className="px-5 pb-5">
            {children}
          </div>
        )}
      </div>
    );
  }

  if (variant === 'faq') {
    return (
      <div className="border-b border-border last:border-b-0">
        <button
          type="button"
          onClick={toggle}
          className="flex w-full items-center justify-between py-4 text-left text-sm font-semibold text-primary hover:text-accent transition-colors"
          aria-expanded={open}
        >
          <span>{title}</span>
          <svg
            width="14"
            height="14"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
            className={`shrink-0 ml-3 transition-transform text-muted-foreground ${open ? 'rotate-180' : ''}`}
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
          <div className="pb-5 text-sm leading-relaxed text-slate-700">
            {children}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <button
        type="button"
        onClick={toggle}
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
