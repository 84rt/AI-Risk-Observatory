'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function PageNav() {
  const pathname = usePathname();

  const links = [
    { href: '/', label: 'Home' },
    { href: '/data', label: 'Data Dashboard' },
    { href: '/about', label: 'Methodology' },
  ];

  return (
    <nav className="border-b border-border bg-white">
      <div className="mx-auto max-w-7xl px-6 flex items-center justify-between">
        <div className="flex items-center gap-10">
          <Link href="/" className="py-5 text-lg font-bold tracking-tight text-primary">
            AI Risk Observatory
          </Link>
          <div className="hidden md:flex items-center gap-8">
            {links.map(link => {
              const isActive =
                link.href === '/'
                  ? pathname === '/'
                  : pathname.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`relative py-5 text-sm font-semibold tracking-normal transition-colors ${
                    isActive
                      ? 'text-accent'
                      : 'text-muted-foreground hover:text-primary'
                  }`}
                >
                  {link.label}
                  {isActive && (
                    <span className="absolute inset-x-0 bottom-0 h-1 bg-accent" />
                  )}
                </Link>
              );
            })}
          </div>
        </div>
        
        <div className="flex items-center">
          <div className="inline-flex flex-col items-end border border-red-200 bg-red-50 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-accent shadow-sm">
            <span className="text-accent">
              Beta v0.9.9
            </span>
            <span className="text-[9px] tracking-[0.18em] text-accent/70">
              Updated: 04.04.2026
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}
