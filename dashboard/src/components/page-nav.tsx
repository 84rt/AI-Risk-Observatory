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
          <Link href="/" className="py-5 text-lg font-bold tracking-tighter text-primary uppercase">
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
                  className={`relative py-5 text-[11px] font-bold uppercase tracking-[0.15em] transition-colors ${
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
          <div className="inline-flex flex-col items-end rounded-full border border-border bg-secondary/80 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-muted shadow-sm">
            <span className="inline-flex items-center gap-2 text-muted">
              <span className="text-accent">●</span>
              Beta v0.9.6
            </span>
            <span className="text-[9px] tracking-[0.18em] text-muted-foreground">
              Updated: 31.03.2026
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}
