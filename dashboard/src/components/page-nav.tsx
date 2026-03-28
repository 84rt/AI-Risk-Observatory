'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function PageNav() {
  const pathname = usePathname();

  const links = [
    { href: '/', label: 'Home' },
    { href: '/data', label: 'Data' },
    { href: '/about', label: 'About' },
  ];

  return (
    <nav className="border-b border-slate-200/70 bg-[#f6f3ef]">
      <div className="mx-auto max-w-7xl px-6 flex items-center gap-6">
        <Link href="/" className="py-3 font-semibold text-slate-900 hover:text-slate-700">
          AI Risk Observatory
        </Link>
        <div className="h-4 w-px bg-slate-300" />
        {links.map(link => {
          const isActive =
            link.href === '/'
              ? pathname === '/'
              : pathname.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`relative py-3 text-sm font-medium transition-colors ${
                isActive
                  ? 'text-slate-900'
                  : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              {link.label}
              {isActive && (
                <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-amber-500" />
              )}
            </Link>
          );
        })}
        <div className="ml-auto flex items-center gap-2 text-[10px] sm:text-[11px]">
          <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 font-semibold uppercase tracking-[0.14em] text-amber-800">
            Beta v0.8.1
          </span>
          <span className="whitespace-nowrap font-medium text-slate-500">
            Updated: 28.03.2026
          </span>
        </div>
      </div>
    </nav>
  );
}
