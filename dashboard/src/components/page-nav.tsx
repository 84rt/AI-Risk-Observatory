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
    <nav className="border-b border-slate-200/70 bg-white">
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
                <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-slate-900" />
              )}
            </Link>
          );
        })}
        <div className="ml-auto flex items-center gap-2">
          <span className="rounded bg-slate-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
            BETA v0.8
          </span>
          <span className="text-[10px] font-medium text-slate-400">
            Last update: 27.03.2026
          </span>
        </div>
      </div>
    </nav>
  );
}
