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
      <div className="mx-auto max-w-7xl px-6 py-3 flex items-center gap-6">
        <Link href="/" className="font-semibold text-slate-900 hover:text-slate-700">
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
              className={`text-sm font-medium transition-colors ${
                isActive
                  ? 'text-slate-900'
                  : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
