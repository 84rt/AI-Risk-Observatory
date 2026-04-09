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
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-10 gap-y-3 px-6 py-4 sm:py-0">
        <Link href="/" className="py-1 text-lg font-bold tracking-tight text-primary sm:py-5">
          AI Risk Observatory
        </Link>
        <div className="flex w-full flex-wrap items-center gap-x-6 gap-y-2 pb-1 sm:w-auto sm:gap-x-8 sm:pb-0">
          {links.map(link => {
            const isActive =
              link.href === '/'
                ? pathname === '/'
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative py-1 text-sm font-semibold tracking-normal transition-colors sm:py-5 ${
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
    </nav>
  );
}
