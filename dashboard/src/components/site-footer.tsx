'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

type FooterLink = {
  href: string;
  label: string;
  external?: boolean;
};

const primaryLinks: FooterLink[] = [
  { href: '/', label: 'Home' },
  { href: '/data', label: 'Data Dashboard' },
  { href: '/about', label: 'Methodology' },
];

const aboutSectionLinks: FooterLink[] = [
  { href: '/about#dataset', label: 'Dataset' },
  { href: '/about#pipeline', label: 'Pipeline' },
  { href: '/about#examples', label: 'Examples' },
  { href: '/about#taxonomies', label: 'Taxonomies' },
  { href: '/about#quality-controls', label: 'Quality Controls' },
  { href: '/about#baseline-analysis', label: 'Baseline Analysis' },
];

const dashboardSectionLinks: FooterLink[] = [
  { href: '/data#risk', label: 'Risk' },
  { href: '/data#adoption', label: 'Adoption' },
  { href: '/data#vendors', label: 'Vendors' },
  { href: '/data#signal-quality', label: 'Signal Quality' },
];

const resourceLinks: FooterLink[] = [
  {
    href: 'https://github.com/84rt/AI-Risk-Observatory',
    label: 'GitHub Repository',
    external: true,
  },
  {
    href: '/about',
    label: 'Report (WIP)',
  },
];

function FooterLinkList({
  title,
  links,
}: {
  title: string;
  links: FooterLink[];
}) {
  return (
    <div>
      <h2 className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">{title}</h2>
      <ul className="mt-4 space-y-3">
        {links.map(link => (
          <li key={link.href}>
            {link.external ? (
              <a
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground transition-colors hover:text-primary"
              >
                {link.label}
              </a>
            ) : (
              <Link
                href={link.href}
                className="text-sm text-muted-foreground transition-colors hover:text-primary"
              >
                {link.label}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SiteFooter() {
  const pathname = usePathname();

  const pageSectionLinks = pathname === '/about'
    ? aboutSectionLinks
    : pathname === '/data'
      ? dashboardSectionLinks
      : null;
  const pageSectionTitle = pathname === '/data' ? 'Dashboard Views' : 'On This Page';
  const footerGridClassName = pageSectionLinks
    ? 'mx-auto grid max-w-7xl gap-12 px-6 py-12 lg:grid-cols-[minmax(0,1.25fr)_repeat(3,minmax(0,1fr))]'
    : 'mx-auto grid max-w-7xl gap-12 px-6 py-12 lg:grid-cols-[minmax(0,1.25fr)_repeat(2,minmax(0,1fr))]';

  return (
    <footer className="border-t border-border bg-secondary">
      <div className={footerGridClassName}>
        <div className="max-w-sm">
          <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted">
            AI Risk Observatory
          </p>
          <p className="mt-4 text-sm leading-6 text-muted-foreground">
            Monitoring AI adoption, risk, vendor concentration, and disclosure blind spots across UK public-company reporting.
          </p>
          <p className="mt-6 text-[10px] uppercase tracking-widest text-muted-foreground">
            &copy; {new Date().getFullYear()} AI Risk Observatory
          </p>
          <div className="mt-6 space-y-1 text-[11px] leading-tight text-muted-foreground">
            <p>Last updated 10 April 2026</p>
            <p>
              Currently in public beta.{' '}
              <a
                href="https://forms.gle/qgwZSUPrMhxeTxcEA"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2 hover:text-primary transition-colors"
              >
                Share your feedback.
              </a>
            </p>
          </div>
        </div>

        <FooterLinkList title="Pages" links={primaryLinks} />
        {pageSectionLinks ? <FooterLinkList title={pageSectionTitle} links={pageSectionLinks} /> : null}
        <FooterLinkList title="Resources" links={resourceLinks} />
      </div>
    </footer>
  );
}
