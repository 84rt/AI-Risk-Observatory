import type { Metadata } from "next";
import { Onest } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const onest = Onest({
  variable: "--font-onest",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Risk Observatory",
  description: "Monitoring AI adoption and risk in UK Public Companies",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${onest.variable} antialiased bg-background text-foreground min-h-screen flex flex-col`}
      >
        <div className="border-b border-red-300/60 bg-red-100/80 px-6 py-2.5 text-center text-xs font-semibold uppercase tracking-[0.2em] text-red-900">
          WIP — Data and labels are in active iteration. Do not treat as final.
        </div>
        <nav className="border-b border-slate-200/70 bg-[#f6f3ef]">
          <div className="mx-auto max-w-7xl px-6 py-3 flex items-center gap-6">
            <Link href="/" className="font-semibold text-slate-900 hover:text-slate-700">
              AI Risk Observatory
            </Link>
            <div className="h-4 w-px bg-slate-300" />
            <Link href="/about" className="text-sm text-slate-500 hover:text-slate-900">
              About
            </Link>
            <Link href="/dataset" className="text-sm text-slate-500 hover:text-slate-900">
              Dataset
            </Link>
          </div>
        </nav>
        <main className="flex-1">
          {children}
        </main>
      </body>
    </html>
  );
}
