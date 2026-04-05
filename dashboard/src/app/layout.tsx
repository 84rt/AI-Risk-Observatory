import type { Metadata } from "next";
import { Onest } from "next/font/google";
import { PageNav } from "@/components/page-nav";
import { SiteFooter } from "@/components/site-footer";
import "./globals.css";

const emojiFavicon = `data:image/svg+xml,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" font-size="52">📡</text>
  </svg>`,
)}`;

const onest = Onest({
  variable: "--font-onest",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Risk Observatory",
  description: "Monitoring AI adoption and risk in UK Public Companies",
  icons: {
    icon: emojiFavicon,
    shortcut: emojiFavicon,
    apple: emojiFavicon,
  },
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
        <PageNav />
        <main className="flex-1">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
