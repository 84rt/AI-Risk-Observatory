import type { Metadata } from "next";
import { Onest } from "next/font/google";
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
        <main className="flex-1">
          {children}
        </main>
      </body>
    </html>
  );
}
