import { notFound } from 'next/navigation';

import { ReportFigureRenderer } from '@/components/report-figure-renderer';
import { getReportFigure } from '@/lib/report-figures';

export const dynamic = 'force-dynamic';

type PageProps = {
  params: Promise<{
    figureId: string;
  }>;
};

export default async function ReportFigurePage({ params }: PageProps) {
  const { figureId } = await params;
  const figure = getReportFigure(figureId);

  if (!figure) {
    notFound();
  }

  return (
    <main className="bg-white">
      <ReportFigureRenderer figure={figure} mode="export" />
    </main>
  );
}
