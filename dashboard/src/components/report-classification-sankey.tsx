'use client';

import { ResponsiveSankey } from '@nivo/sankey';
import type { DefaultLink, DefaultNode, SankeyLinkDatum, SankeyNodeDatum } from '@nivo/sankey';
import type { ReportClassificationFlow } from '@/lib/golden-set';

const formatNumber = (value: number) => new Intl.NumberFormat('en-GB').format(value);

type SankeyFlowNode = DefaultNode & {
  name: string;
  displayLabel: string;
  stage: 'root' | 'gate' | 'phase1' | 'phase2';
  fill: string;
  reportCount: number;
};

type SankeyFlowLink = DefaultLink;

function NodeTooltip({ node }: { node: SankeyNodeDatum<SankeyFlowNode, SankeyFlowLink> }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-lg">
      <p className="font-bold text-slate-900">{formatNumber(node.reportCount)} reports</p>
      <p className="text-slate-500 mt-0.5">{node.name}</p>
    </div>
  );
}

function LinkTooltip({ link }: { link: SankeyLinkDatum<SankeyFlowNode, SankeyFlowLink> }) {
  const percentage = ((link.value / link.source.reportCount) * 100).toFixed(0);
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-lg">
      <p className="font-bold text-slate-900">
        {formatNumber(link.value)} reports ({percentage}%)
      </p>
      <p className="text-slate-500 mt-0.5">
        categorised as <span className="font-medium text-slate-700">{link.target.name}</span>
      </p>
    </div>
  );
}

export function ReportClassificationSankey({ flow }: { flow: ReportClassificationFlow }) {
  const sankeyData = {
    nodes: flow.nodes.map(node => ({
      id: node.id,
      name: node.name,
      displayLabel: `${node.name} (${formatNumber(node.reportCount)})`,
      stage: node.stage,
      fill: node.fill,
      reportCount: node.reportCount,
    })),
    links: flow.links.map(link => ({
      source: flow.nodes[link.source]?.id || '',
      target: flow.nodes[link.target]?.id || '',
      value: link.value,
    })),
  };

  return (
    <section className="rounded-[1.4rem] border border-slate-200/80 bg-white/90 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_6px_20px_rgba(0,0,0,0.03)] sm:p-6">
      <div className="max-w-3xl">
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
            Report Classification Flow
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">
            A Sankey-style view of the full pipeline: from the total reports examined, through to those with AI mentions,
            into specific categories (Adoption, Risk, Vendor, etc.), and finally into detailed tags.
          </p>
          <p className="mt-2 text-xs leading-relaxed text-slate-500">
            Widths represent actual report counts for each category. Because reports often carry multiple labels,
            flows overlap and the diagram expands downstream to show the true scale of each classification.
          </p>
      </div>

      <div className="mt-6 h-[1280px] w-full rounded-[1.1rem] border border-slate-200 bg-[linear-gradient(180deg,#fffdfa_0%,#fcfbf8_100%)] p-3 sm:h-[1280px] lg:h-[1280px]">
        <ResponsiveSankey<SankeyFlowNode, SankeyFlowLink>
          data={sankeyData}
          margin={{ top: 24, right: 280, bottom: 24, left: 44 }}
          align="start"
          // Strictly respect the order of nodes provided in the nodes array
          sort={(nodeA, nodeB) => {
            // Nivo nodes in the sort function are the internal layout nodes
            // which have an 'index' property matching the original array order
            return nodeA.index - nodeB.index;
          }}
          colors={node => node.fill}
          nodeOpacity={0.92}
          nodeHoverOpacity={1}
          nodeHoverOthersOpacity={0.2}
          nodeThickness={22}
          nodeSpacing={18}
          nodeInnerPadding={1}
          nodeBorderWidth={1}
          nodeBorderColor={{ from: 'color', modifiers: [['darker', 0.25]] }}
          nodeBorderRadius={3}
          linkOpacity={0.3}
          linkHoverOpacity={0.6}
          linkHoverOthersOpacity={0.08}
          linkBlendMode="multiply"
          enableLabels
          label="displayLabel"
          labelPosition="outside"
          labelPadding={16}
          labelTextColor="#334155"
          animate={false}
          nodeTooltip={NodeTooltip}
          linkTooltip={LinkTooltip}
        />
      </div>
    </section>
  );
}
