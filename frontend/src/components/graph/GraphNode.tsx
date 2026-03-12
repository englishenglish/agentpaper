'use client';

import React, { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';

interface GraphNodeData {
  label: string;
  nodeType: string;
  color: string;
}

export const GraphNodeComponent = memo(function GraphNodeComponent({
  data,
  selected,
}: NodeProps<GraphNodeData>) {
  return (
    <div
      style={{ borderColor: data.color }}
      className={`
        relative px-3 py-2 rounded-xl border-2 bg-card shadow-sm cursor-pointer
        transition-all duration-150 min-w-[80px] max-w-[160px] text-center
        ${selected ? 'ring-2 ring-offset-1 ring-offset-background shadow-lg scale-105' : 'hover:shadow-md hover:scale-102'}
      `}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground/50 !w-2 !h-2" />

      {/* Type badge */}
      <div
        style={{ backgroundColor: `${data.color}20`, color: data.color }}
        className="text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded-full mb-1 inline-block"
      >
        {data.nodeType}
      </div>

      {/* Label */}
      <p className="text-xs font-medium leading-tight line-clamp-2 text-foreground">
        {data.label}
      </p>

      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground/50 !w-2 !h-2" />
    </div>
  );
});
