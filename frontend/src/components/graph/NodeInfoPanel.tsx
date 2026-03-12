'use client';

import React from 'react';
import { useGraphStore } from '@/store/graphStore';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { X, Link2 } from 'lucide-react';
import { getNodeColor } from '@/lib/utils';

export function NodeInfoPanel() {
  const { selectedNode, rawGraph, selectNode } = useGraphStore();

  if (!selectedNode) return null;

  const color = getNodeColor(selectedNode.type);

  // Find connected nodes
  const connectedEdges = rawGraph?.edges.filter(
    (e) => e.source === selectedNode.id || e.target === selectedNode.id
  ) ?? [];

  const connectedNodeIds = new Set(
    connectedEdges.flatMap((e) => [e.source, e.target]).filter((id) => id !== selectedNode.id)
  );
  const connectedNodes = [...connectedNodeIds]
    .map((id) => rawGraph?.nodes[id])
    .filter(Boolean);

  return (
    <div className="absolute right-4 top-4 w-72 bg-card border rounded-2xl shadow-xl z-10 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b" style={{ borderLeftColor: color, borderLeftWidth: 3 }}>
        <div className="flex items-center gap-2 min-w-0">
          <Badge
            style={{ backgroundColor: `${color}20`, color }}
            className="text-[10px] border-0 shrink-0"
          >
            {selectedNode.type}
          </Badge>
          <span className="text-sm font-semibold truncate">{selectedNode.label}</span>
        </div>
        <button
          onClick={() => selectNode(null)}
          className="p-1 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors shrink-0"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4 max-h-[400px] overflow-y-auto">
        {/* Properties */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">属性</p>
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">ID</span>
              <span className="font-mono text-[10px] truncate max-w-[160px]" title={selectedNode.id}>
                {selectedNode.id}
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">标准化标签</span>
              <span className="truncate max-w-[160px]">{selectedNode.norm_label}</span>
            </div>
          </div>
        </div>

        {/* Connections */}
        {connectedNodes.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <Link2 className="w-3.5 h-3.5 text-muted-foreground" />
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  关联节点 ({connectedNodes.length})
                </p>
              </div>
              <div className="space-y-1.5">
                {connectedNodes.slice(0, 8).map((node) => {
                  if (!node) return null;
                  const edge = connectedEdges.find(
                    (e) => e.source === node.id || e.target === node.id
                  );
                  const nodeColor = getNodeColor(node.type);
                  return (
                    <div
                      key={node.id}
                      className="flex items-center gap-2 p-2 rounded-lg bg-muted/40 hover:bg-muted transition-colors cursor-pointer"
                      onClick={() => selectNode(node)}
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: nodeColor }}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium truncate">{node.label}</p>
                        <p className="text-[10px] text-muted-foreground">
                          {edge?.type ?? ''} · {node.type}
                        </p>
                      </div>
                    </div>
                  );
                })}
                {connectedNodes.length > 8 && (
                  <p className="text-xs text-muted-foreground text-center py-1">
                    还有 {connectedNodes.length - 8} 个关联节点
                  </p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
