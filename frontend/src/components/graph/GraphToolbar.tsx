'use client';

import React from 'react';
import { useGraphStore } from '@/store/graphStore';
import { useRagStore } from '@/store/ragStore';
import { Button } from '@/components/ui/button';
import { RefreshCw, ZoomIn, ZoomOut, Maximize2, Loader2 } from 'lucide-react';
import type { ReactFlowInstance } from 'reactflow';

interface GraphToolbarProps {
  rfInstance: ReactFlowInstance | null;
}

export function GraphToolbar({ rfInstance }: GraphToolbarProps) {
  const { databases } = useRagStore();
  const { dbId, isLoading, rawGraph, fetchGraph } = useGraphStore();

  const stats = rawGraph?.stats;
  const activeDb = databases.find((kb) => kb.db_id === dbId);
  const neo4jUrl = process.env.NEXT_PUBLIC_NEO4J_BROWSER_URL ?? 'http://localhost:7474/browser';

  return (
    <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
      {/* DB selector + summary */}
      <div className="bg-card border rounded-xl shadow-sm p-2 flex flex-col gap-2 min-w-[220px] max-w-[260px]">
        <p className="text-xs font-medium text-muted-foreground px-1">选择知识库</p>
        <div className="flex flex-col gap-1 max-h-44 overflow-y-auto pr-1">
          {databases.length === 0 && (
            <p className="text-xs text-muted-foreground px-1 py-2 text-center">暂无知识库</p>
          )}
          {databases.map((kb) => (
            <button
              key={kb.db_id}
              onClick={() => fetchGraph(kb.db_id)}
              disabled={isLoading}
              className={`text-left text-xs px-2 py-1.5 rounded-lg transition-colors ${
                dbId === kb.db_id
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-muted text-muted-foreground hover:text-foreground'
              }`}
              title={kb.description || kb.name}
            >
              <div className="font-medium leading-snug line-clamp-2">{kb.name}</div>
              {kb.description && (
                <div className="mt-0.5 text-[10px] opacity-80 line-clamp-1">
                  {kb.description}
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Active DB brief description */}
        {activeDb && (
          <div className="mt-1 rounded-lg bg-muted/70 px-2 py-1.5">
            <p className="text-[11px] font-semibold text-foreground truncate">{activeDb.name}</p>
            {activeDb.description && (
              <p className="mt-0.5 text-[10px] text-muted-foreground line-clamp-2">
                {activeDb.description}
              </p>
            )}
          </div>
        )}

        {/* Stats */}
        {stats && (
          <div className="border-t pt-2 space-y-2">
            <div className="grid grid-cols-3 gap-1 text-center">
              {[
                { label: '节点', value: stats.node_count },
                { label: '边', value: stats.edge_count },
                { label: '论文', value: stats.paper_count },
              ].map(({ label, value }) => (
                <div key={label}>
                  <p className="text-xs font-bold text-foreground">{value}</p>
                  <p className="text-[10px] text-muted-foreground">{label}</p>
                </div>
              ))}
            </div>

            {/* Neo4j helper */}
            {dbId && (
              <div className="mt-1 rounded-lg bg-muted/70 px-2 py-1.5">
                <p className="text-[10px] font-medium text-muted-foreground mb-1">
                  在 Neo4j Browser 中执行：
                </p>
                <pre className="bg-background/80 rounded px-2 py-1 text-[9px] leading-snug overflow-auto max-h-20">
{`MATCH (n:Entity {db_id: '${dbId}'})-[r:RELATION]->(m:Entity {db_id: '${dbId}'})
RETURN n,r,m
LIMIT 300;`}
                </pre>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-1 h-6 px-2 text-[10px]"
                  onClick={() => window.open(neo4jUrl, '_blank')}
                >
                  打开 Neo4j Browser
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Zoom controls */}
      <div className="bg-card border rounded-xl shadow-sm p-1.5 flex flex-col gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => rfInstance?.zoomIn()}
          title="放大"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => rfInstance?.zoomOut()}
          title="缩小"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => rfInstance?.fitView({ padding: 0.1 })}
          title="适应窗口"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </Button>
        {isLoading && (
          <div className="h-7 w-7 flex items-center justify-center">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
          </div>
        )}
      </div>
    </div>
  );
}
