'use client';

import React, { useEffect } from 'react';
import { GraphViewer } from './GraphViewer';
import { useRagStore } from '@/store/ragStore';
import { Network } from 'lucide-react';

export function GraphRagPage() {
  const { fetchDatabases, databases } = useRagStore();

  useEffect(() => {
    if (databases.length === 0) fetchDatabases();
  }, [databases.length, fetchDatabases]);

  return (
    <div className="w-full h-full flex flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-muted-foreground" />
          <div>
            <h1 className="text-sm font-semibold tracking-tight">GraphRAG 知识图谱</h1>
            <p className="text-[11px] text-muted-foreground">
              左侧选择知识库查看实体图谱，或复制 Cypher 到 Neo4j Browser 中进行高级分析。
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 rounded-xl border bg-background overflow-hidden">
        <GraphViewer />
      </div>
    </div>
  );
}
