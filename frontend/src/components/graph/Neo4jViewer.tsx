'use client';

import React, { useEffect, useState } from 'react';
import { useRagStore } from '@/store/ragStore';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { AlertCircle } from 'lucide-react';

const NEO4J_BROWSER_URL = process.env.NEXT_PUBLIC_NEO4J_BROWSER_URL ?? 'http://localhost:7474/browser';

export function Neo4jViewer() {
  const { databases, fetchDatabases } = useRagStore();
  const [dbId, setDbId] = useState<string | null>(null);

  useEffect(() => {
    if (databases.length === 0) {
      fetchDatabases().catch(() => undefined);
    } else if (!dbId && databases.length > 0) {
      setDbId(databases[0].db_id);
    }
  }, [databases, dbId, fetchDatabases]);

  const activeDb = databases.find((kb) => kb.db_id === dbId) ?? null;

  const cypher = dbId
    ? `MATCH (n:Entity {db_id: '${dbId}'})-[r:RELATION]->(m:Entity {db_id: '${dbId}'})\nRETURN n,r,m\nLIMIT 300;`
    : '请选择一个知识库后，复制这里的 Cypher 到 Neo4j Browser 中执行。';

  return (
    <div className="w-full h-full flex gap-4">
      <div className="w-72 shrink-0 flex flex-col gap-4">
        <Card className="p-3 flex flex-col gap-2">
          <p className="text-xs font-medium text-muted-foreground mb-1">选择知识库</p>
          <select
            className="h-8 rounded-md border bg-background px-2 text-xs"
            value={dbId ?? ''}
            onChange={(e) => setDbId(e.target.value || null)}
          >
            <option value="">请选择知识库</option>
            {databases.map((kb) => (
              <option key={kb.db_id} value={kb.db_id}>
                {kb.name}
              </option>
            ))}
          </select>

          {activeDb && (
            <div className="mt-2 rounded-md bg-muted/70 px-2 py-1.5">
              <p className="text-[11px] font-semibold truncate">{activeDb.name}</p>
              {activeDb.description && (
                <p className="mt-0.5 text-[11px] text-muted-foreground line-clamp-2">
                  {activeDb.description}
                </p>
              )}
            </div>
          )}
        </Card>

        <Card className="p-3 flex flex-col gap-2">
          <p className="text-xs font-medium text-muted-foreground">在 Neo4j Browser 中执行</p>
          <pre className="mt-1 bg-muted rounded-md p-2 text-[11px] leading-snug overflow-auto max-h-40">
            {cypher}
          </pre>
          <Button
            variant="outline"
            size="sm"
            className="mt-1 text-xs"
            onClick={() => window.open(NEO4J_BROWSER_URL, '_blank')}
          >
            打开 Neo4j Browser
          </Button>
        </Card>

        <Card className="p-3 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5" />
          <div className="space-y-1">
            <p className="text-xs font-semibold">说明</p>
            <p className="text-[11px] text-muted-foreground leading-snug">
              我已经将知识图谱同步到 Neo4j。你可以在 Neo4j Browser 中使用上面的 Cypher 查询，并利用 Neo4j 自带的力导向布局查看图谱。
            </p>
          </div>
        </Card>
      </div>

      <div className="flex-1 h-full bg-muted/10 rounded-xl border overflow-hidden">
        <iframe
          src={NEO4J_BROWSER_URL}
          className="w-full h-full border-0"
          title="Neo4j Browser"
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        />
      </div>
    </div>
  );
}


