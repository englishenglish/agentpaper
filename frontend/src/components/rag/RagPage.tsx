'use client';

import React, { useEffect } from 'react';
import { useRagStore } from '@/store/ragStore';
import { KnowledgeBaseCard } from './KnowledgeBaseCard';
import { CreateKnowledgeBaseModal } from './CreateKnowledgeBaseModal';
import { DocumentTable } from './DocumentTable';
import { Separator } from '@/components/ui/separator';
import { Database, Loader2, AlertCircle } from 'lucide-react';

export function RagPage() {
  const {
    databases,
    selectedDbId,
    isLoading,
    error,
    fetchDatabases,
    fetchBuildOptions,
    selectDatabase,
  } = useRagStore();

  useEffect(() => {
    fetchDatabases();
    fetchBuildOptions();
  }, [fetchDatabases, fetchBuildOptions]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left panel: KB list */}
      <div className="w-80 shrink-0 border-r flex flex-col h-full bg-card">
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-primary" />
            <h2 className="font-semibold text-sm">知识库</h2>
            <span className="text-xs text-muted-foreground bg-muted rounded-full px-2 py-0.5">
              {databases.length}
            </span>
          </div>
          <CreateKnowledgeBaseModal />
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {isLoading && databases.length === 0 && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-xs">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}

          {!isLoading && databases.length === 0 && !error && (
            <div className="text-center py-12">
              <Database className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">暂无知识库</p>
              <p className="text-xs text-muted-foreground/60 mt-1">点击右上角创建</p>
            </div>
          )}

          {databases.map((kb) => (
            <KnowledgeBaseCard
              key={kb.db_id}
              kb={kb}
              isSelected={selectedDbId === kb.db_id}
              onSelect={() => selectDatabase(kb.db_id)}
            />
          ))}
        </div>
      </div>

      {/* Right panel: Document management */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {selectedDbId ? (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <div className="mb-4">
                <h2 className="text-lg font-semibold">
                  {databases.find((d) => d.db_id === selectedDbId)?.name ?? '文档管理'}
                </h2>
                <p className="text-sm text-muted-foreground mt-0.5">
                  {databases.find((d) => d.db_id === selectedDbId)?.description}
                </p>
              </div>
              <Separator className="mb-4" />
              <DocumentTable dbId={selectedDbId} />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center p-8">
            <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center">
              <Database className="w-8 h-8 text-muted-foreground/50" />
            </div>
            <div>
              <h3 className="font-semibold">选择一个知识库</h3>
              <p className="text-sm text-muted-foreground mt-1">
                从左侧选择知识库以查看和管理文档
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
