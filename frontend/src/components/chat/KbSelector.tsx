'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Database, ChevronDown, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/store/chatStore';
import { useRagStore } from '@/store/ragStore';

export function KbSelector() {
  const { activeSessionId, sessions, patchSession } = useChatStore();
  const { databases, fetchDatabases } = useRagStore();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const panelRef = useRef<HTMLDivElement>(null);

  const session = sessions.find((s) => s.id === activeSessionId);
  const selectedId = session?.selectedDbId ?? null;
  const kbBinding = session?.kbBinding ?? 'none';
  const locked = kbBinding === 'manual' || kbBinding === 'built';

  useEffect(() => {
    fetchDatabases();
  }, [fetchDatabases]);

  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
        setQuery('');
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [open]);

  const selectedKb = databases.find((db) => db.db_id === selectedId);

  const filtered = databases.filter(
    (db) =>
      query === '' ||
      db.name.toLowerCase().includes(query.toLowerCase()) ||
      db.description?.toLowerCase().includes(query.toLowerCase())
  );

  const pickKb = (id: string) => {
    if (!activeSessionId || locked) return;
    patchSession(activeSessionId, {
      selectedDbId: id,
      kbBinding: 'manual',
      enableWebSearch: false,
    });
    setQuery('');
    setOpen(false);
  };

  const clearKb = () => {
    if (!activeSessionId || locked) return;
    patchSession(activeSessionId, {
      selectedDbId: null,
      kbBinding: 'none',
    });
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 min-h-[28px]">
      {selectedKb && (
        <span
          className="inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded-full text-xs font-medium
                     bg-primary/10 text-primary border border-primary/25 max-w-[200px]"
        >
          <Database className="w-3 h-3 shrink-0" />
          <span className="truncate" title={selectedKb.name}>
            {selectedKb.name}
          </span>
          {!locked && (
            <button
              type="button"
              onClick={clearKb}
              className="ml-0.5 text-[10px] text-muted-foreground hover:text-foreground px-1 rounded"
              title="清除选择"
            >
              清除
            </button>
          )}
        </span>
      )}

      {!locked && (
        <div className="relative" ref={panelRef}>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className={cn(
              'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs transition-colors',
              open
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            )}
          >
            <Database className="w-3 h-3" />
            {selectedKb ? '更换知识库' : '选择知识库'}
            <ChevronDown className={cn('w-3 h-3 transition-transform', open && 'rotate-180')} />
          </button>

          {open && (
            <div
              className="absolute bottom-full mb-2 left-0 z-50 w-64 rounded-xl border border-border
                         bg-popover shadow-lg overflow-hidden animate-in fade-in-0 zoom-in-95"
            >
              <div className="p-2 border-b border-border">
                <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-muted text-xs">
                  <Search className="w-3 h-3 text-muted-foreground shrink-0" />
                  <input
                    autoFocus
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="搜索知识库..."
                    className="flex-1 bg-transparent outline-none text-foreground placeholder:text-muted-foreground"
                  />
                </div>
              </div>

              <div className="max-h-52 overflow-y-auto py-1">
                {filtered.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    {databases.length === 0 ? '暂无知识库' : '无匹配结果'}
                  </p>
                ) : (
                  filtered.map((db) => (
                    <button
                      key={db.db_id}
                      type="button"
                      onClick={() => pickKb(db.db_id)}
                      className={cn(
                        'w-full flex items-start gap-2 px-3 py-2 hover:bg-muted transition-colors text-left',
                        selectedId === db.db_id && 'bg-muted/80'
                      )}
                    >
                      <Database className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-xs font-medium truncate">{db.name}</p>
                        {db.description && (
                          <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                            {db.description}
                          </p>
                        )}
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {locked && (
        <span className="text-[10px] text-muted-foreground">
          {kbBinding === 'built' ? '已绑定联网创建的知识库' : '已绑定手动选择的知识库'}
        </span>
      )}
    </div>
  );
}
