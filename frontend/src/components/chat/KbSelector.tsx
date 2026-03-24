'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Database, Plus, X, ChevronDown, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/store/chatStore';
import { useRagStore } from '@/store/ragStore';

export function KbSelector() {
  const { settings, updateSettings } = useChatStore();
  const { databases, fetchDatabases } = useRagStore();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const panelRef = useRef<HTMLDivElement>(null);

  // Load databases once on mount
  useEffect(() => {
    fetchDatabases();
  }, [fetchDatabases]);

  // Close panel on outside click
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

  const selectedIds = settings.selectedDbIds ?? [];

  const selectedKbs = databases.filter((db) => selectedIds.includes(db.db_id));
  const filtered = databases.filter(
    (db) =>
      !selectedIds.includes(db.db_id) &&
      (query === '' ||
        db.name.toLowerCase().includes(query.toLowerCase()) ||
        db.description?.toLowerCase().includes(query.toLowerCase()))
  );

  const removeKb = (id: string) => {
    updateSettings({ selectedDbIds: selectedIds.filter((x) => x !== id) });
  };

  const addKb = (id: string) => {
    updateSettings({ selectedDbIds: [...selectedIds, id] });
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 min-h-[28px]">
      {/* Selected KB chips */}
      {selectedKbs.map((kb) => (
        <span
          key={kb.db_id}
          className="inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded-full text-xs font-medium
                     bg-primary/10 text-primary border border-primary/25 max-w-[180px]"
        >
          <Database className="w-3 h-3 shrink-0" />
          <span className="truncate" title={kb.name}>
            {kb.name}
          </span>
          <button
            onClick={() => removeKb(kb.db_id)}
            className="ml-0.5 rounded-full hover:bg-primary/20 p-0.5 transition-colors"
            title="移除"
          >
            <X className="w-2.5 h-2.5" />
          </button>
        </span>
      ))}

      {/* Add button + popover */}
      <div className="relative" ref={panelRef}>
        <button
          onClick={() => setOpen((v) => !v)}
          className={cn(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs transition-colors',
            open
              ? 'bg-muted text-foreground'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          )}
        >
          <Plus className="w-3 h-3" />
          添加知识库
          <ChevronDown className={cn('w-3 h-3 transition-transform', open && 'rotate-180')} />
        </button>

        {open && (
          <div
            className="absolute bottom-full mb-2 left-0 z-50 w-64 rounded-xl border border-border
                       bg-popover shadow-lg overflow-hidden animate-in fade-in-0 zoom-in-95"
          >
            {/* Search */}
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

            {/* List */}
            <div className="max-h-52 overflow-y-auto py-1">
              {filtered.length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-4">
                  {databases.length === 0 ? '暂无知识库' : '无匹配结果'}
                </p>
              ) : (
                filtered.map((db) => (
                  <button
                    key={db.db_id}
                    onClick={() => {
                      addKb(db.db_id);
                      setQuery('');
                    }}
                    className="w-full flex items-start gap-2 px-3 py-2 hover:bg-muted transition-colors text-left"
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
    </div>
  );
}
