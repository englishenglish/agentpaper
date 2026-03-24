'use client';

import React, { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { knowledgeApi } from '@/lib/api';
import { Loader2, FileText } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface Chunk {
  id: string;
  content: string;
  metadata: Record<string, any>;
  chunk_order_index?: number;
}

interface DocumentChunksModalProps {
  isOpen: boolean;
  onClose: () => void;
  dbId: string;
  docId: string;
  filename: string;
}

export function DocumentChunksModal({
  isOpen,
  onClose,
  dbId,
  docId,
  filename,
}: DocumentChunksModalProps) {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && dbId && docId) {
      fetchChunks();
    } else {
      setChunks([]);
      setError(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, dbId, docId]);

  const fetchChunks = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await knowledgeApi.getDocumentContent(dbId, docId);
      if (res && res.lines) {
        setChunks(res.lines);
      } else {
        setChunks([]);
      }
    } catch (err: any) {
      console.error('Failed to fetch chunks', err);
      setError(err?.response?.data?.detail || err.message || '获取切片失败');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            <span className="truncate max-w-[500px]" title={filename}>
              {filename}
            </span>
            {!isLoading && chunks.length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {chunks.length} 个切片
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-hidden mt-2">
          {isLoading ? (
            <div className="flex items-center justify-center h-40">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-40 text-destructive text-sm">
              {error}
            </div>
          ) : chunks.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-muted-foreground">
              <FileText className="w-8 h-8 mb-2 opacity-20" />
              <p className="text-sm">该文档暂无切片数据</p>
            </div>
          ) : (
            <ScrollArea className="h-[60vh] pr-4">
              <div className="space-y-4">
                {chunks.map((chunk, idx) => (
                  <div
                    key={chunk.id || idx}
                    className="p-4 rounded-lg border bg-card text-sm space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                      <span className="font-mono">
                        #{chunk.chunk_order_index ?? idx + 1}
                      </span>
                      <span className="font-mono opacity-50 truncate max-w-[200px]" title={chunk.id}>
                        {chunk.id}
                      </span>
                    </div>
                    <div className="whitespace-pre-wrap text-foreground/90 leading-relaxed">
                      {chunk.content}
                    </div>
                    {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                      <div className="mt-3 pt-3 border-t border-border/50 flex flex-wrap gap-2">
                        {Object.entries(chunk.metadata).map(([k, v]) => {
                          if (['source', 'file_id', 'chunk_index', 'chunk_id', 'full_doc_id', 'paper_index'].includes(k)) return null;
                          return (
                            <Badge key={k} variant="outline" className="text-[10px] bg-muted/50">
                              <span className="opacity-70 mr-1">{k}:</span>
                              <span className="truncate max-w-[150px]" title={String(v)}>
                                {String(v)}
                              </span>
                            </Badge>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
