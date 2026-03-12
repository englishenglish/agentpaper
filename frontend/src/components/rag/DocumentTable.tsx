'use client';

import React, { useRef } from 'react';
import { useRagStore } from '@/store/ragStore';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Trash2, Upload, RefreshCw, FileText, Loader2 } from 'lucide-react';
import { formatDate } from '@/lib/utils';
import type { EmbeddingStatus } from '@/types';

const STATUS_CONFIG: Record<string, { label: string; variant: 'success' | 'info' | 'warning' | 'destructive' | 'secondary' }> = {
  completed: { label: '已完成', variant: 'success' },
  done: { label: '已完成', variant: 'success' },
  processing: { label: '处理中', variant: 'info' },
  pending: { label: '等待中', variant: 'warning' },
  failed: { label: '失败', variant: 'destructive' },
  error: { label: '错误', variant: 'destructive' },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? { label: status, variant: 'secondary' as const };
  return <Badge variant={cfg.variant}>{cfg.label}</Badge>;
}

interface DocumentTableProps {
  dbId: string;
}

export function DocumentTable({ dbId }: DocumentTableProps) {
  const { documents, deleteDocument, uploadAndAddDocument, fetchDocuments, isUploading } = useRagStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docs = documents[dbId] ?? [];

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadAndAddDocument(dbId, file);
    e.target.value = '';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Table header toolbar */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-muted-foreground">
          {docs.length} 个文档
        </h3>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 text-xs"
            onClick={() => fetchDocuments(dbId)}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            刷新
          </Button>
          <Button
            size="sm"
            className="h-8 gap-1.5 text-xs"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Upload className="w-3.5 h-3.5" />
            )}
            上传文档
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.txt,.md,.docx,.csv,.json"
            onChange={handleUpload}
          />
        </div>
      </div>

      {/* Table */}
      {docs.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center py-12 border border-dashed rounded-xl">
          <FileText className="w-10 h-10 text-muted-foreground/40" />
          <div>
            <p className="text-sm font-medium text-muted-foreground">暂无文档</p>
            <p className="text-xs text-muted-foreground/60 mt-1">点击「上传文档」添加文件</p>
          </div>
        </div>
      ) : (
        <div className="border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/40">
                <th className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground">文档名称</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground">类型</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground">状态</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-muted-foreground">上传时间</th>
                <th className="text-right px-4 py-2.5 text-xs font-medium text-muted-foreground">操作</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((doc, idx) => (
                <tr
                  key={doc.file_id}
                  className={`border-b last:border-0 hover:bg-muted/30 transition-colors ${
                    idx % 2 === 0 ? '' : 'bg-muted/10'
                  }`}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
                      <span className="font-medium truncate max-w-[200px]" title={doc.filename}>
                        {doc.filename}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-muted-foreground uppercase">
                      {doc.file_type || doc.filename.split('.').pop() || '—'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={doc.status} />
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {doc.created_at ? formatDate(doc.created_at) : '—'}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 hover:text-destructive"
                      onClick={() => deleteDocument(dbId, doc.file_id)}
                      title="删除文档"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
