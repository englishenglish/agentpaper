'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Database, Trash2, Pencil, FileText, RefreshCw, ChevronRight } from 'lucide-react';
import { useRagStore } from '@/store/ragStore';
import type { KnowledgeBase } from '@/types';
import { cn } from '@/lib/utils';

interface KnowledgeBaseCardProps {
  kb: KnowledgeBase;
  isSelected: boolean;
  onSelect: () => void;
}

export function KnowledgeBaseCard({ kb, isSelected, onSelect }: KnowledgeBaseCardProps) {
  const { deleteDatabase, updateDatabase, rebuildDatabase } = useRagStore();
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editName, setEditName] = useState(kb.name);
  const [editDesc, setEditDesc] = useState(kb.description);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async () => {
    if (!confirm(`确定要删除知识库「${kb.name}」吗？此操作不可撤销。`)) return;
    setIsDeleting(true);
    await deleteDatabase(kb.db_id);
  };

  const handleUpdate = async () => {
    await updateDatabase(kb.db_id, { name: editName, description: editDesc });
    setIsEditOpen(false);
  };

  const handleRebuild = async () => {
    if (!confirm(`确定要重建知识库「${kb.name}」的索引吗？`)) return;
    await rebuildDatabase(kb.db_id);
  };

  const fileCount = kb.row_count ?? kb.file_count ?? 0;

  return (
    <>
      <Card
        onClick={onSelect}
        className={cn(
          'cursor-pointer transition-all hover:shadow-md hover:border-primary/50 group',
          isSelected ? 'border-primary ring-1 ring-primary' : ''
        )}
      >
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <Database className="w-4 h-4 text-primary" />
              </div>
              <div className="min-w-0">
                <CardTitle className="text-sm font-semibold truncate">{kb.name}</CardTitle>
                <p className="text-xs text-muted-foreground truncate mt-0.5">{kb.description}</p>
              </div>
            </div>
            <ChevronRight className={cn(
              'w-4 h-4 text-muted-foreground shrink-0 transition-transform',
              isSelected ? 'rotate-90 text-primary' : 'group-hover:translate-x-0.5'
            )} />
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <FileText className="w-3 h-3" />
                {fileCount} 个文档
              </span>
              <Badge variant="secondary" className="text-[10px] h-4 px-1.5">
                {kb.kb_type ?? 'chroma'}
              </Badge>
            </div>
            {/* Action buttons - visible on hover */}
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={(e) => { e.stopPropagation(); handleRebuild(); }}
                title="重建索引"
              >
                <RefreshCw className="w-3 h-3" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={(e) => { e.stopPropagation(); setIsEditOpen(true); }}
                title="编辑"
              >
                <Pencil className="w-3 h-3" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 hover:text-destructive"
                onClick={(e) => { e.stopPropagation(); handleDelete(); }}
                disabled={isDeleting}
                title="删除"
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>编辑知识库</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div className="space-y-1.5">
              <Label htmlFor="kb-name">名称</Label>
              <Input
                id="kb-name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="知识库名称"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="kb-desc">描述</Label>
              <Input
                id="kb-desc"
                value={editDesc}
                onChange={(e) => setEditDesc(e.target.value)}
                placeholder="知识库描述"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setIsEditOpen(false)}>取消</Button>
              <Button onClick={handleUpdate}>保存</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
