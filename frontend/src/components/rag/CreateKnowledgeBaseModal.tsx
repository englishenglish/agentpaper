'use client';

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Plus } from 'lucide-react';
import { useRagStore } from '@/store/ragStore';

export function CreateKnowledgeBaseModal() {
  const { createDatabase, isLoading, buildOptions } = useRagStore();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [buildMethod, setBuildMethod] = useState('default_chunk');
  const [retrievalMethod, setRetrievalMethod] = useState('rag');

  const handleCreate = async () => {
    if (!name.trim()) return;
    await createDatabase({
      database_name: name.trim(),
      description: description.trim() || `${name.trim()} 知识库`,
      additional_params: { build_method: buildMethod, retrieval_method: retrievalMethod },
    });
    setOpen(false);
    setName('');
    setDescription('');
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="gap-2">
          <Plus className="w-4 h-4" />
          新建知识库
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>新建知识库</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 pt-2">
          <div className="space-y-1.5">
            <Label htmlFor="new-name">名称 *</Label>
            <Input
              id="new-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例如：深度学习论文库"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="new-desc">描述</Label>
            <Input
              id="new-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="知识库用途说明（可选）"
            />
          </div>

          {buildOptions && (
            <>
              <div className="space-y-1.5">
                <Label>切片方式</Label>
                <div className="flex gap-2 flex-wrap">
                  {buildOptions.build_methods.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setBuildMethod(m.id)}
                      className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                        buildMethod === m.id
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'border-border text-muted-foreground hover:border-primary/50'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>检索方式</Label>
                <div className="flex gap-2 flex-wrap">
                  {buildOptions.retrieval_methods.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setRetrievalMethod(m.id)}
                      className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                        retrievalMethod === m.id
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'border-border text-muted-foreground hover:border-primary/50'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setOpen(false)}>取消</Button>
            <Button onClick={handleCreate} disabled={!name.trim() || isLoading}>
              {isLoading ? '创建中...' : '创建'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
