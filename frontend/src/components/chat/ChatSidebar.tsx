'use client';

import React, { useState } from 'react';
import { useChatStore } from '@/store/chatStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Plus,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Trash2,
  Pencil,
  Check,
  X,
} from 'lucide-react';
import { cn, formatDate } from '@/lib/utils';

export function ChatSidebar() {
  const {
    sessions,
    activeSessionId,
    isSidebarOpen,
    createSession,
    deleteSession,
    renameSession,
    setActiveSession,
    toggleSidebar,
  } = useChatStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');

  const handleStartEdit = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditValue(currentTitle);
  };

  const handleConfirmEdit = () => {
    if (editingId && editValue.trim()) {
      renameSession(editingId, editValue.trim());
    }
    setEditingId(null);
  };

  const handleCancelEdit = () => setEditingId(null);

  return (
    <TooltipProvider delayDuration={300}>
      <aside
        className={cn(
          'h-full border-r bg-card flex flex-col transition-all duration-300 ease-in-out shrink-0',
          isSidebarOpen ? 'w-64' : 'w-[52px]'
        )}
      >
        {/* Header */}
        <div className="p-2 flex items-center gap-2 border-b">
          {isSidebarOpen && (
            <Button
              onClick={createSession}
              className="flex-1 h-8 text-xs gap-1.5"
              variant="default"
            >
              <Plus className="w-3.5 h-3.5" />
              新建对话
            </Button>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={toggleSidebar}>
                {isSidebarOpen ? (
                  <PanelLeftClose className="w-4 h-4" />
                ) : (
                  <PanelLeftOpen className="w-4 h-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              {isSidebarOpen ? '收起侧边栏' : '展开侧边栏'}
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Collapsed: new chat icon */}
        {!isSidebarOpen && (
          <div className="p-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={createSession}
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">新建对话</TooltipContent>
            </Tooltip>
          </div>
        )}

        {/* Session List */}
        <div className="flex-1 overflow-y-auto py-2 px-1.5 space-y-0.5">
          {sessions.length === 0 && isSidebarOpen && (
            <p className="text-xs text-muted-foreground text-center py-8 px-4">
              点击「新建对话」开始
            </p>
          )}
          {sessions.map((session) => {
            const isActive = session.id === activeSessionId;
            const isEditing = editingId === session.id;

            return (
              <div
                key={session.id}
                onClick={() => !isEditing && setActiveSession(session.id)}
                className={cn(
                  'group relative flex items-center gap-2 rounded-lg cursor-pointer transition-colors',
                  isSidebarOpen ? 'px-2 py-1.5' : 'px-1.5 py-1.5 justify-center',
                  isActive
                    ? 'bg-secondary text-secondary-foreground'
                    : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                )}
              >
                <MessageSquare className="w-4 h-4 shrink-0" />

                {isSidebarOpen && (
                  <>
                    {isEditing ? (
                      <div className="flex-1 flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                        <Input
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleConfirmEdit();
                            if (e.key === 'Escape') handleCancelEdit();
                          }}
                          className="h-6 text-xs px-1.5 flex-1"
                          autoFocus
                        />
                        <button onClick={handleConfirmEdit} className="text-green-500 hover:text-green-400">
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button onClick={handleCancelEdit} className="text-muted-foreground hover:text-foreground">
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium truncate">{session.title}</p>
                          <p className="text-[10px] text-muted-foreground">
                            {formatDate(session.updatedAt)}
                          </p>
                        </div>
                        {/* Action buttons */}
                        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartEdit(session.id, session.title);
                            }}
                            className="p-1 rounded hover:bg-muted-foreground/20 text-muted-foreground hover:text-foreground"
                          >
                            <Pencil className="w-3 h-3" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteSession(session.id);
                            }}
                            className="p-1 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Footer info */}
        {isSidebarOpen && (
          <div className="p-3 border-t">
            <p className="text-[10px] text-muted-foreground text-center">
              {sessions.length} 个对话
            </p>
          </div>
        )}
      </aside>
    </TooltipProvider>
  );
}
