'use client';

import React, { useRef, useEffect, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Square, Globe, Paperclip } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/store/chatStore';
import { KbSelector } from './KbSelector';

interface ChatInputProps {
  onSend: (content: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, onStop, isGenerating, disabled }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { settings, updateSettings } = useChatStore();
  const [value, setValue] = React.useState('');

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || isGenerating) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t bg-background/95 backdrop-blur px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* Toolbar row */}
        <div className="flex items-center gap-2 mb-2">
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              'h-7 text-xs gap-1.5 rounded-full px-3',
              settings.enableWebSearch
                ? 'bg-primary/10 text-primary hover:bg-primary/20'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            )}
            onClick={() => updateSettings({ enableWebSearch: !settings.enableWebSearch })}
          >
            <Globe
              className={cn(
                'w-3.5 h-3.5',
                settings.enableWebSearch ? '' : 'opacity-60'
              )}
            />
            联网搜索
          </Button>

          <div className="flex items-center gap-1 ml-auto">
            <span className="text-xs text-muted-foreground">检索：</span>
            {(['rag', 'graphrag', 'both'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => updateSettings({ retrievalMode: mode })}
                className={cn(
                  'text-xs px-2 py-0.5 rounded-full transition-colors',
                  settings.retrievalMode === mode
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                {mode === 'rag' ? 'RAG' : mode === 'graphrag' ? 'GraphRAG' : 'Both'}
              </button>
            ))}
          </div>
        </div>

        {/* Selected knowledge base tags */}
        <div className="mb-2">
          <KbSelector />
        </div>

        {/* Input area */}
        <div className="relative flex items-end gap-2 rounded-2xl border border-input bg-card px-3 py-2 focus-within:ring-1 focus-within:ring-ring transition-shadow">
          <button className="p-1.5 text-muted-foreground hover:text-foreground transition-colors shrink-0 mb-0.5">
            <Paperclip className="w-4 h-4" />
          </button>

          <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="发送消息... (Shift+Enter 换行)"
            disabled={disabled}
            className="flex-1 border-0 bg-transparent focus-visible:ring-0 shadow-none resize-none min-h-[36px] max-h-[200px] py-1 text-sm"
            rows={1}
          />

          <div className="shrink-0 mb-0.5">
            {isGenerating ? (
              <Button
                onClick={onStop}
                size="icon"
                variant="destructive"
                className="h-8 w-8 rounded-xl"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
              </Button>
            ) : (
              <Button
                onClick={handleSend}
                size="icon"
                disabled={!value.trim() || disabled}
                className="h-8 w-8 rounded-xl"
              >
                <Send className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        </div>

        <p className="text-[10px] text-muted-foreground text-center mt-2">
          AI 可能会犯错，请核实重要信息
        </p>
      </div>
    </div>
  );
}
