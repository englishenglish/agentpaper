'use client';

import React, { useEffect, useRef } from 'react';
import { MessageItem } from './MessageItem';
import { Bot } from 'lucide-react';
import type { Message } from '@/types';

interface MessageListProps {
  messages: Message[];
  isGenerating: boolean;
}

export function MessageList({ messages, isGenerating }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center p-8">
        <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
          <Bot className="w-7 h-7 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Paper Agent</h2>
          <p className="text-sm text-muted-foreground mt-1">
            基于 RAG 和 GraphRAG 的 AI 科研助手
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 max-w-md w-full mt-4">
          {[
            '帮我分析最新的 LLM 论文',
            '什么是 GraphRAG？',
            '知识图谱在 RAG 中的应用',
            '比较 RAG 和 GraphRAG 的优劣',
          ].map((prompt) => (
            <button
              key={prompt}
              className="text-left text-xs p-3 rounded-xl border border-border bg-card hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto">
        {messages.map((msg) => (
          <MessageItem key={msg.id} message={msg} />
        ))}
        {isGenerating && messages[messages.length - 1]?.role !== 'assistant' && (
          <div className="flex gap-3 py-4 px-4">
            <div className="w-8 h-8 rounded-full bg-muted border border-border flex items-center justify-center shrink-0 mt-0.5">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1.5 items-center h-4">
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:0ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
