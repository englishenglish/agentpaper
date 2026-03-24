'use client';

import React, { useCallback, useEffect } from 'react';
import { useChatStore } from '@/store/chatStore';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { generateId } from '@/lib/utils';

export function ChatWindow() {
  const {
    sessions,
    activeSessionId,
    isGenerating,
    settings,
    createSession,
    addMessage,
    updateLastAssistantMessage,
    setLastMessageKbs,
    setIsGenerating,
    setAbortController,
    abortController,
  } = useChatStore();

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Create initial session if none
  useEffect(() => {
    if (!activeSessionId) createSession();
  }, [activeSessionId, createSession]);

  const handleSend = useCallback(
    async (content: string) => {
      if (!activeSessionId) return;

      // Add user message
      addMessage(activeSessionId, { role: 'user', content });

      // Add placeholder assistant message
      addMessage(activeSessionId, { role: 'assistant', content: '', isStreaming: true });

      setIsGenerating(true);

      const params = new URLSearchParams({
        question: content,
        session_id: activeSessionId,
        enable_web_search: String(settings.enableWebSearch),
        retrieval_mode: settings.retrievalMode,
      });
      settings.selectedDbIds.forEach((id) => params.append('selected_db_ids', id));

      const es = new EventSource(`/api/research/chat?${params}`);
      let accumulated = '';

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const { step, state, data: payload } = data;

          if (step === 'qa_answering') {
            if (state === 'kb_context' && Array.isArray(payload)) {
              setLastMessageKbs(activeSessionId, payload as { db_id: string; name: string }[]);
            } else if (state === 'generating' && typeof payload === 'string') {
              accumulated += payload;
              updateLastAssistantMessage(activeSessionId, accumulated, true);
            } else if (state === 'completed') {
              const finalText = typeof payload === 'string' ? payload : accumulated;
              updateLastAssistantMessage(activeSessionId, finalText, false);
              es.close();
              setIsGenerating(false);
            }
          } else if (state === 'finished') {
            if (!accumulated) {
              updateLastAssistantMessage(activeSessionId, '（无回复）', false);
            }
            es.close();
            setIsGenerating(false);
          }
        } catch {
          // ignore parse errors
        }
      };

      es.onerror = () => {
        if (!accumulated) {
          updateLastAssistantMessage(activeSessionId, '连接失败，请检查后端服务是否启动。', false);
        } else {
          updateLastAssistantMessage(activeSessionId, accumulated, false);
        }
        es.close();
        setIsGenerating(false);
      };
    },
    [activeSessionId, settings, addMessage, updateLastAssistantMessage, setLastMessageKbs, setIsGenerating]
  );

  const handleStop = useCallback(() => {
    abortController?.abort();
    setIsGenerating(false);
    if (activeSessionId) {
      updateLastAssistantMessage(activeSessionId, '（已中断）', false);
    }
  }, [abortController, activeSessionId, setIsGenerating, updateLastAssistantMessage]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background">
      {/* Messages */}
      <MessageList
        messages={activeSession?.messages ?? []}
        isGenerating={isGenerating}
      />

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isGenerating={isGenerating}
        disabled={!activeSessionId}
      />
    </div>
  );
}
