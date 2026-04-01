'use client';

import React, { useCallback, useEffect } from 'react';
import { useChatStore } from '@/store/chatStore';
import type { CitationChunk, KbBinding } from '@/types';
import { useChatStoreHydration } from '@/hooks/useChatStoreHydration';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

export function ChatWindow() {
  const hydrated = useChatStoreHydration();
  const {
    sessions,
    activeSessionId,
    isGenerating,
    createSession,
    addMessage,
    updateLastAssistantMessage,
    setLastMessageKbs,
    setLastMessageCitationChunks,
    setIsGenerating,
    abortController,
    patchSession,
  } = useChatStore();

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  useEffect(() => {
    if (!hydrated || activeSessionId) return;
    createSession();
  }, [hydrated, activeSessionId, createSession]);

  const handleSend = useCallback(
    async (content: string) => {
      if (!activeSessionId || !activeSession) return;

      addMessage(activeSessionId, { role: 'user', content });
      addMessage(activeSessionId, { role: 'assistant', content: '', isStreaming: true });

      setIsGenerating(true);

      const enableWeb = activeSession.enableWebSearch ?? true;
      const retrievalMode = activeSession.retrievalMode ?? 'rag';
      const selectedDbId = activeSession.selectedDbId ?? null;

      const params = new URLSearchParams({
        question: content,
        session_id: activeSessionId,
        enable_web_search: String(enableWeb),
        retrieval_mode: retrievalMode,
      });
      if (selectedDbId) {
        params.set('selected_db_id', selectedDbId);
      }

      const es = new EventSource(`/api/research/chat?${params}`);
      let accumulated = '';

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const { step, state, data: payload } = data;

          if (step === 'qa_answering') {
            if (state === 'session_profile' && payload && typeof payload === 'object') {
              const p = payload as {
                kb_binding?: string;
                selected_db_ids?: string[];
                enable_web_search?: boolean;
              };
              let kb: KbBinding = 'none';
              if (p.kb_binding === 'manual') kb = 'manual';
              else if (p.kb_binding === 'built') kb = 'built';
              patchSession(activeSessionId, {
                kbBinding: kb,
                selectedDbId: p.selected_db_ids?.[0] ?? null,
                enableWebSearch: p.enable_web_search ?? false,
              });
            } else if (state === 'kb_context' && Array.isArray(payload)) {
              setLastMessageKbs(activeSessionId, payload as { db_id: string; name: string }[]);
            } else if (state === 'citation_chunks' && Array.isArray(payload)) {
              setLastMessageCitationChunks(activeSessionId, payload as CitationChunk[]);
            } else if (state === 'stream_delta' && typeof payload === 'string') {
              accumulated += payload;
              updateLastAssistantMessage(activeSessionId, accumulated, true);
            } else if (state === 'generating') {
              /* 仅状态提示，不拼进正文；正文靠 stream_delta */
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
    [
      activeSessionId,
      activeSession,
      addMessage,
      updateLastAssistantMessage,
      setLastMessageKbs,
      setLastMessageCitationChunks,
      setIsGenerating,
      patchSession,
    ]
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
      <MessageList
        messages={activeSession?.messages ?? []}
        isGenerating={isGenerating}
      />

      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isGenerating={isGenerating}
        disabled={!activeSessionId}
      />
    </div>
  );
}
