import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

/** SSR/Node 下无 `localStorage` 时仍提供 Storage 形态，否则 persist 不会挂载 `api.persist`（见 zustand middleware `newImpl`）。 */
function getPersistStorage(): Storage {
  if (typeof window !== 'undefined') {
    return localStorage;
  }
  return {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
    length: 0,
    clear: () => {},
    key: () => null,
  } as Storage;
}
import type { ChatSession, CitationChunk, KbBinding, Message, UsedKb } from '@/types';
import { generateId } from '@/lib/utils';

export type SessionPatch = Partial<
  Pick<ChatSession, 'selectedDbId' | 'enableWebSearch' | 'retrievalMode' | 'kbBinding'>
>;

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isSidebarOpen: boolean;
  isGenerating: boolean;
  abortController: AbortController | null;

  createSession: () => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  setActiveSession: (id: string) => void;
  getActiveSession: () => ChatSession | null;
  patchSession: (sessionId: string, patch: SessionPatch) => void;

  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'createdAt'>) => Message;
  updateLastAssistantMessage: (sessionId: string, content: string, isStreaming?: boolean) => void;
  setLastMessageKbs: (sessionId: string, kbs: UsedKb[]) => void;
  setLastMessageCitationChunks: (sessionId: string, chunks: CitationChunk[]) => void;
  clearMessages: (sessionId: string) => void;

  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setIsGenerating: (v: boolean) => void;
  setAbortController: (ctrl: AbortController | null) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      isSidebarOpen: true,
      isGenerating: false,
      abortController: null,

      createSession: () => {
        const id = generateId();
        const now = Date.now();
        const session: ChatSession = {
          id,
          title: '新对话',
          messages: [],
          createdAt: now,
          updatedAt: now,
          selectedDbId: null,
          enableWebSearch: true,
          retrievalMode: 'rag',
          kbBinding: 'none',
        };
        set((state) => ({
          sessions: [session, ...state.sessions],
          activeSessionId: id,
        }));
        return id;
      },

      deleteSession: (id) =>
        set((state) => {
          const remaining = state.sessions.filter((s) => s.id !== id);
          const newActive =
            state.activeSessionId === id ? (remaining[0]?.id ?? null) : state.activeSessionId;
          return { sessions: remaining, activeSessionId: newActive };
        }),

      renameSession: (id, title) =>
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? { ...s, title } : s)),
        })),

      setActiveSession: (id) => set({ activeSessionId: id }),

      getActiveSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find((s) => s.id === activeSessionId) ?? null;
      },

      patchSession: (sessionId, patch) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, ...patch, updatedAt: Date.now() } : s
          ),
        })),

      addMessage: (sessionId, messageData) => {
        const message: Message = {
          ...messageData,
          id: generateId(),
          createdAt: Date.now(),
        };
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, message],
                  updatedAt: Date.now(),
                  title:
                    s.messages.length === 0 && messageData.role === 'user'
                      ? messageData.content.slice(0, 30) || s.title
                      : s.title,
                }
              : s
          ),
        }));
        return message;
      },

      updateLastAssistantMessage: (sessionId, content, isStreaming = false) =>
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id !== sessionId) return s;
            const msgs = [...s.messages];
            const lastIdx = msgs.length - 1;
            if (lastIdx >= 0 && msgs[lastIdx].role === 'assistant') {
              msgs[lastIdx] = { ...msgs[lastIdx], content, isStreaming };
            }
            return { ...s, messages: msgs, updatedAt: Date.now() };
          }),
        })),

      setLastMessageKbs: (sessionId, kbs) =>
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id !== sessionId) return s;
            const msgs = [...s.messages];
            const lastIdx = msgs.length - 1;
            if (lastIdx >= 0 && msgs[lastIdx].role === 'assistant') {
              msgs[lastIdx] = { ...msgs[lastIdx], usedKbs: kbs };
            }
            return { ...s, messages: msgs };
          }),
        })),

      setLastMessageCitationChunks: (sessionId, chunks) =>
        set((state) => ({
          sessions: state.sessions.map((s) => {
            if (s.id !== sessionId) return s;
            const msgs = [...s.messages];
            const lastIdx = msgs.length - 1;
            if (lastIdx >= 0 && msgs[lastIdx].role === 'assistant') {
              msgs[lastIdx] = { ...msgs[lastIdx], citationChunks: chunks };
            }
            return { ...s, messages: msgs };
          }),
        })),

      clearMessages: (sessionId) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, messages: [], updatedAt: Date.now() } : s
          ),
        })),

      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      setSidebarOpen: (open) => set({ isSidebarOpen: open }),
      setIsGenerating: (v) => set({ isGenerating: v }),
      setAbortController: (ctrl) => set({ abortController: ctrl }),
    }),
    {
      name: 'paper-agent-chat',
      storage: createJSONStorage(getPersistStorage),
      version: 4,
      migrate: (persistedState: unknown, version: number) => {
        const s = persistedState as {
          settings?: {
            selectedDbId?: string | null;
            selectedDbIds?: string[];
            enableWebSearch?: boolean;
            retrievalMode?: 'rag' | 'graphrag' | 'both';
          };
          sessions?: ChatSession[];
        };
        if (version < 2 && s.settings?.selectedDbIds && s.settings.selectedDbId === undefined) {
          s.settings.selectedDbId = s.settings.selectedDbIds[0] ?? null;
          delete s.settings.selectedDbIds;
        }
        if (version < 3 && s.sessions) {
          const def = s.settings;
          s.sessions = s.sessions.map((sess) => ({
            ...sess,
            selectedDbId: sess.selectedDbId ?? def?.selectedDbId ?? null,
            enableWebSearch: sess.enableWebSearch ?? def?.enableWebSearch ?? true,
            retrievalMode: sess.retrievalMode ?? def?.retrievalMode ?? 'rag',
            kbBinding: (sess.kbBinding ?? 'none') as KbBinding,
          }));
          if (s.settings) delete (s as { settings?: unknown }).settings;
        }
        return persistedState as ChatState;
      },
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        isSidebarOpen: state.isSidebarOpen,
      }),
    }
  )
);
