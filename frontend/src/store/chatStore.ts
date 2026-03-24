import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatSession, Message, UsedKb } from '@/types';
import { generateId } from '@/lib/utils';

interface ChatSettings {
  enableWebSearch: boolean;
  selectedDbIds: string[];
  retrievalMode: 'rag' | 'graphrag' | 'both';
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isSidebarOpen: boolean;
  settings: ChatSettings;
  isGenerating: boolean;
  abortController: AbortController | null;

  // Session actions
  createSession: () => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  setActiveSession: (id: string) => void;
  getActiveSession: () => ChatSession | null;

  // Message actions
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'createdAt'>) => Message;
  updateLastAssistantMessage: (sessionId: string, content: string, isStreaming?: boolean) => void;
  setLastMessageKbs: (sessionId: string, kbs: UsedKb[]) => void;
  clearMessages: (sessionId: string) => void;

  // UI actions
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setIsGenerating: (v: boolean) => void;
  setAbortController: (ctrl: AbortController | null) => void;

  // Settings
  updateSettings: (patch: Partial<ChatSettings>) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      isSidebarOpen: true,
      isGenerating: false,
      abortController: null,
      settings: {
        enableWebSearch: false,
        selectedDbIds: [],
        retrievalMode: 'rag',
      },

      createSession: () => {
        const id = generateId();
        const now = Date.now();
        const session: ChatSession = {
          id,
          title: '新对话',
          messages: [],
          createdAt: now,
          updatedAt: now,
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
            state.activeSessionId === id
              ? (remaining[0]?.id ?? null)
              : state.activeSessionId;
          return { sessions: remaining, activeSessionId: newActive };
        }),

      renameSession: (id, title) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, title } : s
          ),
        })),

      setActiveSession: (id) => set({ activeSessionId: id }),

      getActiveSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find((s) => s.id === activeSessionId) ?? null;
      },

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
                  // Auto-title from first user message
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
      updateSettings: (patch) =>
        set((state) => ({ settings: { ...state.settings, ...patch } })),
    }),
    {
      name: 'paper-agent-chat',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        isSidebarOpen: state.isSidebarOpen,
        settings: state.settings,
      }),
    }
  )
);
