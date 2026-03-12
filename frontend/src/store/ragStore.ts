import { create } from 'zustand';
import type { KnowledgeBase, KBDocument, BuildOptions } from '@/types';
import { knowledgeApi } from '@/lib/api';

interface RagState {
  databases: KnowledgeBase[];
  selectedDbId: string | null;
  documents: Record<string, KBDocument[]>;
  buildOptions: BuildOptions | null;
  isLoading: boolean;
  isUploading: boolean;
  error: string | null;

  // Actions
  fetchDatabases: () => Promise<void>;
  createDatabase: (params: {
    database_name: string;
    description: string;
    additional_params?: Record<string, unknown>;
  }) => Promise<void>;
  updateDatabase: (
    db_id: string,
    params: { name: string; description: string; additional_params?: Record<string, unknown> }
  ) => Promise<void>;
  deleteDatabase: (db_id: string) => Promise<void>;
  selectDatabase: (db_id: string | null) => void;

  fetchDocuments: (db_id: string) => Promise<void>;
  deleteDocument: (db_id: string, doc_id: string) => Promise<void>;
  uploadAndAddDocument: (db_id: string, file: File) => Promise<void>;
  rebuildDatabase: (db_id: string) => Promise<void>;

  fetchBuildOptions: () => Promise<void>;
  clearError: () => void;
}

export const useRagStore = create<RagState>((set, get) => ({
  databases: [],
  selectedDbId: null,
  documents: {},
  buildOptions: null,
  isLoading: false,
  isUploading: false,
  error: null,

  fetchDatabases: async () => {
    set({ isLoading: true, error: null });
    try {
      const res = await knowledgeApi.getDatabases();
      set({ databases: res.databases ?? [] });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  createDatabase: async (params) => {
    set({ isLoading: true, error: null });
    try {
      await knowledgeApi.createDatabase(params);
      await get().fetchDatabases();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  updateDatabase: async (db_id, params) => {
    set({ isLoading: true, error: null });
    try {
      await knowledgeApi.updateDatabase(db_id, params);
      await get().fetchDatabases();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  deleteDatabase: async (db_id) => {
    set({ isLoading: true, error: null });
    try {
      await knowledgeApi.deleteDatabase(db_id);
      set((state) => {
        const docs = { ...state.documents };
        delete docs[db_id];
        return {
          databases: state.databases.filter((d) => d.db_id !== db_id),
          selectedDbId: state.selectedDbId === db_id ? null : state.selectedDbId,
          documents: docs,
        };
      });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  selectDatabase: (db_id) => {
    set({ selectedDbId: db_id });
    if (db_id) get().fetchDocuments(db_id);
  },

  fetchDocuments: async (db_id) => {
    set({ isLoading: true, error: null });
    try {
      const res = await knowledgeApi.listDocuments(db_id);
      set((state) => ({
        documents: { ...state.documents, [db_id]: res.documents ?? [] },
      }));
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  deleteDocument: async (db_id, doc_id) => {
    try {
      await knowledgeApi.deleteDocument(db_id, doc_id);
      set((state) => ({
        documents: {
          ...state.documents,
          [db_id]: (state.documents[db_id] ?? []).filter((d) => d.file_id !== doc_id),
        },
      }));
    } catch (e) {
      set({ error: String(e) });
    }
  },

  uploadAndAddDocument: async (db_id, file) => {
    set({ isUploading: true, error: null });
    try {
      const uploaded = await knowledgeApi.uploadFile(file, db_id);
      await knowledgeApi.addDocuments(db_id, [uploaded.file_path], { content_type: 'file' });
      await get().fetchDocuments(db_id);
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isUploading: false });
    }
  },

  rebuildDatabase: async (db_id) => {
    set({ isLoading: true, error: null });
    try {
      await knowledgeApi.rebuildDatabase(db_id);
      await get().fetchDocuments(db_id);
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  fetchBuildOptions: async () => {
    try {
      const opts = await knowledgeApi.getBuildOptions();
      set({ buildOptions: opts });
    } catch {
      // non-critical
    }
  },

  clearError: () => set({ error: null }),
}));
