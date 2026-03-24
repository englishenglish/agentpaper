import axios from 'axios';
import type { KnowledgeBase, KBDocument, BuildOptions, EntityGraph } from '@/types';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? '';

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ============================================================
// Knowledge Base API
// ============================================================

export const knowledgeApi = {
  getDatabases: async (): Promise<{ databases: KnowledgeBase[] }> => {
    const res = await http.get('/knowledge/databases');
    return res.data;
  },

  createDatabase: async (params: {
    database_name: string;
    description: string;
    additional_params?: Record<string, unknown>;
  }) => {
    const res = await http.post('/knowledge/databases', params);
    return res.data;
  },

  updateDatabase: async (
    db_id: string,
    params: { name: string; description: string; additional_params?: Record<string, unknown> }
  ) => {
    const res = await http.put(`/knowledge/databases/${db_id}`, params);
    return res.data;
  },

  deleteDatabase: async (db_id: string) => {
    const res = await http.delete(`/knowledge/databases/${db_id}`);
    return res.data;
  },

  getDatabaseInfo: async (db_id: string): Promise<KnowledgeBase> => {
    const res = await http.get(`/knowledge/databases/${db_id}`);
    return res.data;
  },

  listDocuments: async (db_id: string): Promise<{ db_id: string; documents: KBDocument[] }> => {
    const res = await http.get(`/knowledge/databases/${db_id}/documents`);
    return res.data;
  },

  deleteDocument: async (db_id: string, doc_id: string) => {
    const res = await http.delete(`/knowledge/databases/${db_id}/documents/${doc_id}`);
    return res.data;
  },

  getDocumentContent: async (db_id: string, doc_id: string): Promise<{ lines: { id: string; content: string; metadata: any; chunk_order_index?: number }[] }> => {
    const res = await http.get(`/knowledge/databases/${db_id}/documents/${doc_id}/content`);
    return res.data;
  },

  rebuildDatabase: async (db_id: string, params?: Record<string, unknown>) => {
    const res = await http.post(`/knowledge/databases/${db_id}/rebuild`, params ?? {});
    return res.data;
  },

  getBuildOptions: async (): Promise<BuildOptions> => {
    const res = await http.get('/knowledge/build-options');
    return res.data;
  },

  uploadFile: async (file: File, db_id?: string): Promise<{ file_path: string; content_hash: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await http.post(
      `/knowledge/files/upload${db_id ? `?db_id=${db_id}` : ''}`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return res.data;
  },

  addDocuments: async (db_id: string, items: string[], params: Record<string, unknown> = {}) => {
    const res = await http.post(`/knowledge/databases/${db_id}/documents`, { items, params });
    return res.data;
  },

  getEntityGraph: async (db_id: string): Promise<{ db_id: string; graph: EntityGraph | null; summary: Record<string, unknown> }> => {
    const res = await http.get(`/knowledge/databases/${db_id}/graph`);
    return res.data;
  },
};

// ============================================================
// Chat API (SSE streaming)
// ============================================================

export const chatApi = {
  initResearch: (
    query: string,
    options: {
      enable_web_search?: boolean;
      selected_db_ids?: string[];
      retrieval_mode?: string;
    } = {}
  ): EventSource => {
    const params = new URLSearchParams({
      query,
      enable_web_search: String(options.enable_web_search ?? true),
      retrieval_mode: options.retrieval_mode ?? 'rag',
    });
    if (options.selected_db_ids?.length) {
      options.selected_db_ids.forEach((id) => params.append('selected_db_ids', id));
    }
    return new EventSource(`${BASE_URL}/api/research/init?${params}`);
  },

  chat: (
    question: string,
    session_id: string,
    options: {
      enable_web_search?: boolean;
      selected_db_ids?: string[];
      retrieval_mode?: string;
    } = {}
  ): EventSource => {
    const params = new URLSearchParams({
      question,
      session_id,
      enable_web_search: String(options.enable_web_search ?? false),
      retrieval_mode: options.retrieval_mode ?? 'rag',
    });
    if (options.selected_db_ids?.length) {
      options.selected_db_ids.forEach((id) => params.append('selected_db_ids', id));
    }
    return new EventSource(`${BASE_URL}/api/research/chat?${params}`);
  },

  health: async () => {
    const res = await http.get('/health');
    return res.data;
  },
};
