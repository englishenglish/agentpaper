// ============================================================
// Chat Types
// ============================================================

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface ChatRequestBody {
  question: string;
  session_id: string;
  enable_web_search?: boolean;
  selected_db_ids?: string[];
  retrieval_mode?: 'rag' | 'graphrag' | 'both';
}

// ============================================================
// RAG / Knowledge Base Types
// ============================================================

export type EmbeddingStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface KnowledgeBase {
  db_id: string;
  name: string;
  description: string;
  kb_type: string;
  file_count?: number;
  row_count?: number;
  created_at: string;
  status?: string;
  additional_params?: {
    build_method?: string;
    retrieval_method?: string;
    [key: string]: unknown;
  };
}

export interface KBDocument {
  file_id: string;
  filename: string;
  path: string;
  file_type: string;
  status: EmbeddingStatus | string;
  created_at: string;
  chunk_count?: number;
}

export interface BuildOption {
  id: string;
  label: string;
}

export interface BuildOptions {
  build_methods: BuildOption[];
  retrieval_methods: BuildOption[];
}

// ============================================================
// GraphRAG Types
// ============================================================

export interface GraphNode {
  id: string;
  type: 'Paper' | 'Method' | 'Dataset' | 'Metric' | 'Topic' | 'Contribution';
  label: string;
  norm_label: string;
  // React Flow position
  position?: { x: number; y: number };
  data?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  label?: string;
}

export interface EntityGraph {
  db_id: string;
  nodes: Record<string, GraphNode>;
  edges: GraphEdge[];
  paper_entities: Record<string, string[]>;
  entity_aliases: Record<string, string>;
  stats: {
    node_count: number;
    edge_count: number;
    paper_count: number;
    node_type_count: Record<string, number>;
  };
}

// ============================================================
// API Response Types
// ============================================================

export interface ApiResponse<T = unknown> {
  data?: T;
  message?: string;
  status?: string;
}

export interface SSEEvent {
  step: string;
  state: string;
  data: unknown;
}
