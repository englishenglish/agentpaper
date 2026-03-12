import { create } from 'zustand';
import type { Node, Edge } from 'reactflow';
import type { EntityGraph, GraphNode } from '@/types';
import { knowledgeApi } from '@/lib/api';
import { getNodeColor } from '@/lib/utils';

interface GraphState {
  dbId: string | null;
  rawGraph: EntityGraph | null;
  nodes: Node[];
  edges: Edge[];
  selectedNode: GraphNode | null;
  isLoading: boolean;
  error: string | null;

  fetchGraph: (db_id: string) => Promise<void>;
  selectNode: (node: GraphNode | null) => void;
  clearGraph: () => void;
}

/**
 * 更易读的布局：
 * - 按实体类型分“同心圆”分层：Paper 在中心，Method/Model 靠内圈，Task/Dataset 中圈，其它类型外圈
 * - 同一层节点按角度均匀排布，并加入少量抖动，避免完全重叠
 */
function layoutNodes(rawNodes: Record<string, GraphNode>): Node[] {
  const nodeList = Object.values(rawNodes);
  const total = nodeList.length || 1;

  const layers: { types: string[]; radius: number }[] = [
    { types: ['Paper'], radius: 0 }, // 中心
    { types: ['Method', 'Model'], radius: 220 },
    { types: ['Task', 'Dataset'], radius: 360 },
    { types: ['Metric', 'Finding', 'Concept'], radius: 500 },
    { types: ['Other'], radius: 640 },
  ];

  // 将节点按类型分层
  const layerBuckets: Record<number, GraphNode[]> = {};
  nodeList.forEach((n) => {
    const layerIndex =
      layers.findIndex((l) => l.types.includes(n.type)) !== -1
        ? layers.findIndex((l) => l.types.includes(n.type))
        : layers.findIndex((l) => l.types.includes('Other'));
    if (!layerBuckets[layerIndex]) layerBuckets[layerIndex] = [];
    layerBuckets[layerIndex].push(n);
  });

  const centerX = 0;
  const centerY = 0;

  const positioned: Node[] = [];

  Object.entries(layerBuckets).forEach(([layerKey, nodesInLayer]) => {
    const layer = layers[Number(layerKey)];
    const count = nodesInLayer.length;
    const radius = layer.radius;

    if (radius === 0) {
      // 中心层（通常是 Paper），最多放少量节点，稍微打散
      const step = (2 * Math.PI) / Math.max(count, 1);
      nodesInLayer.forEach((n, idx) => {
        const angle = step * idx;
        const jitter = 20;
        positioned.push({
          id: n.id,
          type: 'graphNode',
          position: {
            x: centerX + Math.cos(angle) * jitter,
            y: centerY + Math.sin(angle) * jitter,
          },
          data: {
            label: n.label,
            nodeType: n.type,
            color: getNodeColor(n.type),
            raw: n,
          },
        });
      });
    } else {
      const step = (2 * Math.PI) / Math.max(count, 1);
      nodesInLayer.forEach((n, idx) => {
        const angle = step * idx;
        const jitterRadius = 25; // 让位置稍微有点随机，避免完全重叠
        const jitterAngle = (Math.random() - 0.5) * (Math.PI / 24);
        const r = radius + (Math.random() - 0.5) * jitterRadius;
        positioned.push({
          id: n.id,
          type: 'graphNode',
          position: {
            x: centerX + r * Math.cos(angle + jitterAngle),
            y: centerY + r * Math.sin(angle + jitterAngle),
          },
          data: {
            label: n.label,
            nodeType: n.type,
            color: getNodeColor(n.type),
            raw: n,
          },
        });
      });
    }
  });

  // 简单的碰撞处理，尽量避免节点重叠：如果两个节点太近，就沿连线方向轻微推开
  const minDist = 70; // 最小间距像素
  for (let i = 0; i < positioned.length; i++) {
    for (let j = i + 1; j < positioned.length; j++) {
      const a = positioned[i];
      const b = positioned[j];
      const dx = a.position.x - b.position.x;
      const dy = a.position.y - b.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001;
      if (dist < minDist) {
        const overlap = (minDist - dist) / 2;
        const ux = dx / dist;
        const uy = dy / dist;
        a.position.x += ux * overlap;
        a.position.y += uy * overlap;
        b.position.x -= ux * overlap;
        b.position.y -= uy * overlap;
      }
    }
  }

  return positioned;
}

function buildEdges(rawEdges: EntityGraph['edges']): Edge[] {
  return rawEdges.map((e, i) => ({
    id: `edge-${i}`,
    source: e.source,
    target: e.target,
    label: e.type,
    // 使用贝塞尔曲线展示更平滑的关系，而不是折线
    type: 'bezier',
    animated: e.type === 'USES_METHOD',
    style: { stroke: '#64748b', strokeWidth: 1.5 },
    labelStyle: { fontSize: 10, fill: '#94a3b8' },
    data: { weight: e.weight },
  }));
}

export const useGraphStore = create<GraphState>((set) => ({
  dbId: null,
  rawGraph: null,
  nodes: [],
  edges: [],
  selectedNode: null,
  isLoading: false,
  error: null,

  fetchGraph: async (db_id) => {
    set({ isLoading: true, error: null, dbId: db_id });
    try {
      const res = await knowledgeApi.getEntityGraph(db_id);
      if (!res.graph) {
        set({ rawGraph: null, nodes: [], edges: [], isLoading: false });
        return;
      }
      const graph = res.graph;
      const nodes = layoutNodes(graph.nodes);
      const edges = buildEdges(graph.edges);
      set({ rawGraph: graph, nodes, edges });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ isLoading: false });
    }
  },

  selectNode: (node) => set({ selectedNode: node }),

  clearGraph: () =>
    set({ rawGraph: null, nodes: [], edges: [], selectedNode: null, dbId: null }),
}));
