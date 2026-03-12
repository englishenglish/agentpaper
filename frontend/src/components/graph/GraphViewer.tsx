'use client';

import React, { useCallback, useState } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  type NodeTypes,
  type ReactFlowInstance,
  type Node,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useGraphStore } from '@/store/graphStore';
import { GraphNodeComponent } from './GraphNode';
import { NodeInfoPanel } from './NodeInfoPanel';
import { GraphToolbar } from './GraphToolbar';
import { Network } from 'lucide-react';

const nodeTypes: NodeTypes = {
  graphNode: GraphNodeComponent,
};

export function GraphViewer() {
  const { nodes: storeNodes, edges: storeEdges, selectNode, rawGraph } = useGraphStore();
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);

  // Sync store nodes/edges into local RF state when they change
  React.useEffect(() => {
    setNodes(storeNodes);
  }, [storeNodes, setNodes]);

  React.useEffect(() => {
    setEdges(storeEdges);
  }, [storeEdges, setEdges]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const rawNode = rawGraph?.nodes[node.id];
      if (rawNode) selectNode(rawNode);
    },
    [rawGraph, selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  return (
    <div className="w-full h-full relative bg-muted/5">
      {/* Empty state */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center pointer-events-none z-10">
          <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center">
            <Network className="w-8 h-8 text-muted-foreground/40" />
          </div>
          <div>
            <h3 className="font-semibold text-muted-foreground">暂无图谱数据</h3>
            <p className="text-sm text-muted-foreground/60 mt-1">
              从左侧选择知识库以加载实体图谱
            </p>
          </div>
        </div>
      )}

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onInit={setRfInstance}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.1}
        maxZoom={2}
        className="bg-transparent"
        proOptions={{ hideAttribution: true }}
      >
        <Controls
          className="!bg-card !border-border !shadow-sm !rounded-xl overflow-hidden"
          showInteractive={false}
        />
        <MiniMap
          nodeColor={(node) => (node.data as { color: string }).color ?? '#94a3b8'}
          maskColor="rgba(0,0,0,0.15)"
          className="!bg-card !border-border !rounded-xl overflow-hidden"
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="hsl(var(--muted-foreground) / 0.2)"
        />
      </ReactFlow>

      {/* Overlays */}
      <GraphToolbar rfInstance={rfInstance} />
      <NodeInfoPanel />
    </div>
  );
}
