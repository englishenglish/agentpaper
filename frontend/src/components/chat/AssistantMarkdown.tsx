'use client';

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { cn } from '@/lib/utils';
import type { CitationChunk } from '@/types';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

const MD_COMPONENTS = {
  pre: ({ children }: { children?: React.ReactNode }) => (
    <pre className="relative group/code overflow-x-auto rounded-lg bg-muted p-3 text-xs my-2">
      {children}
    </pre>
  ),
  code: ({
    inline,
    className,
    children,
    ...props
  }: {
    inline?: boolean;
    className?: string;
    children?: React.ReactNode;
  }) =>
    inline ? (
      <code
        className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-violet-400"
        {...props}
      >
        {children}
      </code>
    ) : (
      <code className={cn('text-xs font-mono', className)} {...props}>
        {children}
      </code>
    ),
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="overflow-x-auto my-2">
      <table className="border-collapse w-full text-xs">{children}</table>
    </div>
  ),
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="my-1 first:mt-0 last:mb-0">{children}</p>
  ),
};

function CitationMark({ refNum, meta }: { refNum: number; meta: CitationChunk }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <sup
          className="cursor-help text-primary font-semibold mx-0.5 align-super text-[0.75em] hover:underline underline-offset-2"
          tabIndex={0}
        >
          [{refNum}]
        </sup>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-[min(28rem,92vw)] bg-popover text-popover-foreground border border-border p-3 text-xs shadow-md z-[100]"
      >
        <div className="space-y-1 font-normal">
          <div>
            <span className="text-muted-foreground">Chunk ID</span>{' '}
            <span className="font-mono text-[11px] break-all">{meta.chunk_id}</span>
          </div>
          {meta.title ? (
            <div className="font-medium text-foreground leading-snug">{meta.title}</div>
          ) : null}
          {meta.paper_id ? (
            <div>
              <span className="text-muted-foreground">Paper</span> {meta.paper_id}
            </div>
          ) : null}
          {meta.section ? (
            <div>
              <span className="text-muted-foreground">章节</span> {meta.section}
            </div>
          ) : null}
          <div>
            <span className="text-muted-foreground">文件</span>{' '}
            <span className="break-all">{meta.source || '—'}</span>
          </div>
          <div>
            <span className="text-muted-foreground">相似度</span> {meta.score}
          </div>
          <div className="pt-2 mt-1 border-t border-border text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground max-h-40 overflow-y-auto">
            {meta.preview}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

interface AssistantMarkdownProps {
  content: string;
  citationChunks?: CitationChunk[];
  isStreaming?: boolean;
}

/**
 * 将模型输出的 [1]、[2] 等与检索 chunk 元数据对齐，悬停展示片段摘要。
 */
export function AssistantMarkdown({ content, citationChunks, isStreaming }: AssistantMarkdownProps) {
  const refMap = useMemo(() => {
    const m = new Map<number, CitationChunk>();
    for (const c of citationChunks || []) {
      m.set(c.ref, c);
    }
    return m;
  }, [citationChunks]);

  if (!citationChunks?.length) {
    return (
      <>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={MD_COMPONENTS}
        >
          {content}
        </ReactMarkdown>
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-foreground/70 ml-0.5 animate-typing rounded-sm" />
        )}
      </>
    );
  }

  const parts = content.split(/(\[\d+\])/g);

  return (
    <TooltipProvider delayDuration={180}>
      <div className="inline">
        {parts.map((part, i) => {
          if (!part) return null;
          const br = part.match(/^\[(\d+)\]$/);
          if (br) {
            const n = parseInt(br[1], 10);
            const meta = refMap.get(n);
            if (meta) {
              return <CitationMark key={i} refNum={n} meta={meta} />;
            }
          }
          return (
            <ReactMarkdown
              key={i}
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={MD_COMPONENTS}
            >
              {part}
            </ReactMarkdown>
          );
        })}
      </div>
      {isStreaming && (
        <span className="inline-block w-2 h-4 bg-foreground/70 ml-0.5 animate-typing rounded-sm" />
      )}
    </TooltipProvider>
  );
}
