'use client';

import { useEffect, useState } from 'react';
import { useChatStore } from '@/store/chatStore';

/**
 * zustand persist 从 localStorage 异步恢复；在恢复完成前 sessions 可能仍为空，
 * 若此时自动「新建对话」会覆盖用户已保存的会话与每会话的检索/联网/知识库设置。
 */
export function useChatStoreHydration(): boolean {
  const [hydrated, setHydrated] = useState(() => useChatStore.persist.hasHydrated());

  useEffect(() => {
    if (useChatStore.persist.hasHydrated()) {
      setHydrated(true);
      return;
    }
    const unsub = useChatStore.persist.onFinishHydration(() => setHydrated(true));
    return unsub;
  }, []);

  return hydrated;
}
