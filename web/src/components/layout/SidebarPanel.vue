<template>
  <aside :class="['sidebar', { collapsed }]">
    <div class="sidebar-header">
      <div class="logo">Paper-Agent</div>
      <button class="icon-button" @click="$emit('toggle-sidebar')">
        {{ collapsed ? '▶' : '◀' }}
      </button>
    </div>

    <div v-if="!collapsed" class="sidebar-content">
      <section class="section">
        <div class="section-title">
          会话
          <button class="link-button" @click="$emit('create-chat')">新建</button>
        </div>
        <ul class="chat-list">
          <li
            v-for="chat in chats"
            :key="chat.id"
            :class="['chat-item', { active: chat.id === activeChatId }]"
            @click="$emit('select-chat', chat.id)"
          >
            <div class="chat-title">{{ chat.title || '新会话' }}</div>
            <div class="chat-subtitle">{{ chatPreview(chat) }}</div>
          </li>
        </ul>
      </section>

      <section class="section">
        <div class="section-title">知识库</div>
        <div v-if="kbLoading" class="muted">加载知识库中...</div>
        <div v-else-if="knowledgeBases.length === 0" class="muted">暂无知识库</div>
        <ul v-else class="kb-list">
          <li
            v-for="kb in knowledgeBases"
            :key="kb.db_id || kb.id"
            :class="['kb-item', { active: selectedDbIds.includes(kb.db_id || kb.id) }]"
            @click="$emit('toggle-kb', kb)"
          >
            <div class="kb-name">{{ kb.name }}</div>
            <div class="kb-desc">{{ kb.description }}</div>
          </li>
        </ul>
      </section>
    </div>
  </aside>
</template>

<script setup>
defineProps({
  collapsed: Boolean,
  chats: Array,
  activeChatId: String,
  chatPreview: Function,
  knowledgeBases: Array,
  kbLoading: Boolean,
  selectedDbIds: Array
});

defineEmits(['toggle-sidebar', 'create-chat', 'select-chat', 'toggle-kb']);
</script>

