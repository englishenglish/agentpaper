<template>
  <header class="main-header">
    <div class="header-left">
      <h1 class="title">科研助手</h1>
      <div class="subtitle">多轮对话 · 知识库 RAG · 联网搜索（可关闭）</div>
      <div class="view-tabs">
        <button :class="['tab-button', { active: viewMode === 'chat' }]" @click="$emit('change-view', 'chat')">
          聊天
        </button>
        <button :class="['tab-button', { active: viewMode === 'knowledge' }]" @click="$emit('change-view', 'knowledge')">
          知识库管理
        </button>
      </div>
    </div>
    <div class="header-right">
      <div class="control-group">
        <label class="control">
          <span>模式</span>
          <select :value="settings.mode" @change="$emit('change-mode', $event.target.value)">
            <option value="stream">流式响应</option>
            <option value="normal">普通响应</option>
          </select>
        </label>
        <label class="control">
          <span>检索策略</span>
          <select :value="settings.retrievalMode" @change="$emit('change-retrieval-mode', $event.target.value)">
            <option value="rag">RAG</option>
            <option value="graphrag">GraphRAG</option>
            <option value="both">RAG + GraphRAG</option>
          </select>
        </label>
        <label class="control checkbox">
          <input :checked="settings.enableWebSearch" type="checkbox" @change="$emit('change-web-search', $event.target.checked)" />
          <span>联网搜索</span>
        </label>
      </div>
      <button class="icon-button" @click="$emit('toggle-theme')">
        {{ theme === 'light' ? '🌙' : '☀️' }}
      </button>
    </div>
  </header>
</template>

<script setup>
defineProps({
  viewMode: String,
  settings: Object,
  theme: String
});

defineEmits(['change-view', 'change-mode', 'change-retrieval-mode', 'change-web-search', 'toggle-theme']);
</script>

