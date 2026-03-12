<template>
  <section class="chat-window">
    <div class="messages" :ref="setMessagesEl">
      <div v-for="msg in activeChat?.messages || []" :key="msg.id" :class="['message', msg.role]">
        <div class="avatar">{{ msg.role === 'user' ? '我' : 'AI' }}</div>
        <div class="bubble">
          <div v-if="msg.metadata?.status && !msg.metadata?.isFinished" class="tag">
            {{ msg.metadata.status }} <span class="dot" v-if="activeChat?.streaming">…</span>
          </div>
          <div class="content" v-show="msg.content">{{ msg.content }}</div>
        </div>
      </div>
    </div>

    <div class="chat-footer">
      <div class="upload-row">
        <input ref="fileInputRef" class="hidden-input" type="file" multiple @change="$emit('files-picked', $event)" />
        <button class="secondary-button" @click="fileInputRef?.click()">选择文件</button>
        <button class="primary-button" :disabled="pendingFiles.length === 0 || uploadingFiles" @click="$emit('upload-files')">
          {{ uploadingFiles ? '上传处理中...' : '上传并构建知识库' }}
        </button>
        <span class="muted">支持多文件，未选知识库时会自动新建一个</span>
      </div>
      <div v-if="pendingFiles.length" class="file-list">
        <span v-for="(file, idx) in pendingFiles" :key="file.name + idx" class="file-chip">
          {{ file.name }}
          <button class="chip-close" @click="$emit('remove-file', idx)">×</button>
        </span>
      </div>

      <div class="input-row">
        <textarea
          :value="input"
          class="chat-input"
          rows="2"
          placeholder="请输入你的科研问题或追问内容，回车发送，Shift+回车换行"
          @input="$emit('update-input', $event.target.value)"
          @keydown.enter.prevent.exact="$emit('send')"
          @keydown.enter.shift.exact.stop
        />
        <div class="input-actions">
          <button v-if="activeChat?.streaming" class="secondary-button" @click="$emit('interrupt')">中断生成</button>
          <button :disabled="!input.trim() || sending" class="primary-button" @click="$emit('send')">
            {{ activeChat?.sessionId ? '追问' : '提问并建库' }}
          </button>
        </div>
      </div>

      <div class="hint-row">
        <span class="muted">当前知识库：<strong>{{ currentDbNames || '未选择（将自动匹配）' }}</strong></span>
        <span class="muted"> 回车发送 · Shift+回车换行 </span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref } from 'vue';

const fileInputRef = ref(null);

defineProps({
  activeChat: Object,
  input: String,
  sending: Boolean,
  pendingFiles: Array,
  uploadingFiles: Boolean,
  currentDbNames: String,
  setMessagesEl: Function
});

defineEmits([
  'update-input',
  'send',
  'interrupt',
  'files-picked',
  'remove-file',
  'upload-files'
]);
</script>

