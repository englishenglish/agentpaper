<template>
  <div :class="['app-root', theme]">
    <div class="app-container">
      <SidebarPanel
        :collapsed="sidebarCollapsed"
        :chats="chats"
        :active-chat-id="activeChatId"
        :chat-preview="chatPreview"
        :knowledge-bases="knowledgeBases"
        :kb-loading="kbLoading"
        :selected-db-ids="selectedDbIds"
        @toggle-sidebar="sidebarCollapsed = !sidebarCollapsed"
        @create-chat="createNewChat"
        @select-chat="setActiveChat"
        @toggle-kb="toggleKnowledgeBaseSelection"
      />

      <main class="main">
        <AppHeader
          :view-mode="viewMode"
          :settings="settings"
          :theme="theme"
          @change-view="viewMode = $event"
          @change-mode="settings.mode = $event"
          @change-retrieval-mode="settings.retrievalMode = $event"
          @change-web-search="settings.enableWebSearch = $event"
          @toggle-theme="toggleTheme"
        />

        <ChatView
          v-if="viewMode === 'chat'"
          :active-chat="activeChat"
          :input="input"
          :sending="sending"
          :pending-files="pendingFiles"
          :uploading-files="uploadingFiles"
          :current-db-names="currentDbNames"
          :set-messages-el="setMessagesEl"
          @update-input="input = $event"
          @send="handleSend"
          @interrupt="interruptStream"
          @files-picked="onFilePicked"
          @remove-file="removePendingFile"
          @upload-files="uploadFilesAndBuildKb"
        />

        <KnowledgeManagerView
          v-else
          :kb-loading="kbLoading"
          :knowledge-bases="knowledgeBases"
          :new-kb-name="newKbName"
          :new-kb-desc="newKbDesc"
          :new-build-method="newBuildMethod"
          :new-retrieval-method="newRetrievalMethod"
          :editing-kb-id="editingKbId"
          :edit-kb-name="editKbName"
          :edit-kb-desc="editKbDesc"
          :edit-build-method="editBuildMethod"
          :edit-retrieval-method="editRetrievalMethod"
          :build-options="buildOptions"
          :documents-by-db="documentsByDb"
          :expanded-db-ids="expandedDbIds"
          @update:new-kb-name="newKbName = $event"
          @update:new-kb-desc="newKbDesc = $event"
          @update:new-build-method="newBuildMethod = $event"
          @update:new-retrieval-method="newRetrievalMethod = $event"
          @update:edit-kb-name="editKbName = $event"
          @update:edit-kb-desc="editKbDesc = $event"
          @update:edit-build-method="editBuildMethod = $event"
          @update:edit-retrieval-method="editRetrievalMethod = $event"
          @create-kb="createKnowledgeBase"
          @start-edit-kb="startEditKnowledgeBase"
          @cancel-edit-kb="cancelEditKnowledgeBase"
          @save-kb="saveKnowledgeBase"
          @delete-kb="onDeleteKnowledgeBase"
          @toggle-db-expanded="toggleDatabaseExpanded"
          @add-db-files="onAddDbFiles"
          @delete-doc="onDeleteDoc"
          @rebuild-kb="onRebuildKb"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue';
import AppHeader from './components/layout/AppHeader.vue';
import SidebarPanel from './components/layout/SidebarPanel.vue';
import ChatView from './views/ChatView.vue';
import KnowledgeManagerView from './views/KnowledgeManagerView.vue';
import { useTheme } from './composables/useTheme';
import { useKnowledgeBase } from './composables/useKnowledgeBase';
import { useChat } from './composables/useChat';

const sidebarCollapsed = ref(false);
const viewMode = ref('chat');
const settings = ref(
  reactive({
    mode: 'stream',
    retrievalMode: 'rag',
    enableWebSearch: true
  })
);

const { theme, loadTheme, toggleTheme } = useTheme();
const kb = useKnowledgeBase();
const chat = useChat({
  settings,
  selectedDbIds: kb.selectedDbIds,
  refreshKnowledgeBases: kb.refreshKnowledgeBases,
  currentDbNames: kb.currentDbNames
});

const {
  knowledgeBases,
  kbLoading,
  selectedDbIds,
  currentDbNames,
  newKbName,
  newKbDesc,
  newBuildMethod,
  newRetrievalMethod,
  editingKbId,
  editKbName,
  editKbDesc,
  editBuildMethod,
  editRetrievalMethod,
  buildOptions,
  documentsByDb,
  expandedDbIds,
  refreshKnowledgeBases,
  refreshBuildOptions,
  toggleKnowledgeBaseSelection,
  createKnowledgeBase,
  startEditKnowledgeBase,
  cancelEditKnowledgeBase,
  saveKnowledgeBase,
  removeKnowledgeBase,
  toggleDatabaseExpanded,
  addFilesToDatabase,
  removeDocument,
  rebuildKnowledgeBase
} = kb;

const {
  chats,
  activeChatId,
  activeChat,
  input,
  sending,
  uploadingFiles,
  pendingFiles,
  createNewChat,
  setActiveChat,
  chatPreview,
  setMessagesEl,
  interruptStream,
  onFilePicked,
  removePendingFile,
  uploadFilesAndBuildKb,
  handleSend
} = chat;

async function onDeleteKnowledgeBase(kbItem) {
  if (!window.confirm(`确认删除知识库「${kbItem.name}」吗？`)) return;
  await removeKnowledgeBase(kbItem);
}

async function onAddDbFiles(kbItem, event) {
  const files = Array.from(event.target.files || []);
  if (files.length === 0) return;
  await addFilesToDatabase(kbItem, files);
  event.target.value = '';
}

async function onDeleteDoc(kbItem, doc) {
  if (!window.confirm(`确认删除论文/文档「${doc.filename}」吗？`)) return;
  await removeDocument(kbItem, doc);
}

async function onRebuildKb(kbItem) {
  await rebuildKnowledgeBase(kbItem);
}

onMounted(async () => {
  loadTheme();
  await refreshBuildOptions();
  await refreshKnowledgeBases();
  createNewChat();
});
</script>