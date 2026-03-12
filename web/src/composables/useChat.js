import { computed, nextTick, ref } from 'vue';
import { createChatEventSource } from '../api/chatApi';
import {
  addDocumentsToDatabase,
  createDatabase,
  selectMultipleDatabases,
  uploadFileToDatabase
} from '../api/knowledgeApi';

export function useChat({ settings, selectedDbIds, refreshKnowledgeBases, currentDbNames }) {
  const chats = ref([]);
  const activeChatId = ref(null);
  const input = ref('');
  const sending = ref(false);
  const uploadingFiles = ref(false);
  const pendingFiles = ref([]);
  const messagesEl = ref(null);
  let currentEventSource = null;

  const activeChat = computed(() => chats.value.find(c => c.id === activeChatId.value));

  function createNewChat() {
    const id = Date.now().toString();
    chats.value.unshift({
      id,
      title: '新会话',
      messages: [],
      sessionId: null,
      streaming: false,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    });
    activeChatId.value = id;
  }

  function setActiveChat(id) {
    activeChatId.value = id;
  }

  function chatPreview(chat) {
    const last = chat.messages[chat.messages.length - 1];
    if (!last) return '暂无消息';
    const text = last.content || '';
    return text.length > 20 ? `${text.slice(0, 20)}...` : text;
  }

  function scrollToBottom() {
    nextTick(() => {
      if (messagesEl.value) {
        messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
      }
    });
  }

  function appendMessage(chat, message) {
    chat.messages.push({ id: `${Date.now()}-${Math.random()}`, ...message });
    chat.updatedAt = new Date().toISOString();
    scrollToBottom();
  }

  function setMessagesEl(el) {
    messagesEl.value = el;
  }

  function interruptStream() {
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
    if (activeChat.value) {
      activeChat.value.streaming = false;
      const lastMsg = activeChat.value.messages[activeChat.value.messages.length - 1];
      if (lastMsg && lastMsg.role === 'assistant') {
        lastMsg.metadata.isFinished = true;
        lastMsg.metadata.status = '已中断';
      }
    }
    sending.value = false;
  }

  function onFilePicked(event) {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    pendingFiles.value = [...pendingFiles.value, ...files];
    event.target.value = '';
  }

  function removePendingFile(index) {
    pendingFiles.value.splice(index, 1);
  }

  async function uploadFilesAndBuildKb() {
    if (pendingFiles.value.length === 0) return;
    uploadingFiles.value = true;
    try {
      let targetDbId = selectedDbIds.value[0] || '';
      if (!targetDbId) {
        const autoName = `上传知识库-${new Date().toLocaleString()}`;
        const created = await createDatabase(autoName, '由聊天上传文件自动创建');
        targetDbId = created.db_id;
        if (targetDbId) {
          selectedDbIds.value = [targetDbId];
          await selectMultipleDatabases(selectedDbIds.value);
        }
      }
      if (!targetDbId) throw new Error('未能创建或选中知识库');

      const uploadedPaths = [];
      for (const file of pendingFiles.value) {
        const uploadData = await uploadFileToDatabase(file, targetDbId);
        if (uploadData.file_path) uploadedPaths.push(uploadData.file_path);
      }
      if (uploadedPaths.length > 0) {
        await addDocumentsToDatabase(targetDbId, uploadedPaths);
      }

      if (activeChat.value) {
        appendMessage(activeChat.value, {
          role: 'assistant',
          content: `已上传 ${pendingFiles.value.length} 个文件，并构建到知识库 ${targetDbId}。`,
          metadata: { status: '知识库构建完成', isFinished: true }
        });
      }
      pendingFiles.value = [];
      await refreshKnowledgeBases();
    } finally {
      uploadingFiles.value = false;
    }
  }

  function handleStreamEvent(chat, data) {
    const { step, state, data: payload } = data;
    let lastMsg = chat.messages[chat.messages.length - 1];
    if (!lastMsg || lastMsg.role !== 'assistant' || lastMsg.metadata?.isFinished) {
      lastMsg = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: '',
        metadata: { status: '准备就绪', isFinished: false, buffer: '' }
      };
      chat.messages.push(lastMsg);
    }

    if (step === 'system' && state === 'session_created') {
      chat.sessionId = payload?.session_id || payload;
    }

    if (step === 'qa_answering' && payload) {
      const text = String(payload);
      if (settings.value.mode === 'stream') {
        lastMsg.content += text;
        lastMsg.metadata.status = '正在生成内容';
      } else {
        lastMsg.metadata.buffer += text;
        lastMsg.metadata.status = '正在思考，请稍候';
      }
    } else if (state && state !== 'finished' && state !== 'session_created') {
      lastMsg.metadata.status = `[${step}] ${state}`;
    }

    if (state === 'finished') {
      if (settings.value.mode === 'normal') {
        lastMsg.content = lastMsg.metadata.buffer;
      }
      lastMsg.metadata.isFinished = true;
      chat.streaming = false;
      sending.value = false;
      if (chat.title === '新会话' && chat.messages.length >= 2) {
        const firstUserMsg = chat.messages.find(m => m.role === 'user');
        if (firstUserMsg) {
          chat.title = firstUserMsg.content.length > 10 ? `${firstUserMsg.content.slice(0, 10)}...` : firstUserMsg.content;
        }
      }
    }
    scrollToBottom();
  }

  function handleSend() {
    const text = input.value.trim();
    if (!text || !activeChat.value || sending.value) return;

    const chat = activeChat.value;
    appendMessage(chat, { role: 'user', content: text });
    input.value = '';
    sending.value = true;
    chat.streaming = true;

    if (currentEventSource) currentEventSource.close();

    const isInit = !chat.sessionId;
    const endpoint = isInit ? '/api/research/init' : '/api/research/chat';
    const params = isInit ? { query: text } : { question: text, session_id: chat.sessionId };
    currentEventSource = createChatEventSource(endpoint, params, {
      enableWebSearch: settings.value.enableWebSearch,
      retrievalMode: settings.value.retrievalMode,
      selectedDbIds: selectedDbIds.value
    });

    currentEventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data);
        handleStreamEvent(chat, data);
        if (data.state === 'finished') {
          currentEventSource?.close();
          currentEventSource = null;
        }
      } catch (e) {
        console.error('解析 SSE 消息失败', e);
      }
    };

    currentEventSource.onerror = () => {
      currentEventSource?.close();
      currentEventSource = null;
      chat.streaming = false;
      sending.value = false;
      const lastMsg = chat.messages[chat.messages.length - 1];
      if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.metadata.isFinished) {
        lastMsg.metadata.status = '连接异常断开';
        lastMsg.metadata.isFinished = true;
      }
    };
  }

  return {
    chats,
    activeChatId,
    activeChat,
    input,
    sending,
    uploadingFiles,
    pendingFiles,
    currentDbNames,
    createNewChat,
    setActiveChat,
    chatPreview,
    setMessagesEl,
    interruptStream,
    onFilePicked,
    removePendingFile,
    uploadFilesAndBuildKb,
    handleSend
  };
}

