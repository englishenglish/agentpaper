<template>
  <section class="chat-window">
    <div class="messages">
      <div class="kb-toolbar">
        <input :value="newKbName" class="chat-input" placeholder="新知识库标题" @input="$emit('update:newKbName', $event.target.value)" />
        <input :value="newKbDesc" class="chat-input" placeholder="新知识库详细介绍" @input="$emit('update:newKbDesc', $event.target.value)" />
        <select :value="newBuildMethod" @change="$emit('update:newBuildMethod', $event.target.value)">
          <option v-for="m in buildOptions.build_methods" :key="m.id" :value="m.id">{{ m.label }}</option>
        </select>
        <select :value="newRetrievalMethod" @change="$emit('update:newRetrievalMethod', $event.target.value)">
          <option v-for="m in buildOptions.retrieval_methods" :key="m.id" :value="m.id">{{ m.label }}</option>
        </select>
        <button class="primary-button" @click="$emit('create-kb')">新建知识库</button>
      </div>

      <div v-if="kbLoading" class="muted">加载中...</div>
      <div v-else-if="knowledgeBases.length === 0" class="muted">暂无知识库</div>
      <div v-else class="kb-manage-list">
        <div v-for="kb in knowledgeBases" :key="kb.db_id || kb.id" class="kb-manage-item">
          <template v-if="editingKbId === (kb.db_id || kb.id)">
            <input :value="editKbName" class="chat-input" @input="$emit('update:editKbName', $event.target.value)" />
            <textarea :value="editKbDesc" class="chat-input" rows="2" @input="$emit('update:editKbDesc', $event.target.value)" />
            <div class="kb-methods">
              <label class="control">
                <span>构建方法</span>
                <select :value="editBuildMethod" @change="$emit('update:editBuildMethod', $event.target.value)">
                  <option v-for="m in buildOptions.build_methods" :key="m.id" :value="m.id">{{ m.label }}</option>
                </select>
              </label>
              <label class="control">
                <span>检索方法</span>
                <select :value="editRetrievalMethod" @change="$emit('update:editRetrievalMethod', $event.target.value)">
                  <option v-for="m in buildOptions.retrieval_methods" :key="m.id" :value="m.id">{{ m.label }}</option>
                </select>
              </label>
            </div>
            <div class="kb-actions">
              <button class="primary-button" @click="$emit('save-kb', kb)">保存</button>
              <button class="secondary-button" @click="$emit('rebuild-kb', kb)">重新构建</button>
              <button class="secondary-button" @click="$emit('cancel-edit-kb')">取消</button>
            </div>
          </template>
          <template v-else>
            <div class="kb-name">{{ kb.name }}</div>
            <div class="kb-desc">{{ kb.description }}</div>
            <div class="kb-actions">
              <button class="secondary-button" @click="$emit('start-edit-kb', kb)">编辑</button>
              <button class="secondary-button" @click="$emit('toggle-db-expanded', kb)">
                {{ expandedDbIds.includes(kb.db_id || kb.id) ? '收起论文' : '查看论文' }}
              </button>
              <button class="secondary-button danger" @click="$emit('delete-kb', kb)">删除</button>
            </div>
            <div v-if="expandedDbIds.includes(kb.db_id || kb.id)" class="kb-docs">
              <div class="upload-row">
                <input type="file" multiple @change="$emit('add-db-files', kb, $event)" />
              </div>
              <div v-if="(documentsByDb[kb.db_id || kb.id] || []).length === 0" class="muted">暂无论文</div>
              <ul v-else class="doc-list">
                <li v-for="doc in documentsByDb[kb.db_id || kb.id]" :key="doc.file_id" class="doc-item">
                  <span class="doc-name">{{ doc.filename }}</span>
                  <button class="secondary-button danger" @click="$emit('delete-doc', kb, doc)">删除</button>
                </li>
              </ul>
            </div>
          </template>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  kbLoading: Boolean,
  knowledgeBases: Array,
  newKbName: String,
  newKbDesc: String,
  newBuildMethod: String,
  newRetrievalMethod: String,
  editingKbId: String,
  editKbName: String,
  editKbDesc: String,
  editBuildMethod: String,
  editRetrievalMethod: String,
  buildOptions: Object,
  documentsByDb: Object,
  expandedDbIds: Array
});

defineEmits([
  'update:newKbName',
  'update:newKbDesc',
  'update:newBuildMethod',
  'update:newRetrievalMethod',
  'update:editKbName',
  'update:editKbDesc',
  'update:editBuildMethod',
  'update:editRetrievalMethod',
  'create-kb',
  'start-edit-kb',
  'cancel-edit-kb',
  'save-kb',
  'delete-kb',
  'toggle-db-expanded',
  'add-db-files',
  'delete-doc',
  'rebuild-kb'
]);
</script>

