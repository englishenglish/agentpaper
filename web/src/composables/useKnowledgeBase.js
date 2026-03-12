import { computed, ref } from 'vue';
import {
  addDocumentsToDatabase,
  createDatabase,
  deleteDocumentFromDatabase,
  deleteDatabase,
  fetchBuildOptions,
  fetchDatabaseDocuments,
  fetchDatabases,
  rebuildDatabase,
  selectMultipleDatabases,
  updateDatabaseWithParams,
  uploadFileToDatabase
} from '../api/knowledgeApi';

export function useKnowledgeBase() {
  const knowledgeBases = ref([]);
  const kbLoading = ref(false);
  const selectedDbIds = ref([]);

  const newKbName = ref('');
  const newKbDesc = ref('');
  const newBuildMethod = ref('default_chunk');
  const newRetrievalMethod = ref('vector');
  const editingKbId = ref('');
  const editKbName = ref('');
  const editKbDesc = ref('');
  const buildOptions = ref({ build_methods: [], retrieval_methods: [] });
  const documentsByDb = ref({});
  const expandedDbIds = ref([]);
  const editBuildMethod = ref('default_chunk');
  const editRetrievalMethod = ref('vector');

  const currentDbNames = computed(() => {
    if (selectedDbIds.value.length === 0) return '';
    return knowledgeBases.value
      .filter(k => selectedDbIds.value.includes(k.db_id || k.id))
      .map(k => k.name)
      .join('、');
  });

  async function refreshKnowledgeBases() {
    kbLoading.value = true;
    try {
      knowledgeBases.value = await fetchDatabases();
    } finally {
      kbLoading.value = false;
    }
  }

  async function refreshBuildOptions() {
    try {
      buildOptions.value = await fetchBuildOptions();
    } catch {
      buildOptions.value = {
        build_methods: [],
        retrieval_methods: []
      };
    }
  }

  async function toggleKnowledgeBaseSelection(kb) {
    const dbId = kb.db_id || kb.id;
    if (selectedDbIds.value.includes(dbId)) {
      selectedDbIds.value = selectedDbIds.value.filter(id => id !== dbId);
    } else {
      selectedDbIds.value = [...selectedDbIds.value, dbId];
    }
    await selectMultipleDatabases(selectedDbIds.value);
  }

  async function createKnowledgeBase() {
    const name = newKbName.value.trim();
    const desc = newKbDesc.value.trim();
    if (!name || !desc) return;
    await createDatabase(name, desc, {
      build_method: newBuildMethod.value,
      retrieval_method: newRetrievalMethod.value
    });
    newKbName.value = '';
    newKbDesc.value = '';
    await refreshKnowledgeBases();
  }

  function startEditKnowledgeBase(kb) {
    editingKbId.value = kb.db_id || kb.id;
    editKbName.value = kb.name || '';
    editKbDesc.value = kb.description || '';
    const params = kb.additional_params || {};
    editBuildMethod.value = params.build_method || 'default_chunk';
    editRetrievalMethod.value = params.retrieval_method || 'vector';
  }

  function cancelEditKnowledgeBase() {
    editingKbId.value = '';
    editKbName.value = '';
    editKbDesc.value = '';
    editBuildMethod.value = 'default_chunk';
    editRetrievalMethod.value = 'vector';
  }

  async function saveKnowledgeBase(kb) {
    const dbId = kb.db_id || kb.id;
    await updateDatabaseWithParams(dbId, editKbName.value.trim(), editKbDesc.value.trim(), {
      build_method: editBuildMethod.value,
      retrieval_method: editRetrievalMethod.value
    });
    cancelEditKnowledgeBase();
    await refreshKnowledgeBases();
  }

  async function removeKnowledgeBase(kb) {
    const dbId = kb.db_id || kb.id;
    await deleteDatabase(dbId);
    selectedDbIds.value = selectedDbIds.value.filter(id => id !== dbId);
    await selectMultipleDatabases(selectedDbIds.value);
    await refreshKnowledgeBases();
  }

  async function toggleDatabaseExpanded(kb) {
    const dbId = kb.db_id || kb.id;
    if (expandedDbIds.value.includes(dbId)) {
      expandedDbIds.value = expandedDbIds.value.filter(id => id !== dbId);
      return;
    }
    expandedDbIds.value = [...expandedDbIds.value, dbId];
    const result = await fetchDatabaseDocuments(dbId);
    documentsByDb.value = {
      ...documentsByDb.value,
      [dbId]: result.documents || []
    };
  }

  async function addFilesToDatabase(kb, files) {
    const dbId = kb.db_id || kb.id;
    const uploadedPaths = [];
    for (const file of files) {
      const uploadData = await uploadFileToDatabase(file, dbId);
      if (uploadData.file_path) uploadedPaths.push(uploadData.file_path);
    }
    if (uploadedPaths.length > 0) {
      await addDocumentsToDatabase(dbId, uploadedPaths);
    }
    const result = await fetchDatabaseDocuments(dbId);
    documentsByDb.value = { ...documentsByDb.value, [dbId]: result.documents || [] };
    await refreshKnowledgeBases();
  }

  async function removeDocument(kb, doc) {
    const dbId = kb.db_id || kb.id;
    const docId = doc.file_id;
    await deleteDocumentFromDatabase(dbId, docId);
    const result = await fetchDatabaseDocuments(dbId);
    documentsByDb.value = { ...documentsByDb.value, [dbId]: result.documents || [] };
    await refreshKnowledgeBases();
  }

  async function rebuildKnowledgeBase(kb) {
    const dbId = kb.db_id || kb.id;
    await rebuildDatabase(dbId, {
      use_qa_split: editBuildMethod.value === 'qa_chunk',
      build_method: editBuildMethod.value,
      retrieval_method: editRetrievalMethod.value,
      content_type: 'file'
    });
    const result = await fetchDatabaseDocuments(dbId);
    documentsByDb.value = { ...documentsByDb.value, [dbId]: result.documents || [] };
  }

  return {
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
  };
}

