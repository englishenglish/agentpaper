export async function fetchDatabases() {
  const res = await fetch('/knowledge/databases');
  const data = await res.json();
  return Array.isArray(data) ? data : data.databases || [];
}

export async function selectMultipleDatabases(dbIds) {
  const res = await fetch('/knowledge/databases/select-multiple', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbIds)
  });
  if (!res.ok) {
    throw new Error('选择知识库失败');
  }
  return res.json();
}

export async function createDatabase(name, description, additionalParams = {}) {
  const res = await fetch('/knowledge/databases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      database_name: name,
      description,
      additional_params: additionalParams
    })
  });
  if (!res.ok) {
    throw new Error('创建知识库失败');
  }
  return res.json();
}

export async function updateDatabase(dbId, name, description) {
  return updateDatabaseWithParams(dbId, name, description, null);
}

export async function updateDatabaseWithParams(dbId, name, description, additionalParams = null) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, additional_params: additionalParams })
  });
  if (!res.ok) {
    throw new Error('更新知识库失败');
  }
  return res.json();
}

export async function deleteDatabase(dbId) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}`, {
    method: 'DELETE'
  });
  if (!res.ok) {
    throw new Error('删除知识库失败');
  }
  return res.json();
}

export async function uploadFileToDatabase(file, dbId) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`/knowledge/files/upload?db_id=${encodeURIComponent(dbId)}`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    throw new Error(`上传失败: ${file.name}`);
  }
  return res.json();
}

export async function addDocumentsToDatabase(dbId, filePaths) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}/documents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      items: filePaths,
      params: { content_type: 'file' }
    })
  });
  if (!res.ok) {
    throw new Error('文档入库失败');
  }
  return res.json();
}

export async function fetchDatabaseDocuments(dbId) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}/documents`);
  if (!res.ok) {
    throw new Error('获取知识库文档失败');
  }
  return res.json();
}

export async function deleteDocumentFromDatabase(dbId, docId) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE'
  });
  if (!res.ok) {
    throw new Error('删除文档失败');
  }
  return res.json();
}

export async function rebuildDatabase(dbId, params) {
  const res = await fetch(`/knowledge/databases/${encodeURIComponent(dbId)}/rebuild`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params || {})
  });
  if (!res.ok) {
    throw new Error('重建知识库失败');
  }
  return res.json();
}

export async function fetchBuildOptions() {
  const res = await fetch('/knowledge/build-options');
  if (!res.ok) {
    throw new Error('获取构建选项失败');
  }
  return res.json();
}

