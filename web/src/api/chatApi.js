export function buildChatQuery(params, options) {
  const usp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') {
      usp.set(k, String(v));
    }
  });
  usp.set('enable_web_search', options.enableWebSearch ? '1' : '0');
  if (options.retrievalMode) {
    usp.set('retrieval_mode', options.retrievalMode);
  }
  if (options.selectedDbIds && options.selectedDbIds.length > 0) {
    usp.set('selected_db_ids', options.selectedDbIds.join(','));
  }
  return usp.toString();
}

export function createChatEventSource(endpoint, params, options) {
  const query = buildChatQuery(params, options);
  return new EventSource(`${endpoint}?${query}`);
}

