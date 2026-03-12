import { ref, watch } from 'vue';

export function useTheme() {
  const theme = ref('light');

  function loadTheme() {
    const saved = window.localStorage.getItem('paper_agent_theme');
    if (saved === 'light' || saved === 'dark') {
      theme.value = saved;
      return;
    }
    theme.value = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  function toggleTheme() {
    theme.value = theme.value === 'light' ? 'dark' : 'light';
  }

  watch(theme, () => {
    window.localStorage.setItem('paper_agent_theme', theme.value);
  });

  return {
    theme,
    loadTheme,
    toggleTheme
  };
}

