'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { MessageSquare, Database, Network, Moon, Sun, Zap } from 'lucide-react';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { name: 'Chat', path: '/chat', icon: MessageSquare },
  { name: 'RAG', path: '/rag', icon: Database },
  { name: 'GraphRAG', path: '/graphrag', icon: Network },
];

export function TopNav() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();

  return (
    <header className="h-14 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center px-4 sticky top-0 z-50 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2 font-bold text-base mr-6 select-none">
        <div className="w-7 h-7 bg-primary rounded-lg flex items-center justify-center">
          <Zap className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="hidden sm:block">Paper Agent</span>
      </div>

      {/* Nav Links */}
      <nav className="flex items-center gap-1 flex-1">
        {NAV_ITEMS.map(({ name, path, icon: Icon }) => {
          const active = pathname.startsWith(path);
          return (
            <Link key={path} href={path}>
              <div
                className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  active
                    ? 'bg-secondary text-secondary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:block">{name}</span>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Theme Toggle */}
      <button
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        aria-label="Toggle theme"
      >
        <Sun className="w-4 h-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
        <Moon className="absolute w-4 h-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      </button>
    </header>
  );
}
