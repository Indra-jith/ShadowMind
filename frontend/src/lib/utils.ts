import { type ClassValue, clsx } from './clsx';

/**
 * Merge class names with conflict resolution.
 * Lightweight alternative to `tailwind-merge` + `clsx`.
 */
export function cn(...inputs: ClassValue[]): string {
  return clsx(...inputs);
}

/**
 * Format a confidence score as a percentage string.
 */
export function formatConfidence(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * Format a timestamp string to a human-readable time.
 */
export function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return '--:--:--';
  }
}

/**
 * Truncate text to a maximum length with ellipsis.
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

/**
 * Get the appropriate status color class.
 */
export function getStatusColor(status: string): string {
  switch (status) {
    case 'active':
    case 'scanning':
    case 'generating':
    case 'retrieving':
      return 'text-electric-cyan';
    case 'surviving':
    case 'concluded':
    case 'complete':
      return 'text-electric-cyan';
    case 'eliminated':
    case 'failed':
      return 'text-crimson-burn';
    case 'scoring':
      return 'text-toxic-violet';
    default:
      return 'text-ghost-dim';
  }
}
