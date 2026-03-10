'use client';

import { formatMessageContent } from '@/lib/utils';

export interface Message {
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isError: boolean;
}

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const { content, sender, timestamp, isError } = message;

  const timeStr = timestamp.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const formatted = formatMessageContent(content);

  return (
    <div className={`message ${sender}-message${isError ? ' error-message' : ''}`}>
      <div className="message-avatar">
        <i className={sender === 'user' ? 'fas fa-user' : 'fas fa-robot'}></i>
      </div>
      <div className="message-content">
        <div
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatted }}
        />
        <div className="message-time">{timeStr}</div>
      </div>
    </div>
  );
}
