'use client';

import { useRef, useEffect } from 'react';

interface InputAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  variant: 'centered' | 'fixed-bottom';
}

export default function InputArea({
  value,
  onChange,
  onSend,
  isLoading,
  variant,
}: InputAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [value]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }

  const disabled = !value.trim() || isLoading;

  return (
    <div className={`input-container ${variant}`}>
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="message-input"
          placeholder="Энд бичнэ үү..."
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className={`send-button${isLoading ? ' loading' : ''}`}
          disabled={disabled}
          onClick={onSend}
        >
          <i className="fas fa-paper-plane"></i>
        </button>
      </div>
    </div>
  );
}
