'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './Sidebar';
import ChatHeader from './ChatHeader';
import WelcomeScreen from './WelcomeScreen';
import MessageBubble, { Message } from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import InputArea from './InputArea';
import ErrorToast from './ErrorToast';
import { generateUUID } from '@/lib/utils';

function getOrCreateSessionId(): string {
  if (typeof window === 'undefined') return generateUUID();
  let id = sessionStorage.getItem('chatSessionId');
  if (!id) {
    id = generateUUID();
    sessionStorage.setItem('chatSessionId', id);
  }
  return id;
}

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [connected, setConnected] = useState(true);
  const [toast, setToast] = useState({ visible: false, message: '' });
  const [sessionId, setSessionId] = useState('');

  const chatMessagesRef = useRef<HTMLDivElement>(null);

  // initialise session id client-side only
  useEffect(() => {
    setSessionId(getOrCreateSessionId());
  }, []);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (chatMessagesRef.current) {
        chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
      }
    });
  }, []);

  useEffect(() => {
    if (messages.length > 0) scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  const showErrorToast = useCallback((message: string) => {
    setToast({ visible: true, message });
    setTimeout(() => setToast({ visible: false, message: '' }), 5000);
  }, []);

  const checkConnection = useCallback(async () => {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      const ok = res.ok && data.status === 'healthy';
      setConnected(ok);
      if (ok && !data.pipeline_ready) {
        showErrorToast('Service initializing. Please wait...');
      }
    } catch {
      setConnected(false);
    }
  }, [showErrorToast]);

  // initial health check + periodic
  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  // visibility / online / offline
  useEffect(() => {
    const onVisible = () => { if (!document.hidden) checkConnection(); };
    const onOnline = () => checkConnection();
    const onOffline = () => {
      setConnected(false);
      showErrorToast('You are offline.');
    };
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
    };
  }, [checkConnection, showErrorToast]);

  async function sendMessage() {
    const message = inputValue.trim();
    if (!message || isLoading) return;

    const userMsg: Message = {
      content: message,
      sender: 'user',
      timestamp: new Date(),
      isError: false,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const botMsg: Message = {
        content: data.response || 'Уучлаарай, алдаа гарлаа.',
        sender: 'bot',
        timestamp: new Date(),
        isError: data.status !== 'success',
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        {
          content: 'Сүлжээний алдаа. Дахин оролдоно уу.',
          sender: 'bot',
          timestamp: new Date(),
          isError: true,
        },
      ]);
      showErrorToast('Connection error. Please try again.');
      setConnected(false);
    } finally {
      setIsLoading(false);
    }
  }

  function clearChat() {
    const newId = generateUUID();
    sessionStorage.setItem('chatSessionId', newId);
    setSessionId(newId);
    setMessages([]);
    setInputValue('');
  }

  const hasMessages = messages.length > 0;

  return (
    <>
      <div className="chat-container">
        {/* Sidebar */}
        {sidebarVisible && (
          <Sidebar
            onNewChat={clearChat}
            onClose={() => setSidebarVisible(false)}
          />
        )}

        {/* Main Chat */}
        <div className="main-chat">
          <ChatHeader
            connected={connected}
            onMenuToggle={() => setSidebarVisible((v) => !v)}
          />

          {/* Messages area */}
          <div
            ref={chatMessagesRef}
            className={`chat-messages${hasMessages ? ' has-messages' : ''}`}
            id="chatMessages"
          >
            {!hasMessages ? (
              /* Empty state: welcome + centered input */
              <div className="chat-center-wrapper">
                <WelcomeScreen />
                <InputArea
                  value={inputValue}
                  onChange={setInputValue}
                  onSend={sendMessage}
                  isLoading={isLoading}
                  variant="centered"
                />
              </div>
            ) : (
              /* Messages list */
              messages.map((msg, idx) => (
                <MessageBubble key={idx} message={msg} />
              ))
            )}
          </div>

          {/* Typing indicator */}
          {isLoading && <TypingIndicator />}

          {/* Fixed-bottom input (only when messages exist) */}
          {hasMessages && (
            <InputArea
              value={inputValue}
              onChange={setInputValue}
              onSend={sendMessage}
              isLoading={isLoading}
              variant="fixed-bottom"
            />
          )}

          {/* Footer */}
          <div
            className="input-footer"
            style={{ left: sidebarVisible ? '260px' : '0' }}
          >
            <span className="footer-text">
              <i className="fas fa-exclamation-circle"></i>
              {' '}Хиймэл оюун ухаан алдаа гаргаж болно.
            </span>
          </div>
        </div>
      </div>

      {/* Error Toast */}
      <ErrorToast
        visible={toast.visible}
        message={toast.message}
        onClose={() => setToast({ visible: false, message: '' })}
      />
    </>
  );
}
