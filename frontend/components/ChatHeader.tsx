'use client';

interface ChatHeaderProps {
  connected: boolean;
  onMenuToggle: () => void;
}

export default function ChatHeader({ connected, onMenuToggle }: ChatHeaderProps) {
  return (
    <div className="chat-header">
      <div className="header-left">
        <button className="menu-toggle" onClick={onMenuToggle}>
          <i className="fas fa-bars"></i>
        </button>
        <h1 className="chat-title">DBM Assistant</h1>
      </div>
      <div className="header-right">
        <div className="connection-indicator">
          <span className={`status-dot${connected ? '' : ' disconnected'}`}></span>
          <span className="status-text">{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>
    </div>
  );
}
