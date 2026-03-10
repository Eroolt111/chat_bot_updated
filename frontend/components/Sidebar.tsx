'use client';

interface SidebarProps {
  onNewChat: () => void;
  onClose: () => void;
}

export default function Sidebar({ onNewChat, onClose }: SidebarProps) {
  return (
    <div className="sidebar" id="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <i className="fas fa-university"></i>
        </div>
        <button className="sidebar-close-btn" onClick={onClose}>
          <i className="fas fa-xmark"></i>
        </button>
      </div>
      <div className="sidebar-content">
        <button className="new-chat-btn" onClick={onNewChat}>
          <i className="fas fa-plus"></i>
          <span>New chat</span>
        </button>
      </div>
    </div>
  );
}
