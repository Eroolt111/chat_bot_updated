'use client';

interface ErrorToastProps {
  visible: boolean;
  message: string;
  onClose: () => void;
}

export default function ErrorToast({ visible, message, onClose }: ErrorToastProps) {
  if (!visible) return null;

  return (
    <div className="toast" style={{ display: 'flex' }}>
      <div className="toast-content">
        <i className="fas fa-exclamation-circle"></i>
        <span>{message}</span>
      </div>
      <button className="toast-close" onClick={onClose}>
        <i className="fas fa-times"></i>
      </button>
    </div>
  );
}
