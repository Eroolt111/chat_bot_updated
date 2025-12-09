// Modern ChatGPT-Style Chat Application
class ChatApp {
    constructor() {
        // DOM Elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.inputContainer = document.getElementById('inputContainer');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.menuToggle = document.getElementById('menuToggle');
        this.sidebar = document.getElementById('sidebar');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
        this.inputFooter = document.querySelector('.input-footer');

        this.sessionId = this.getOrCreateSessionId();
        console.log(`Session ID: ${this.sessionId}`);
        
        // State
        this.isLoading = false;
        this.messageHistory = [];
        
        // Initialize
        this.initializeEventListeners();
        this.checkConnection();
        this.autoResizeTextarea();
        this.updateFooterPosition();
    }

    getOrCreateSessionId() {
        let sessionId = sessionStorage.getItem('chatSessionId');
        if (!sessionId) {
            sessionId = this.generateUUID();
            sessionStorage.setItem('chatSessionId', sessionId);
            console.log(`New session ID generated and stored: ${sessionId}`);
        } else {
            console.log(`Existing session ID retrieved: ${sessionId}`);
        }
        return sessionId;
    }

    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });

    }

    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter (without Shift)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea and validate input
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.validateInput();
        });

        // Sidebar close button
        if (this.sidebarCloseBtn) {
            this.sidebarCloseBtn.addEventListener('click', () => {
                this.sidebar.style.display = 'none';
                this.updateFooterPosition();
            });
        }

        // New chat button
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => {
                this.clearChat();
            });
        }

        // Sidebar toggle (hamburger menu)
        if (this.menuToggle) {
            this.menuToggle.addEventListener('click', () => {
                if (this.sidebar.style.display === 'none') {
                    this.sidebar.style.display = 'flex';
                } else {
                    this.sidebar.style.display = 'none';
                }
                this.updateFooterPosition();
            });
        }
    }
    
    updateFooterPosition() {
        if (this.inputFooter) {
            if (this.sidebar.style.display === 'none') {
                this.inputFooter.style.left = '0';
            } else {
                this.inputFooter.style.left = '260px';
            }
        }
    }

    validateInput() {
        const message = this.messageInput.value.trim();
        this.sendButton.disabled = !message || this.isLoading;
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || this.isLoading) {
            return;
        }

        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeTextarea();
        this.validateInput();
        
        // Show loading
        this.setLoadingState(true);
        this.showTypingIndicator();

        try {
            // ✅ CHANGED: Pass session_id to API call
            const response = await this.callChatAPI(message);
            
            this.hideTypingIndicator();
            
            if (response.status === 'success') {
                this.addMessage(response.response, 'bot');
            } else {
                this.addMessage(response.response || 'Уучлаарай, алдаа гарлаа.', 'bot', true);
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('Сүлжээний алдаа. Дахин оролдоно уу.', 'bot', true);
            this.showErrorToast('Connection error. Please try again.');
            this.updateConnectionStatus(false);
        } finally {
            this.setLoadingState(false);
        }
    }

    async callChatAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                session_id: this.sessionId  // ✅ 
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    addMessage(content, sender, isError = false) {
        // Move input to bottom on first message
        const centerWrapper = this.chatMessages.querySelector('.chat-center-wrapper');
        if (centerWrapper) {
            const welcomeScreen = centerWrapper.querySelector('.welcome-screen');
            if (welcomeScreen) {
                welcomeScreen.remove();
            }

            const mainChat = document.querySelector('.main-chat');
            if (this.inputContainer && mainChat) {
                mainChat.appendChild(this.inputContainer);
                this.inputContainer.classList.add('fixed-bottom');
                this.inputContainer.classList.remove('centered');
            }

            centerWrapper.remove();
        }

        this.chatMessages.classList.add('has-messages');

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message${isError ? ' error-message' : ''}`;

        const timestamp = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        const formattedContent = this.formatMessageContent(content);

        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        // Create content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content';
        contentWrapper.innerHTML = `
            <div class="message-text">${formattedContent}</div>
            <div class="message-time">${timestamp}</div>
        `;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentWrapper);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in history
        this.messageHistory.push({
            content: content,
            sender: sender,
            timestamp: new Date(),
            isError: isError
        });
    }
    
    formatMessageContent(text) {
        const escaped = this.escapeHtml(text);
        return escaped.replace(/\n/g, '<br>');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    setLoadingState(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading || !this.messageInput.value.trim();
        
        if (loading) {
            this.sendButton.classList.add('loading');
        } else {
            this.sendButton.classList.remove('loading');
        }
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        });
    }

    showErrorToast(message) {
        const toast = document.getElementById('errorToast');
        const errorMessage = document.getElementById('errorMessage');
        
        errorMessage.textContent = message;
        toast.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            toast.style.display = 'none';
        }, 5000);
    }

    async checkConnection() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            this.updateConnectionStatus(response.ok && data.status === 'healthy');
            
            if (!data.pipeline_ready) {
                this.showErrorToast('Service initializing. Please wait...');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateConnectionStatus(false);
        }
    }
                                        
    updateConnectionStatus(connected) {
        if (connected) {
            this.statusDot.classList.remove('disconnected');
            this.statusText.textContent = 'Connected';
        } else {
            this.statusDot.classList.add('disconnected');
            this.statusText.textContent = 'Disconnected';
        }
    }

    clearChat() {
        // Generate new session ID for new chat
        this.sessionId = this.generateUUID();
        sessionStorage.setItem('chatSessionId', this.sessionId);
        console.log(`New chat started with session: ${this.sessionId}`);
        
        this.chatMessages.classList.remove('has-messages');
        this.chatMessages.innerHTML = `
            <div class="chat-center-wrapper">
                <div class="welcome-screen">
                    <div class="welcome-icon">
                        <img src="/static/images/14.png" alt="DBM Logo">
                    </div>
                   
                </div>
            </div>
        `;

        const centerWrapper = this.chatMessages.querySelector('.chat-center-wrapper');

        if (centerWrapper && this.inputContainer) {
            this.inputContainer.classList.remove('fixed-bottom');
            this.inputContainer.classList.add('centered');
            centerWrapper.appendChild(this.inputContainer);
        }
        
        this.messageHistory = [];
    }

    focusInput() {
        this.messageInput.focus();
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const chatApp = new ChatApp();
    
    // Global reference for debugging
    window.chatApp = chatApp;
    
    // Periodic connection check (every 30 seconds)
    setInterval(() => {
        chatApp.checkConnection();
    }, 30000);
    
    // Focus input on load
    chatApp.focusInput();
});

// Handle visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.chatApp) {
        window.chatApp.checkConnection();
    }
});

// Handle online/offline events
window.addEventListener('online', () => {
    if (window.chatApp) {
        window.chatApp.checkConnection();
    }
});

window.addEventListener('offline', () => {
    if (window.chatApp) {
        window.chatApp.updateConnectionStatus(false);
        window.chatApp.showErrorToast('You are offline.');
    }
});
