class AzureChat {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.thinkingIndicator = document.getElementById('thinkingIndicator');
        this.messages = [];
        this.isProcessing = false;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.handleSendMessage());
        
        // Send message on Enter key (but allow Shift+Enter for new lines)
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
    }

    async handleSendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;

        // Add user message to chat
        this.addMessage('user', message);
        this.chatInput.value = '';
        this.isProcessing = true;
        this.showThinking(true);

        try {
            // Get response from the server
            const response = await this.getLLMResponse(message);
            
            // Add assistant's response to chat
            this.addMessage('assistant', response);
        } catch (error) {
            console.error('Error getting LLM response:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        } finally {
            this.isProcessing = false;
            this.showThinking(false);
        }
    }

    async getLLMResponse(message) {
        try {
            const API_BASE_URL = window.location.origin + '/api/llm';
            
            // Use the chat endpoint with the expected format
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [
                        { role: 'user', content: message }
                    ],
                    temperature: 0.7,
                    max_tokens: 2048
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.error || 'Failed to get response from server');
            }

            const data = await response.json();
            
            // Format the response with command results
            let formattedResponse = data.response || 'I apologize, but I could not generate a response.';
            
            // Add command results if available
            if (data.commands && data.commands.length > 0) {
                formattedResponse += '\n\nCommand Results:\n';
                data.commands.forEach(cmd => {
                    formattedResponse += `\nCommand: ${cmd.command}`;
                    if (cmd.success) {
                        formattedResponse += `\nStatus: SUCCESS\n${cmd.stdout || 'No output'}`;
                    } else {
                        formattedResponse += `\nStatus: FAILED\n${cmd.stderr || 'No error details'}`;
                    }
                    formattedResponse += '\n---\n';
                });
            }

            return formattedResponse;
        } catch (error) {
            console.error('Error in getLLMResponse:', error);
            if (error.message.includes('Failed to fetch')) {
                return 'Error: Could not connect to the AI service. Please make sure the Ollama server is running at http://localhost:11434';
            }
            return `Error: ${error.message}`;
        }
    }

    addMessage(role, content) {
        this.messages.push({ role, content });
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        // Format the content with proper line breaks and markdown if needed
        const formattedContent = this.formatMessage(content);
        messageElement.innerHTML = formattedContent;
        
        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    formatMessage(content) {
        // Simple markdown to HTML conversion (basic implementation)
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')                // Italic
            .replace(/`([^`]+)`/g, '<code>$1</code>')            // Inline code
            .replace(/\n/g, '<br>');                             // Line breaks

        // Code blocks (```code```)
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)\n```/g, 
            (match, lang, code) => {
                return `<pre><code class="language-${lang || ''}">${code}</code></pre>`;
            }
        );

        return formatted;
    }

    showThinking(show) {
        if (show) {
            this.thinkingIndicator.style.display = 'flex';
        } else {
            this.thinkingIndicator.style.display = 'none';
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Initialize chat when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.azureChat = new AzureChat();
});
