from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from sales_rag import SalesAssistant

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Initialize assistant
print("Initializing Sales Assistant...")
try:
    assistant = SalesAssistant(
        os.environ.get('NOTION_API_KEY'), 
        os.environ.get('GOOGLE_API_KEY')
    )
    
    databases = {
        'playbooks': os.environ.get('PLAYBOOKS_DB_ID'),
        'competitive': os.environ.get('COMPETITIVE_DB_ID'),
        'objections': os.environ.get('OBJECTIONS_DB_ID'),
        'case_studies': os.environ.get('CASE_STUDIES_DB_ID'),
        'products': os.environ.get('PRODUCTS_DB_ID')
    }
    
    doc_count = assistant.load_from_notion(databases)
    print(f"Assistant initialized! Loaded {doc_count} documents")
    
except Exception as e:
    print(f"Error initializing assistant: {e}")
    assistant = None

@app.route('/')
def chat_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Knowledge Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Space Mono', monospace;
                background: #fafafa;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: 600px;
                margin: 0 auto;
                background: white;
                border: 1px solid #e0e0e0;
                width: 100%;
            }
            
            .chat-header {
                background: #1a1a1a;
                color: #fafafa;
                padding: 20px;
                text-align: center;
            }
            
            .chat-title {
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 5px;
            }
            
            .chat-subtitle {
                font-size: 12px;
                opacity: 0.8;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
                min-height: 400px;
            }
            
            .message {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 8px;
                line-height: 1.4;
                font-size: 14px;
            }
            
            .message.user {
                align-self: flex-end;
                background: #1a1a1a;
                color: #fafafa;
            }
            
            .message.assistant {
                align-self: flex-start;
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                color: #1a1a1a;
            }
            
            .message-meta {
                font-size: 10px;
                opacity: 0.7;
                margin-top: 5px;
            }
            
            .typing-indicator {
                align-self: flex-start;
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 12px 16px;
                border-radius: 8px;
                display: none;
            }
            
            .typing-dots {
                display: inline-flex;
                gap: 4px;
            }
            
            .typing-dot {
                width: 6px;
                height: 6px;
                background: #666;
                border-radius: 50%;
                animation: typing 1.4s ease-in-out infinite both;
            }
            
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
                40% { transform: scale(1); opacity: 1; }
            }
            
            .chat-input-container {
                border-top: 1px solid #e0e0e0;
                padding: 20px;
                background: #fafafa;
            }
            
            .chat-input-form {
                display: flex;
                gap: 10px;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Space Mono', monospace;
                font-size: 14px;
                background: white;
            }
            
            .chat-input:focus {
                outline: none;
                border-color: #1a1a1a;
            }
            
            .chat-send {
                padding: 12px 20px;
                background: #1a1a1a;
                color: #fafafa;
                border: none;
                border-radius: 4px;
                font-family: 'Space Mono', monospace;
                font-size: 14px;
                font-weight: 700;
                cursor: pointer;
                transition: background 0.2s;
            }
            
            .chat-send:hover {
                background: #333;
            }
            
            .chat-send:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .error-message {
                background: #ffebee;
                border: 1px solid #f8bbd9;
                color: #c62828;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14px;
            }
            
            @media (max-width: 768px) {
                .chat-container {
                    height: 100vh;
                    max-width: 100%;
                }
                
                .chat-messages {
                    min-height: 300px;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title">Sales Knowledge Assistant</div>
                <div class="chat-subtitle">Ask about objections, competitive positioning, case studies, and more</div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div>Hello! I'm your Sales Knowledge Assistant. I can help you with:</div>
                    <div style="margin-top: 10px; font-size: 12px;">
                      • Objection handling strategies<br>
                      • Competitive positioning<br>
                      • Case study examples<br>
                      • Product information<br>
                      • Sales playbooks
                    </div>
                    <div style="margin-top: 10px; font-size: 12px; opacity: 0.8;">
                      Try asking: "How do I handle pricing objections?"
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <form class="chat-input-form" id="chatForm">
                    <input 
                        type="text" 
                        class="chat-input" 
                        id="chatInput" 
                        placeholder="Ask a question..."
                        autocomplete="off"
                        maxlength="500"
                    />
                    <button type="submit" class="chat-send" id="sendButton">Send</button>
                </form>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const chatForm = document.getElementById('chatForm');
            const chatInput = document.getElementById('chatInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');

            function addMessage(role, content, metadata = {}) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.textContent = content;
                messageDiv.appendChild(contentDiv);
                
                if (metadata.response_time_ms) {
                    const metaDiv = document.createElement('div');
                    metaDiv.className = 'message-meta';
                    metaDiv.textContent = `${metadata.response_time_ms}ms • ${metadata.question_type || 'general'}`;
                    messageDiv.appendChild(metaDiv);
                }
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function showTyping() {
                typingIndicator.style.display = 'block';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function hideTyping() {
                typingIndicator.style.display = 'none';
            }

            function showError(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = message;
                chatMessages.appendChild(errorDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            async function sendMessage(question) {
                if (!question.trim()) return;
                
                // Add user message
                addMessage('user', question);
                
                // Show typing indicator
                showTyping();
                
                // Disable input
                sendButton.disabled = true;
                chatInput.disabled = true;
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    hideTyping();
                    
                    if (data.error) {
                        showError(data.error);
                    } else {
                        addMessage('assistant', data.answer, {
                            response_time_ms: Math.round(data.response_time_ms),
                            question_type: data.question_type
                        });
                    }
                    
                } catch (error) {
                    hideTyping();
                    showError('Sorry, there was an error processing your request. Please try again.');
                    console.error('Error:', error);
                } finally {
                    // Re-enable input
                    sendButton.disabled = false;
                    chatInput.disabled = false;
                    chatInput.focus();
                }
            }

            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const question = chatInput.value.trim();
                if (question) {
                    chatInput.value = '';
                    await sendMessage(question);
                }
            });

            // Focus input on load
            chatInput.focus();
        </script>
    </body>
    </html>
    """

@app.route('/ask', methods=['POST'])
def ask_question():
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
            
        if len(question) > 500:
            return jsonify({'error': 'Question too long. Please keep under 500 characters.'}), 400
        
        result = assistant.ask(question)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'documents_loaded': assistant is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
