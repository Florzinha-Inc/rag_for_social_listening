import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from sales_assistant import SalesAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variable to store the assistant
assistant = None

def initialize_assistant():
    """Initialize the sales assistant with environment variables"""
    global assistant
    
    notion_api_key = os.getenv('NOTION_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not notion_api_key or not google_api_key:
        logger.error("Missing required environment variables")
        return False
    
    try:
        assistant = SalesAssistant(notion_api_key, google_api_key)
        
        # Load databases
        database_configs = {
            'playbooks': os.getenv('PLAYBOOKS_DB_ID', ''),
            'competitive': os.getenv('COMPETITIVE_DB_ID', ''),
            'case_studies': os.getenv('CASE_STUDIES_DB_ID', ''),
            'products': os.getenv('PRODUCTS_DB_ID', ''),
            'objections': os.getenv('OBJECTIONS_DB_ID', '')
        }
        
        # Filter out empty database IDs
        database_configs = {k: v for k, v in database_configs.items() if v}
        
        if database_configs:
            docs_loaded = assistant.load_from_notion(database_configs)
            logger.info(f"Loaded {docs_loaded} documents from Notion")
        else:
            logger.warning("No database IDs provided")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing assistant: {e}")
        return False

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({'status': 'healthy', 'service': 'sales-rag-assistant'})

# Main interface
@app.route('/')
def index():
    """Main interface for the sales assistant"""
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sales Knowledge Assistant</title>
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
                color: #1a1a1a;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }

            /* Slack-style header */
            .header {
                background: #1a1a1a;
                color: #fafafa;
                padding: 12px 20px;
                border-bottom: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-shrink: 0;
            }

            .header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .channel-name {
                font-weight: 700;
                font-size: 16px;
            }

            .channel-desc {
                font-size: 12px;
                color: #ccc;
                margin-left: 8px;
            }

            .header-right {
                font-size: 12px;
                color: #ccc;
            }

            /* Main container */
            .container { 
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: none;
                margin: 0;
                padding: 0;
                background: #fafafa;
            }

            /* Messages area */
            .messages-container {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            /* Message styles */
            .message {
                display: flex;
                gap: 12px;
                max-width: 800px;
            }

            .message-avatar {
                width: 36px;
                height: 36px;
                background: #1a1a1a;
                color: #fafafa;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 14px;
                flex-shrink: 0;
            }

            .message-content {
                flex: 1;
            }

            .message-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 4px;
            }

            .message-author {
                font-weight: 700;
                font-size: 14px;
                color: #1a1a1a;
            }

            .message-time {
                font-size: 12px;
                color: #666;
            }

            .message-text {
                font-size: 14px;
                line-height: 1.5;
                color: #1a1a1a;
                white-space: pre-wrap;
            }

            .user-message .message-text {
                background: #f5f5f5;
                padding: 12px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }

            .assistant-message .message-text {
                background: none;
                padding: 0;
                border: none;
            }

            /* Sources */
            .sources {
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid #e0e0e0;
            }

            .sources-header {
                font-size: 12px;
                font-weight: 700;
                color: #666;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .source {
                margin: 4px 0;
                padding: 8px 12px;
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-size: 12px;
                color: #666;
            }

            .source-title {
                font-weight: 700;
                color: #1a1a1a;
            }

            /* Input area */
            .input-area {
                padding: 20px;
                border-top: 1px solid #e0e0e0;
                background: #fafafa;
                flex-shrink: 0;
            }

            .input-container {
                max-width: 800px;
                margin: 0 auto;
                position: relative;
            }

            textarea { 
                width: 100%;
                min-height: 44px;
                max-height: 120px;
                padding: 12px 80px 12px 16px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                font-family: 'Space Mono', monospace;
                font-size: 14px;
                resize: none;
                background: #ffffff;
                color: #1a1a1a;
            }

            textarea:focus {
                outline: none;
                border-color: #1a1a1a;
            }

            textarea::placeholder {
                color: #999;
            }

            .send-button {
                position: absolute;
                right: 8px;
                bottom: 8px;
                background: #1a1a1a;
                color: #fafafa;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-family: 'Space Mono', monospace;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: background 0.2s;
            }

            .send-button:hover:not(:disabled) {
                background: #333;
            }

            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }

            /* Loading indicator */
            .loading-message {
                display: none;
            }

            .loading-message .message-text {
                color: #666;
                font-style: italic;
            }

            /* Typing indicator */
            .typing-indicator {
                display: none;
                align-items: center;
                gap: 4px;
                color: #666;
                font-size: 12px;
                padding: 8px 0;
            }

            .typing-dots {
                display: flex;
                gap: 2px;
            }

            .typing-dot {
                width: 4px;
                height: 4px;
                background: #666;
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }

            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }

            @keyframes typing {
                0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
                40% { transform: scale(1); opacity: 1; }
            }

            /* Welcome message */
            .welcome-message {
                text-align: center;
                color: #666;
                font-size: 14px;
                margin: 40px 0;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }

            .welcome-title {
                font-weight: 700;
                color: #1a1a1a;
                margin-bottom: 8px;
            }

            /* Error states */
            .error .message-text {
                color: #dc3545;
                background: #f8d7da;
                border: 1px solid #dc3545;
                padding: 12px;
                border-radius: 6px;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .header {
                    padding: 12px 15px;
                }

                .channel-desc {
                    display: none;
                }

                .messages-container {
                    padding: 15px;
                }

                .input-area {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-left">
                <div class="channel-name"># sales-assistant</div>
                <div class="channel-desc">AI-powered knowledge retrieval for revenue teams</div>
            </div>
            <div class="header-right">Production • Live</div>
        </div>

        <div class="container">
            <div class="messages-container" id="messagesContainer">
                <div class="welcome-message">
                    <div class="welcome-title">Sales Knowledge Assistant</div>
                    <div>Ask questions about competitive positioning, objection handling, case studies, and sales playbooks.</div>
                </div>
            </div>

            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span>Assistant is typing...</span>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <textarea 
                        id="question" 
                        placeholder="Ask about objection handling, competitive advantages, case studies..."
                        rows="1"
                    ></textarea>
                    <button class="send-button" onclick="askQuestion()" id="askBtn">Send</button>
                </div>
            </div>
        </div>

        <script>
            let messageId = 0;

            function getCurrentTime() {
                return new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }

            function addMessage(author, text, type = 'user', sources = null) {
                const messagesContainer = document.getElementById('messagesContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                messageDiv.id = `message-${messageId++}`;

                const avatar = type === 'user' ? 'YOU' : 'AI';
                const authorName = type === 'user' ? 'You' : 'Sales Assistant';

                messageDiv.innerHTML = `
                    <div class="message-avatar">${avatar}</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-author">${authorName}</span>
                            <span class="message-time">${getCurrentTime()}</span>
                        </div>
                        <div class="message-text">${text}</div>
                        ${sources ? createSourcesHtml(sources) : ''}
                    </div>
                `;

                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                return messageDiv;
            }

            function createSourcesHtml(sources) {
                if (!sources || sources.length === 0) return '';
                
                let sourcesHtml = '<div class="sources"><div class="sources-header">Sources</div>';
                sources.forEach(source => {
                    sourcesHtml += `<div class="source"><span class="source-title">${source.title}</span> (${source.type}) - ${source.relevance}</div>`;
                });
                sourcesHtml += '</div>';
                return sourcesHtml;
            }

            function showTypingIndicator() {
                const indicator = document.getElementById('typingIndicator');
                indicator.style.display = 'flex';
            }

            function hideTypingIndicator() {
                const indicator = document.getElementById('typingIndicator');
                indicator.style.display = 'none';
            }

            async function askQuestion() {
                const questionInput = document.getElementById('question');
                const askBtn = document.getElementById('askBtn');
                const question = questionInput.value.trim();
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                // Add user message
                addMessage('user', question, 'user');
                
                // Clear input and disable button
                questionInput.value = '';
                askBtn.disabled = true;
                askBtn.textContent = 'Sending...';
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    hideTypingIndicator();
                    
                    if (data.error) {
                        addMessage('assistant', `Error: ${data.error}`, 'error');
                    } else {
                        addMessage('assistant', data.answer, 'assistant', data.sources);
                    }
                } catch (error) {
                    hideTypingIndicator();
                    addMessage('assistant', `Error: ${error.message}`, 'error');
                }
                
                askBtn.disabled = false;
                askBtn.textContent = 'Send';
                questionInput.focus();
            }
            
            // Auto-resize textarea
            document.getElementById('question').addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });
            
            // Allow Enter to submit (Shift+Enter for new line)
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });

            // Focus on input when page loads
            window.onload = function() {
                document.getElementById('question').focus();
            };
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

# API endpoint for questions
@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question API requests"""
    global assistant
    
    if not assistant:
        if not initialize_assistant():
            return jsonify({'error': 'Assistant not initialized. Check environment variables.'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        result = assistant.ask(question)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': str(e)}), 500

# Stats endpoint
@app.route('/stats')
def get_stats():
    """Get system statistics"""
    global assistant
    
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
    
    try:
        stats = assistant.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize assistant on startup
def create_app():
    """Application factory pattern"""
    logger.info("Starting Sales RAG Assistant...")
    success = initialize_assistant()
    if success:
        logger.info("Assistant initialized successfully")
    else:
        logger.warning("Assistant initialization failed - will retry on first request")
    return app

# Initialize the app
create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
