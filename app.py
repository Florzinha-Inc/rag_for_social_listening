import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from reddit_gtm_assistant import GTMIntelligenceAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variable to store the assistant
assistant = None

# Note: We're using hardcoded Reddit data from the provided JSON files
# instead of fetching from URLs. The data is loaded directly in the
# GTMIntelligenceAssistant.load_real_reddit_data() method
REDDIT_DATA_SOURCE = "hardcoded_from_json_files"

def initialize_assistant():
    """Initialize the GTM intelligence assistant"""
    global assistant
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        logger.error("Missing GOOGLE_API_KEY environment variable")
        return False
    
    try:
        logger.info("Creating GTMIntelligenceAssistant instance...")
        assistant = GTMIntelligenceAssistant(google_api_key)
        
        logger.info("Loading real Reddit data directly...")
        # Load the real Reddit data directly, bypass URL fetching completely
        threads = assistant.load_real_reddit_data()
        
        if threads:
            logger.info(f"Got {len(threads)} threads from real Reddit data")
            assistant.threads_data = threads
            assistant.vector_store.add_threads(threads)
            
            total_posts = len(threads)
            total_comments = sum(len(t['comments']) for t in threads)
            
            logger.info(f"Successfully loaded {total_posts} Reddit posts with {total_comments} comments")
            
            # Debug: Check vector store
            if hasattr(assistant.vector_store, 'documents'):
                logger.info(f"Vector store now has {len(assistant.vector_store.documents)} documents")
            else:
                logger.error("Vector store has no documents attribute")
            
            return True
        else:
            logger.error("load_real_reddit_data() returned empty list")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing GTM assistant: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'gtm-intelligence-assistant'})

# Main interface
@app.route('/')
def index():
    """Main interface for the GTM intelligence assistant"""
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GTM Intelligence Assistant</title>
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

            /* Header styling - GTM Intelligence theme */
            .header {
                background: linear-gradient(135deg, #1a1a1a 0%, #333 100%);
                color: #fafafa;
                padding: 16px 20px;
                border-bottom: 3px solid #4CAF50;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-shrink: 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .header-left {
                display: flex;
                align-items: center;
                gap: 15px;
            }

            .header-icon {
                width: 32px;
                height: 32px;
                background: #4CAF50;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 16px;
            }

            .channel-name {
                font-weight: 700;
                font-size: 18px;
                color: #4CAF50;
            }

            .channel-desc {
                font-size: 12px;
                color: #ccc;
                margin-left: 8px;
            }

            .header-right {
                font-size: 11px;
                color: #aaa;
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 2px;
            }

            .status-indicator {
                width: 8px;
                height: 8px;
                background: #4CAF50;
                border-radius: 50%;
                display: inline-block;
                margin-right: 6px;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
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
                max-width: 900px;
            }

            .message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 12px;
                flex-shrink: 0;
            }

            .user-message .message-avatar {
                background: #e3f2fd;
                color: #1976d2;
            }

            .assistant-message .message-avatar {
                background: #4CAF50;
                color: #fff;
            }

            .message-content {
                flex: 1;
            }

            .message-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }

            .message-author {
                font-weight: 700;
                font-size: 14px;
                color: #1a1a1a;
            }

            .message-time {
                font-size: 11px;
                color: #666;
            }

            .analysis-type {
                background: #4CAF50;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .message-text {
                font-size: 14px;
                line-height: 1.6;
                color: #1a1a1a;
                white-space: pre-wrap;
            }

            .user-message .message-text {
                background: #f0f8ff;
                padding: 12px 16px;
                border: 1px solid #e3f2fd;
                border-radius: 8px;
                border-left: 4px solid #1976d2;
            }

            .assistant-message .message-text {
                background: #fff;
                padding: 16px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }

            /* Sources styling */
            .sources {
                margin-top: 16px;
                padding-top: 16px;
                border-top: 2px solid #f0f0f0;
            }

            .sources-header {
                font-size: 12px;
                font-weight: 700;
                color: #4CAF50;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .sources-header::before {
                content: "📊";
                font-size: 14px;
            }

            .source {
                margin: 6px 0;
                padding: 12px 16px;
                background: #f9f9f9;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
                font-size: 12px;
                color: #555;
                position: relative;
            }

            .source:hover {
                background: #f0f8ff;
                border-color: #e3f2fd;
            }

            .source-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
            }

            .source-title {
                font-weight: 700;
                color: #1a1a1a;
                font-size: 13px;
            }

            .source-meta {
                display: flex;
                gap: 12px;
                font-size: 11px;
                color: #666;
            }

            .subreddit-tag {
                background: #e8f5e8;
                color: #2e7d32;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: 600;
            }

            .score-badge {
                background: #fff3e0;
                color: #f57c00;
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
            }

            /* Input area */
            .input-area {
                padding: 20px;
                border-top: 2px solid #f0f0f0;
                background: linear-gradient(to bottom, #fafafa, #f5f5f5);
                flex-shrink: 0;
            }

            .input-container {
                max-width: 900px;
                margin: 0 auto;
                position: relative;
            }

            .input-helper {
                font-size: 11px;
                color: #666;
                margin-bottom: 8px;
                display: flex;
                gap: 15px;
            }

            .helper-examples {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .example-query {
                background: #e8f5e8;
                color: #2e7d32;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 10px;
                cursor: pointer;
                border: 1px solid transparent;
                transition: all 0.2s;
            }

            .example-query:hover {
                background: #4CAF50;
                color: white;
                transform: translateY(-1px);
            }

            textarea { 
                width: 100%;
                min-height: 48px;
                max-height: 140px;
                padding: 14px 90px 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-family: 'Space Mono', monospace;
                font-size: 14px;
                resize: none;
                background: #ffffff;
                color: #1a1a1a;
                transition: border-color 0.2s;
            }

            textarea:focus {
                outline: none;
                border-color: #4CAF50;
                box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
            }

            textarea::placeholder {
                color: #999;
            }

            .send-button {
                position: absolute;
                right: 8px;
                bottom: 8px;
                background: #4CAF50;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-family: 'Space Mono', monospace;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .send-button:hover:not(:disabled) {
                background: #45a049;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
            }

            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            /* Typing indicator */
            .typing-indicator {
                display: none;
                align-items: center;
                gap: 8px;
                color: #666;
                font-size: 12px;
                padding: 12px 20px;
                background: #f9f9f9;
                border-top: 1px solid #eee;
            }

            .typing-dots {
                display: flex;
                gap: 3px;
            }

            .typing-dot {
                width: 6px;
                height: 6px;
                background: #4CAF50;
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
                margin: 40px 20px;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
                background: #fff;
                padding: 30px;
                border-radius: 12px;
                border: 2px solid #f0f0f0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }

            .welcome-title {
                font-weight: 700;
                color: #1a1a1a;
                margin-bottom: 12px;
                font-size: 18px;
            }

            .welcome-subtitle {
                color: #4CAF50;
                font-weight: 600;
                margin-bottom: 16px;
                font-size: 14px;
            }

            .data-sources {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #f0f0f0;
            }

            .data-sources-title {
                font-size: 12px;
                font-weight: 700;
                color: #666;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .source-list {
                font-size: 11px;
                color: #888;
                line-height: 1.4;
            }

            /* Error states */
            .error .message-text {
                color: #d32f2f;
                background: #ffebee;
                border-left-color: #d32f2f;
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

                .helper-examples {
                    display: none;
                }

                .welcome-message {
                    margin: 20px 10px;
                    padding: 20px;
                }
            }

            /* Analysis formatting */
            .message-text h1, .message-text h2, .message-text h3 {
                color: #1a1a1a;
                margin: 16px 0 8px 0;
                font-weight: 700;
            }

            .message-text h2 {
                font-size: 16px;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 4px;
            }

            .message-text h3 {
                font-size: 14px;
                color: #4CAF50;
            }

            .message-text ul, .message-text ol {
                margin: 8px 0 8px 20px;
            }

            .message-text li {
                margin: 4px 0;
                line-height: 1.5;
            }

            .message-text strong {
                color: #1a1a1a;
                font-weight: 700;
            }

            .message-text em {
                color: #4CAF50;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-left">
                <div class="header-icon">GTM</div>
                <div>
                    <div class="channel-name">GTM Intelligence Assistant</div>
                    <div class="channel-desc">Reddit community analysis for growth marketing teams</div>
                </div>
            </div>
            <div class="header-right">
                <div><span class="status-indicator"></span>Live Analysis</div>
                <div>Cybersecurity • Pentesting • AI Tools</div>
            </div>
        </div>

        <div class="container">
            <div class="messages-container" id="messagesContainer">
                <div class="welcome-message">
                    <div class="welcome-title">GTM Intelligence from Reddit Communities</div>
                    <div class="welcome-subtitle">Analyze authentic developer conversations for strategic insights</div>
                    <div>Ask questions about pain points, feature gaps, competitive positioning, and market sentiment from real community discussions.</div>
                    
                    <div class="data-sources">
                        <div class="data-sources-title">Data Sources</div>
                        <div class="source-list">
                            r/cybersecurity • r/Pentesting • r/Hacking_Tutorials • r/bugbounty<br>
                            Analyzing discussions about AI tools, penetration testing, and cybersecurity workflows
                        </div>
                    </div>
                </div>
            </div>

            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span>Analyzing community discussions...</span>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <div class="input-helper">
                        <span style="font-weight: 700; color: #4CAF50;">Try:</span>
                        <div class="helper-examples">
                            <div class="example-query" onclick="setQuery(this.textContent)">What pain points do users mention?</div>
                            <div class="example-query" onclick="setQuery(this.textContent)">How do users feel about AI in pentesting?</div>
                            <div class="example-query" onclick="setQuery(this.textContent)">What features are users asking for?</div>
                            <div class="example-query" onclick="setQuery(this.textContent)">Competitive concerns about XBOW</div>
                        </div>
                    </div>
                    <textarea 
                        id="question" 
                        placeholder="Analyze community discussions for GTM insights..."
                        rows="1"
                    ></textarea>
                    <button class="send-button" onclick="askQuestion()" id="askBtn">Analyze</button>
                </div>
            </div>
        </div>

        <script>
            let messageId = 0;

            function getCurrentTime() {
                return new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }

            function setQuery(text) {
                document.getElementById('question').value = text;
                document.getElementById('question').focus();
            }

            function addMessage(author, text, type = 'user', sources = null, analysisType = null) {
                const messagesContainer = document.getElementById('messagesContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                messageDiv.id = `message-${messageId++}`;

                const avatar = type === 'user' ? 'YOU' : 'GTM';
                const authorName = type === 'user' ? 'You' : 'GTM Intelligence';

                let analysisTypeBadge = '';
                if (analysisType) {
                    analysisTypeBadge = `<span class="analysis-type">${analysisType.replace('_', ' ')}</span>`;
                }

                messageDiv.innerHTML = `
                    <div class="message-avatar">${avatar}</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-author">${authorName}</span>
                            <span class="message-time">${getCurrentTime()}</span>
                            ${analysisTypeBadge}
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
                
                let sourcesHtml = '<div class="sources"><div class="sources-header">Discussion Sources</div>';
                sources.forEach(source => {
                    sourcesHtml += `
                        <div class="source">
                            <div class="source-header">
                                <span class="source-title">${source.title}</span>
                                <div class="source-meta">
                                    <span class="subreddit-tag">r/${source.subreddit}</span>
                                    <span class="score-badge">↑${source.score}</span>
                                    <span>relevance: ${source.relevance}</span>
                                </div>
                            </div>
                            <div style="font-size: 10px; color: #888; margin-top: 4px;">
                                ${source.type} • <a href="${source.url}" target="_blank" style="color: #4CAF50;">view discussion</a>
                            </div>
                        </div>
                    `;
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
                    alert('Please enter a question for GTM analysis');
                    return;
                }
                
                // Add user message
                addMessage('user', question, 'user');
                
                // Clear input and disable button
                questionInput.value = '';
                askBtn.disabled = true;
                askBtn.textContent = 'Analyzing...';
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: question })
                    });
                    
                    const data = await response.json();
                    
                    hideTypingIndicator();
                    
                    if (data.error) {
                        addMessage('assistant', `Error: ${data.error}`, 'error');
                    } else {
                        addMessage('assistant', data.analysis, 'assistant', data.sources, data.analysis_type);
                    }
                } catch (error) {
                    hideTypingIndicator();
                    addMessage('assistant', `Error: ${error.message}`, 'error');
                }
                
                askBtn.disabled = false;
                askBtn.textContent = 'Analyze';
                questionInput.focus();
            }
            
            // Auto-resize textarea
            document.getElementById('question').addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 140) + 'px';
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

# API endpoint for GTM intelligence analysis
@app.route('/analyze', methods=['POST'])
def analyze_discussions():
    """Handle GTM analysis requests"""
    global assistant
    
    if not assistant:
        if not initialize_assistant():
            return jsonify({'error': 'GTM Assistant not initialized. Check GOOGLE_API_KEY environment variable.'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Analysis query is required'}), 400
        
        result = assistant.analyze(query)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing GTM analysis: {e}")
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

# Simple test endpoint to bypass all initialization and directly test the system
@app.route('/test_direct')
def test_direct():
    """Direct test bypassing initialization"""
    try:
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            return jsonify({'error': 'No Google API key'})

        from reddit_gtm_assistant import GTMVectorStore, GTMIntelligenceRAG
        
        # Create vector store directly
        vector_store = GTMVectorStore()
        
        # Add test data directly
        test_threads = [
            {
                'post': {
                    'id': 'test1',
                    'title': 'XBOW AI Pain Points Discussion',
                    'selftext': 'Users are frustrated with XBOW generating too many false positives and low quality vulnerability reports',
                    'author': 'test_user',
                    'score': 10,
                    'upvote_ratio': 0.8,
                    'num_comments': 5,
                    'created_utc': 1700000000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/test/',
                    'url': 'https://reddit.com/test/'
                },
                'comments': [
                    {
                        'id': 'comment1',
                        'author': 'security_expert',
                        'body': 'The main pain point with XBOW is that it generates too many false positives. We spend more time triaging bad reports than finding real vulnerabilities. The quality is questionable and it strains our security team resources.',
                        'score': 15,
                        'created_utc': 1700001000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comment2', 
                        'author': 'pentester_pro',
                        'body': 'XBOW cannot understand business logic and design flaws. It only finds basic vulnerabilities that any scanner would catch. Human creativity and intuition are still needed for complex security testing.',
                        'score': 12,
                        'created_utc': 1700002000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'test',
                'extracted_at': '2024-01-01T00:00:00'
            }
        ]
        
        # Add threads to vector store
        vector_store.add_threads(test_threads)
        
        # Test search
        if vector_store.documents and len(vector_store.documents) > 0:
            search_results = vector_store.search("pain points XBOW", n_results=3)
            
            # Test RAG system
            rag = GTMIntelligenceRAG(vector_store, google_api_key)
            analysis_result = rag.analyze("What pain points do users mention about XBOW?")
            
            return jsonify({
                'success': True,
                'documents_created': len(vector_store.documents),
                'search_results': len(search_results['documents']),
                'analysis': analysis_result['analysis'][:500] + '...' if len(analysis_result['analysis']) > 500 else analysis_result['analysis'],
                'sources': analysis_result['sources']
            })
        else:
            return jsonify({'error': 'No documents in vector store'})
            
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})

# Test endpoint to force reload and verify data
@app.route('/force_reload')
def force_reload():
    """Force reload of the assistant with debugging"""
    global assistant
    
    try:
        logger.info("=== FORCE RELOAD TRIGGERED ===")
        
        # Reset assistant
        assistant = None
        
        # Initialize fresh
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            return jsonify({'error': 'No Google API key'})
        
        logger.info("Creating new assistant instance...")
        from reddit_gtm_assistant import GTMIntelligenceAssistant
        assistant = GTMIntelligenceAssistant(google_api_key)
        
        logger.info("Loading real Reddit data...")
        threads = assistant.load_real_reddit_data()
        
        if not threads:
            return jsonify({'error': 'No threads loaded'})
        
        logger.info(f"Got {len(threads)} threads")
        assistant.threads_data = threads
        assistant.vector_store.add_threads(threads)
        
        # Test search immediately
        if hasattr(assistant.vector_store, 'documents') and assistant.vector_store.documents:
            search_results = assistant.vector_store.search("XBOW AI pentesting pain points", n_results=3)
            
            return jsonify({
                'success': True,
                'threads_loaded': len(threads),
                'total_documents': len(assistant.vector_store.documents),
                'vector_store_ready': assistant.vector_store.document_vectors is not None,
                'search_test': {
                    'query': 'XBOW AI pentesting pain points',
                    'results_found': len(search_results['documents']),
                    'sample_results': [doc[:150] + '...' for doc in search_results['documents'][:2]]
                }
            })
        else:
            return jsonify({'error': 'Vector store has no documents'})
            
    except Exception as e:
        import traceback
        logger.error(f"Force reload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})
@app.route('/debug')
def debug_data():
    """Debug endpoint to check data loading"""
    global assistant
    
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'})
    
    try:
        # Get basic stats
        stats = assistant.get_stats()
        
        # Get some sample documents from vector store
        sample_docs = []
        sample_metadata = []
        
        if hasattr(assistant.vector_store, 'documents') and assistant.vector_store.documents:
            sample_docs = [doc[:200] + '...' for doc in assistant.vector_store.documents[:5]]
        
        if hasattr(assistant.vector_store, 'metadata') and assistant.vector_store.metadata:
            sample_metadata = assistant.vector_store.metadata[:5]
        
        # Test a simple search
        search_test = None
        if assistant.vector_store.documents:
            try:
                test_results = assistant.vector_store.search("XBOW AI pentesting", n_results=3)
                search_test = {
                    'query': 'XBOW AI pentesting',
                    'results_found': len(test_results['documents']),
                    'sample_results': [doc[:100] + '...' for doc in test_results['documents'][:2]]
                }
            except Exception as e:
                search_test = {'error': str(e)}
        
        debug_info = {
            'assistant_initialized': assistant is not None,
            'stats': stats,
            'vector_store_has_documents': len(assistant.vector_store.documents) if hasattr(assistant.vector_store, 'documents') else 0,
            'vector_store_has_metadata': len(assistant.vector_store.metadata) if hasattr(assistant.vector_store, 'metadata') else 0,
            'vector_store_initialized': assistant.vector_store.document_vectors is not None if hasattr(assistant.vector_store, 'document_vectors') else False,
            'sample_documents': sample_docs,
            'sample_metadata': sample_metadata,
            'search_test': search_test,
            'threads_loaded': len(assistant.threads_data) if assistant.threads_data else 0,
            'data_source': REDDIT_DATA_SOURCE,
        }
        
        return jsonify(debug_info)
    except Exception as e:
        import traceback
        return jsonify({'error': f'Debug error: {str(e)}', 'traceback': traceback.format_exc()})

# Thread summaries endpoint
@app.route('/threads')
def get_threads():
    """Get Reddit thread summaries"""
    global assistant
    
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
    
    try:
        summaries = assistant.get_thread_summaries()
        return jsonify({'threads': summaries})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize assistant on startup
def create_app():
    """Application factory pattern"""
    logger.info("Starting GTM Intelligence Assistant...")
    success = initialize_assistant()
    if success:
        logger.info("GTM Assistant initialized successfully")
    else:
        logger.warning("GTM Assistant initialization failed - will retry on first request")
    return app

# Initialize the app
create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
