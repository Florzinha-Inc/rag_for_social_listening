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
        <title>Sales RAG Assistant</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px; 
            }
            .input-group { 
                margin-bottom: 20px; 
            }
            label { 
                display: block; 
                margin-bottom: 5px; 
                font-weight: 600; 
                color: #555; 
            }
            input, textarea { 
                width: 100%; 
                padding: 12px; 
                border: 2px solid #ddd; 
                border-radius: 6px; 
                font-size: 16px; 
                box-sizing: border-box; 
            }
            textarea { 
                height: 100px; 
                resize: vertical; 
            }
            button { 
                background: #007cba; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 6px; 
                font-size: 16px; 
                cursor: pointer; 
                width: 100%; 
            }
            button:hover { 
                background: #005a87; 
            }
            button:disabled { 
                background: #ccc; 
                cursor: not-allowed; 
            }
            .response { 
                margin-top: 20px; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 6px; 
                border-left: 4px solid #007cba; 
            }
            .sources { 
                margin-top: 15px; 
                font-size: 14px; 
                color: #666; 
            }
            .source { 
                margin: 5px 0; 
                padding: 5px; 
                background: #e9ecef; 
                border-radius: 4px; 
            }
            .loading { 
                display: none; 
                text-align: center; 
                margin-top: 10px; 
            }
            .error { 
                color: #dc3545; 
                background: #f8d7da; 
                border-color: #dc3545; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Sales RAG Assistant</h1>
            
            <div class="input-group">
                <label for="question">Ask your sales question:</label>
                <textarea id="question" placeholder="e.g., How do I handle price objections? What are our competitive advantages vs CompetitorX?"></textarea>
            </div>
            
            <button onclick="askQuestion()" id="askBtn">Ask Question</button>
            
            <div class="loading" id="loading">
                <p>🤔 Thinking...</p>
            </div>
            
            <div id="response"></div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                const askBtn = document.getElementById('askBtn');
                const loading = document.getElementById('loading');
                const responseDiv = document.getElementById('response');
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                askBtn.disabled = true;
                askBtn.textContent = 'Thinking...';
                loading.style.display = 'block';
                responseDiv.innerHTML = '';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        responseDiv.innerHTML = `<div class="response error"><strong>Error:</strong> ${data.error}</div>`;
                    } else {
                        let sourcesHtml = '';
                        if (data.sources && data.sources.length > 0) {
                            sourcesHtml = '<div class="sources"><strong>Sources:</strong>';
                            data.sources.forEach(source => {
                                sourcesHtml += `<div class="source">${source.title} (${source.type}) - Relevance: ${source.relevance}</div>`;
                            });
                            sourcesHtml += '</div>';
                        }
                        
                        responseDiv.innerHTML = `
                            <div class="response">
                                <strong>Answer:</strong><br>
                                ${data.answer.replace(/\\n/g, '<br>')}
                                ${sourcesHtml}
                            </div>
                        `;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `<div class="response error"><strong>Error:</strong> ${error.message}</div>`;
                }
                
                askBtn.disabled = false;
                askBtn.textContent = 'Ask Question';
                loading.style.display = 'none';
            }
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    askQuestion();
                }
            });
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
