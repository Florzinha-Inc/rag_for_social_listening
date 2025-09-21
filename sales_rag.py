import os
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time

# AI and Vector Search
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotionExtractor:
    """Extract content from Notion databases"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Notion-Version': '2022-06-28'
        }
    
    def query_database(self, database_id: str) -> List[Dict]:
        """Get all pages from a Notion database with pagination"""
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        all_results = []
        has_more = True
        start_cursor = None
        
        while has_more:
            payload = {}
            if start_cursor:
                payload['start_cursor'] = start_cursor
            
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code != 200:
                logger.error(f"Error querying database: {response.text}")
                break
            
            data = response.json()
            all_results.extend(data.get('results', []))
            has_more = data.get('has_more', False)
            start_cursor = data.get('next_cursor')
            time.sleep(0.1)  # Rate limiting
        
        return all_results
    
    def extract_text_from_rich_text(self, rich_text_array: List[Dict]) -> str:
        """Convert Notion rich text to plain text"""
        if not rich_text_array:
            return ""
        
        text_parts = []
        for text_obj in rich_text_array:
            if text_obj.get('type') == 'text':
                text_parts.append(text_obj['text']['content'])
        
        return ''.join(text_parts)
    
    def process_page_properties(self, properties: Dict) -> Dict:
        """Extract data from Notion page properties"""
        processed = {}
        
        for prop_name, prop_data in properties.items():
            prop_type = prop_data.get('type')
            
            if prop_type == 'title':
                processed[prop_name] = self.extract_text_from_rich_text(prop_data['title'])
            elif prop_type == 'rich_text':
                processed[prop_name] = self.extract_text_from_rich_text(prop_data['rich_text'])
            elif prop_type == 'select':
                processed[prop_name] = prop_data['select']['name'] if prop_data['select'] else ""
            elif prop_type == 'multi_select':
                processed[prop_name] = [item['name'] for item in prop_data['multi_select']]
            elif prop_type == 'date':
                processed[prop_name] = prop_data['date']['start'] if prop_data['date'] else None
            elif prop_type == 'number':
                processed[prop_name] = prop_data['number']
            else:
                processed[prop_name] = str(prop_data.get(prop_type, ''))
        
        return processed
    
    def extract_from_database(self, database_id: str, content_type: str) -> List[Dict]:
        """Extract and process all pages from a database"""
        pages = self.query_database(database_id)
        processed_pages = []
        
        for page in pages:
            properties = self.process_page_properties(page['properties'])
            
            # Get title from first title property
            title = ""
            for prop_name, value in properties.items():
                if isinstance(value, str) and value and not title:
                    title = value
                    break
            
            # Combine all text content
            content_parts = []
            for prop_name, value in properties.items():
                if isinstance(value, str) and value and len(value) > 5:
                    content_parts.append(f"{prop_name}: {value}")
                elif isinstance(value, list) and value:
                    content_parts.append(f"{prop_name}: {', '.join(map(str, value))}")
            
            full_content = '\n\n'.join(content_parts)
            
            if title and full_content and len(full_content) > 20:
                processed_pages.append({
                    'id': page['id'],
                    'title': title,
                    'content': full_content,
                    'content_type': content_type,
                    'url': page['url'],
                    'created_time': page['created_time'],
                    'last_edited_time': page['last_edited_time']
                })
        
        return processed_pages


class VectorStore:
    """Manage vector embeddings and search"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="sales_knowledge",
            metadata={"description": "Sales knowledge base"}
        )
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store"""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_{chunk_idx}"
                metadata = {
                    'title': doc['title'],
                    'content_type': doc['content_type'],
                    'url': doc.get('url', ''),
                    'chunk_index': chunk_idx,
                    'original_doc_id': doc['id']
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)
        
        # Create embeddings and add to collection
        embeddings = self.embedding_model.encode(all_chunks)
        
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids
        )
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {'total_chunks': self.collection.count()}


class SalesRAG:
    """RAG system for sales knowledge using Google Gemini"""
    
    def __init__(self, vector_store: VectorStore, google_api_key: str):
        self.vector_store = vector_store
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.conversation_history = []
        
        # System prompts for different question types
        self.prompts = {
            'general': """You are a helpful sales assistant. Provide specific, actionable advice based on the context.
Keep responses focused and practical for sales situations.""",
            
            'objections': """You are a sales coach specializing in objection handling. 
Provide specific responses with supporting evidence and follow-up questions.""",
            
            'competitive': """You are a competitive intelligence expert. 
Provide factual information about competitors and positioning strategies.""",
            
            'product': """You are a product expert helping with demos and positioning.
Provide clear explanations with business benefits and use cases."""
        }
    
    def determine_question_type(self, question: str) -> str:
        """Classify question type for appropriate response"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['objection', 'pushback', 'concern']):
            return 'objections'
        elif any(word in question_lower for word in ['competitor', 'competition', 'versus']):
            return 'competitive'
        elif any(word in question_lower for word in ['demo', 'product', 'feature']):
            return 'product'
        else:
            return 'general'
    
    def format_context(self, search_results: Dict) -> str:
        """Format search results for AI context"""
        if not search_results['documents']:
            return "No relevant information found."
        
        context_parts = []
        for doc, metadata, distance in zip(
            search_results['documents'], 
            search_results['metadatas'], 
            search_results['distances']
        ):
            if distance < 1.2:  # Only include relevant results
                context_parts.append(f"[{metadata['content_type'].upper()}] {metadata['title']}\n{doc}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str, question_type: str) -> str:
        """Generate answer using Gemini"""
        system_prompt = self.prompts[question_type]
        
        full_prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {question}

Provide a helpful answer based on the context. If no relevant context, say so clearly."""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Main method to ask questions"""
        start_time = datetime.now()
        
        # Determine question type
        question_type = self.determine_question_type(question)
        
        # Search for relevant context
        search_results = self.vector_store.search(question)
        
        if not search_results['documents']:
            return {
                'question': question,
                'answer': "No relevant information found in knowledge base.",
                'sources': [],
                'question_type': question_type,
                'timestamp': start_time.isoformat()
            }
        
        # Generate answer
        context = self.format_context(search_results)
        answer = self.generate_answer(question, context, question_type)
        
        # Format sources
        sources = []
        seen_titles = set()
        for metadata, distance in zip(search_results['metadatas'], search_results['distances']):
            title = metadata['title']
            if title not in seen_titles and distance < 1.2:
                sources.append({
                    'title': title,
                    'type': metadata['content_type'],
                    'url': metadata.get('url', ''),
                    'relevance': round(1 - distance, 3)
                })
                seen_titles.add(title)
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'question_type': question_type,
            'timestamp': start_time.isoformat(),
            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.conversation_history.append(result)
        return result


class SalesAssistant:
    """Main application class"""
    
    def __init__(self, notion_api_key: str, google_api_key: str):
        self.notion_extractor = NotionExtractor(notion_api_key)
        self.vector_store = VectorStore()
        self.rag_system = SalesRAG(self.vector_store, google_api_key)
        self.documents = []  # Store documents as a simple list instead of DataFrame
    
    def load_from_notion(self, database_configs: Dict[str, str]):
        """Load content from Notion databases"""
        all_documents = []
        
        for content_type, database_id in database_configs.items():
            if database_id:  # Skip if database ID not provided
                try:
                    docs = self.notion_extractor.extract_from_database(database_id, content_type)
                    all_documents.extend(docs)
                    logger.info(f"Extracted {len(docs)} documents from {content_type}")
                except Exception as e:
                    logger.error(f"Error extracting {content_type}: {e}")
        
        if all_documents:
            self.documents = all_documents
            
            # Add to vector store
            self.vector_store.add_documents(all_documents)
            
            return len(all_documents)
        
        return 0
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question"""
        return self.rag_system.ask(question)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        # Count documents by content type manually
        content_types = {}
        for doc in self.documents:
            content_type = doc.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        stats = {
            'knowledge_base': {
                'total_documents': len(self.documents),
                'content_types': content_types
            },
            'vector_store': self.vector_store.get_stats(),
            'conversations': len(self.rag_system.conversation_history)
        }
        return stats
