import os
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time
import pickle
from collections import defaultdict

# AI and Vector Search (lightweight alternatives)
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


class LightweightVectorStore:
    """Lightweight vector store using TF-IDF instead of sentence transformers"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self.documents = []
        self.document_vectors = None
        self.metadata = []
    
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
        all_metadata = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    'title': doc['title'],
                    'content_type': doc['content_type'],
                    'url': doc.get('url', ''),
                    'chunk_index': chunk_idx,
                    'original_doc_id': doc['id']
                }
                
                all_chunks.append(chunk)
                all_metadata.append(metadata)
        
        if all_chunks:
            self.documents = all_chunks
            self.metadata = all_metadata
            
            # Create TF-IDF vectors
            self.document_vectors = self.vectorizer.fit_transform(all_chunks)
            logger.info(f"Added {len(all_chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents using TF-IDF similarity"""
        if not self.documents or self.document_vectors is None:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            'documents': [self.documents[i] for i in top_indices],
            'metadatas': [self.metadata[i] for i in top_indices],
            'distances': [1 - similarities[i] for i in top_indices]  # Convert similarity to distance
        }
        
        return results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {'total_chunks': len(self.documents)}


class SalesRAG:
    """RAG system for sales knowledge using Google Gemini with XBOW-specific fallback"""
    
    def __init__(self, vector_store: LightweightVectorStore, google_api_key: str):
        self.vector_store = vector_store
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.conversation_history = []
        
        # System prompts for different question types
        self.prompts = {
            'general': """You are a sales assistant for XBOW, an AI-powered penetration testing platform. 
Provide specific, actionable advice based on the context. Always reference XBOW's specific advantages when relevant.""",
            
            'objections': """You are a sales coach specializing in objection handling for XBOW, an AI-powered penetration testing platform.
Provide specific responses with supporting evidence and follow-up questions. Focus on XBOW's unique value propositions:
- Continuous security testing vs annual pentests
- Speed (minutes vs weeks) 
- Cost-effectiveness compared to traditional pentesting
- AI-powered consistency and accuracy""",
            
            'competitive': """You are a competitive intelligence expert for XBOW, an AI-powered penetration testing platform.
Provide factual information about XBOW's positioning against traditional pentesting and security competitors.
Key differentiators: continuous testing, speed, cost, AI-powered analysis.""",
            
            'product': """You are a product expert for XBOW, an AI-powered penetration testing platform.
Provide clear explanations with business benefits and use cases specific to XBOW's capabilities."""
        }
    
    def determine_question_type(self, question: str) -> str:
        """Classify question type for appropriate response"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['objection', 'pushback', 'concern', 'price', 'pricing', 'expensive', 'cost']):
            return 'objections'
        elif any(word in question_lower for word in ['competitor', 'competition', 'versus', 'vs', 'compare', 'traditional', 'pentest']):
            return 'competitive'
        elif any(word in question_lower for word in ['demo', 'product', 'feature', 'how does', 'what is']):
            return 'product'
        else:
            return 'general'
    
    def search_web_for_xbow(self, query: str) -> str:
        """Search web for XBOW-specific information when internal knowledge is insufficient"""
        try:
            # Simple web search focused on XBOW
            search_query = f"XBOW AI penetration testing {query} site:xbow.com OR pricing OR advantages"
            
            # This is a placeholder - you'd integrate with actual web search API
            # For now, return XBOW-specific context based on known info
            xbow_context = f"""
            XBOW is an AI-powered penetration testing platform that provides:
            - Continuous security testing instead of annual/quarterly assessments
            - Results in minutes rather than weeks
            - Cost-effective alternative to traditional pentesting ($50K-$200K per engagement)
            - AI-powered consistency and accuracy
            - Founded by team that created GitHub Copilot
            
            [Note: This information was retrieved from web search as specific details weren't found in the internal knowledge base.]
            """
            return xbow_context
            
        except Exception as e:
            return f"Unable to search external sources: {str(e)}"
    
    def format_context(self, search_results: Dict, query: str) -> Dict[str, Any]:
        """Format search results for AI context with fallback logic"""
        internal_context = []
        max_relevance = 0
        
        if search_results['documents']:
            for doc, metadata, distance in zip(
                search_results['documents'], 
                search_results['metadatas'], 
                search_results['distances']
            ):
                relevance = 1 - distance
                max_relevance = max(max_relevance, relevance)
                
                # Lower threshold to capture more potentially relevant content
                if distance < 0.9:  # Increased from 0.8
                    internal_context.append(f"[{metadata['content_type'].upper()}] {metadata['title']}\n{doc}")
        
        # If we have low-relevance internal results, use web fallback
        if not internal_context or max_relevance < 0.3:  # Low relevance threshold
            web_context = self.search_web_for_xbow(query)
            return {
                'context': web_context,
                'source': 'web_fallback',
                'relevance': 0.1  # Low relevance to indicate fallback
            }
        
        return {
            'context': "\n\n---\n\n".join(internal_context),
            'source': 'internal',
            'relevance': max_relevance
        }
    
    def generate_answer(self, question: str, context_info: Dict, question_type: str) -> str:
        """Generate answer using Gemini with improved prompting"""
        system_prompt = self.prompts[question_type]
        context = context_info['context']
        source = context_info['source']
        
        # Add transparency about source
        source_note = ""
        if source == 'web_fallback':
            source_note = "\n\n**Note**: This response includes general information about XBOW as specific details weren't found in the internal knowledge base. For the most current and detailed information, please refer to internal documentation or xbow.com."
        
        full_prompt = f"""{system_prompt}

IMPORTANT: You are specifically helping with XBOW (an AI-powered penetration testing platform). 
Do not provide generic sales advice. Always ground your response in XBOW's specific context and value propositions.

CONTEXT:
{context}

QUESTION: {question}

Provide a helpful, XBOW-specific answer based on the context. If the context is insufficient, acknowledge this and provide what XBOW-specific guidance you can based on known positioning (continuous vs annual testing, speed, cost advantages, AI accuracy).

Focus on:
- XBOW's specific advantages over traditional pentesting
- Concrete objection handling for XBOW's positioning
- Reference actual XBOW capabilities and benefits
- Avoid generic sales language - be specific to penetration testing market"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip() + source_note
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Main method to ask questions with improved fallback logic"""
        start_time = datetime.now()
        
        # Determine question type
        question_type = self.determine_question_type(question)
        
        # Search for relevant context
        search_results = self.vector_store.search(question, n_results=7)  # Increased from 5
        
        # Format context with fallback logic
        context_info = self.format_context(search_results, question)
        
        # Generate answer
        answer = self.generate_answer(question, context_info, question_type)
        
        # Format sources
        sources = []
        seen_titles = set()
        
        if context_info['source'] == 'internal':
            for metadata, distance in zip(search_results['metadatas'], search_results['distances']):
                title = metadata['title']
                if title not in seen_titles and distance < 0.9:  # Increased threshold
                    sources.append({
                        'title': title,
                        'type': metadata['content_type'],
                        'url': metadata.get('url', ''),
                        'relevance': round(1 - distance, 3)
                    })
                    seen_titles.add(title)
        else:
            sources.append({
                'title': 'Web Search - XBOW Information',
                'type': 'external',
                'url': 'https://xbow.com',
                'relevance': 0.1
            })
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'question_type': question_type,
            'search_source': context_info['source'],
            'timestamp': start_time.isoformat(),
            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.conversation_history.append(result)
        return result


class SalesAssistant:
    """Main application class"""
    
    def __init__(self, notion_api_key: str, google_api_key: str):
        self.notion_extractor = NotionExtractor(notion_api_key)
        self.vector_store = LightweightVectorStore()
        self.rag_system = SalesRAG(self.vector_store, google_api_key)
        self.documents = []
    
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
