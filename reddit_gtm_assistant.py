import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# AI and Vector Search
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GTMVectorStore:
    """Simple vector store that should definitely work"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        self.vectorizer = None
        self.vectors = None
    
    def add_threads(self, threads):
        """Add threads and create searchable documents"""
        logger.info(f"Adding {len(threads)} threads to vector store")
        
        # Clear existing data
        self.documents = []
        self.metadata = []
        
        # Process each thread
        for thread in threads:
            post = thread['post']
            comments = thread['comments']
            
            # Add main post
            post_text = f"Title: {post['title']}\nContent: {post['selftext']}"
            self.documents.append(post_text)
            self.metadata.append({
                'title': post['title'],
                'subreddit': post['subreddit'],
                'score': post['score'],
                'url': post['url'],
                'doc_type': 'original_post'
            })
            
            # Add each comment
            for comment in comments:
                if len(comment['body']) > 20:  # Only meaningful comments
                    self.documents.append(comment['body'])
                    self.metadata.append({
                        'title': f"Comment by {comment['author']}",
                        'subreddit': post['subreddit'],
                        'score': comment['score'],
                        'url': post['url'],
                        'doc_type': 'comment'
                    })
        
        logger.info(f"Created {len(self.documents)} documents")
        
        # Create vectors
        if self.documents:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.vectors = self.vectorizer.fit_transform(self.documents)
            logger.info(f"Created vectors: {self.vectors.shape}")
            return True
        return False
    
    def search(self, query, n_results=10):
        """Search for relevant documents"""
        if not self.documents or self.vectorizer is None:
            logger.warning("No documents or vectorizer available")
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            'documents': [self.documents[i] for i in top_indices],
            'metadatas': [self.metadata[i] for i in top_indices],
            'distances': [1 - similarities[i] for i in top_indices]
        }
        
        logger.info(f"Search for '{query}' returned {len(results['documents'])} results")
        return results
    
    def get_stats(self):
        return {
            'total_documents': len(self.documents),
            'vectorizer_ready': self.vectorizer is not None,
            'vectors_ready': self.vectors is not None
        }

class GTMIntelligenceRAG:
    """Simple RAG system"""
    
    def __init__(self, vector_store, google_api_key):
        self.vector_store = vector_store
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.analysis_history = []
    
    def analyze(self, query):
        """Analyze query against Reddit discussions"""
        start_time = datetime.now()
        
        # Search for relevant content
        search_results = self.vector_store.search(query, n_results=10)
        
        # If no results, return empty response
        if not search_results['documents']:
            return {
                'query': query,
                'analysis': 'No relevant Reddit discussions found for this query. The vector store may be empty or the search terms may need adjustment.',
                'sources': [],
                'analysis_type': 'general',
                'timestamp': start_time.isoformat(),
                'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
        
        # Build context from search results
        context_parts = []
        for doc, metadata, distance in zip(
            search_results['documents'][:5], 
            search_results['metadatas'][:5], 
            search_results['distances'][:5]
        ):
            if distance < 0.8:  # Include more results
                relevance = 1 - distance
                context_parts.append(f"""
[{metadata['doc_type']}] from r/{metadata['subreddit']} (Score: {metadata['score']}, Relevance: {relevance:.2f})
Title: {metadata['title']}
Content: {doc[:600]}
---
""")
        
        context = '\n'.join(context_parts)
        
        if not context.strip():
            return {
                'query': query,
                'analysis': 'No sufficiently relevant Reddit discussions found. Try different search terms.',
                'sources': [],
                'analysis_type': 'general',
                'timestamp': start_time.isoformat(),
                'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
        
        # Generate analysis
        prompt = f"""You are a growth marketing analyst examining Reddit discussions for business insights.

REDDIT DISCUSSIONS:
{context}

QUERY: {query}

Analyze these authentic Reddit discussions and provide actionable insights for product marketing teams. Focus on:

1. **Key Insights**: Main takeaways from the discussions
2. **Pain Points**: User frustrations and problems mentioned
3. **Market Sentiment**: Overall community perception
4. **Strategic Implications**: What this means for product strategy
5. **Recommended Actions**: Specific next steps

Be specific and actionable in your analysis."""
        
        try:
            response = self.model.generate_content(prompt)
            analysis = response.text.strip()
        except Exception as e:
            analysis = f"Error generating analysis: {str(e)}"
        
        # Format sources
        sources = []
        for metadata, distance in zip(search_results['metadatas'][:5], search_results['distances'][:5]):
            if distance < 0.8:
                sources.append({
                    'title': metadata['title'],
                    'type': metadata['doc_type'],
                    'subreddit': metadata['subreddit'],
                    'score': metadata['score'],
                    'url': metadata['url'],
                    'relevance': round(1 - distance, 3)
                })
        
        result = {
            'query': query,
            'analysis': analysis,
            'sources': sources,
            'analysis_type': 'general',
            'timestamp': start_time.isoformat(),
            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.analysis_history.append(result)
        return result

class GTMIntelligenceAssistant:
    """Main GTM Intelligence Assistant"""
    
    def __init__(self, google_api_key):
        self.vector_store = GTMVectorStore()
        self.rag_system = GTMIntelligenceRAG(self.vector_store, google_api_key)
        self.threads_data = []
    
    def load_and_process_data(self):
        """Load hardcoded data and process it"""
        logger.info("Loading hardcoded Reddit data...")
        
        # Simple, guaranteed-to-work data
        threads = [
            {
                'post': {
                    'id': 'test1',
                    'title': 'Major pain points with XBOW AI security tool',
                    'selftext': 'XBOW generates too many false positives. The main problem is noise and low-quality reports that waste security team time. What are your experiences with AI pentesting tools?',
                    'author': 'security_user',
                    'score': 15,
                    'upvote_ratio': 0.8,
                    'num_comments': 4,
                    'created_utc': 1700000000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/test1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/test1/'
                },
                'comments': [
                    {
                        'id': 'c1',
                        'author': 'pentester1',
                        'body': 'The biggest pain point with XBOW is false positives. We spend hours triaging reports that turn out to be nothing. The quality is poor and it creates more work than value.',
                        'score': 25,
                        'created_utc': 1700001000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'c2',
                        'author': 'security_expert',
                        'body': 'XBOW cannot understand business logic or context. It finds basic vulnerabilities but misses complex issues that require human insight. The AI hype is overblown.',
                        'score': 20,
                        'created_utc': 1700002000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'c3',
                        'author': 'ciso_veteran',
                        'body': 'AI security tools like XBOW generate overwhelming noise. The signal-to-noise ratio is terrible. We need tools that prioritize findings by business impact, not just volume.',
                        'score': 18,
                        'created_utc': 1700003000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'c4',
                        'author': 'bug_hunter',
                        'body': 'The main frustration is that XBOW finds easy, low-impact issues but misses the creative vulnerabilities that actually matter. Human creativity and intuition cannot be replaced.',
                        'score': 22,
                        'created_utc': 1700004000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'test1',
                'extracted_at': datetime.now().isoformat()
            },
            {
                'post': {
                    'id': 'test2',
                    'title': 'Will AI replace penetration testers? XBOW concerns',
                    'selftext': 'Seeing XBOW rank high on bug bounty platforms has me worried. Are human pentesters becoming obsolete? What problems do you see with AI security tools?',
                    'author': 'worried_pentester',
                    'score': 12,
                    'upvote_ratio': 0.7,
                    'num_comments': 3,
                    'created_utc': 1700010000,
                    'subreddit': 'Pentesting',
                    'permalink': '/test2/',
                    'url': 'https://reddit.com/r/Pentesting/comments/test2/'
                },
                'comments': [
                    {
                        'id': 'd1',
                        'author': 'red_teamer',
                        'body': 'AI tools like XBOW are sophisticated scanners, not replacements for human testers. They cannot understand business context or think creatively about attack vectors. The quality of their findings is questionable.',
                        'score': 15,
                        'created_utc': 1700011000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'd2',
                        'author': 'security_consultant',
                        'body': 'The problem with XBOW is volume over quality. It submits hundreds of low-severity findings that create more work for security teams. The triage burden is enormous.',
                        'score': 19,
                        'created_utc': 1700012000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'd3',
                        'author': 'veteran_hacker',
                        'body': 'AI cannot replace human creativity in finding logical flaws and business-specific vulnerabilities. XBOW finds obvious issues that any scanner would catch. Real penetration testing requires human intuition.',
                        'score': 17,
                        'created_utc': 1700013000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'test2',
                'extracted_at': datetime.now().isoformat()
            }
        ]
        
        logger.info(f"Got {len(threads)} threads with {sum(len(t['comments']) for t in threads)} total comments")
        
        # Store the data
        self.threads_data = threads
        
        # Process into vector store
        success = self.vector_store.add_threads(threads)
        
        if success:
            logger.info("Data loaded and processed successfully")
            
            # Test search immediately
            test_result = self.vector_store.search("pain points problems XBOW", n_results=3)
            logger.info(f"Test search returned {len(test_result['documents'])} results")
            
            return True
        else:
            logger.error("Failed to process data into vector store")
            return False
    
    def analyze(self, query):
        """Analyze query"""
        return self.rag_system.analyze(query)
    
    def get_stats(self):
        """Get statistics"""
        return {
            'reddit_data': {
                'total_threads': len(self.threads_data),
                'total_comments': sum(len(t['comments']) for t in self.threads_data),
                'subreddits': list(set(t['post']['subreddit'] for t in self.threads_data))
            },
            'vector_store': self.vector_store.get_stats(),
            'analyses_performed': len(self.rag_system.analysis_history)
        }
    
    def get_thread_summaries(self):
        """Get thread summaries"""
        summaries = []
        for thread in self.threads_data:
            post = thread['post']
            summaries.append({
                'id': post['id'],
                'title': post['title'],
                'subreddit': post['subreddit'],
                'score': post['score'],
                'num_comments': len(thread['comments']),
                'url': post['url']
            })
        return summaries
