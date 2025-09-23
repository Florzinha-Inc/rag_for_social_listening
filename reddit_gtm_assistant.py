import os
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time
import re
from collections import defaultdict

# AI and Vector Search
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditExtractor:
    """Extract content from Reddit JSON APIs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'GTMIntelligenceBot/1.0 (Growth Marketing Research)'
        }
    
    def fetch_reddit_json(self, url: str) -> Dict:
        """Fetch Reddit thread JSON data"""
        try:
            logger.info(f"Fetching Reddit data from: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched Reddit data. Response length: {len(str(data))}")
                return data
            else:
                logger.error(f"Error fetching Reddit data from {url}: {response.status_code} - {response.text[:500]}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching Reddit JSON from {url}: {e}")
            return {}


class GTMVectorStore:
    """Vector store optimized for GTM intelligence analysis"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=1
        )
        self.documents = []
        self.document_vectors = None
        self.metadata = []
    
    def create_document_from_thread(self, thread_data: Dict) -> List[Dict]:
        """Convert Reddit thread to searchable documents"""
        documents = []
        
        post = thread_data['post']
        comments = thread_data['comments']
        
        # Main post as document
        post_content = f"Title: {post['title']}\n\n{post['selftext']}"
        if len(post_content.strip()) > 50:
            documents.append({
                'id': f"post_{post['id']}",
                'content': post_content,
                'type': 'original_post',
                'metadata': {
                    'title': post['title'],
                    'subreddit': post['subreddit'],
                    'score': post['score'],
                    'num_comments': post['num_comments'],
                    'upvote_ratio': post.get('upvote_ratio', 0),
                    'url': post['url'],
                    'created_utc': post['created_utc']
                }
            })
        
        # High-value comments as documents
        for comment in comments[:50]:  # Top 50 comments
            if len(comment['body']) > 100 and comment['score'] > 1:
                documents.append({
                    'id': f"comment_{comment['id']}",
                    'content': comment['body'],
                    'type': 'comment',
                    'metadata': {
                        'title': f"Comment by {comment['author']}",
                        'subreddit': post['subreddit'],
                        'score': comment['score'],
                        'depth': comment['depth'],
                        'replies_count': comment['replies_count'],
                        'url': f"{post['url']}/comments/{comment['id']}",
                        'created_utc': comment['created_utc'],
                        'parent_post_title': post['title']
                    }
                })
        
        # Create conversation clusters (groups of related comments)
        if len(comments) > 10:
            cluster_content = []
            cluster_scores = []
            
            for comment in comments[:20]:
                if len(comment['body']) > 50:
                    cluster_content.append(comment['body'])
                    cluster_scores.append(comment['score'])
            
            if cluster_content:
                combined_content = "\n\n---\n\n".join(cluster_content)
                avg_score = sum(cluster_scores) / len(cluster_scores)
                
                documents.append({
                    'id': f"cluster_{post['id']}",
                    'content': f"Discussion thread about: {post['title']}\n\n{combined_content}",
                    'type': 'conversation_cluster',
                    'metadata': {
                        'title': f"Discussion: {post['title']}",
                        'subreddit': post['subreddit'],
                        'score': avg_score,
                        'comment_count': len(cluster_content),
                        'url': post['url'],
                        'created_utc': post['created_utc']
                    }
                })
        
        return documents
    
    def add_threads(self, threads: List[Dict]):
        """Add Reddit threads to vector store"""
        all_documents = []
        all_metadata = []
        
        logger.info(f"Adding {len(threads)} threads to vector store")
        
        for thread_idx, thread in enumerate(threads):
            logger.info(f"Processing thread {thread_idx + 1}: {thread['post']['title'][:50]}...")
            docs = self.create_document_from_thread(thread)
            logger.info(f"Created {len(docs)} documents from thread")
            
            for doc in docs:
                all_documents.append(doc['content'])
                
                metadata = doc['metadata'].copy()
                metadata.update({
                    'doc_id': doc['id'],
                    'doc_type': doc['type']
                })
                all_metadata.append(metadata)
        
        if all_documents:
            self.documents = all_documents
            self.metadata = all_metadata
            
            logger.info(f"Creating TF-IDF vectors for {len(all_documents)} documents")
            
            # Create TF-IDF vectors
            self.document_vectors = self.vectorizer.fit_transform(all_documents)
            logger.info(f"Successfully created vector store with {len(all_documents)} documents")
            
            # Log some sample document info for debugging
            for i, (doc, meta) in enumerate(zip(all_documents[:3], all_metadata[:3])):
                logger.info(f"Sample doc {i+1}: {meta['doc_type']} - {doc[:100]}...")
        else:
            logger.error("No documents were created from threads!")
    
    def search(self, query: str, n_results: int = 10) -> Dict:
        """Search for relevant discussions"""
        if not self.documents or self.document_vectors is None:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            'documents': [self.documents[i] for i in top_indices],
            'metadatas': [self.metadata[i] for i in top_indices],
            'distances': [1 - similarities[i] for i in top_indices]
        }
        
        return results
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        if not self.metadata:
            return {'total_documents': 0}
        
        doc_types = defaultdict(int)
        subreddits = defaultdict(int)
        
        for meta in self.metadata:
            doc_types[meta['doc_type']] += 1
            subreddits[meta['subreddit']] += 1
        
        return {
            'total_documents': len(self.documents),
            'document_types': dict(doc_types),
            'subreddits': dict(subreddits)
        }


class GTMIntelligenceRAG:
    """RAG system specialized for GTM intelligence from Reddit discussions"""
    
    def __init__(self, vector_store: GTMVectorStore, google_api_key: str):
        self.vector_store = vector_store
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.analysis_history = []
        
        # GTM-focused prompts
        self.prompts = {
            'pain_points': """You are a growth marketing analyst specializing in identifying customer pain points from community discussions.
Analyze the Reddit discussions and identify key pain points, frustrations, and unmet needs. Focus on actionable insights for product marketing teams.""",
            
            'feature_gaps': """You are a product marketing strategist analyzing community feedback for feature gaps and opportunities.
Identify missing features, desired functionality, and competitive gaps mentioned in discussions. Prioritize insights that could inform product roadmap decisions.""",
            
            'sentiment': """You are a market research analyst specializing in sentiment analysis from developer communities.
Analyze the overall sentiment, adoption patterns, and community perception. Identify trends in user satisfaction and market reception.""",
            
            'competitive': """You are a competitive intelligence analyst examining community discussions about market positioning.
Identify how products are positioned against alternatives, what users prefer, and competitive advantages/disadvantages mentioned in authentic conversations.""",
            
            'general': """You are a growth marketing intelligence analyst examining Reddit community discussions.
Provide strategic insights for product marketing teams based on authentic user conversations. Focus on actionable intelligence for GTM strategy."""
        }
    
    def determine_analysis_type(self, query: str) -> str:
        """Classify query type for appropriate analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['pain', 'problem', 'frustrat', 'issue', 'challenge', 'difficult']):
            return 'pain_points'
        elif any(word in query_lower for word in ['feature', 'gap', 'missing', 'need', 'want', 'wish', 'roadmap']):
            return 'feature_gaps'
        elif any(word in query_lower for word in ['sentiment', 'opinion', 'feel', 'think', 'perception', 'reaction']):
            return 'sentiment'
        elif any(word in query_lower for word in ['competitor', 'alternative', 'versus', 'vs', 'compare', 'better']):
            return 'competitive'
        else:
            return 'general'
    
    def format_reddit_context(self, search_results: Dict) -> str:
        """Format Reddit search results for GTM analysis"""
        context_parts = []
        
        for doc, metadata, distance in zip(
            search_results['documents'],
            search_results['metadatas'], 
            search_results['distances']
        ):
            if distance < 0.7:  # Relevance threshold
                relevance = 1 - distance
                
                context_part = f"""
[{metadata['doc_type'].upper()}] from r/{metadata['subreddit']} (Score: {metadata['score']}, Relevance: {relevance:.2f})
Title: {metadata['title']}
Content: {doc[:800]}{'...' if len(doc) > 800 else ''}
URL: {metadata['url']}
---
"""
                context_parts.append(context_part)
        
        return '\n'.join(context_parts)
    
    def generate_gtm_analysis(self, query: str, context: str, analysis_type: str) -> str:
        """Generate GTM-focused analysis using Gemini"""
        system_prompt = self.prompts[analysis_type]
        
        full_prompt = f"""{system_prompt}

CONTEXT FROM REDDIT DISCUSSIONS:
{context}

ANALYSIS REQUEST: {query}

Provide actionable GTM intelligence based on these authentic community discussions. Structure your response with:

1. **Key Insights**: Main takeaways from the discussions
2. **Strategic Implications**: How this affects GTM strategy
3. **Recommended Actions**: Specific next steps for product marketing teams
4. **Evidence Summary**: Brief summary of supporting evidence from discussions

Focus on insights that would help product marketers understand market dynamics, user needs, and competitive positioning. Be specific and actionable."""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating GTM analysis: {str(e)}"
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Main method for GTM intelligence analysis"""
        start_time = datetime.now()
        
        # Determine analysis type
        analysis_type = self.determine_analysis_type(query)
        
        # Search Reddit discussions
        search_results = self.vector_store.search(query, n_results=12)
        
        # Format context
        context = self.format_reddit_context(search_results)
        
        if not context.strip():
            return {
                'query': query,
                'analysis': 'No relevant Reddit discussions found for this query. Try adjusting your search terms or check if the data has been loaded.',
                'sources': [],
                'analysis_type': analysis_type,
                'timestamp': start_time.isoformat(),
                'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
        
        # Generate analysis
        analysis = self.generate_gtm_analysis(query, context, analysis_type)
        
        # Format sources
        sources = []
        seen_titles = set()
        
        for metadata, distance in zip(search_results['metadatas'], search_results['distances']):
            if distance < 0.7:
                title = metadata['title']
                if title not in seen_titles:
                    sources.append({
                        'title': title,
                        'type': metadata['doc_type'],
                        'subreddit': metadata['subreddit'],
                        'score': metadata['score'],
                        'url': metadata['url'],
                        'relevance': round(1 - distance, 3)
                    })
                    seen_titles.add(title)
        
        result = {
            'query': query,
            'analysis': analysis,
            'sources': sources,
            'analysis_type': analysis_type,
            'timestamp': start_time.isoformat(),
            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.analysis_history.append(result)
        return result


class GTMIntelligenceAssistant:
    """Main GTM Intelligence Assistant application"""
    
    def __init__(self, google_api_key: str):
        self.reddit_extractor = RedditExtractor()
        self.vector_store = GTMVectorStore()
        self.rag_system = GTMIntelligenceRAG(self.vector_store, google_api_key)
        self.threads_data = []
    
    def load_real_reddit_data(self) -> List[Dict]:
        """Load the actual Reddit data from the provided JSON files"""
        logger.info("Starting to load real Reddit data...")
        
        real_threads = []
        
        # Data from r/cybersecurity
        cybersec_thread = {
            'post': {
                'id': '1ly9nxf',
                'title': 'Is Penetration Testing still worth it after seeing XBOW at work?',
                'selftext': 'So...Recently, some of my seniors conducted a workshop about "Agentic AI" and its scope. The speaker at that particular workshop actually worked at a company developing AI agents. He asked us a question along the lines "Can AI take jobs of a Bug Bounty hunter or a Pentester in the near future?". I know AI has a lot of capabilities but these things seem to complex for an AI so I said no. Well, then he told us about XBOW and its accuracy in finding bugs, I had second thoughts. XBOW is an AI agent and it is currently at the third place in the United States of America on Hackerone. It was at first place but dropped down to third. I am pursuing my cybersecurity career as a penetration tester and I just wanna know should I proceed on that path or should I change it?',
                'author': 'RayyanRA7',
                'score': 0,
                'upvote_ratio': 0.39,
                'num_comments': 17,
                'created_utc': 1752350549,
                'subreddit': 'cybersecurity',
                'permalink': '/r/cybersecurity/comments/1ly9nxf/is_penetration_testing_still_worth_it_after/',
                'url': 'https://reddit.com/r/cybersecurity/comments/1ly9nxf/is_penetration_testing_still_worth_it_after/'
            },
            'comments': [
                {
                    'id': 'n2vbczt',
                    'author': 'cloudfox1',
                    'body': 'That is the sales person job, to sell you their tools, ofc they are going to talk it up',
                    'score': 13,
                    'created_utc': 1752397400,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                },
                {
                    'id': 'n2vemyp',
                    'author': 'Strange-Mountain1810',
                    'body': 'It found easy to find xss and sprayed vdp hardly pentest replacement.',
                    'score': 9,
                    'created_utc': 1752399364,
                    'parent_id': None,
                    'replies_count': 2,
                    'depth': 0
                },
                {
                    'id': 'n9tlq5h',
                    'author': 'greybrimstone',
                    'body': 'XBOW does not perform nearly as well as they make it sound. It is, for all intents and purposes, a really cool automated vulnerability scanner. The experiments they have done pitting their AI against humans have been heavily biased. Their rankings on bug bounty programs, speak to quantity of easy-to-find issues, not quality. AI cannot compete with human creativity, intuition, and ability to make leaps. AI only knows what it was trained on. Sure, it is great and useful, but not at all as capable as a real tester.',
                    'score': 2,
                    'created_utc': 1755741602,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/cybersecurity/comments/1ly9nxf/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        # Data from r/Pentesting
        pentesting_thread = {
            'post': {
                'id': '1lmzvx8',
                'title': 'Will XBOW or AIs be able to replace Pentesters?',
                'selftext': 'How do you see the future of Pentesters with this trend of AIs that do not stop coming out.',
                'author': 'bjnc_',
                'score': 0,
                'upvote_ratio': 0.33,
                'num_comments': 15,
                'created_utc': 1751151764,
                'subreddit': 'Pentesting',
                'permalink': '/r/Pentesting/comments/1lmzvx8/will_xbow_or_ais_be_able_to_replace_pentesters/',
                'url': 'https://reddit.com/r/Pentesting/comments/1lmzvx8/will_xbow_or_ais_be_able_to_replace_pentesters/'
            },
            'comments': [
                {
                    'id': 'n0g90j3',
                    'author': 'Clean-Drop9629',
                    'body': 'I recently spoke with a contact at one of the organizations that has received a significant number of vulnerability reports from XBOW. They shared that tools like XBOW have made their work substantially more difficult, as they now spend countless hours triaging and validating reports many of which turn out to be false positives or issues of such low criticality that they fall outside the organization risk threshold. While XBOW may appear impressive due to the volume of submissions, the quality and relevance of many of these findings are questionable, ultimately straining the receiving team resources.',
                    'score': 3,
                    'created_utc': 1751223076,
                    'parent_id': None,
                    'replies_count': 1,
                    'depth': 0
                },
                {
                    'id': 'n0bj9ad',
                    'author': 'AttackForge',
                    'body': 'They will never be able to test for business logic and design flaws.',
                    'score': 4,
                    'created_utc': 1751152037,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/Pentesting/comments/1lmzvx8/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        # Data from r/bugbounty
        bugbounty_thread = {
            'post': {
                'id': '1l97esk',
                'title': 'How AI is affecting pentesting and bug bounties',
                'selftext': 'Recently, I came across with a project named XBOW and it is actually the current top US-based hacker on Hackerone leaderboard. It is a fully automated AI agent trained on real vulnerability data and will be available soon. Do you think it is still worth to learn pentesting and get into bug bounties?',
                'author': 'S4vz4d',
                'score': 14,
                'upvote_ratio': 0.7,
                'num_comments': 12,
                'created_utc': 1749684165,
                'subreddit': 'bugbounty',
                'permalink': '/r/bugbounty/comments/1l97esk/how_ai_is_affecting_pentesting_and_bug_bounties/',
                'url': 'https://reddit.com/r/bugbounty/comments/1l97esk/how_ai_is_affecting_pentesting_and_bug_bounties/'
            },
            'comments': [
                {
                    'id': 'mxcbgt2',
                    'author': 'chopper332nd',
                    'body': 'As a customer of hacker one I am more worried about the crap we are gunna have to sort through now. We have scanners and other companies that offer AI agents for pentesting which find the low hanging fruit. We have a Bug Bounty program to find more nuanced vulnerabilities in our products that other security testing cannot find.',
                    'score': 21,
                    'created_utc': 1749712425,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                },
                {
                    'id': 'mxbvg2p',
                    'author': 'k4lashhnikov',
                    'body': 'The human factor is always required for logic errors, vertical or horizontal scaling, AI and automated tools cannot understand the business context. If AIs have vulnerabilities and are not imperfect, what makes you think they will replace the human hacker?',
                    'score': 14,
                    'created_utc': 1749703837,
                    'parent_id': None,
                    'replies_count': 1,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/bugbounty/comments/1l97esk/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        real_threads.extend([cybersec_thread, pentesting_thread, bugbounty_thread])
        
        logger.info(f"Loaded {len(real_threads)} threads with real Reddit data")
        for thread in real_threads:
            logger.info(f"Thread: {thread['post']['title']} from r/{thread['post']['subreddit']} with {len(thread['comments'])} comments")
        
        return real_threads
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze Reddit discussions for GTM intelligence"""
        return self.rag_system.analyze(query)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'reddit_data': {
                'total_threads': len(self.threads_data),
                'total_comments': sum(len(t['comments']) for t in self.threads_data),
                'subreddits': list(set(t['post']['subreddit'] for t in self.threads_data))
            },
            'vector_store': self.vector_store.get_stats(),
            'analyses_performed': len(self.rag_system.analysis_history)
        }
        return stats
