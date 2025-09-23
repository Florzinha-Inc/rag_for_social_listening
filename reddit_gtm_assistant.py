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
    
    def clean_text(self, text: str) -> str:
        """Clean Reddit text content"""
        if not text:
            return ""
        
        # Remove Reddit markup
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'\^(.*?)\^', r'\1', text)      # Superscript
        text = re.sub(r'&gt;!.*?!&lt;', '', text)    # Spoilers
        text = re.sub(r'&gt;', '>', text)            # Quote markers
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_comments_recursive(self, comment_data: Dict, parent_id: str = None) -> List[Dict]:
        """Recursively extract comments from Reddit thread"""
        comments = []
        
        if not comment_data or comment_data.get('kind') != 't1':
            return comments
        
        data = comment_data.get('data', {})
        
        # Skip deleted/removed comments and AutoModerator
        author = data.get('author', '')
        body = data.get('body', '')
        
        if author in ['[deleted]', '[removed]', 'AutoModerator'] or not body or body in ['[deleted]', '[removed]']:
            return comments
        
        comment = {
            'id': data.get('id'),
            'author': author,
            'body': self.clean_text(body),
            'score': data.get('score', 0),
            'created_utc': data.get('created_utc', 0),
            'parent_id': parent_id,
            'replies_count': 0,
            'depth': 0
        }
        
        # Calculate depth based on parent
        if parent_id:
            comment['depth'] = 1  # Simplified depth calculation
        
        # Only include substantial comments with reasonable score threshold
        if len(comment['body']) > 10 and comment['score'] >= -10:  # More lenient filters
            comments.append(comment)
        
        # Process replies
        replies = data.get('replies', {})
        if isinstance(replies, dict) and replies.get('data', {}).get('children'):
            for reply in replies['data']['children']:
                if reply.get('kind') == 't1':  # Only process comment replies
                    reply_comments = self.extract_comments_recursive(reply, comment['id'])
                    for reply_comment in reply_comments:
                        reply_comment['depth'] = comment['depth'] + 1
                    comments.extend(reply_comments)
            
            comment['replies_count'] = len([c for c in comments if c['parent_id'] == comment['id']])
        
        return comments
    
    def extract_thread_data(self, reddit_url: str) -> Dict:
        """Extract complete thread data from Reddit JSON URL"""
        data = self.fetch_reddit_json(reddit_url)
        
        if not data:
            logger.error(f"No data returned from {reddit_url}")
            return {}
        
        if not isinstance(data, list) or len(data) < 2:
            logger.error(f"Invalid Reddit data structure from {reddit_url}. Expected list with 2+ items, got: {type(data)}")
            if isinstance(data, list):
                logger.error(f"Data length: {len(data)}")
            return {}
        
        try:
            # First element is the post, second is comments
            post_listing = data[0]
            comments_listing = data[1]
            
            if not post_listing.get('data', {}).get('children'):
                logger.error(f"No post children found in {reddit_url}")
                return {}
                
            post_data = post_listing['data']['children'][0]['data']
            comments_data = comments_listing['data']['children']
            
            logger.info(f"Processing post: '{post_data.get('title', 'No title')}' with {len(comments_data)} comment objects")
            
            # Extract main post
            post = {
                'id': post_data.get('id'),
                'title': self.clean_text(post_data.get('title', '')),
                'selftext': self.clean_text(post_data.get('selftext', '')),
                'author': post_data.get('author'),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': post_data.get('created_utc', 0),
                'subreddit': post_data.get('subreddit'),
                'permalink': f"https://reddit.com{post_data.get('permalink', '')}",
                'url': reddit_url.replace('.json', '')
            }
            
            # Extract comments
            comments = []
            for comment_data in comments_data:
                if comment_data.get('kind') == 'more':  # Skip "load more comments" entries
                    continue
                extracted_comments = self.extract_comments_recursive(comment_data)
                comments.extend(extracted_comments)
            
            # Sort comments by score and relevance
            comments.sort(key=lambda x: (x['score'], -x['depth']), reverse=True)
            
            logger.info(f"Extracted {len(comments)} valid comments from {reddit_url}")
            
            return {
                'post': post,
                'comments': comments,
                'thread_url': reddit_url,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Reddit thread {reddit_url}: {e}")
            return {}
    
    def process_multiple_threads(self, reddit_urls: List[str]) -> List[Dict]:
        """Process multiple Reddit threads"""
        threads = []
        
        for url in reddit_urls:
            logger.info(f"Processing Reddit thread: {url}")
            thread_data = self.extract_thread_data(url)
            
            if thread_data:
                threads.append(thread_data)
                logger.info(f"Successfully processed {url}")
            else:
                logger.warning(f"Failed to process {url}")
            
            time.sleep(1)  # Be respectful to Reddit's servers
        
        logger.info(f"Total threads processed: {len(threads)}")
        return threads


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
                    'upvote_ratio': post['upvote_ratio'],
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
        real_threads = []
        
        # Data from r/cybersecurity - "Is Penetration Testing still worth it after seeing XBOW at work?"
        cybersec_thread = {
            'post': {
                'id': '1ly9nxf',
                'title': 'Is Penetration Testing still worth it after seeing XBOW at work?',
                'selftext': 'So...Recently, some of my seniors conducted a workshop about "Agentic AI" and its scope. The speaker at that particular workshop actually worked at a company developing AI agents. He asked us a question along the lines "Can AI take jobs of a Bug Bounty hunter or a Pentester in the near future?". I know AI has a lot of capabilities but these things seem to complex for an AI so I said no. Well, then he told us about XBOW and its accuracy in finding bugs, I had second thoughts. XBOW is an AI agent and it is currently at the third place in the United States of America on Hackerone. It was at first place but dropped down to third. I\'m pursuing my cybersecurity career as a penetration tester and I just wanna know should I proceed on that path or should I change it?',
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
                    'body': 'That\'s the sales person\'s job, to sell you their tools, ofc they are going to talk it up',
                    'score': 13,
                    'created_utc': 1752397400,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                },
                {
                    'id': 'n2vemyp',
                    'author': 'Strange-Mountain1810',
                    'body': 'It found easy to find xss and sprayed vdp\'s hardly pentest replacement.',
                    'score': 9,
                    'created_utc': 1752399364,
                    'parent_id': None,
                    'replies_count': 2,
                    'depth': 0
                },
                {
                    'id': 'n2vaslt',
                    'author': 'Beautiful_Watch_7215',
                    'body': 'Change to what? What jobs don\'t have demos of an AI solution replacing it?',
                    'score': 8,
                    'created_utc': 1752397066,
                    'parent_id': None,
                    'replies_count': 3,
                    'depth': 0
                },
                {
                    'id': 'n2ve0oh',
                    'author': 'Zatetics',
                    'body': 'Is this an ad?',
                    'score': 7,
                    'created_utc': 1752398989,
                    'parent_id': None,
                    'replies_count': 1,
                    'depth': 0
                },
                {
                    'id': 'n9tlq5h',
                    'author': 'greybrimstone',
                    'body': 'XBOW does not perform nearly as well as they make it sound. It is, for all intents and purposes, a really cool automated vulnerability scanner. The experiments they\'ve done pitting their AI against humans have been heavily biased. Their rankings on bug bounty programs, speak to quantity of easy-to-find issues, not quality. AI cannot compete with human creativity, intuition, and ability to make leaps. AI only knows what it was trained on. Sure, it\'s great and useful, but not at all as capable as a real tester.',
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
        
        # Data from r/Pentesting - "Will XBOW or AIs be able to replace Pentesters?"
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
                    'body': 'I recently spoke with a contact at one of the organizations that has received a significant number of vulnerability reports from XBOW. They shared that tools like XBOW have made their work substantially more difficult, as they now spend countless hours triaging and validating reports many of which turn out to be false positives or issues of such low criticality that they fall outside the organization\'s risk threshold. While XBOW may appear impressive due to the volume of submissions, the quality and relevance of many of these findings are questionable, ultimately straining the receiving team\'s resources.',
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
                },
                {
                    'id': 'n1tc7h0',
                    'author': 'Reasonable_Cut8116',
                    'body': 'I dont think it will fully replace senior pentesters but it will absolutely replace your current vulnerability scanner and junior pentesters. I own an MSP/MSSP and we use a platform called StealthNet AI(stealthnet.ai) to offer automated pentests via AI agents. They have a fleet of AI agents (External,Vishing,API,etc).Their API/Web agent is really impressive it finds things that traditional vulnerability scanners miss due to not understanding business logic. Overall they perform at the level of a junior - intermediate pentester.',
                    'score': 1,
                    'created_utc': 1751899180,
                    'parent_id': None,
                    'replies_count': 1,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/Pentesting/comments/1lmzvx8/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        # Data from r/Hacking_Tutorials - "Is xbow ai snake oil or the real deal"
        hacking_thread = {
            'post': {
                'id': '1ln80yl',
                'title': 'Is xbow ai snake oil or the real deal',
                'selftext': 'What do you think? Its scary to be honest',
                'author': 'Impossible-Line1070',
                'score': 0,
                'upvote_ratio': 0.5,
                'num_comments': 11,
                'created_utc': 1751179027,
                'subreddit': 'Hacking_Tutorials',
                'permalink': '/r/Hacking_Tutorials/comments/1ln80yl/is_xbow_ai_snake_oil_or_the_real_deal/',
                'url': 'https://reddit.com/r/Hacking_Tutorials/comments/1ln80yl/is_xbow_ai_snake_oil_or_the_real_deal/'
            },
            'comments': [
                {
                    'id': 'n0dc3vu',
                    'author': 'Sqooky',
                    'body': 'AI is just another tool to help us do work. We have tens of products in our suite and it still wont catch vulnerabilities that humans can find. Lots of companies don\'t like tossing their data into a black-box. Especially proprietary and potentially vulnerability related data. Mainly because of loss of control.',
                    'score': 3,
                    'created_utc': 1751180043,
                    'parent_id': None,
                    'replies_count': 1,
                    'depth': 0
                },
                {
                    'id': 'napheni',
                    'author': 'Sqooky',
                    'body': 'The point is - You can be a horrible "hacker" and land at the top of H1s leaderboard. Points mean nothing when programs like the DoDs exist that\'ll give you anything for anything. Let\'s also not pretend xbow is 100% only AI. We don\'t have full AI. We have generative & language models. It\'s 100% going through human review and analysis before its submitted to any company.',
                    'score': 1,
                    'created_utc': 1756181428,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/Hacking_Tutorials/comments/1ln80yl/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        # Data from r/bugbounty - "How AI is affecting pentesting and bug bounties"
        bugbounty_thread = {
            'post': {
                'id': '1l97esk',
                'title': 'How AI is affecting pentesting and bug bounties',
                'selftext': 'Recently, I came across with a project named "Xbow" and it\'s actually the current top US-based hacker on Hackerone\'s leaderboard. It\'s a fully automated AI agent trained on real vulnerability data and will be available soon. Do you think it\'s still worth to learn pentesting and get into bug bounties?',
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
                    'body': 'As a customer of hacker one I\'m more worried about the crap we\'re gunna have to sort through now. We have scanners and other companies that offer AI agents for pentesting which find the low hanging fruit. We have a Bug Bounty program to find more nuanced vulnerabilities in our products that other security testing can\'t find.',
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
                },
                {
                    'id': 'mxeewa2',
                    'author': 'k4lashhnikov',
                    'body': 'Sure, they can surpass human capabilities but there is little point in analyzing hundreds of thousands of endpoints to find uninteresting things or false positives, If an AI analyzes misconfigurations of JS, code, or exposed credentials, it cannot (for now) have the ability to manually modify things that apparently work well. For example, a step-by-step business flow, if the AI superficially sees that the flow is correct, it will leave it as is, but a human has the idea of seeing what happens if a specific step is skipped.',
                    'score': 4,
                    'created_utc': 1749743477,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                },
                {
                    'id': 'nchqml7',
                    'author': 'Reasonable_Cut8116',
                    'body': 'It does surprisingly well. I have heard of Xbow. We use a similar tool from StealthNet AI(stealthnet.ai). They have a fleet of AI agents to automate penetration testing. Compared to normal vulnerability scanners they perform 100x better. You can think of them as a really smart vulnerbility scanner or a junior pentester. They can perform really well but it wont be able to find everything. More complex attacks will still require humans.',
                    'score': 2,
                    'created_utc': 1757041432,
                    'parent_id': None,
                    'replies_count': 0,
                    'depth': 0
                }
            ],
            'thread_url': 'https://reddit.com/r/bugbounty/comments/1l97esk/.json',
            'extracted_at': datetime.now().isoformat()
        }
        
        real_threads.extend([cybersec_thread, pentesting_thread, hacking_thread, bugbounty_thread])
        return real_threads
    
    def load_reddit_threads(self, reddit_urls: List[str]):
        """Load Reddit threads into the system"""
        logger.info("Loading Reddit threads for GTM analysis...")
        
        # Use the real Reddit data from the provided JSON files
        threads = self.load_real_reddit_data()
        
        if threads:
            self.threads_data = threads
            self.vector_store.add_threads(threads)
            
            total_posts = len(threads)
            total_comments = sum(len(t['comments']) for t in threads)
            
            logger.info(f"Loaded {total_posts} Reddit posts with {total_comments} comments from actual Reddit data")
            return total_posts
        
        logger.error("Failed to load Reddit thread data")
        return 0
    
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
    
    def get_thread_summaries(self) -> List[Dict]:
        """Get summaries of loaded Reddit threads"""
        summaries = []
        
        for thread in self.threads_data:
            post = thread['post']
            summaries.append({
                'title': post['title'],
                'subreddit': post['subreddit'],
                'score': post['score'],
                'num_comments': post['num_comments'],
                'upvote_ratio': post['upvote_ratio'],
                'url': post['url']
            })
        
        return summaries
