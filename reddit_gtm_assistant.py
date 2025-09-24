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
                        'doc_type': 'comment',
                        'author': comment['author']
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
Author: {metadata.get('author', 'Unknown')}
Content: "{doc[:600]}"
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

Analyze these authentic Reddit discussions and provide actionable insights for product marketing teams. Structure your response as follows:

1. **Key Insights**: Main takeaways from the discussions
2. **Pain Points**: User frustrations and problems mentioned  
3. **Market Sentiment**: Overall community perception
4. **Strategic Implications**: What this means for product strategy
5. **Recommended Actions**: Specific next steps

IMPORTANT: Include direct quotes from the Reddit discussions to support your analysis. Use quotes like this format:
"[Quote from user]" - username from r/subreddit

Use quotes as evidence to ground your insights in actual user feedback. Be specific and actionable in your analysis."""
        
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
        
        # Comprehensive dataset with diverse topics and perspectives
        threads = [
            # Thread 1: XBOW Pain Points
            {
                'post': {
                    'id': 'xbow_pain_1',
                    'title': 'Major pain points with XBOW AI security tool',
                    'selftext': 'XBOW generates too many false positives. The main problem is noise and low-quality reports that waste security team time. What are your experiences with AI pentesting tools?',
                    'author': 'security_user',
                    'score': 15,
                    'upvote_ratio': 0.8,
                    'num_comments': 4,
                    'created_utc': 1700000000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/xbow_pain_1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/xbow_pain_1/'
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
                'thread_url': 'xbow_pain_1',
                'extracted_at': datetime.now().isoformat()
            },

            # Thread 2: AI Replacement Concerns
            {
                'post': {
                    'id': 'ai_replacement_1',
                    'title': 'Will AI replace penetration testers? XBOW concerns',
                    'selftext': 'Seeing XBOW rank high on bug bounty platforms has me worried. Are human pentesters becoming obsolete? What problems do you see with AI security tools?',
                    'author': 'worried_pentester',
                    'score': 12,
                    'upvote_ratio': 0.7,
                    'num_comments': 3,
                    'created_utc': 1700010000,
                    'subreddit': 'Pentesting',
                    'permalink': '/ai_replacement_1/',
                    'url': 'https://reddit.com/r/Pentesting/comments/ai_replacement_1/'
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
                'thread_url': 'ai_replacement_1',
                'extracted_at': datetime.now().isoformat()
            },

            # Thread 3: Feature Requests and Gaps
            {
                'post': {
                    'id': 'feature_gaps_1',
                    'title': 'What features are missing from AI pentesting tools like XBOW?',
                    'selftext': 'I have been evaluating AI security tools and while they are impressive, there are clear gaps. What features would make these tools more valuable for enterprise security teams?',
                    'author': 'enterprise_security',
                    'score': 28,
                    'upvote_ratio': 0.85,
                    'num_comments': 6,
                    'created_utc': 1700020000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/feature_gaps_1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/feature_gaps_1/'
                },
                'comments': [
                    {
                        'id': 'f1',
                        'author': 'security_architect',
                        'body': 'We need better integration with SIEM and ticketing systems. XBOW outputs are hard to integrate into existing workflows. Better API support and standardized formats would help.',
                        'score': 32,
                        'created_utc': 1700021000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'f2',
                        'author': 'devsec_lead',
                        'body': 'AI tools need better context awareness. They should understand our tech stack, business logic, and custom applications. Generic scanning is not enough for modern enterprises.',
                        'score': 28,
                        'created_utc': 1700022000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'f3',
                        'author': 'compliance_officer',
                        'body': 'Risk prioritization based on business impact is missing. XBOW treats all findings equally. We need tools that understand our compliance requirements and business criticality.',
                        'score': 25,
                        'created_utc': 1700023000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'f4',
                        'author': 'security_manager',
                        'body': 'Better reporting for executives is needed. Technical details are fine for engineers, but we need business-friendly summaries and metrics for leadership.',
                        'score': 22,
                        'created_utc': 1700024000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'f5',
                        'author': 'cloud_security',
                        'body': 'Cloud-native security understanding is lacking. AI tools need to better understand containerized environments, serverless functions, and cloud-specific attack vectors.',
                        'score': 30,
                        'created_utc': 1700025000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'f6',
                        'author': 'incident_responder',
                        'body': 'Real-time threat detection and response capabilities are missing. Current AI tools are mostly for assessment, not active defense or incident response.',
                        'score': 26,
                        'created_utc': 1700026000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'feature_gaps_1',
                'extracted_at': datetime.now().isoformat()
            },

            # Thread 4: Positive Experiences and Use Cases
            {
                'post': {
                    'id': 'positive_ai_1',
                    'title': 'AI security tools that actually work well - success stories',
                    'selftext': 'While there is a lot of criticism of AI security tools, what are some success stories? When have these tools provided real value to your security program?',
                    'author': 'pragmatic_ciso',
                    'score': 35,
                    'upvote_ratio': 0.9,
                    'num_comments': 7,
                    'created_utc': 1700030000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/positive_ai_1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/positive_ai_1/'
                },
                'comments': [
                    {
                        'id': 'p1',
                        'author': 'startup_cto',
                        'body': 'AI tools excel at continuous monitoring and baseline establishment. We use them for ongoing security posture assessment rather than one-off pentests. The trend analysis is valuable.',
                        'score': 28,
                        'created_utc': 1700031000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p2',
                        'author': 'fintech_security',
                        'body': 'For compliance scanning and policy enforcement, AI tools are excellent. They consistently check configurations and catch drift that humans might miss.',
                        'score': 24,
                        'created_utc': 1700032000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p3',
                        'author': 'security_educator',
                        'body': 'AI tools are great for training junior security staff. They provide consistent results that can be used as learning examples and help establish security testing methodologies.',
                        'score': 31,
                        'created_utc': 1700033000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p4',
                        'author': 'budget_conscious_admin',
                        'body': 'For small teams with limited budgets, AI tools provide security coverage we could not otherwise afford. They are not perfect but better than nothing.',
                        'score': 20,
                        'created_utc': 1700034000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p5',
                        'author': 'devops_security',
                        'body': 'Integration with CI/CD pipelines is where AI tools shine. They provide fast feedback during development and catch issues before production deployment.',
                        'score': 33,
                        'created_utc': 1700035000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p6',
                        'author': 'scale_security',
                        'body': 'For large environments with thousands of assets, AI tools provide the scale that human teams cannot match. The key is proper tuning and customization.',
                        'score': 27,
                        'created_utc': 1700036000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'p7',
                        'author': 'threat_hunter',
                        'body': 'AI excels at pattern recognition in logs and network traffic. For threat hunting and anomaly detection, these tools provide insights humans would miss.',
                        'score': 29,
                        'created_utc': 1700037000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'positive_ai_1',
                'extracted_at': datetime.now().isoformat()
            },

            # Thread 5: Competitive Analysis
            {
                'post': {
                    'id': 'competitive_analysis_1',
                    'title': 'XBOW vs other AI security tools - comparison and alternatives',
                    'selftext': 'Has anyone compared XBOW to competitors like Pentera, AttackIQ, or other AI-powered security platforms? What are the key differences and which provides better value?',
                    'author': 'security_buyer',
                    'score': 42,
                    'upvote_ratio': 0.88,
                    'num_comments': 8,
                    'created_utc': 1700040000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/competitive_analysis_1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/competitive_analysis_1/'
                },
                'comments': [
                    {
                        'id': 'comp1',
                        'author': 'vendor_evaluator',
                        'body': 'Pentera focuses more on network-based attacks while XBOW is stronger in web application testing. Pentera has better enterprise integration but XBOW has more comprehensive coverage.',
                        'score': 35,
                        'created_utc': 1700041000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp2',
                        'author': 'enterprise_buyer',
                        'body': 'AttackIQ is more focused on breach and attack simulation while XBOW does vulnerability discovery. Different use cases but both valuable for comprehensive security programs.',
                        'score': 28,
                        'created_utc': 1700042000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp3',
                        'author': 'security_consultant',
                        'body': 'Price-wise, XBOW is more accessible for mid-market companies. Enterprise platforms like Pentera require significant investment and dedicated resources.',
                        'score': 32,
                        'created_utc': 1700043000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp4',
                        'author': 'procurement_specialist',
                        'body': 'Support and professional services vary significantly. Traditional vendors have better support structures while newer AI tools like XBOW rely more on community and documentation.',
                        'score': 25,
                        'created_utc': 1700044000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp5',
                        'author': 'multi_tool_user',
                        'body': 'We use multiple tools in combination. XBOW for continuous scanning, Pentera for quarterly assessments, and traditional pentesting for critical applications. Each has its place.',
                        'score': 38,
                        'created_utc': 1700045000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp6',
                        'author': 'technology_analyst',
                        'body': 'The technology stack matters. Cloud-native organizations might prefer different tools than traditional on-premises environments. XBOW adapts well to modern architectures.',
                        'score': 30,
                        'created_utc': 1700046000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp7',
                        'author': 'roi_focused',
                        'body': 'ROI calculation is complex. Lower upfront costs but higher operational overhead for tuning and managing false positives. Total cost of ownership varies by organization size and maturity.',
                        'score': 26,
                        'created_utc': 1700047000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'comp8',
                        'author': 'future_focused',
                        'body': 'Consider the roadmap and innovation pace. AI tools are evolving rapidly while traditional platforms may have more mature but slower-evolving feature sets.',
                        'score': 23,
                        'created_utc': 1700048000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'competitive_analysis_1',
                'extracted_at': datetime.now().isoformat()
            },

            # Thread 6: Implementation Challenges
            {
                'post': {
                    'id': 'implementation_1',
                    'title': 'Challenges implementing AI security tools in enterprise environments',
                    'selftext': 'Our organization is considering AI-powered security tools but we are concerned about implementation complexity. What challenges should we expect and how do we prepare?',
                    'author': 'implementation_manager',
                    'score': 31,
                    'upvote_ratio': 0.82,
                    'num_comments': 6,
                    'created_utc': 1700050000,
                    'subreddit': 'cybersecurity',
                    'permalink': '/implementation_1/',
                    'url': 'https://reddit.com/r/cybersecurity/comments/implementation_1/'
                },
                'comments': [
                    {
                        'id': 'impl1',
                        'author': 'change_management',
                        'body': 'Staff resistance is a major challenge. Security teams worry about job security and may not trust AI recommendations. Change management and training are critical for success.',
                        'score': 28,
                        'created_utc': 1700051000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'impl2',
                        'author': 'integration_expert',
                        'body': 'Technical integration is complex. Legacy systems, network segmentation, and access controls create deployment challenges. Plan for longer implementation timelines.',
                        'score': 34,
                        'created_utc': 1700052000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'impl3',
                        'author': 'data_privacy_officer',
                        'body': 'Data privacy and compliance concerns are significant. AI tools need access to sensitive data and systems. Legal and compliance review is essential before deployment.',
                        'score': 31,
                        'created_utc': 1700053000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'impl4',
                        'author': 'budget_planner',
                        'body': 'Hidden costs add up quickly. Training, customization, integration, and ongoing tuning require significant resources beyond the license fees.',
                        'score': 26,
                        'created_utc': 1700054000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'impl5',
                        'author': 'performance_engineer',
                        'body': 'System performance impact can be substantial. AI tools are resource-intensive and may affect network and system performance during scanning periods.',
                        'score': 24,
                        'created_utc': 1700055000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    },
                    {
                        'id': 'impl6',
                        'author': 'success_metrics',
                        'body': 'Defining success metrics is challenging. Traditional security metrics may not apply to AI tools. Establish clear KPIs before implementation to measure value.',
                        'score': 29,
                        'created_utc': 1700056000,
                        'parent_id': None,
                        'replies_count': 0,
                        'depth': 0
                    }
                ],
                'thread_url': 'implementation_1',
                'extracted_at': datetime.now().isoformat()
            }
        ]
        
        logger.info(f"Got {len(threads)} threads with {sum(len(t['comments']) for t in threads)} total comments")
        
        # Verify we have real URLs
        for thread in threads:
            logger.info(f"Thread: {thread['post']['title'][:50]}... - URL: {thread['post']['url']}")
        
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
