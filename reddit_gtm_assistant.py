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
        """Load real Reddit data and process it"""
        logger.info("Loading real Reddit data...")
        
        # Real Reddit data from the JSON files provided
        threads = []
        
        # Parse cybersecurity subreddit data (document 1)
        cybersecurity_data = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "subreddit": "cybersecurity",
                            "selftext": "So...Recently, some of my seniors conducted a workshop about \"Agentic AI\" and its scope. The speaker at that particular workshop actually worked at a company developing AI agents. He asked us a question along the lines \"Can AI take jobs of a Bug Bounty hunter or a Pentester in the near future?\". I know AI has a lot of capabilities but these things seem to complex for an AI so I said no. Well, then he told us about XBOW and its accuracy in finding bugs, I had second thoughts. XBOW is an AI agent and it is currently at the third place in the United States of America on Hackerone. It was at first place but dropped down to third. I'm pursuing my cybersecurity career as a penetration tester and I just wanna know should I proceed on that path or should I change it? I still want to do it as I am more interested in it but now I'm having double thoughts.",
                            "title": "Is Penetration Testing still worth it after seeing XBOW at work?",
                            "score": 0,
                            "upvote_ratio": 0.39,
                            "num_comments": 17,
                            "url": "https://www.reddit.com/r/cybersecurity/comments/1ly9nxf/is_penetration_testing_still_worth_it_after/",
                            "author": "RayyanRA7",
                            "created_utc": 1752350549
                        }
                    }
                ]
            }
        }
        
        # Get comments from document 1
        cybersecurity_comments = [
            {
                "author": "cloudfox1",
                "body": "That's the sales person's job, to sell you their tools, ofc they are going to talk it up",
                "score": 13,
                "created_utc": 1752397400
            },
            {
                "author": "Strange-Mountain1810",
                "body": "It found easy to find xss and sprayed vdp's hardly pentest replacement.",
                "score": 9,
                "created_utc": 1752399364
            },
            {
                "author": "Beautiful_Watch_7215",
                "body": "Change to what? What jobs don't have demos of an AI solution replacing it?",
                "score": 8,
                "created_utc": 1752397066
            },
            {
                "author": "Zatetics",
                "body": "Is this an ad?",
                "score": 7,
                "created_utc": 1752398989
            },
            {
                "author": "JustNobre",
                "body": "If you adapt and AI doesn't take your job you win. If you adapt and AI takes your job you might be able to work with AI or pivot to another area. If AI doesn't take your job and you don't adapt you will be in the same place, probably with less career choices.",
                "score": 1,
                "created_utc": 1752402627
            },
            {
                "author": "Loud-Run-9725",
                "body": "Penetration testing requiring more creativity and human intervention (business logic flaws, broken auth, etc) are going to require humans. It's just that so many penetration testers spend more time on the classes of vulns XBOW is finding since they are easier to find and exploit. Just like everything else in tech, AI is going to be great at augmenting human activities but not fully replace.",
                "score": 1,
                "created_utc": 1752665437
            },
            {
                "author": "greybrimstone",
                "body": "XBOW does not perform nearly as well as they make it sound. It is, for all intents and purposes, a really cool automated vulnerability scanner. The experiments they've done pitting their AI against humans have been heavily biased. Their rankings on bug bounty programs, speak to quantity of easy-to-find issues, not quality. AI cannot compete with human creativity, intuition, and ability to make leaps. AI only knows what it was trained on. Sure, it's great and useful, but not at all as capable as a real tester. If XBOW wants a real experiment, we can design one that has no bias and we can provide the testers too.",
                "score": 2,
                "created_utc": 1755741602
            }
        ]
        
        # Add cybersecurity thread
        threads.append({
            'post': {
                'id': '1ly9nxf',
                'title': 'Is Penetration Testing still worth it after seeing XBOW at work?',
                'selftext': 'So...Recently, some of my seniors conducted a workshop about "Agentic AI" and its scope. The speaker at that particular workshop actually worked at a company developing AI agents. He asked us a question along the lines "Can AI take jobs of a Bug Bounty hunter or a Pentester in the near future?". I know AI has a lot of capabilities but these things seem to complex for an AI so I said no. Well, then he told us about XBOW and its accuracy in finding bugs, I had second thoughts. XBOW is an AI agent and it is currently at the third place in the United States of America on Hackerone. It was at first place but dropped down to third. I\'m pursuing my cybersecurity career as a penetration tester and I just wanna know should I proceed on that path or should I change it? I still want to do it as I am more interested in it but now I\'m having double thoughts.',
                'author': 'RayyanRA7',
                'score': 0,
                'upvote_ratio': 0.39,
                'num_comments': 17,
                'created_utc': 1752350549,
                'subreddit': 'cybersecurity',
                'permalink': '/r/cybersecurity/comments/1ly9nxf/is_penetration_testing_still_worth_it_after/',
                'url': 'https://www.reddit.com/r/cybersecurity/comments/1ly9nxf/is_penetration_testing_still_worth_it_after/'
            },
            'comments': cybersecurity_comments,
            'thread_url': '1ly9nxf',
            'extracted_at': datetime.now().isoformat()
        })
        
        # Parse pentesting subreddit data (document 2)
        pentesting_comments = [
            {
                "author": "Clean-Drop9629",
                "body": "I recently spoke with a contact at one of the organizations that has received a significant number of vulnerability reports from XBOW. They shared that tools like XBOW have made their work substantially more difficult, as they now spend countless hours triaging and validating reports many of which turn out to be false positives or issues of such low criticality that they fall outside the organization's risk threshold. While XBOW may appear impressive due to the volume of submissions, the quality and relevance of many of these findings are questionable, ultimately straining the receiving team's resources.",
                "score": 3,
                "created_utc": 1751223076
            },
            {
                "author": "619Smitty",
                "body": "I think eventually it will, like most things. Once AI's complex problem solving issues improve, in conjunction with self-improvement capabilities, it will be able to do a lot of automation across the board.",
                "score": 2,
                "created_utc": 1751156217
            },
            {
                "author": "Vivid_Cod_2109",
                "body": "Not yet and the reason is not because of critical thinking of AI but it is hard to set up for XBOW to pentest in a complex environment.",
                "score": 2,
                "created_utc": 1751156372
            },
            {
                "author": "AttackForge",
                "body": "They will never be able to test for business logic and design flaws.",
                "score": 4,
                "created_utc": 1751152037
            },
            {
                "author": "Reasonable_Cut8116",
                "body": "I dont think it will fully replace senior pentesters but it will absolutely replace your current vulnerability scanner and junior pentesters. I own an MSP/MSSP and we use a platform called StealthNet AI(stealthnet.ai) to offer automated pentests via AI agents. They have a fleet of AI agents (External,Vishing,API,etc).Their API/Web agent is really impressive it finds things that traditional vulnerability scanners miss due to not understanding business logic. It also writes some of the best pentest reports iv seen, they look human written its impossible to tell. Overall they perform at the level of a junior - intermediate pentester which is really good considering your average junior pentester is going to cost you $50 an hour. Its defiantly game changing technology but its not going to replace a senior pentester. This type of tech is still brand new. Now 5 years from now things might be different I think most pentest agents will be senior level by then.",
                "score": 1,
                "created_utc": 1751899180
            },
            {
                "author": "latnGemin616",
                "body": "If its anything like running a Nessus scan, I go with the consensus and say NO. Why? We have Snyk that can check code quality for vulnerabilties. We have SAST/DAST solutions that require human intervention to interpret findings and rule out false-negatives/false-positives. And to the point about Nessus scans, there's still a human that as to filter the signal from the noise. Not everything in a scan is a legit finding. Where I can see AI being a benefit is for those who are stuck with a finding or need a way to proofread their work.",
                "score": 1,
                "created_utc": 1751157432
            },
            {
                "author": "Strange-Mountain1810",
                "body": "Maybe the shitty pentesters lol",
                "score": 1,
                "created_utc": 1751163164
            }
        ]
        
        # Add pentesting thread
        threads.append({
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
                'url': 'https://www.reddit.com/r/Pentesting/comments/1lmzvx8/will_xbow_or_ais_be_able_to_replace_pentesters/'
            },
            'comments': pentesting_comments,
            'thread_url': '1lmzvx8',
            'extracted_at': datetime.now().isoformat()
        })
        
        # Parse hacking tutorials subreddit data (document 3)
        hacking_tutorials_comments = [
            {
                "author": "Sqooky",
                "body": "AI is just another tool to help us do work. We have tens of products in our suite and it still wont catch vulnerabilities that humans can find. Lots of companies don't like tossing their data into a black-box. Especially proprietary and potentially vulnerability related data. Mainly because of loss of control.",
                "score": 3,
                "created_utc": 1751180043
            },
            {
                "author": "Safe_Nobody_760",
                "body": "so why did you call it snake oil? by default snake oil is not the real deal, yet you are saying this is something else?",
                "score": 2,
                "created_utc": 1751196514
            }
        ]
        
        # Add hacking tutorials thread
        threads.append({
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
                'url': 'https://www.reddit.com/r/Hacking_Tutorials/comments/1ln80yl/is_xbow_ai_snake_oil_or_the_real_deal/'
            },
            'comments': hacking_tutorials_comments,
            'thread_url': '1ln80yl',
            'extracted_at': datetime.now().isoformat()
        })
        
        # Parse bug bounty subreddit data (document 4)
        bugbounty_comments = [
            {
                "author": "chopper332nd",
                "body": "As a customer of hacker one I'm more worried about the crap we're gunna have to sort through now 🤷‍♂️ We have scanners and other companies that offer AI agents for pentesting which find the low hanging fruit. We have a Bug Bounty program to find more nuanced vulnerabilities in our products that other security testing can't find.",
                "score": 21,
                "created_utc": 1749712425
            },
            {
                "author": "k4lashhnikov",
                "body": "The human factor is always required for logic errors, vertical or horizontal scaling, AI and automated tools cannot understand the business context. If AIs have vulnerabilities and are not imperfect, what makes you think they will replace the human hacker?",
                "score": 14,
                "created_utc": 1749703837
            },
            {
                "author": "k4lashhnikov",
                "body": "Sure, they can surpass human capabilities but there is little point in analyzing hundreds of thousands of endpoints to find uninteresting things or false positives, If an AI analyzes misconfigurations of JS, code, or exposed credentials, it cannot (for now) have the ability to manually modify things that apparently work well. For example, a step-by-step business flow, if the AI superficially sees that the flow is correct, it will leave it as is, but a human has the idea of seeing what happens if a specific step is skipped, or if you decide to give a random input with random characters and cause an error on purpose, those kinds of subtle things are the ones that from my perspective are impossible to replace the human hacker. But of course, AI will advance without precedent, this is where we as hackers have time to study and look for vulnerabilities in the AI itself, in fact there are bug bounty or red team programs There is an OWASP 10 for AI especially, it is advancing faster than its security is advancing with it, so don't be discouraged, there are enough bugs for everyone. 😃",
                "score": 4,
                "created_utc": 1749743477
            },
            {
                "author": "6W99ocQnb8Zy17",
                "body": "The post-AI world is just another pivot point, same as post-printing-press, or post-computer blah. When the technology changes rapidly, there will always be people who struggle to make the change. But there will also be people who not only accept, but embrace the change. The choice boils down to whether you want to be an unemployed cinema pianist or not ;)",
                "score": 7,
                "created_utc": 1749713809
            },
            {
                "author": "InvestmentOk1962",
                "body": "yea bro i think u should just leave if u have this intense dilemma if u want money learn anything else or if want love this then do even if the whole world doesnt",
                "score": 3,
                "created_utc": 1749702658
            },
            {
                "author": "Reasonable_Cut8116",
                "body": "It does surprisingly well. I have heard of Xbow. We use a similar tool from StealthNet AI(stealthnet.ai). They have a fleet of AI agents to automate penetration testing. Compared to normal vulnerability scanners they perform 100x better. You can think of them as a really smart vulnerbility scanner or a junior pentester. They can perform really well but it wont be able to find everything. More complex attacks will still require humans and I dont think thats going to change any time soon. I actually think you will see a lot of AI + Human as you get the best of both worlds. A senior pentester who has access to an army of junior level AI pentesters will be able to do 100x the amount of work. Just like anything else AI agents are a tool that will enable senior pentesters to move 100x faster. It might replace junior level testers but you will always need a human to do more complex attacks.",
                "score": 2,
                "created_utc": 1757041432
            }
        ]
        
        # Add bug bounty thread
        threads.append({
            'post': {
                'id': '1l97esk',
                'title': 'How AI is affecting pentesting and bug bounties',
                'selftext': 'Recently, I came across with a project named "Xbow" and it\'s actually the current top US-based hacker on Hackerone\'s leaderboard. It\'s a fully automated AI agent trained on real vulnerability data and will be available soon. Do you think it\'s still worth to learn pentesting and get into bug bounties? I\'m currently learning and seeing this got me thinking if I should continue or maybe move to another field inside red team.',
                'author': 'S4vz4d',
                'score': 14,
                'upvote_ratio': 0.7,
                'num_comments': 12,
                'created_utc': 1749684165,
                'subreddit': 'bugbounty',
                'permalink': '/r/bugbounty/comments/1l97esk/how_ai_is_affecting_pentesting_and_bug_bounties/',
                'url': 'https://www.reddit.com/r/bugbounty/comments/1l97esk/how_ai_is_affecting_pentesting_and_bug_bounties/'
            },
            'comments': bugbounty_comments,
            'thread_url': '1l97esk',
            'extracted_at': datetime.now().isoformat()
        })
        
        logger.info(f"Loaded {len(threads)} threads with {sum(len(t['comments']) for t in threads)} total comments")
        
        # Store the data
        self.threads_data = threads
        
        # Process into vector store
        success = self.vector_store.add_threads(threads)
        
        if success:
            logger.info("Real Reddit data loaded and processed successfully")
            
            # Test search immediately
            test_result = self.vector_store.search("XBOW pain points problems", n_results=3)
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
