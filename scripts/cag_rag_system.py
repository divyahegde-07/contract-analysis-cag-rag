import time
import hashlib
import numpy as np
import chromadb
#from chromadb.config import Settings
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from redis_setup import RedisCache

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class CAGRAGSystem:
    """CAG-RAG Hybrid system: Redis cache + ChromaDB + LLM"""
    
    def __init__(self, redis_cache: RedisCache, collection_name: str = "cag_rag_clauses"):
        """Initialize CAG-RAG system with Redis cache and ChromaDB"""
        
        self.cache = redis_cache
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_cag_rag"
            )
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        # Create fresh collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "CAG-RAG hybrid clauses"}
        )
        
        # Statistics tracking
        self.query_count = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.total_latency = 0
        self.total_cost = 0
        self.similarity_threshold = 0.85
        
        print(f"CAG-RAG: Initialized Redis cache + ChromaDB")
    
    def add_documents(self, clauses: List[Dict]):
        """Add clauses to vector database"""
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, clause in enumerate(clauses):
            documents.append(clause['clause_text'])
            metadatas.append({
                'clause_type': clause['clause_type'],
                'clause_id': clause.get('clause_id', f'clause_{idx}')
            })
            ids.append(f"doc_{idx}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"CAG-RAG: Added {len(documents)} clauses to vector DB")
    
    def generate_cache_key(self, clause_text: str) -> str:
        """Generate consistent cache key from clause text"""
        normalized = clause_text.lower().strip()
        return f"clause:{hashlib.md5(normalized.encode()).hexdigest()}"
    
    def find_similar_cached_clause(self, clause_text: str) -> Optional[Dict]:
        """Search Redis cache for semantically similar clause"""
        query_embedding = self.model.encode(clause_text)
        all_keys = self.cache.list_all_keys()
        
        best_match = None
        highest_similarity = 0
        
        for key in all_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            
            cached_data = self.cache.get_cache(key)
            if not cached_data or 'embedding' not in cached_data:
                continue
            
            cached_embedding = np.array(cached_data['embedding'])
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = cached_data
        
        # Return match if above threshold
        if highest_similarity >= self.similarity_threshold:
            best_match['similarity'] = highest_similarity
            return best_match
        
        return None
    
    def analyze_clause(self, clause_text: str) -> Dict:
        """Analyze clause - checks cache first, then falls back to RAG"""
        
        start_time = time.time()
        self.query_count += 1
        
        # Step 1: Check exact cache match
        cache_key = self.generate_cache_key(clause_text)
        cached_result = self.cache.get_cache(cache_key)
        
        if cached_result:
            self.cache_hits += 1
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            
            return {
                'analysis': f"CACHED ANALYSIS:\n\nViolation: {cached_result['compliance_violation']['violation_type']}\nSeverity: {cached_result['compliance_violation']['severity']}\n\n{cached_result['redline_comment']}",
                'source': 'cache_exact',
                'latency_ms': latency,
                'cost': 0,
                'cache_hit': True,
                'similarity': 1.0
            }
        
        # Step 2: Check for semantic similarity in cache
        similar_cached = self.find_similar_cached_clause(clause_text)
        if similar_cached:
            self.cache_hits += 1
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            
            return {
                'analysis': f"CACHED ANALYSIS (Similar: {similar_cached['similarity']:.1%}):\n\nViolation: {similar_cached['compliance_violation']['violation_type']}\nSeverity: {similar_cached['compliance_violation']['severity']}\n\n{similar_cached['redline_comment']}",
                'source': 'cache_semantic',
                'latency_ms': latency,
                'cost': 0,
                'cache_hit': True,
                'similarity': similar_cached['similarity']
            }
        
        # Step 3: Cache miss - do RAG (retrieve from vector DB)
        results = self.collection.query(
            query_texts=[clause_text],
            n_results=2
        )
        
        context = ""
        if results['documents'] and results['documents'][0]:
            context = "Similar clauses:\n\n"
            for i, doc in enumerate(results['documents'][0][:2], 1):
                context += f"{i}. {doc[:150]}...\n\n"
        
        # Step 4: Call LLM with retrieved context
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a legal contract analyst. Analyze for compliance issues: missing indemnification, non-standard liability caps, jurisdiction problems, payment terms issues."
                    },
                    {
                        "role": "user", 
                        "content": f"{context}\nAnalyze this clause:\n\n{clause_text}\n\nProvide: violation type, severity, and redline comments."
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            self.llm_calls += 1
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
            self.total_cost += cost
            
            analysis = response.choices[0].message.content
            
            # Step 5: Cache the result for future queries
            embedding = self.model.encode(clause_text)
            cache_value = {
                'clause_text': clause_text,
                'analysis': analysis,
                'embedding': embedding.tolist(),
                'compliance_violation': {'violation_type': 'Generated', 'severity': 'Unknown'},
                'redline_comment': analysis,
                'source': 'llm_generated'
            }
            self.cache.set_cache(cache_key, cache_value)
            
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            
            return {
                'analysis': f"LLM ANALYSIS (now cached):\n\n{analysis}",
                'source': 'vector_db + llm',
                'latency_ms': latency,
                'cost': cost,
                'cache_hit': False
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            return {
                'analysis': f"Error: {str(e)}",
                'source': 'error',
                'latency_ms': latency,
                'cost': 0,
                'cache_hit': False
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        avg_latency = self.total_latency / self.query_count if self.query_count > 0 else 0
        cache_hit_rate = (self.cache_hits / self.query_count * 100) if self.query_count > 0 else 0
        
        return {
            'system': 'CAG-RAG Hybrid',
            'vector_db_size': self.collection.count(),
            'total_queries': self.query_count,
            'llm_calls': self.llm_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_latency_ms': round(avg_latency, 2),
            'total_cost': round(self.total_cost, 4),
            'cost_per_query': round(self.total_cost / self.query_count, 4) if self.query_count > 0 else 0
        }


def test_cag_rag():
    """Test CAG-RAG system"""
    print("="*70)
    print("Testing CAG-RAG System")
    print("="*70)
    
    from redis_setup import RedisCache
    
    # Initialize
    redis_cache = RedisCache()
    cag_system = CAGRAGSystem(redis_cache)
    
    # Load and add documents
    import json
    with open('./data/cached_clauses.json', 'r') as f:
        data = json.load(f)
        standard_clauses = data['standard_clauses']
    
    cag_system.add_documents(standard_clauses)
    
    # Test query
    test_clause = "Vendor's total liability shall not exceed fees paid in preceding three months."
    
    print(f"\nTesting with clause:\n{test_clause}\n")
    result = cag_system.analyze_clause(test_clause)
    
    print(f"Source: {result['source']}")
    print(f"Cache hit: {result['cache_hit']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print(f"Cost: ${result['cost']:.4f}")
    
    print("\nCAG-RAG test complete!")


if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in .env file")
        exit(1)
    
    test_cag_rag()