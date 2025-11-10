import json
import time
import chromadb
#from chromadb.config import Settings
from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class PureRAGSystem:
    
    def __init__(self, collection_name: str = "legal_clauses"):
        """Initialize ChromaDB and create collection"""
        
        # Initialize ChromaDB client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_pure_rag"
            )
        
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Legal contract clauses with compliance data"}
            )
            print(f"Created new collection: {collection_name}")
        
        # Statistics tracking
        self.query_count = 0
        self.total_latency = 0
        self.llm_calls = 0
        self.total_cost = 0
        
    def add_documents(self, clauses: List[Dict]):
        """Add clauses to vector database"""
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, clause in enumerate(clauses):
            # Document text
            documents.append(clause['clause_text'])
            
            # Metadata
            metadatas.append({
                'clause_id': clause.get('clause_id', f'clause_{idx}'),
                'clause_type': clause['clause_type'],
                'violation_type': clause.get('compliance_violation', {}).get('violation_type', 'unknown'),
                'severity': clause.get('compliance_violation', {}).get('severity', 'unknown')
            })
            
            # Unique ID
            ids.append(f"doc_{idx}_{clause.get('clause_id', idx)}")
        
        # Add to ChromaDB (automatically generates embeddings)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} clauses to vector database")
        print(f"Total documents in DB: {self.collection.count()}")
    
    def retrieve_similar_clauses(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve similar clauses from vector database"""
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Format results
        similar_clauses = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similar_clauses.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return similar_clauses
    
    def analyze_clause(self, clause_text: str, verbose: bool = False) -> Dict:
        """Analyze clause using RAG - retrieves context then calls LLM"""
        
        start_time = time.time()
        self.query_count += 1
        
        # Step 1: Retrieve similar clauses from vector DB
        similar_clauses = self.retrieve_similar_clauses(clause_text, top_k=2)
        
        if verbose:
            print(f"\n Retrieved {len(similar_clauses)} similar clauses from vector DB")
        
        # Step 2: Build context from retrieved clauses
        context = ""
        if similar_clauses:
            context = "Similar clauses and their compliance issues:\n\n"
            for i, clause in enumerate(similar_clauses, 1):
                context += f"{i}. {clause['text'][:200]}...\n"
                context += f"   Issue: {clause['metadata'].get('violation_type', 'N/A')}\n\n"
        
        # Step 3: Call LLM with retrieved context
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a legal contract analyst. Analyze clauses for compliance issues related to: missing indemnification, non-standard liability caps, jurisdiction problems, and payment terms issues."
                },
                {
                    "role": "user", 
                    "content": f"{context}\nNow analyze this clause:\n\n{clause_text}\n\nProvide: violation type, severity (HIGH/MEDIUM/LOW), and detailed redline comments."
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            self.llm_calls += 1
            
            # Calculate cost (gpt-3.5-turbo pricing: $0.0015/1K prompt, $0.002/1K completion)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
            self.total_cost += cost
            
            analysis = response.choices[0].message.content
            
            latency = (time.time() - start_time) * 1000
            self.total_latency += latency
            
            return {
                'analysis': analysis,
                'source': 'vector_db + llm',
                'retrieved_context': len(similar_clauses),
                'latency_ms': latency,
                'cost': cost,
                'tokens': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': prompt_tokens + completion_tokens
                }
            }
            
        except Exception as e:
            print(f"LLM Error: {e}")
            latency = (time.time() - start_time) * 1000
            return {
                'analysis': f"Error: {str(e)}",
                'source': 'error',
                'retrieved_context': 0,
                'latency_ms': latency,
                'cost': 0
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        avg_latency = self.total_latency / self.query_count if self.query_count > 0 else 0
        
        return {
            'system': 'Pure RAG',
            'vector_db_size': self.collection.count(),
            'total_queries': self.query_count,
            'llm_calls': self.llm_calls,
            'cache_hits': 0,
            'avg_latency_ms': round(avg_latency, 2),
            'total_latency_ms': round(self.total_latency, 2),
            'total_cost': round(self.total_cost, 4),
            'cost_per_query': round(self.total_cost / self.query_count, 4) if self.query_count > 0 else 0
        }
    
    def clear_database(self):
        """Clear all documents from the collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection.name)
            print(f"Cleared collection: {self.collection.name}")
        except:
            pass


def test_pure_rag():
    
    print("Testing Pure RAG System with ChromaDB")
    
    # Initialize system
    print("\nInitializing Pure RAG system...")
    rag_system = PureRAGSystem(collection_name="test_legal_clauses")
    
    # Load standard clauses to populate vector DB
    print("\n Loading standard clauses...")
    with open('data/cached_clauses.json', 'r') as f:
        data = json.load(f)
        standard_clauses = data['standard_clauses']
    
    # Clear existing data and add fresh
    rag_system.clear_database()
    rag_system = PureRAGSystem(collection_name="test_legal_clauses")
    rag_system.add_documents(standard_clauses)
    
    # Test with a sample query
    print("\n" + "="*70)
    print("Test Query")
    print("="*70)
    
    test_clause = """
    Vendor's total liability under this agreement shall not exceed 
    the fees paid by Client in the preceding three (3) months.
    """
    
    print(f"\nAnalyzing clause:\n{test_clause}")
    
    result = rag_system.analyze_clause(test_clause, verbose=True)
    
    print("\nResult:")
    print(f"Source: {result['source']}")
    print(f"Retrieved context: {result['retrieved_context']} similar clauses")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print(f"Cost: ${result['cost']:.4f}")
    print(f"Tokens: {result['tokens']['total']}")
    
    print(f"\nAnalysis:\n{result['analysis']}")
    
    # Show stats
    print("\n" + "="*70)
    print("System Statistics")
    print("="*70)
    stats = rag_system.get_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print("\nPure RAG system test complete!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with: OPENAI_API_KEY=your-key-here")
        exit(1)
    
    test_pure_rag()