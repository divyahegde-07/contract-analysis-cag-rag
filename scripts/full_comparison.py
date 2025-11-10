import json
from typing import List, Dict
from redis_setup import RedisCache
from rag_system import PureRAGSystem
from cag_rag_system import CAGRAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_test_contracts(filepath: str = './data/test_contracts.json') -> List[Dict]:
    """Load test contracts from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['contracts']


def load_standard_clauses(filepath: str = './data/cached_clauses.json') -> List[Dict]:
    """Load standard clauses from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['standard_clauses']


def run_full_experiment():
    """Run complete CAG vs RAG comparison experiment"""
    
    print("=" * 70)
    print("STEP 3: Full CAG vs RAG Comparison Experiment")
    print("=" * 70)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\nError: OPENAI_API_KEY not found in .env file")
        return
    
    # Initialize systems
    print("\nInitializing systems...")
    redis_cache = RedisCache()
    pure_rag = PureRAGSystem()
    cag_rag = CAGRAGSystem(redis_cache)
    print()
    
    # Load data
    print("Loading data...")
    standard_clauses = load_standard_clauses()
    contracts = load_test_contracts()
    
    # Add standard clauses to both vector DBs
    pure_rag.add_documents(standard_clauses)
    cag_rag.add_documents(standard_clauses)
    print()
    
    # Extract all test clauses
    all_clauses = []
    for contract in contracts:
        for clause in contract['clauses']:
            all_clauses.append({
                'contract_id': contract['contract_id'],
                'clause_number': clause['clause_number'],
                'clause_type': clause['clause_type'],
                'clause_text': clause['clause_text'],
                'is_standard': clause['is_standard']
            })
    
    standard_count = sum(1 for c in all_clauses if c['is_standard'])
    print(f"Test dataset:")
    print(f"  • Total clauses: {len(all_clauses)}")
    print(f"  • Standard (pre-cached in CAG): {standard_count}")
    print(f"  • Variations: {len(all_clauses) - standard_count}")
    print()
    
    # Run Pure RAG
    print("="*70)
    print("PURE RAG SYSTEM (Always: Retrieve + LLM)")
    print("="*70)
    
    for idx, clause in enumerate(all_clauses, 1):
        print(f"\n[{idx}/{len(all_clauses)}] {clause['contract_id']} - {clause['clause_type']}")
        print(f"  Standard: {clause['is_standard']}")
        
        result = pure_rag.analyze_clause(clause['clause_text'])
        print(f" {result['latency_ms']:.0f}ms |  ${result['cost']:.4f} |  {result['source']}")
    
    # Run CAG-RAG
    print("\n" + "="*70)
    print("CAG-RAG HYBRID (Cache → Retrieve → LLM)")
    print("="*70)
    
    for idx, clause in enumerate(all_clauses, 1):
        print(f"\n[{idx}/{len(all_clauses)}] {clause['contract_id']} - {clause['clause_type']}")
        print(f"  Standard: {clause['is_standard']}")
        
        result = cag_rag.analyze_clause(clause['clause_text'])
        
        cache_indicator = "CACHE HIT!" if result['cache_hit'] else "Cache miss"
        print(f"  {result['latency_ms']:.0f}ms |  ${result['cost']:.4f} | {result['source']} | {cache_indicator}")
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    rag_stats = pure_rag.get_stats()
    cag_stats = cag_rag.get_stats()
    
    print("\nPure RAG:")
    for key, value in rag_stats.items():
        print(f"  • {key}: {value}")
    
    print("\nCAG-RAG Hybrid:")
    for key, value in cag_stats.items():
        print(f"  • {key}: {value}")
    
    # Savings Analysis
    print("\n" + "="*70)
    print("SAVINGS ANALYSIS")
    print("="*70)
    
    latency_saved = rag_stats['avg_latency_ms'] - cag_stats['avg_latency_ms']
    latency_improvement = (latency_saved / rag_stats['avg_latency_ms'] * 100) if rag_stats['avg_latency_ms'] > 0 else 0
    
    cost_saved = rag_stats['total_cost'] - cag_stats['total_cost']
    cost_improvement = (cost_saved / rag_stats['total_cost'] * 100) if rag_stats['total_cost'] > 0 else 0
    
    print(f"\nLatency:")
    print(f"  • Pure RAG: {rag_stats['avg_latency_ms']:.0f}ms avg")
    print(f"  • CAG-RAG: {cag_stats['avg_latency_ms']:.0f}ms avg")
    print(f"  • Improvement: {latency_saved:.0f}ms faster ({latency_improvement:.1f}%)")
    
    print(f"\nCost:")
    print(f"  • Pure RAG: ${rag_stats['total_cost']:.4f}")
    print(f"  • CAG-RAG: ${cag_stats['total_cost']:.4f}")
    print(f"  • Savings: ${cost_saved:.4f} ({cost_improvement:.1f}%)")
    
    print(f"\nEfficiency:")
    print(f"  • Cache hits: {cag_stats['cache_hits']}/{cag_stats['total_queries']} ({cag_stats['cache_hit_rate']})")
    print(f"  • LLM calls avoided: {rag_stats['llm_calls'] - cag_stats['llm_calls']}")
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    print(f"\nKey Result: CAG-RAG is {latency_improvement:.1f}% faster and saves {cost_improvement:.1f}% on costs!")
    
    return {
        'rag_stats': rag_stats,
        'cag_stats': cag_stats,
        'savings': {
            'latency_ms': latency_saved,
            'latency_pct': latency_improvement,
            'cost_usd': cost_saved,
            'cost_pct': cost_improvement
        }
    }


if __name__ == "__main__":
    try:
        results = run_full_experiment()
        print("\nReady for Step 4: Generate visualization and report!")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()