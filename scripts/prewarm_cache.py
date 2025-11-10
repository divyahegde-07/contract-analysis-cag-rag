import json
import hashlib
from redis_setup import RedisCache
from sentence_transformers import SentenceTransformer

class CachePreWarmer:
    """Pre-warm Redis cache with standard clause analyses"""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence transformer model")
    
    def generate_cache_key(self, clause_text: str) -> str:
        """Generate consistent cache key from clause text"""
        # Use hash of normalized text as key
        normalized = clause_text.lower().strip()
        return f"clause:{hashlib.md5(normalized.encode()).hexdigest()}"
    
    def load_standard_clauses(self, filepath: str = 'data/cached_clauses.json') -> list:
        """Load standard clauses from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data['standard_clauses']
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            print("Make sure cached_clauses.json is in the data directory")
            raise
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            raise
    
    def prewarm_cache(self, clauses: list):
        """Pre-load standard clauses into Redis cache"""
        print(f"\n Pre-warming cache with {len(clauses)} standard clauses...\n")
        
        for idx, clause in enumerate(clauses, 1):
            clause_text = clause['clause_text']
            
            # Generate cache key
            cache_key = self.generate_cache_key(clause_text)
            
            # Generate embedding for semantic search later
            embedding = self.model.encode(clause_text)
            
            # Prepare cache value with all relevant data
            cache_value = {
                'clause_id': clause['clause_id'],
                'clause_type': clause['clause_type'],
                'clause_text': clause_text,
                'compliance_violation': clause['compliance_violation'],
                'redline_comment': clause['redline_comment'],
                'embedding': embedding.tolist(),  # Store as list for JSON
                'source': 'prewarmed'
            }
            
            # Store in Redis
            success = self.cache.set_cache(cache_key, cache_value)
            
            if success:
                print(f" [{idx}/{len(clauses)}] Cached: {clause['clause_id']} ({clause['clause_type']})")
                print(f"    Key: {cache_key}")
                print(f"    Violation: {clause['compliance_violation']['violation_type']}")
            else:
                print(f" Failed to cache: {clause['clause_id']}")
            
            print()
        
        print(" Cache pre-warming complete!")
    
    def verify_cache(self, clauses: list):
        """Verify all clauses were cached successfully"""
        print("\n🔍 Verifying cached clauses...\n")
        
        success_count = 0
        for clause in clauses:
            cache_key = self.generate_cache_key(clause['clause_text'])
            cached_data = self.cache.get_cache(cache_key)
            
            if cached_data:
                success_count += 1
                print(f" {clause['clause_id']}: Found in cache")
            else:
                print(f" {clause['clause_id']}: NOT found in cache")
        
        print(f"\n Verification: {success_count}/{len(clauses)} clauses cached successfully")
        return success_count == len(clauses)
    
    def show_cache_summary(self):
        """Display cache statistics and contents"""
        print("\n" + "="*60)
        print("CACHE SUMMARY")
        print("="*60)
        
        # Get Redis stats
        stats = self.cache.get_stats()
        print("\n Redis Statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        # List all cached keys
        all_keys = self.cache.list_all_keys()
        print(f"\n Cached Keys ({len(all_keys)} total):")
        for key in all_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            
            # Get cached data
            data = self.cache.get_cache(key)
            if data:
                clause_type = data.get('clause_type', 'unknown')
                clause_id = data.get('clause_id', 'unknown')
                print(f"  • {key[:20]}... → {clause_id} ({clause_type})")
        
        print("\n Cache is ready for testing!")


def main():
    print("=" * 60)
    print("STEP 2: Pre-warm Cache with Standard Clauses")
    print("=" * 60)
    
    try:
        # Connect to Redis
        print("\n Connecting to Redis...")
        cache = RedisCache()
        
        # Initialize pre-warmer
        prewarmer = CachePreWarmer(cache)
        
        # Load standard clauses
        print("\n Loading standard clauses from cached_clauses.json...")
        standard_clauses = prewarmer.load_standard_clauses()
        print(f" Loaded {len(standard_clauses)} standard clauses")
        
        # Clear existing cache (optional - comment out to keep previous data)
        print("\n Clearing existing cache...")
        cache.clear_all()
        print(" Cache cleared")
        
        # Pre-warm the cache
        prewarmer.prewarm_cache(standard_clauses)
        
        # Verify cache
        all_cached = prewarmer.verify_cache(standard_clauses)
        
        if all_cached:
            print("\n All standard clauses successfully cached!")
        else:
            print("\n Some clauses failed to cache - check errors above")
        
        # Show summary
        prewarmer.show_cache_summary()
        
        print("\n" + "="*60)
        print(" Step 2 Complete!")
        print("="*60)
        print("\nNext: Step 3 - Run CAG vs RAG comparison experiment")
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure Redis is running (Step 1)")
        print("- Verify cached_clauses.json exists in current directory")
        print("- Check that sentence-transformers is installed:")
        print("  pip install sentence-transformers")


if __name__ == "__main__":
    main()