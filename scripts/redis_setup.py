import redis
import json
from typing import Dict, Optional

class RedisCache:    
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            print("Successfully connected to Redis!")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            print("\nMake sure Redis is running:")
            print("  docker run -d --name redis-cache -p 6379:6379 redis:latest")
            raise
    
    def set_cache(self, key: str, value: Dict, expiry_seconds: Optional[int] = None):
        try:
            # Convert dict to JSON string
            json_value = json.dumps(value)
            
            if expiry_seconds:
                self.client.setex(key, expiry_seconds, json_value)
            else:
                self.client.set(key, json_value)
            
            return True
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Dict]:
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Error getting cache: {e}")
            return None
    
    def delete_cache(self, key: str):
        return self.client.delete(key)
    
    def clear_all(self):
        return self.client.flushdb()
    
    def get_stats(self) -> Dict:
        info = self.client.info()
        return {
            'total_keys': self.client.dbsize(),
            'used_memory': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 0),
            'uptime_days': info.get('uptime_in_days', 0)
        }
    
    def list_all_keys(self):
        return self.client.keys('*')


def test_redis_connection():
    print("\nTesting Redis Setup...\n")
    
    # Initialize Redis
    cache = RedisCache()
    
    # Clear any existing data
    cache.clear_all()
    print("Cleared existing cache\n")
    
    # # Basic set/get
    # print("Test 1: Basic Set/Get")
    # test_data = {
    #     'clause': 'Test indemnification clause',
    #     'violation': 'Missing data breach coverage',
    #     'comment': 'Add data breach indemnification'
    # }
    # cache.set_cache('test_clause_1', test_data)
    # retrieved = cache.get_cache('test_clause_1')
    # assert retrieved == test_data, "Set/Get test failed!"
    # print("Basic set/get working\n")
    
    # print("Test 2: Expiry (5 seconds)")
    # cache.set_cache('temp_clause', {'data': 'temporary'}, expiry_seconds=5)
    # assert cache.get_cache('temp_clause') is not None
    # print("Expiry set working\n")
    
    # # Non-existent key
    # print("Test 3: Non-existent key")
    # result = cache.get_cache('does_not_exist')
    # assert result is None, "Should return None for missing key"
    # print("Handles missing keys correctly\n")
    
    print('clause_1:', cache.get_cache('clause_1'))

    # # Multiple keys
    # print("Test 4: Multiple keys")
    # for i in range(5):
    #     cache.set_cache(f'clause_{i}', {'id': i, 'text': f'Clause {i}'})
    # all_keys = cache.list_all_keys()
    # print(f"Stored {len(all_keys)} keys: {all_keys}\n")
    
    # Show stats
    print("Redis Stats:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nRedis is ready for the experiment!")
    return cache


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Redis Setup for CAG-RAG Experiment")
    print("=" * 60)
    
    print("\n Prerequisites:")
    print("1. Docker installed and running")
    print("2. Redis container started:")
    print("   docker run -d --name redis-cache -p 6379:6379 redis:latest")
    print("3. Python redis package installed:")
    print("   pip install redis")
    
    input("\nPress Enter when Redis is running...")
    
    try:
        cache = test_redis_connection()
        print("\n Step 1 Complete! Redis is configured and ready.")
        print("\nNext: Step 2 - Pre-warm cache with standard clauses")
    except Exception as e:
        print(f"\n Setup failed: {e}")
        print("\nTroubleshooting:")
        print("- Ensure Docker is running: docker ps")
        print("- Check Redis logs: docker logs redis-cache")
        print("- Restart Redis: docker restart redis-cache")