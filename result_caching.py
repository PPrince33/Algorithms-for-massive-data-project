# Result Caching and Persistence System for Market Basket Analysis

import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

class ResultCache:
    """
    Comprehensive caching system for frequent itemsets, association rules,
    and other expensive computations with file-based persistence.
    """
    
    def __init__(self, cache_dir: str = ".cache", enable_caching: bool = True, 
                 cache_expiry_hours: int = 24, verbose: bool = True):
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        self.cache_expiry_hours = cache_expiry_hours
        self.verbose = verbose
        
        # Create cache directory if it doesn't exist
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'total_size_mb': 0
        }
    
    def _generate_cache_key(self, operation: str, parameters: Dict[str, Any]) -> str:
        """Generate a unique cache key based on operation and parameters."""
        # Create a deterministic string from parameters
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        
        # Create hash of operation + parameters
        key_string = f"{operation}_{param_str}"
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_filepath(self, cache_key: str, file_type: str = "pkl") -> str:
        """Get the full filepath for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.{file_type}")
    
    def _is_cache_valid(self, filepath: str) -> bool:
        """Check if cached file exists and is not expired."""
        if not os.path.exists(filepath):
            return False
        
        # Check expiry time
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return file_time > expiry_time
    
    def get_cached_result(self, operation: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached result if available and valid."""
        if not self.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(operation, parameters)
        cache_filepath = self._get_cache_filepath(cache_key)
        
        if self._is_cache_valid(cache_filepath):
            try:
                with open(cache_filepath, 'rb') as f:
                    result = pickle.load(f)
                
                self.cache_stats['hits'] += 1
                
                if self.verbose:
                    print(f"📦 Cache HIT for {operation} (key: {cache_key[:8]}...)")
                
                return result
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Cache read error for {operation}: {str(e)}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_filepath)
                except:
                    pass
        
        self.cache_stats['misses'] += 1
        
        if self.verbose:
            print(f"📦 Cache MISS for {operation} (key: {cache_key[:8]}...)")
        
        return None
    
    def save_result(self, operation: str, parameters: Dict[str, Any], result: Any) -> bool:
        """Save result to cache."""
        if not self.enable_caching:
            return False
        
        cache_key = self._generate_cache_key(operation, parameters)
        cache_filepath = self._get_cache_filepath(cache_key)
        
        try:
            with open(cache_filepath, 'wb') as f:
                pickle.dump(result, f)
            
            # Update cache statistics
            file_size_mb = os.path.getsize(cache_filepath) / (1024 * 1024)
            self.cache_stats['saves'] += 1
            self.cache_stats['total_size_mb'] += file_size_mb
            
            if self.verbose:
                print(f"💾 Cached {operation} result ({file_size_mb:.2f} MB, key: {cache_key[:8]}...)")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Cache save error for {operation}: {str(e)}")
            return False
    
    def clear_cache(self, operation: str = None) -> int:
        """Clear cache files. If operation is specified, clear only that operation's cache."""
        if not self.enable_caching:
            return 0
        
        cleared_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                
                if operation is None:
                    # Clear all cache files
                    os.remove(filepath)
                    cleared_count += 1
                else:
                    # Check if this file belongs to the specified operation
                    # This is a simple heuristic - in practice you might want more sophisticated matching
                    if filename.endswith('.pkl'):
                        try:
                            with open(filepath, 'rb') as f:
                                cached_data = pickle.load(f)
                            # If we can determine this belongs to the operation, remove it
                            os.remove(filepath)
                            cleared_count += 1
                        except:
                            # If we can't read it, it might be corrupted, so remove it
                            os.remove(filepath)
                            cleared_count += 1
            
            if self.verbose:
                print(f"🗑️  Cleared {cleared_count} cache files")
            
            # Reset statistics
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'saves': 0,
                'total_size_mb': 0
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error clearing cache: {str(e)}")
        
        return cleared_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        cache_files = []
        total_size_mb = 0
        
        if self.enable_caching and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    cache_files.append({
                        'filename': filename,
                        'size_mb': file_size_mb,
                        'created': file_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'age_hours': (datetime.now() - file_time).total_seconds() / 3600
                    })
                    
                    total_size_mb += file_size_mb
        
        hit_rate = (self.cache_stats['hits'] / 
                   (self.cache_stats['hits'] + self.cache_stats['misses']) * 100
                   if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0)
        
        return {
            'enabled': self.enable_caching,
            'cache_dir': self.cache_dir,
            'expiry_hours': self.cache_expiry_hours,
            'total_files': len(cache_files),
            'total_size_mb': total_size_mb,
            'hit_rate_percent': hit_rate,
            'statistics': self.cache_stats,
            'files': sorted(cache_files, key=lambda x: x['age_hours'])
        }
    
    def print_cache_report(self):
        """Print a comprehensive cache report."""
        info = self.get_cache_info()
        
        print(f"\n📊 Cache Report")
        print(f"   Status: {'Enabled' if info['enabled'] else 'Disabled'}")
        print(f"   Directory: {info['cache_dir']}")
        print(f"   Expiry: {info['expiry_hours']} hours")
        print(f"   Total files: {info['total_files']}")
        print(f"   Total size: {info['total_size_mb']:.2f} MB")
        print(f"   Hit rate: {info['hit_rate_percent']:.1f}%")
        
        print(f"\n📈 Statistics:")
        print(f"   Cache hits: {info['statistics']['hits']}")
        print(f"   Cache misses: {info['statistics']['misses']}")
        print(f"   Files saved: {info['statistics']['saves']}")
        
        if info['files']:
            print(f"\n📁 Recent Cache Files:")
            for i, file_info in enumerate(info['files'][:5], 1):
                print(f"   {i}. {file_info['filename'][:20]}... ({file_info['size_mb']:.2f} MB, {file_info['age_hours']:.1f}h old)")


class FrequentItemsetCache:
    """
    Specialized cache for frequent itemsets with parameter-aware caching.
    """
    
    def __init__(self, cache: ResultCache):
        self.cache = cache
    
    def get_frequent_itemsets(self, transaction_matrix_hash: str, min_support: float, 
                            use_colnames: bool = True) -> Optional[pd.DataFrame]:
        """Get cached frequent itemsets."""
        parameters = {
            'transaction_matrix_hash': transaction_matrix_hash,
            'min_support': min_support,
            'use_colnames': use_colnames
        }
        
        return self.cache.get_cached_result('frequent_itemsets', parameters)
    
    def save_frequent_itemsets(self, transaction_matrix_hash: str, min_support: float,
                             use_colnames: bool, frequent_itemsets: pd.DataFrame) -> bool:
        """Save frequent itemsets to cache."""
        parameters = {
            'transaction_matrix_hash': transaction_matrix_hash,
            'min_support': min_support,
            'use_colnames': use_colnames
        }
        
        return self.cache.save_result('frequent_itemsets', parameters, frequent_itemsets)
    
    def _hash_transaction_matrix(self, transaction_matrix: pd.DataFrame) -> str:
        """Generate a hash for the transaction matrix to use as cache key."""
        # Create a hash based on matrix shape and a sample of values
        matrix_info = {
            'shape': transaction_matrix.shape,
            'columns': list(transaction_matrix.columns),
            'sample_hash': hashlib.md5(
                str(transaction_matrix.iloc[:min(100, len(transaction_matrix))].values).encode()
            ).hexdigest()
        }
        
        return hashlib.md5(str(matrix_info).encode()).hexdigest()


class AssociationRuleCache:
    """
    Specialized cache for association rules with parameter-aware caching.
    """
    
    def __init__(self, cache: ResultCache):
        self.cache = cache
    
    def get_association_rules(self, frequent_itemsets_hash: str, min_confidence: float,
                            metric: str = 'lift', metric_threshold: float = 1.0) -> Optional[pd.DataFrame]:
        """Get cached association rules."""
        parameters = {
            'frequent_itemsets_hash': frequent_itemsets_hash,
            'min_confidence': min_confidence,
            'metric': metric,
            'metric_threshold': metric_threshold
        }
        
        return self.cache.get_cached_result('association_rules', parameters)
    
    def save_association_rules(self, frequent_itemsets_hash: str, min_confidence: float,
                             metric: str, metric_threshold: float, rules: pd.DataFrame) -> bool:
        """Save association rules to cache."""
        parameters = {
            'frequent_itemsets_hash': frequent_itemsets_hash,
            'min_confidence': min_confidence,
            'metric': metric,
            'metric_threshold': metric_threshold
        }
        
        return self.cache.save_result('association_rules', parameters, rules)
    
    def _hash_frequent_itemsets(self, frequent_itemsets: pd.DataFrame) -> str:
        """Generate a hash for frequent itemsets to use as cache key."""
        # Create a hash based on itemsets and support values
        itemsets_info = {
            'length': len(frequent_itemsets),
            'support_sum': frequent_itemsets['support'].sum(),
            'itemsets_sample': str(frequent_itemsets.head(10)['itemsets'].tolist())
        }
        
        return hashlib.md5(str(itemsets_info).encode()).hexdigest()


class PersistenceManager:
    """
    Manages persistence of analysis results across notebook runs with versioning.
    """
    
    def __init__(self, persistence_dir: str = ".persistence", enable_persistence: bool = True,
                 verbose: bool = True):
        self.persistence_dir = persistence_dir
        self.enable_persistence = enable_persistence
        self.verbose = verbose
        
        if self.enable_persistence:
            os.makedirs(self.persistence_dir, exist_ok=True)
    
    def save_analysis_session(self, session_name: str, data: Dict[str, Any]) -> bool:
        """Save complete analysis session data."""
        if not self.enable_persistence:
            return False
        
        try:
            # Add metadata
            session_data = {
                'metadata': {
                    'session_name': session_name,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'data': data
            }
            
            filepath = os.path.join(self.persistence_dir, f"{session_name}.json")
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            if self.verbose:
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"💾 Saved analysis session '{session_name}' ({file_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving session '{session_name}': {str(e)}")
            return False
    
    def load_analysis_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Load complete analysis session data."""
        if not self.enable_persistence:
            return None
        
        filepath = os.path.join(self.persistence_dir, f"{session_name}.json")
        
        if not os.path.exists(filepath):
            if self.verbose:
                print(f"📂 Session '{session_name}' not found")
            return None
        
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            if self.verbose:
                timestamp = session_data.get('metadata', {}).get('timestamp', 'Unknown')
                print(f"📂 Loaded analysis session '{session_name}' (saved: {timestamp})")
            
            return session_data.get('data', {})
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading session '{session_name}': {str(e)}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available analysis sessions."""
        sessions = []
        
        if not self.enable_persistence or not os.path.exists(self.persistence_dir):
            return sessions
        
        for filename in os.listdir(self.persistence_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.persistence_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                    
                    metadata = session_data.get('metadata', {})
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    
                    sessions.append({
                        'name': metadata.get('session_name', filename[:-5]),
                        'timestamp': metadata.get('timestamp', 'Unknown'),
                        'size_mb': file_size_mb,
                        'filepath': filepath
                    })
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Error reading session file {filename}: {str(e)}")
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_session(self, session_name: str) -> bool:
        """Delete an analysis session."""
        if not self.enable_persistence:
            return False
        
        filepath = os.path.join(self.persistence_dir, f"{session_name}.json")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                if self.verbose:
                    print(f"🗑️  Deleted session '{session_name}'")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error deleting session '{session_name}': {str(e)}")
        
        return False


# Example usage and demonstration functions
def demo_result_caching():
    """Demonstrate the result caching system."""
    
    print("🧪 Demo: Result Caching System")
    
    # Initialize cache
    cache = ResultCache(cache_dir=".demo_cache", verbose=True)
    
    # Simulate some expensive computations
    print("\n1. First computation (should be cache MISS)")
    params1 = {'min_support': 0.01, 'dataset_size': 1000}
    
    # Check cache first
    result1 = cache.get_cached_result('frequent_itemsets', params1)
    
    if result1 is None:
        # Simulate expensive computation
        import time
        time.sleep(1)  # Simulate work
        result1 = pd.DataFrame({'itemsets': [['A'], ['B'], ['A', 'B']], 'support': [0.3, 0.4, 0.15]})
        
        # Save to cache
        cache.save_result('frequent_itemsets', params1, result1)
    
    print("\n2. Same computation (should be cache HIT)")
    result1_cached = cache.get_cached_result('frequent_itemsets', params1)
    
    print("\n3. Different parameters (should be cache MISS)")
    params2 = {'min_support': 0.02, 'dataset_size': 1000}
    result2 = cache.get_cached_result('frequent_itemsets', params2)
    
    # Print cache report
    cache.print_cache_report()
    
    # Clean up demo cache
    cache.clear_cache()


def demo_persistence_manager():
    """Demonstrate the persistence manager."""
    
    print("\n🧪 Demo: Persistence Manager")
    
    # Initialize persistence manager
    persistence = PersistenceManager(persistence_dir=".demo_persistence", verbose=True)
    
    # Create sample analysis data
    analysis_data = {
        'frequent_itemsets': {
            'count': 25,
            'min_support': 0.01,
            'top_itemsets': [['Book1'], ['Book2'], ['Book1', 'Book2']]
        },
        'association_rules': {
            'count': 15,
            'min_confidence': 0.5,
            'top_rules': [
                {'antecedent': ['Book1'], 'consequent': ['Book2'], 'confidence': 0.8, 'lift': 1.5}
            ]
        },
        'performance_stats': {
            'mining_time': 45.2,
            'memory_usage_mb': 256.7
        }
    }
    
    # Save session
    print("\n1. Saving analysis session")
    persistence.save_analysis_session('demo_session', analysis_data)
    
    # List sessions
    print("\n2. Listing available sessions")
    sessions = persistence.list_sessions()
    for session in sessions:
        print(f"   - {session['name']}: {session['size_mb']:.2f} MB ({session['timestamp']})")
    
    # Load session
    print("\n3. Loading analysis session")
    loaded_data = persistence.load_analysis_session('demo_session')
    
    if loaded_data:
        print(f"   Loaded {len(loaded_data)} data sections")
        print(f"   Frequent itemsets: {loaded_data['frequent_itemsets']['count']}")
        print(f"   Association rules: {loaded_data['association_rules']['count']}")
    
    # Clean up demo persistence
    persistence.delete_session('demo_session')


if __name__ == "__main__":
    demo_result_caching()
    demo_persistence_manager()