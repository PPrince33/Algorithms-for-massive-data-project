# Performance Monitoring and Optimization System for Market Basket Analysis

import time
import psutil
import tracemalloc
import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from functools import wraps

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for tracking execution time,
    memory usage, and providing optimization recommendations.
    """
    
    def __init__(self, enable_memory_tracking=True, verbose=True):
        self.enable_memory_tracking = enable_memory_tracking
        self.verbose = verbose
        self.execution_history = []
        self.memory_snapshots = []
        self.current_operation = None
        
        if self.enable_memory_tracking:
            tracemalloc.start()
    
    @contextmanager
    def monitor_operation(self, operation_name: str, expected_memory_mb: float = None):
        """Context manager for monitoring a specific operation."""
        self.current_operation = operation_name
        
        # Get initial system state
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        if self.enable_memory_tracking:
            tracemalloc_start = tracemalloc.get_traced_memory()
        
        if self.verbose:
            print(f"🔍 Starting operation: {operation_name}")
            print(f"   Initial memory: {initial_memory:.1f} MB")
            if expected_memory_mb:
                print(f"   Expected memory usage: {expected_memory_mb:.1f} MB")
        
        try:
            yield self
        finally:
            # Calculate final metrics
            end_time = time.time()
            execution_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            final_cpu = process.cpu_percent()
            
            if self.enable_memory_tracking:
                tracemalloc_end = tracemalloc.get_traced_memory()
                peak_memory_mb = tracemalloc_end[1] / 1024 / 1024
            else:
                peak_memory_mb = final_memory
            
            # Store execution record
            execution_record = {
                'operation': operation_name,
                'execution_time': execution_time,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_delta_mb': memory_delta,
                'peak_memory_mb': peak_memory_mb,
                'cpu_usage': final_cpu,
                'timestamp': time.time()
            }
            
            self.execution_history.append(execution_record)
            
            if self.verbose:
                self._print_operation_summary(execution_record, expected_memory_mb)
            
            self.current_operation = None
    
    def _print_operation_summary(self, record: Dict, expected_memory_mb: float = None):
        """Print a formatted summary of the operation performance."""
        print(f"✅ Completed: {record['operation']}")
        print(f"   Execution time: {record['execution_time']:.2f} seconds")
        print(f"   Memory usage: {record['memory_delta_mb']:+.1f} MB (peak: {record['peak_memory_mb']:.1f} MB)")
        
        if expected_memory_mb and record['peak_memory_mb'] > expected_memory_mb * 1.5:
            print(f"   ⚠️  Memory usage exceeded expected by {record['peak_memory_mb'] - expected_memory_mb:.1f} MB")
        
        if record['execution_time'] > 60:
            print(f"   ⏰ Long execution time detected - consider optimization")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.execution_history:
            return {'error': 'No operations recorded'}
        
        df = pd.DataFrame(self.execution_history)
        
        summary = {
            'total_operations': len(self.execution_history),
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean(),
            'peak_memory_usage': df['peak_memory_mb'].max(),
            'total_memory_allocated': df['memory_delta_mb'].sum(),
            'slowest_operations': df.nlargest(3, 'execution_time')[['operation', 'execution_time']].to_dict('records'),
            'memory_intensive_operations': df.nlargest(3, 'peak_memory_mb')[['operation', 'peak_memory_mb']].to_dict('records'),
            'operations_by_type': df['operation'].value_counts().to_dict()
        }
        
        return summary
    
    def print_performance_report(self):
        """Print a comprehensive performance report."""
        summary = self.get_performance_summary()
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"\n📊 Performance Report")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Total execution time: {summary['total_execution_time']:.2f} seconds")
        print(f"   Average execution time: {summary['average_execution_time']:.2f} seconds")
        print(f"   Peak memory usage: {summary['peak_memory_usage']:.1f} MB")
        
        print(f"\n🐌 Slowest Operations:")
        for i, op in enumerate(summary['slowest_operations'], 1):
            print(f"   {i}. {op['operation']}: {op['execution_time']:.2f}s")
        
        print(f"\n💾 Most Memory-Intensive Operations:")
        for i, op in enumerate(summary['memory_intensive_operations'], 1):
            print(f"   {i}. {op['operation']}: {op['peak_memory_mb']:.1f} MB")
    
    def suggest_optimizations(self) -> List[str]:
        """Provide optimization suggestions based on performance data."""
        suggestions = []
        summary = self.get_performance_summary()
        
        if 'error' in summary:
            return ['No performance data available for analysis']
        
        # Memory optimization suggestions
        if summary['peak_memory_usage'] > 1000:  # > 1GB
            suggestions.append("Consider chunked processing for large datasets (peak memory > 1GB)")
        
        # Execution time suggestions
        if summary['average_execution_time'] > 30:
            suggestions.append("Consider distributed processing or algorithm optimization (avg time > 30s)")
        
        # Specific operation suggestions
        for op in summary['slowest_operations']:
            if 'mining' in op['operation'].lower() and op['execution_time'] > 60:
                suggestions.append(f"Consider increasing min_support for {op['operation']} (execution time: {op['execution_time']:.1f}s)")
        
        return suggestions if suggestions else ['Performance looks good - no specific optimizations needed']


class DatasetOptimizer:
    """
    Handles efficient dataset processing with prototype vs full dataset modes,
    memory-efficient transaction matrix handling, and progress indicators.
    """
    
    def __init__(self, use_prototype=True, prototype_size=10000, verbose=True):
        self.use_prototype = use_prototype
        self.prototype_size = prototype_size
        self.verbose = verbose
        self.optimization_stats = {}
    
    def optimize_dataset_loading(self, data_loader_func, *args, **kwargs):
        """Optimize dataset loading based on prototype vs full dataset mode."""
        
        if self.verbose:
            mode = "Prototype" if self.use_prototype else "Full Dataset"
            print(f"📊 Dataset Loading Mode: {mode}")
            if self.use_prototype:
                print(f"   Sample size: {self.prototype_size:,} records")
        
        # Load data using the provided function
        data = data_loader_func(*args, **kwargs)
        
        # Apply prototype sampling if enabled
        if self.use_prototype and len(data) > self.prototype_size:
            original_size = len(data)
            data = data.sample(n=self.prototype_size, random_state=42)
            
            if self.verbose:
                reduction_pct = (1 - self.prototype_size / original_size) * 100
                print(f"   Sampled {self.prototype_size:,} from {original_size:,} records ({reduction_pct:.1f}% reduction)")
        
        return data
    
    def create_memory_efficient_transaction_matrix(self, user_baskets: Dict, 
                                                  chunk_size: int = 1000) -> pd.DataFrame:
        """Create transaction matrix with memory optimization for large datasets."""
        
        total_users = len(user_baskets)
        
        if self.verbose:
            print(f"🔄 Creating memory-efficient transaction matrix")
            print(f"   Total users: {total_users:,}")
            print(f"   Chunk size: {chunk_size:,}")
        
        # Determine if chunking is needed
        use_chunking = total_users > chunk_size * 2
        
        if use_chunking:
            return self._create_chunked_transaction_matrix(user_baskets, chunk_size)
        else:
            return self._create_standard_transaction_matrix(user_baskets)
    
    def _create_standard_transaction_matrix(self, user_baskets: Dict) -> pd.DataFrame:
        """Standard transaction matrix creation for smaller datasets."""
        from mlxtend.preprocessing import TransactionEncoder
        
        transactions = [basket.book_ids for basket in user_baskets.values()]
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        
        return pd.DataFrame(te_ary, columns=te.columns_)
    
    def _create_chunked_transaction_matrix(self, user_baskets: Dict, 
                                         chunk_size: int) -> pd.DataFrame:
        """Memory-efficient chunked transaction matrix creation."""
        
        user_ids = list(user_baskets.keys())
        num_chunks = (len(user_ids) + chunk_size - 1) // chunk_size
        
        if self.verbose:
            print(f"   Processing in {num_chunks} chunks")
        
        # First pass: collect all unique items
        all_items = set()
        for basket in user_baskets.values():
            all_items.update(basket.book_ids)
        
        all_items = sorted(list(all_items))
        
        if self.verbose:
            print(f"   Total unique items: {len(all_items):,}")
        
        # Create transaction matrix chunk by chunk
        transaction_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(user_ids))
            chunk_user_ids = user_ids[start_idx:end_idx]
            
            # Create chunk transaction matrix
            chunk_data = []
            for user_id in chunk_user_ids:
                user_items = set(user_baskets[user_id].book_ids)
                row = [item in user_items for item in all_items]
                chunk_data.append(row)
            
            chunk_df = pd.DataFrame(chunk_data, columns=all_items)
            transaction_chunks.append(chunk_df)
            
            if self.verbose and (i + 1) % max(1, num_chunks // 10) == 0:
                progress = ((i + 1) / num_chunks) * 100
                print(f"   Progress: {progress:.1f}% ({i + 1}/{num_chunks} chunks)")
            
            # Force garbage collection after each chunk
            gc.collect()
        
        # Combine all chunks
        if self.verbose:
            print(f"   Combining {len(transaction_chunks)} chunks...")
        
        final_matrix = pd.concat(transaction_chunks, ignore_index=True)
        
        # Clean up
        del transaction_chunks
        gc.collect()
        
        return final_matrix
    
    def estimate_memory_requirements(self, num_users: int, num_items: int) -> Dict[str, float]:
        """Estimate memory requirements for transaction matrix."""
        
        # Estimate memory for boolean matrix (1 byte per cell)
        matrix_size_bytes = num_users * num_items
        matrix_size_mb = matrix_size_bytes / (1024 * 1024)
        
        # Add overhead for pandas DataFrame (approximately 2x)
        estimated_memory_mb = matrix_size_mb * 2
        
        # Add buffer for processing (50% extra)
        recommended_memory_mb = estimated_memory_mb * 1.5
        
        return {
            'matrix_size_mb': matrix_size_mb,
            'estimated_memory_mb': estimated_memory_mb,
            'recommended_memory_mb': recommended_memory_mb,
            'num_users': num_users,
            'num_items': num_items
        }
    
    def get_optimization_recommendations(self, num_users: int, num_items: int) -> List[str]:
        """Get optimization recommendations based on dataset size."""
        
        memory_est = self.estimate_memory_requirements(num_users, num_items)
        recommendations = []
        
        if memory_est['recommended_memory_mb'] > 2000:  # > 2GB
            recommendations.append("Consider using chunked processing (estimated memory > 2GB)")
            recommendations.append("Enable prototype mode for development and testing")
        
        if num_users > 100000:
            recommendations.append("Consider distributed processing with PySpark or Dask")
        
        if num_items > 50000:
            recommendations.append("Consider item filtering to reduce dimensionality")
            recommendations.append("Increase min_support to focus on more frequent items")
        
        return recommendations if recommendations else ['Dataset size is manageable with current configuration']


class ProgressIndicator:
    """
    Provides progress indicators for long-running operations with time estimates.
    """
    
    def __init__(self, total_steps: int, operation_name: str = "Operation", verbose: bool = True):
        self.total_steps = total_steps
        self.operation_name = operation_name
        self.verbose = verbose
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def update(self, steps: int = 1, message: str = None):
        """Update progress by specified number of steps."""
        self.current_step += steps
        current_time = time.time()
        
        # Update every 5% or every 5 seconds, whichever comes first
        progress_pct = (self.current_step / self.total_steps) * 100
        time_since_update = current_time - self.last_update_time
        
        should_update = (
            progress_pct % 5 < (steps / self.total_steps) * 100 or  # Every 5%
            time_since_update >= 5 or  # Every 5 seconds
            self.current_step >= self.total_steps  # Final update
        )
        
        if should_update and self.verbose:
            elapsed_time = current_time - self.start_time
            
            if self.current_step < self.total_steps:
                # Estimate remaining time
                rate = self.current_step / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = self.total_steps - self.current_step
                eta_seconds = remaining_steps / rate if rate > 0 else 0
                eta_str = f", ETA: {eta_seconds:.0f}s" if eta_seconds > 0 else ""
            else:
                eta_str = ""
            
            status_msg = f"   {self.operation_name}: {progress_pct:.1f}% ({self.current_step:,}/{self.total_steps:,})"
            status_msg += f" - {elapsed_time:.1f}s elapsed{eta_str}"
            
            if message:
                status_msg += f" - {message}"
            
            print(status_msg)
            self.last_update_time = current_time
    
    def finish(self, message: str = None):
        """Mark operation as complete."""
        if self.verbose:
            total_time = time.time() - self.start_time
            final_msg = f"✅ {self.operation_name} completed in {total_time:.2f}s"
            if message:
                final_msg += f" - {message}"
            print(final_msg)


def performance_decorator(operation_name: str, monitor: PerformanceMonitor):
    """Decorator for automatically monitoring function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.monitor_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage functions
def demo_performance_monitoring():
    """Demonstrate the performance monitoring system."""
    
    print("🧪 Demo: Performance Monitoring System")
    
    # Initialize monitor
    monitor = PerformanceMonitor(enable_memory_tracking=True, verbose=True)
    
    # Simulate some operations
    with monitor.monitor_operation("Data Loading", expected_memory_mb=100):
        # Simulate data loading
        time.sleep(1)
        data = pd.DataFrame(np.random.randn(10000, 50))
    
    with monitor.monitor_operation("Data Processing"):
        # Simulate processing
        time.sleep(0.5)
        processed_data = data.sum(axis=1)
    
    with monitor.monitor_operation("Model Training", expected_memory_mb=200):
        # Simulate model training
        time.sleep(2)
        model_result = processed_data.mean()
    
    # Print performance report
    monitor.print_performance_report()
    
    # Get optimization suggestions
    suggestions = monitor.suggest_optimizations()
    print(f"\n💡 Optimization Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")


def demo_dataset_optimization():
    """Demonstrate the dataset optimization system."""
    
    print("\n🧪 Demo: Dataset Optimization System")
    
    # Initialize optimizer
    optimizer = DatasetOptimizer(use_prototype=True, prototype_size=5000, verbose=True)
    
    # Simulate dataset size analysis
    num_users = 50000
    num_items = 10000
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Users: {num_users:,}")
    print(f"   Items: {num_items:,}")
    
    # Get memory estimates
    memory_est = optimizer.estimate_memory_requirements(num_users, num_items)
    print(f"\n💾 Memory Estimates:")
    print(f"   Matrix size: {memory_est['matrix_size_mb']:.1f} MB")
    print(f"   Estimated memory: {memory_est['estimated_memory_mb']:.1f} MB")
    print(f"   Recommended memory: {memory_est['recommended_memory_mb']:.1f} MB")
    
    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations(num_users, num_items)
    print(f"\n💡 Optimization Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")


if __name__ == "__main__":
    demo_performance_monitoring()
    demo_dataset_optimization()