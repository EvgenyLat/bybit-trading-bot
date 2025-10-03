"""
Thread Safety and Concurrency Management
Safe concurrent operations and resource management
"""

import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """Resource state"""
    AVAILABLE = "available"
    LOCKED = "locked"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ResourceLock:
    """Resource lock information"""
    resource_id: str
    thread_id: int
    timestamp: datetime
    timeout: float
    lock_type: str


class ThreadSafeResourceManager:
    """Thread-safe resource management"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.resources: Dict[str, Any] = {}
        self.resource_locks: Dict[str, ResourceLock] = {}
        self.resource_states: Dict[str, ResourceState] = {}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def acquire_resource(self, resource_id: str, timeout: float = 30.0, 
                        lock_type: str = "exclusive") -> bool:
        """Acquire resource with timeout"""
        try:
            with self.condition:
                start_time = time.time()
                
                while self._is_resource_locked(resource_id):
                    if time.time() - start_time > timeout:
                        logger.warning(f"Timeout acquiring resource {resource_id}")
                        return False
                    
                    self.condition.wait(timeout=1.0)
                
                # Acquire the resource
                self.resource_locks[resource_id] = ResourceLock(
                    resource_id=resource_id,
                    thread_id=threading.get_ident(),
                    timestamp=datetime.now(),
                    timeout=timeout,
                    lock_type=lock_type
                )
                
                self.resource_states[resource_id] = ResourceState.LOCKED
                logger.debug(f"Resource {resource_id} acquired by thread {threading.get_ident()}")
                return True
                
        except Exception as e:
            logger.error(f"Error acquiring resource {resource_id}: {e}")
            return False
    
    def release_resource(self, resource_id: str) -> bool:
        """Release resource"""
        try:
            with self.condition:
                if resource_id in self.resource_locks:
                    lock = self.resource_locks[resource_id]
                    
                    # Check if current thread owns the lock
                    if lock.thread_id != threading.get_ident():
                        logger.warning(f"Thread {threading.get_ident()} trying to release "
                                     f"resource {resource_id} owned by thread {lock.thread_id}")
                        return False
                    
                    # Release the resource
                    del self.resource_locks[resource_id]
                    self.resource_states[resource_id] = ResourceState.AVAILABLE
                    
                    # Notify waiting threads
                    self.condition.notify_all()
                    
                    logger.debug(f"Resource {resource_id} released by thread {threading.get_ident()}")
                    return True
                else:
                    logger.warning(f"Resource {resource_id} not locked")
                    return False
                    
        except Exception as e:
            logger.error(f"Error releasing resource {resource_id}: {e}")
            return False
    
    def _is_resource_locked(self, resource_id: str) -> bool:
        """Check if resource is locked"""
        if resource_id not in self.resource_locks:
            return False
        
        lock = self.resource_locks[resource_id]
        
        # Check if lock has expired
        if time.time() - lock.timestamp.timestamp() > lock.timeout:
            logger.warning(f"Resource {resource_id} lock expired, releasing")
            del self.resource_locks[resource_id]
            self.resource_states[resource_id] = ResourceState.AVAILABLE
            return False
        
        return True
    
    @contextmanager
    def resource_lock(self, resource_id: str, timeout: float = 30.0, 
                     lock_type: str = "exclusive"):
        """Context manager for resource locking"""
        acquired = False
        try:
            acquired = self.acquire_resource(resource_id, timeout, lock_type)
            if not acquired:
                raise RuntimeError(f"Failed to acquire resource {resource_id}")
            yield
        finally:
            if acquired:
                self.release_resource(resource_id)


class AsyncResourceManager:
    """Async resource management"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        self.resource_semaphores: Dict[str, asyncio.Semaphore] = {}
        
    async def acquire_resource(self, resource_id: str, timeout: float = 30.0) -> bool:
        """Acquire resource asynchronously"""
        try:
            if resource_id not in self.resource_locks:
                self.resource_locks[resource_id] = asyncio.Lock()
            
            lock = self.resource_locks[resource_id]
            
            # Try to acquire lock with timeout
            try:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
                logger.debug(f"Resource {resource_id} acquired asynchronously")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout acquiring resource {resource_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error acquiring resource {resource_id}: {e}")
            return False
    
    async def release_resource(self, resource_id: str) -> bool:
        """Release resource asynchronously"""
        try:
            if resource_id in self.resource_locks:
                lock = self.resource_locks[resource_id]
                lock.release()
                logger.debug(f"Resource {resource_id} released asynchronously")
                return True
            else:
                logger.warning(f"Resource {resource_id} not locked")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing resource {resource_id}: {e}")
            return False
    
    @asynccontextmanager
    async def resource_lock(self, resource_id: str, timeout: float = 30.0):
        """Async context manager for resource locking"""
        acquired = False
        try:
            acquired = await self.acquire_resource(resource_id, timeout)
            if not acquired:
                raise RuntimeError(f"Failed to acquire resource {resource_id}")
            yield
        finally:
            if acquired:
                await self.release_resource(resource_id)


class ThreadSafeCache:
    """Thread-safe cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.cleanup_lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with self.lock:
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # Check if entry has expired
                    if time.time() - entry['timestamp'] > self.ttl:
                        del self.cache[key]
                        return None
                    
                    return entry['value']
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set value in cache"""
        try:
            with self.lock:
                # Check if cache is full
                if len(self.cache) >= self.max_size:
                    self._evict_oldest()
                
                self.cache[key] = {
                    'value': value,
                    'timestamp': time.time()
                }
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def _evict_oldest(self):
        """Evict oldest entry from cache"""
        try:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            
        except Exception as e:
            logger.error(f"Error evicting cache entry: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        try:
            with self.lock:
                self.cache.clear()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        try:
            with self.cleanup_lock:
                current_time = time.time()
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if current_time - entry['timestamp'] > self.ttl
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")


class SafeQueue:
    """Thread-safe queue with error handling"""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.stats = {
            'put_count': 0,
            'get_count': 0,
            'error_count': 0,
            'timeout_count': 0
        }
    
    def put(self, item: Any, timeout: float = None) -> bool:
        """Put item in queue safely"""
        try:
            self.queue.put(item, timeout=timeout)
            with self.lock:
                self.stats['put_count'] += 1
            return True
            
        except queue.Full:
            with self.lock:
                self.stats['timeout_count'] += 1
            logger.warning("Queue is full, item not added")
            return False
            
        except Exception as e:
            with self.lock:
                self.stats['error_count'] += 1
            logger.error(f"Error putting item in queue: {e}")
            return False
    
    def get(self, timeout: float = None) -> Optional[Any]:
        """Get item from queue safely"""
        try:
            item = self.queue.get(timeout=timeout)
            with self.lock:
                self.stats['get_count'] += 1
            return item
            
        except queue.Empty:
            with self.lock:
                self.stats['timeout_count'] += 1
            return None
            
        except Exception as e:
            with self.lock:
                self.stats['error_count'] += 1
            logger.error(f"Error getting item from queue: {e}")
            return None
    
    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            return self.stats.copy()


class ConcurrentExecutor:
    """Safe concurrent execution manager"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Any] = {}
        self.task_lock = threading.Lock()
        
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> bool:
        """Submit task for execution"""
        try:
            with self.task_lock:
                if task_id in self.active_tasks:
                    logger.warning(f"Task {task_id} already active")
                    return False
                
                future = self.executor.submit(func, *args, **kwargs)
                self.active_tasks[task_id] = {
                    'future': future,
                    'start_time': time.time(),
                    'status': 'running'
                }
                
                logger.debug(f"Task {task_id} submitted")
                return True
                
        except Exception as e:
            logger.error(f"Error submitting task {task_id}: {e}")
            return False
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get task result"""
        try:
            with self.task_lock:
                if task_id not in self.active_tasks:
                    logger.warning(f"Task {task_id} not found")
                    return None
                
                task_info = self.active_tasks[task_id]
                future = task_info['future']
                
                try:
                    result = future.result(timeout=timeout)
                    task_info['status'] = 'completed'
                    del self.active_tasks[task_id]
                    return result
                    
                except Exception as e:
                    task_info['status'] = 'failed'
                    logger.error(f"Task {task_id} failed: {e}")
                    del self.active_tasks[task_id]
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting task result {task_id}: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        try:
            with self.task_lock:
                if task_id in self.active_tasks:
                    task_info = self.active_tasks[task_id]
                    future = task_info['future']
                    
                    if future.cancel():
                        task_info['status'] = 'cancelled'
                        del self.active_tasks[task_id]
                        logger.info(f"Task {task_id} cancelled")
                        return True
                    else:
                        logger.warning(f"Task {task_id} could not be cancelled")
                        return False
                else:
                    logger.warning(f"Task {task_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get active tasks information"""
        with self.task_lock:
            return {
                task_id: {
                    'status': task_info['status'],
                    'start_time': task_info['start_time'],
                    'duration': time.time() - task_info['start_time']
                }
                for task_id, task_info in self.active_tasks.items()
            }
    
    def cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        try:
            with self.task_lock:
                completed_tasks = [
                    task_id for task_id, task_info in self.active_tasks.items()
                    if task_info['future'].done()
                ]
                
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                if completed_tasks:
                    logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")
                    
        except Exception as e:
            logger.error(f"Error cleaning up tasks: {e}")
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor"""
        try:
            self.executor.shutdown(wait=wait)
            logger.info("Concurrent executor shutdown")
            
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")


class RateLimiter:
    """Thread-safe rate limiter"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = queue.Queue()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: float = None) -> bool:
        """Acquire permission to make request"""
        try:
            current_time = time.time()
            
            with self.lock:
                # Remove old requests outside time window
                while not self.requests.empty():
                    try:
                        old_time = self.requests.get_nowait()
                        if current_time - old_time < self.time_window:
                            self.requests.put(old_time)
                            break
                    except queue.Empty:
                        break
                
                # Check if we can make a new request
                if self.requests.qsize() < self.max_requests:
                    self.requests.put(current_time)
                    return True
                else:
                    # Wait for oldest request to expire
                    oldest_time = self.requests.queue[0]
                    wait_time = self.time_window - (current_time - oldest_time)
                    
                    if timeout and wait_time > timeout:
                        return False
                    
                    time.sleep(wait_time)
                    return self.acquire(timeout)
            
        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        try:
            with self.lock:
                if self.requests.empty():
                    return 0.0
                
                current_time = time.time()
                oldest_time = self.requests.queue[0]
                wait_time = self.time_window - (current_time - oldest_time)
                
                return max(0.0, wait_time)
                
        except Exception as e:
            logger.error(f"Error getting wait time: {e}")
            return 0.0

