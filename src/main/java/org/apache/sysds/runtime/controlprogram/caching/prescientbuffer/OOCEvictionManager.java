package org.apache.sysds.runtime.controlprogram.caching.prescientbuffer;

import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheEvictionQueue;

public class OOCEvictionManager {
	private static OOCEvictionManager _instance;

	// Queue of cached OOC stream blocks (similar to LazyWriteBuffer)
	private CacheEvictionQueue _streamQueue;
	private long _cacheLimit;
	private long _currentSize;

	private OOCEvictionManager() {
		_streamQueue = new CacheEvictionQueue();
		_cacheLimit = 0;/* configure based on OOC memory budget */
		_currentSize = 0;
	}

	public static OOCEvictionManager getInstance() {
		if (_instance == null) {
			synchronized (OOCEvictionManager.class) {
				if (_instance == null) {
					_instance = new OOCEvictionManager();
				}
			}
		}
		return _instance;
	}

	// Add a block to the cache
	public synchronized void cacheBlock(String blockID, CacheBlock<?> block) { /* ... */ }

	// Evict blocks to make space
	public synchronized void makeSpace(long requiredSize) { /* ... */ }

	// Get a cached block (updates LRU order)
	public synchronized CacheBlock<?> getBlock(String blockID) { /* ... */ }
}
