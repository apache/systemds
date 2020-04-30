/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.lineage;

import java.io.IOException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCacheStatus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.LocalFileUtils;

public class LineageCacheEviction {
	private static LineageCacheEntry _head = null;
	private static LineageCacheEntry _end = null;
	private static long _cachesize = 0;
	private static long CACHE_LIMIT; //limit in bytes
	protected static final HashSet<LineageItem> _removelist = new HashSet<>();
	private static final Map<LineageItem, SpilledItem> _spillList = new HashMap<>();
	private static String _outdir = null;


	protected static void resetEviction() {
		_head = null;
		_end = null;
		// reset cache size, otherwise the cache clear leads to unusable 
		// space which means evictions could run into endless loops
		_cachesize = 0;
		_spillList.clear();
		weightedQueue.clear();
		_outdir = null;
		if (DMLScript.STATISTICS)
			_removelist.clear();
	}

	private static Comparator<LineageCacheEntry> execTime2SizeComparator = (e1, e2) -> {
		double t2s1 = ((double)e1._computeTime)/e1.getSize();
		double t2s2 = ((double)e2._computeTime)/e2.getSize();
		return t2s1 == t2s2 ? 0 : t2s1 < t2s2 ? -1 : 1;
	};
	
	private static PriorityQueue<LineageCacheEntry> weightedQueue = new PriorityQueue<>(execTime2SizeComparator);

	//--------------- CACHE MAINTENANCE & LOOKUP FUNCTIONS ---------//
	
	protected static void addEntry(LineageCacheEntry entry) {
		if (entry.isNullVal())
			// Placeholders shouldn't be evicted.
			return;

		double exectime = ((double) entry._computeTime) / 1000000; // in milliseconds
		if (!entry.isMatrixValue() && exectime >= LineageCacheConfig.MIN_SPILL_TIME_ESTIMATE)
			// Pin the entries having scalar values and with higher computation time
			// to memory, to save those from eviction. Scalar values are
			// not spilled to disk and are just deleted. Scalar entries associated 
			// with high computation time might contain function outputs. Pinning them
			// will increase chances of multilevel reuse.
			entry.setCacheStatus(LineageCacheStatus.PINNED);

		if (LineageCacheConfig.getCachePolicy().isLRUcache()) //LRU 
			// Maintain linked list.
			setHead(entry);
		else {
			if (entry.isMatrixValue() || exectime < LineageCacheConfig.MIN_SPILL_TIME_ESTIMATE)
				// Don't add the memory pinned entries in weighted queue. 
				// The priorityQueue should contain only entries that can
				// be removed or spilled to disk.
				weightedQueue.add(entry);
		}
	}
	
	protected static void getEntry(LineageCacheEntry entry) {
		if (LineageCacheConfig.getCachePolicy().isLRUcache()) { //LRU 
			// maintain linked list.
			delete(entry);
			setHead(entry);
		}
		// No maintenance is required for weighted scheme
	}

	protected static void removeEntry(Map<LineageItem, LineageCacheEntry> cache, LineageItem key) {
		if (!cache.containsKey(key))
			return;
		if (LineageCacheConfig.getCachePolicy().isLRUcache()) //LRU 
			delete(cache.get(key));
		else
			weightedQueue.remove(cache.get(key));
		cache.remove(key);
	}

	private static void removeEntry(Map<LineageItem, LineageCacheEntry> cache, LineageCacheEntry e) {
		if (DMLScript.STATISTICS)
			_removelist.add(e._key);

		if (LineageCacheConfig.getCachePolicy().isLRUcache()) //LRU 
			delete(e);
		_cachesize -= e.getSize();
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementMemDeletes();
	}

	private static void delete(LineageCacheEntry entry) {
		if (entry._prev != null)
			entry._prev._next = entry._next;
		else
			_head = entry._next;
		if (entry._next != null)
			entry._next._prev = entry._prev;
		else
			_end = entry._prev;
	}
	
	protected static void setHead(LineageCacheEntry entry) {
		entry._next = _head;
		entry._prev = null;
		if (_head != null)
			_head._prev = entry;
		_head = entry;
		if (_end == null)
			_end = _head;
	}
	
	//---------------- CACHE SPACE MANAGEMENT METHODS -----------------
	
	protected static void setCacheLimit(long limit) {
		CACHE_LIMIT = limit;
	}

	protected static long getCacheLimit() {
		return CACHE_LIMIT;
	}
	
	protected static void updateSize(long space, boolean addspace) {
		if (addspace)
			_cachesize += space;
		else
			_cachesize -= space;
	}

	protected static boolean isBelowThreshold(long spaceNeeded) {
		return ((spaceNeeded + _cachesize) <= CACHE_LIMIT);
	}

	protected static void makeSpace(Map<LineageItem, LineageCacheEntry> cache, long spaceNeeded) {
		//Cost based eviction
		boolean isLRU = LineageCacheConfig.getCachePolicy().isLRUcache();
		LineageCacheEntry e = isLRU ? _end : weightedQueue.poll();
		while (e != _head && e != null)
		{
			if ((spaceNeeded + _cachesize) <= CACHE_LIMIT)
				// Enough space recovered.
				break;

			if (!LineageCacheConfig.isSetSpill()) {
				// If eviction is disabled, just delete the entries.
				if (cache.remove(e._key) != null)
					removeEntry(cache, e);
				e = isLRU ? e._prev : weightedQueue.poll();
				continue;
			}

			if (!e.getCacheStatus().canEvict() && isLRU) {
				// Don't delete if the entry's cache status doesn't allow.
				// Note: no action needed for weightedQueue as these entries 
				//       are not part of weightedQueue.
				e = e._prev;
				continue;
			}

			double exectime = ((double) e._computeTime) / 1000000; // in milliseconds

			if (!e.isMatrixValue()) {
				// No spilling for scalar entries. Just delete those.
				// Note: scalar entries with higher computation time are pinned.
				if (cache.remove(e._key) != null)
					removeEntry(cache, e);
				e = isLRU ? e._prev : weightedQueue.poll();
				continue;
			}

			// Estimate time to write to FS + read from FS.
			double spilltime = getDiskSpillEstimate(e) * 1000; // in milliseconds

			if (LineageCache.DEBUG) {
				if (exectime > LineageCacheConfig.MIN_SPILL_TIME_ESTIMATE) {
					System.out.print("LI " + e._key.getOpcode());
					System.out.print(" exec time " + ((double) e._computeTime) / 1000000);
					System.out.print(" estimate time " + getDiskSpillEstimate(e) * 1000);
					System.out.print(" dim " + e.getMBValue().getNumRows() + " " + e.getMBValue().getNumColumns());
					System.out.println(" size " + getDiskSizeEstimate(e));
				}
			}

			if (spilltime < LineageCacheConfig.MIN_SPILL_TIME_ESTIMATE) {
				// Can't trust the estimate if less than 100ms.
				// Spill if it takes longer to recompute.
				if (exectime >= LineageCacheConfig.MIN_SPILL_TIME_ESTIMATE)
					spillToLocalFS(e);
			}
			else {
				// Spill if it takes longer to recompute than spilling.
				if (exectime > spilltime)
					spillToLocalFS(e);
			}

			// Remove the entry from cache.
			if (cache.remove(e._key) != null)
				removeEntry(cache, e);
			e = isLRU ? e._prev : weightedQueue.poll();
		}
	}

	//---------------- COSTING RELATED METHODS -----------------

	private static double getDiskSpillEstimate(LineageCacheEntry e) {
		if (!e.isMatrixValue() || e.isNullVal())
			return 0;
		// This includes sum of writing to and reading from disk
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		double size = getDiskSizeEstimate(e);
		double loadtime = isSparse(e) ? size/LineageCacheConfig.FSREAD_SPARSE : size/LineageCacheConfig.FSREAD_DENSE;
		double writetime = isSparse(e) ? size/LineageCacheConfig.FSWRITE_SPARSE : size/LineageCacheConfig.FSWRITE_DENSE;

		//double loadtime = CostEstimatorStaticRuntime.getFSReadTime(r, c, s);
		//double writetime = CostEstimatorStaticRuntime.getFSWriteTime(r, c, s);
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementCostingTime(System.nanoTime() - t0);
		return loadtime + writetime;
	}

	private static double getDiskSizeEstimate(LineageCacheEntry e) {
		if (!e.isMatrixValue() || e.isNullVal())
			return 0;
		MatrixBlock mb = e.getMBValue();
		long r = mb.getNumRows();
		long c = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		double s = OptimizerUtils.getSparsity(r, c, nnz);
		double disksize = ((double)MatrixBlock.estimateSizeOnDisk(r, c, (long)(s*r*c))) / (1024*1024);
		return disksize;
	}
	
	private static void adjustReadWriteSpeed(LineageCacheEntry e, double IOtime, boolean read) {
		double size = getDiskSizeEstimate(e);
		if (!e.isMatrixValue() || size < LineageCacheConfig.MIN_SPILL_DATA)
			// Scalar or too small
			return; 
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		double newIOSpeed = size / IOtime; // MB per second 
		// Adjust the read/write speed taking into account the last read/write.
		// These constants will eventually converge to the real speed.
		if (read) {
			if (isSparse(e))
				LineageCacheConfig.FSREAD_SPARSE = (LineageCacheConfig.FSREAD_SPARSE + newIOSpeed) / 2;
			else
				LineageCacheConfig.FSREAD_DENSE= (LineageCacheConfig.FSREAD_DENSE+ newIOSpeed) / 2;
		}
		else {
			if (isSparse(e))
				LineageCacheConfig.FSWRITE_SPARSE = (LineageCacheConfig.FSWRITE_SPARSE + newIOSpeed) / 2;
			else
				LineageCacheConfig.FSWRITE_DENSE= (LineageCacheConfig.FSWRITE_DENSE+ newIOSpeed) / 2;
		}
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementCostingTime(System.nanoTime() - t0);
	}
	
	private static boolean isSparse(LineageCacheEntry e) {
		if (!e.isMatrixValue() || e.isNullVal())
			return false;
		MatrixBlock mb = e.getMBValue();
		long r = mb.getNumRows();
		long c = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		double s = OptimizerUtils.getSparsity(r, c, nnz);
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(r, c, (long)(s*r*c));
		return sparse;
	}

	// ---------------- I/O METHODS TO LOCAL FS -----------------
	
	private static void spillToLocalFS(LineageCacheEntry entry) {
		if (!entry.isMatrixValue())
			throw new DMLRuntimeException ("Spilling scalar objects to disk is not allowd. Key: "+entry._key);
		if (entry.isNullVal())
			throw new DMLRuntimeException ("Cannot spill null value to disk. Key: "+entry._key);
		
		long t0 = System.nanoTime();
		if (_outdir == null) {
			_outdir = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_LINEAGE);
			LocalFileUtils.createLocalFileIfNotExist(_outdir);
		}
		String outfile = _outdir+"/"+entry._key.getId();
		try {
			LocalFileUtils.writeMatrixBlockToLocal(outfile, entry.getMBValue());
		} catch (IOException e) {
			throw new DMLRuntimeException ("Write to " + outfile + " failed.", e);
		}
		long t1 = System.nanoTime();
		// Adjust disk writing speed
		adjustReadWriteSpeed(entry, ((double)(t1-t0))/1000000000, false);
		
		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementFSWriteTime(t1-t0);
			LineageCacheStatistics.incrementFSWrites();
		}

		_spillList.put(entry._key, new SpilledItem(outfile, entry._computeTime));
	}

	protected static LineageCacheEntry readFromLocalFS(Map<LineageItem, LineageCacheEntry> cache, LineageItem key) {
		long t0 = System.nanoTime();
		MatrixBlock mb = null;
		// Read from local FS
		try {
			mb = LocalFileUtils.readMatrixBlockFromLocal(_spillList.get(key)._outfile);
		} catch (IOException e) {
			throw new DMLRuntimeException ("Read from " + _spillList.get(key)._outfile + " failed.", e);
		}
		// Restore to cache
		LocalFileUtils.deleteFileIfExists(_spillList.get(key)._outfile, true);
		long t1 = System.nanoTime();
		LineageCache.putIntern(key, DataType.MATRIX, mb, null, _spillList.get(key)._computeTime);
		// Adjust disk reading speed
		adjustReadWriteSpeed(cache.get(key), ((double)(t1-t0))/1000000000, true);
		// TODO: set cache status as RELOADED for this entry
		_spillList.remove(key);
		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementFSReadTime(t1-t0);
			LineageCacheStatistics.incrementFSHits();
		}
		return cache.get(key);
	}
	
	protected static boolean spillListContains(LineageItem key) {
		return _spillList.containsKey(key);
	}

	// ---------------- INTERNAL DATA STRUCTURES FOR EVICTION -----------------

	private static class SpilledItem {
		String _outfile;
		long _computeTime;

		public SpilledItem(String outfile, long computetime) {
			_outfile = outfile;
			_computeTime = computetime;
		}
	}
}
	
