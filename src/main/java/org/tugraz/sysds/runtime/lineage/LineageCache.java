/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.lineage;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.cost.CostEstimatorStaticRuntime;
import org.tugraz.sysds.lops.MMTSJ.MMTSJType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.CPInstructionParser;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.tugraz.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.LocalFileUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class LineageCache {
	private static final Map<LineageItem, Entry> _cache = new HashMap<>();
	private static final Map<LineageItem, String> _spillList = new HashMap<>();
	private static final HashSet<LineageItem> _removelist = new HashSet<>();
	private static final long CACHELIMIT= (long)512*1024*1024; // 500MB
	private static String outdir = null;
	private static long _cachesize = 0;
	private static Entry _head = null;
	private static Entry _end = null;

	//--------------------- CACHE LOGIC METHODS ----------------------

	public static void put(Instruction inst, ExecutionContext ec) {
		if (!DMLScript.LINEAGE_REUSE)
			return;
		if (inst instanceof ComputationCPInstruction && isReusable(inst) ) {
			LineageItem[] items = ((LineageTraceable) inst).getLineageItems(ec);
			for (LineageItem item : items) {
				MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction) inst).output);
				// Safe to pin the object in memory as coming from CPInstruction
				LineageCache.put(inst, item, mo.acquireReadAndRelease());
			}
		}
	}
	
	public static synchronized void put(Instruction inst, LineageItem key, MatrixBlock value) {
		if (_cache.containsKey(key)) {
			// Read from cache, replace value and place the entry at the head of the linkedlist.
			Entry oldItem = _cache.get(key);
			oldItem._value = value;
			delete(oldItem);
			setHead(oldItem);
		}
		else {
			// Create a new entry.
			Entry newItem = new Entry(key, value);
			if (isBelowThreshold(value)) 
				// Place the entry at head position.
				setHead(newItem);
			else {
				// Make space by removing or spilling LRU entries.
				makeSpace(inst, value);
				setHead(newItem);
			}
			_cache.put(key, newItem);
			if (DMLScript.STATISTICS)
				LineageCacheStatistics.incrementMemWrites();
			updateSize(value, true);
		}
	}
	
	public static boolean probe(LineageItem key) {
		boolean p = (_cache.containsKey(key) || _spillList.containsKey(key));
		if (!p && DMLScript.STATISTICS && _removelist.contains(key))
			// The sought entry was in cache but removed later 
			LineageCacheStatistics.incrementDelHits();
		return p;
	}
	
	public static void resetCache() {
		_cache.clear();
		_spillList.clear();
		if (DMLScript.STATISTICS)
			_removelist.clear();
	}
	
	public static boolean reuse(Instruction inst, ExecutionContext ec) {
		if (!DMLScript.LINEAGE_REUSE)
			return false;
		if (LineageCacheConfig.getCacheType().isFullReuse())
			return fullReuse(inst, ec);
		if (LineageCacheConfig.getCacheType().isPartialReuse())
			return RewriteCPlans.executeRewrites(inst, ec);
		return false;
	}

	private static boolean fullReuse (Instruction inst, ExecutionContext ec) {	
		if (inst instanceof ComputationCPInstruction && LineageCache.isReusable(inst)) {
			boolean reused = true;
			LineageItem[] items = ((ComputationCPInstruction) inst).getLineageItems(ec);
			for (LineageItem item : items) {
				if (LineageCache.probe(item)) {
					MatrixBlock d = LineageCache.get(inst, item);
					ec.setMatrixOutput(((ComputationCPInstruction) inst).output.getName(), d);
				} else
					reused = false;
			}
			return reused && items.length > 0;
		} else 
			return false;
	}

	public static synchronized MatrixBlock get(Instruction inst, LineageItem key) {
		// This method is called only when entry is present either in cache or in local FS.
		if (_cache.containsKey(key)) {
			// Read and put the entry at head.
			Entry e = _cache.get(key);
			delete(e);
			setHead(e);
			if (DMLScript.STATISTICS)
				LineageCacheStatistics.incrementMemHits();
			return e._value;
		}
		else
			return readFromLocalFS(inst, key);
	}
	
	public static boolean isReusable (Instruction inst) {
		// TODO: Move this to the new class LineageCacheConfig and extend
		return inst.getOpcode().equalsIgnoreCase("tsmm")
			|| (LineageCacheConfig.getCacheType().isFullReuse() 
				&& inst.getOpcode().equalsIgnoreCase("ba+*"));
		// TODO: Fix getRecomputeEstimate to support ba+* before enabling above code.
	}
	
	//---------------- CACHE SPACE MANAGEMENT METHODS -----------------
	
	public static boolean isBelowThreshold(MatrixBlock value) {
		if (value.getInMemorySize() > CACHELIMIT)
			throw new RuntimeException("Single item larger than the size of lineage cache");
		return ((value.getInMemorySize() + _cachesize) <= CACHELIMIT);
	}
	
	public static void makeSpace(Instruction inst, MatrixBlock value) {
		double valSize = value.getInMemorySize();
		// cost based eviction
		while ((valSize+_cachesize) > CACHELIMIT)
		{
			double reduction = _cache.get(_end._key)._value.getInMemorySize();
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			double spill = getDiskSpillEstimate();
			double comp = getRecomputeEstimate(inst);
			if (DMLScript.STATISTICS) {
				long t1 = System.nanoTime();
				LineageCacheStatistics.incrementCostingTime(t1-t0);
			}
			if (comp > spill) 
				spillToLocalFS(); // If re-computation is more expensive, spill data to disk.

			removeEntry(reduction);
		} 
	}
	
	private static void updateSize(MatrixBlock value, boolean addspace) {
		if (addspace)
			_cachesize += value.getInMemorySize();
		else
			_cachesize -= value.getInMemorySize();
	}

	//---------------- COSTING RELATED METHODS -----------------

	public static double getDiskSpillEstimate() {
		// This includes sum of writing to and reading from disk
		MatrixBlock mb = _cache.get(_end._key)._value;
		long r = mb.getNumRows();
		long c = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		double s = OptimizerUtils.getSparsity(r, c, nnz);
		double loadtime = CostEstimatorStaticRuntime.getFSReadTime(r, c, s);
		double writetime = CostEstimatorStaticRuntime.getFSWriteTime(r, c, s);
		return loadtime+writetime;
	}
	
	public static double getRecomputeEstimate(Instruction inst) {
		MatrixBlock mb = _cache.get(_end._key)._value;
		long r = mb.getNumRows();
		long c = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		double s = OptimizerUtils.getSparsity(r, c, nnz);
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(r, c, nnz);
		
		double nflops = 0;
		CPType cptype = CPInstructionParser.String2CPInstructionType.get(inst.getOpcode());
		//TODO: All other relevant instruction types.
		switch (cptype)
		{
			case MMTSJ:
			//case AggregateBinary:
				MMTSJType type = ((MMTSJCPInstruction)inst).getMMTSJType();
				if (type.isLeft())
					nflops = !sparse ? (r * c * s * c /2):(r * c * s * c * s /2);
				else
					nflops = !sparse ? ((double)r * c * r/2):(r*c*s + r*c*s*c*s /2);
				break;
				
			default:
				throw new DMLRuntimeException("Lineage Cache: unsupported instruction: "+inst.getOpcode());
		}
		return nflops / (2L * 1024 * 1024 * 1024);
	}

	// ---------------- I/O METHODS TO LOCAL FS -----------------
	
	private static void spillToLocalFS() {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if (outdir == null) {
			outdir = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_LINEAGE);
			LocalFileUtils.createLocalFileIfNotExist(outdir);
		}
		String outfile = outdir+"/"+_cache.get(_end._key)._key.getId();
		try {
			LocalFileUtils.writeMatrixBlockToLocal(outfile, _cache.get(_end._key)._value);
		} catch (IOException e) {
			throw new DMLRuntimeException ("Write to " + outfile + " failed.", e);
		}
		if (DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			LineageCacheStatistics.incrementFSWriteTime(t1-t0);
			LineageCacheStatistics.incrementFSWrites();
		}

		_spillList.put(_end._key, outfile);
	}
	
	private static MatrixBlock readFromLocalFS(Instruction inst, LineageItem key) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		MatrixBlock mb = null;
		// Read from local FS
		try {
			mb = LocalFileUtils.readMatrixBlockFromLocal(_spillList.get(key));
		} catch (IOException e) {
			throw new DMLRuntimeException ("Read from " + _spillList.get(key) + " failed.", e);
		}
		// Restore to cache
		LocalFileUtils.deleteFileIfExists(_spillList.get(key), true);
		_spillList.remove(key);
		put(inst, key, mb);
		if (DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			LineageCacheStatistics.incrementFSReadTime(t1-t0);
			LineageCacheStatistics.incrementFSHits();
		}
		return mb;
	}

	//------------------ LINKEDLIST MAINTENANCE METHODS -------------------
	
	public static void delete(Entry entry) {
		if (entry._prev != null)
			entry._prev._next = entry._next;
		else
			_head = entry._next;
		if (entry._next != null)
			entry._next._prev = entry._prev;
		else
			_end = entry._prev;
	}
	
	public static void setHead(Entry entry) {
		entry._next = _head;
		entry._prev = null;
		if (_head != null)
			_head._prev = entry;
		_head = entry;
		if (_end == null)
			_end = _head;
	}

	private static void removeEntry(double space) {
		if (DMLScript.STATISTICS)
			_removelist.add(_end._key);
		_cache.remove(_end._key);
		_cachesize -= space;
		delete(_end);
	}
	
	private static class Entry {
		LineageItem _key;
		MatrixBlock _value;
		Entry _prev;
		Entry _next;
		
		public Entry(LineageItem key, MatrixBlock value) {
			this._key = key;
			this._value = value;
		}
	}
}
