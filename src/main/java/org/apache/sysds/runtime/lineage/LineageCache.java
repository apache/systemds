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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.cost.CostEstimatorStaticRuntime;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.CPInstructionParser;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.InputInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.OutputInfo;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class LineageCache
{
	private static final Map<LineageItem, Entry> _cache = new HashMap<>();
	private static final Map<LineageItem, SpilledItem> _spillList = new HashMap<>();
	private static final HashSet<LineageItem> _removelist = new HashSet<>();
	private static final double CACHE_FRAC = 0.05; // 5% of JVM heap size
	private static final long CACHE_LIMIT; //limit in bytes
	private static String outdir = null;
	private static long _cachesize = 0;
	private static Entry _head = null;
	private static Entry _end = null;

	static {
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		CACHE_LIMIT = (long)(CACHE_FRAC * maxMem);
	}
	
	// Cache Synchronization Approach:
	//   The central static cache is only synchronized in a fine-grained manner
	//   for short get, put, or remove calls or during eviction. All blocking of
	//   threads for computing the values of placeholders is done on the individual
	//   entry objects which reduces contention and prevents deadlocks in case of
	//   function/statement block placeholders which computation itself might be
	//   a complex workflow of operations that accesses the cache as well.
	
	
	///////////////////////////////////////
	// Public Cache API (keep it narrow) //
	///////////////////////////////////////
	
	public static boolean reuse(Instruction inst, ExecutionContext ec) {
		if (ReuseCacheType.isNone())
			return false;
		
		boolean reuse = false;
		//NOTE: the check for computation CP instructions ensures that the output
		// will always fit in memory and hence can be pinned unconditionally
		if (LineageCacheConfig.isReusable(inst, ec)) {
			ComputationCPInstruction cinst = (ComputationCPInstruction) inst;
			LineageItem item = cinst.getLineageItems(ec)[0];
			
			//atomic try reuse full/partial and set placeholder, without
			//obtaining value to avoid blocking in critical section
			Entry e = null;
			synchronized( _cache ) {
				//try to reuse full or partial intermediates
				if (LineageCacheConfig.getCacheType().isFullReuse())
					e = LineageCache.probe(item) ? getIntern(item) : null;
				//TODO need to also move execution of compensation plan out of here
				//(create lazily evaluated entry)
				if (e == null && LineageCacheConfig.getCacheType().isPartialReuse())
					if( LineageRewriteReuse.executeRewrites(inst, ec) )
						e = getIntern(item);
				reuse = (e != null);
				
				//create a placeholder if no reuse to avoid redundancy
				//(e.g., concurrent threads that try to start the computation)
				if(!reuse && isMarkedForCaching(inst, ec)) {
					putIntern(item, cinst.output.getDataType(), null, null,  0);
				}
			}
			
			if( reuse ) { //reuse
				//put reuse value into symbol table (w/ blocking on placeholders)
				if (e.isMatrixValue())
					ec.setMatrixOutput(cinst.output.getName(), e.getMBValue());
				else
					ec.setScalarOutput(cinst.output.getName(), e.getSOValue());
				if (DMLScript.STATISTICS)
					LineageCacheStatistics.incrementInstHits();
				reuse = true;
			}
		}
		
		return reuse;
	}
	
	public static boolean reuse(List<String> outNames, List<DataIdentifier> outParams, int numOutputs, LineageItem[] liInputs, String name, ExecutionContext ec)
	{
		if( !LineageCacheConfig.isMultiLevelReuse())
			return false;
		
		boolean reuse = (outParams.size() != 0);
		HashMap<String, Data> funcOutputs = new HashMap<>();
		HashMap<String, LineageItem> funcLIs = new HashMap<>();
		for (int i=0; i<numOutputs; i++) {
			String opcode = name + String.valueOf(i+1);
			LineageItem li = new LineageItem(outNames.get(i), opcode, liInputs);
			Entry e = null;
			synchronized( _cache ) {
				if (LineageCache.probe(li)) {
					e = LineageCache.getIntern(li);
				}
				else {
					//create a placeholder if no reuse to avoid redundancy
					//(e.g., concurrent threads that try to start the computation)
					putIntern(li, outParams.get(i).getDataType(), null, null, 0);
				}
			}
			//TODO: handling of recursive calls
			
			if ( e != null ) {
				String boundVarName = outNames.get(i);
				Data boundValue = null;
				//convert to matrix object
				if (e.isMatrixValue()) {
					MetaDataFormat md = new MetaDataFormat(e.getMBValue().getDataCharacteristics(),
						OutputInfo.BinaryCellOutputInfo, InputInfo.BinaryCellInputInfo);
					boundValue = new MatrixObject(ValueType.FP64, boundVarName, md);
					((MatrixObject)boundValue).acquireModify(e.getMBValue());
					((MatrixObject)boundValue).release();
				}
				else {
					boundValue = e.getSOValue();
				}

				funcOutputs.put(boundVarName, boundValue);
				LineageItem orig = e._origItem;
				funcLIs.put(boundVarName, orig);
			}
			else {
				// if one output cannot be reused, we need to execute the function
				// NOTE: all outputs need to be prepared for caching and hence,
				// we cannot directly return here
				reuse = false;
			}
		}
		
		if (reuse) {
			funcOutputs.forEach((var, val) -> {
				//cleanup existing data bound to output variable name
				Data exdata = ec.removeVariable(var);
				if( exdata != val)
					ec.cleanupDataObject(exdata);
				//add/replace data in symbol table
				ec.setVariable(var, val);
			});
			//map original lineage items return to the calling site
			funcLIs.forEach((var, li) -> ec.getLineage().set(var, li));
		}
		
		return reuse;
	}
	
	public static boolean probe(LineageItem key) {
		//TODO problematic as after probe the matrix might be kicked out of cache
		boolean p = (_cache.containsKey(key) || _spillList.containsKey(key));
		if (!p && DMLScript.STATISTICS && _removelist.contains(key))
			// The sought entry was in cache but removed later 
			LineageCacheStatistics.incrementDelHits();
		return p;
	}
	
	public static MatrixBlock getMatrix(LineageItem key) {
		Entry e = null;
		synchronized( _cache ) {
			e = getIntern(key);
		}
		return e.getMBValue();
	}
	
	//NOTE: safe to pin the object in memory as coming from CPInstruction
	//TODO why do we need both of these public put methods
	public static void putMatrix(Instruction inst, ExecutionContext ec) {
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			LineageItem item = ((LineageTraceable) inst).getLineageItems(ec)[0];
			//This method is called only to put matrix value
			MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction) inst).output);
			synchronized( _cache ) {
				putIntern(item, DataType.MATRIX, mo.acquireReadAndRelease(),
					null, getRecomputeEstimate(inst, ec));
			}
		}
	}
	
	public static void putValue(Instruction inst, ExecutionContext ec) {
		if (ReuseCacheType.isNone())
			return;
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			//if (!isMarkedForCaching(inst, ec)) return;
			LineageItem item = ((LineageTraceable) inst).getLineageItems(ec)[0];
			Data data = ec.getVariable(((ComputationCPInstruction) inst).output);
			double cest = getRecomputeEstimate(inst, ec);
			synchronized( _cache ) {
				if( data instanceof MatrixObject )
					_cache.get(item).setValue(((MatrixObject)data).acquireReadAndRelease(), cest);
				else
					_cache.get(item).setValue((ScalarObject)data, cest);
				long size = _cache.get(item).getSize();
				
				if( !isBelowThreshold(size) ) 
					makeSpace(size);
				updateSize(size, true);
			}
		}
	}
	
	public static void putValue(List<DataIdentifier> outputs, LineageItem[] liInputs, String name, ExecutionContext ec)
	{
		if( !LineageCacheConfig.isMultiLevelReuse() )
			return;

		HashMap<LineageItem, LineageItem> FuncLIMap = new HashMap<>();
		boolean AllOutputsCacheable = true;
		for (int i=0; i<outputs.size(); i++) {
			String opcode = name + String.valueOf(i+1);
			LineageItem li = new LineageItem(outputs.get(i).getName(), opcode, liInputs);
			String boundVarName = outputs.get(i).getName();
			LineageItem boundLI = ec.getLineage().get(boundVarName);
			if (boundLI != null)
				boundLI.resetVisitStatus();
			if (boundLI == null 
				|| !LineageCache.probe(li)
				//TODO remove this brittle constraint (if the placeholder is removed
				//it might crash threads that are already waiting for its results)
				|| LineageItemUtils.containsRandDataGen(new HashSet<>(Arrays.asList(liInputs)), boundLI)) {
				AllOutputsCacheable = false;
			}
			FuncLIMap.put(li, boundLI);
		}

		//cache either all the outputs, or none.
		synchronized( _cache ) {
			//move or remove placeholders 
			if(AllOutputsCacheable)
				FuncLIMap.forEach((Li, boundLI) -> mvIntern(Li, boundLI));
			else
				FuncLIMap.forEach((Li, boundLI) -> removeEntry(Li));
		}
		
		return;
	}
	
	public static void resetCache() {
		synchronized( _cache ) {
			_cache.clear();
			_spillList.clear();
			_head = null;
			_end = null;
			// reset cache size, otherwise the cache clear leads to unusable 
			// space which means evictions could run into endless loops
			_cachesize = 0;
			if (DMLScript.STATISTICS)
				_removelist.clear();
		}
	}
	
	/////////////////////////////////////////
	// Internal Cache Logic Implementation //
	/////////////////////////////////////////
	
	private static void putIntern(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, double compcost) {
		if (_cache.containsKey(key))
			//can come here if reuse_partial option is enabled
			return;
		
		// Create a new entry.
		Entry newItem = new Entry(key, dt, Mval, Sval, compcost);
		
		// Make space by removing or spilling LRU entries.
		if( Mval != null || Sval != null ) {
			long size = newItem.getSize();
			if( size > CACHE_LIMIT )
				return; //not applicable
			if( !isBelowThreshold(size) ) 
				makeSpace(size);
			updateSize(size, true);
		}
		
		// Place the entry at head position.
		setHead(newItem);
		
		_cache.put(key, newItem);
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementMemWrites();
	}
	
	private static Entry getIntern(LineageItem key) {
		// This method is called only when entry is present either in cache or in local FS.
		if (_cache.containsKey(key)) {
			// Read and put the entry at head.
			Entry e = _cache.get(key);
			delete(e);
			setHead(e);
			if (DMLScript.STATISTICS)
				LineageCacheStatistics.incrementMemHits();
			return e;
		}
		else
			return readFromLocalFS(key);
	}

	
	private static void mvIntern(LineageItem item, LineageItem probeItem) {
		if (ReuseCacheType.isNone())
			return;
		if (LineageCache.probe(probeItem)) {
			Entry oe = getIntern(probeItem);
			Entry e = _cache.get(item);
			//TODO: compute estimate for function
			if (oe.isMatrixValue())
				e.setValue(oe.getMBValue(), 0); 
			else
				e.setValue(oe.getSOValue(), 0);
			e._origItem = probeItem; 

			long size = oe.getSize();
			if(!isBelowThreshold(size)) 
				makeSpace(size);
			updateSize(size, true);
		}
		else
			removeEntry(item);  //remove the placeholder
	}
	
	private static boolean isMarkedForCaching (Instruction inst, ExecutionContext ec) {
		if (!LineageCacheConfig.getCompAssRW())
			return true;

		if (((ComputationCPInstruction)inst).output.isMatrix()) {
			MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)inst).output);
			//limit this to full reuse as partial reuse is applicable even for loop dependent operation
			return !(LineageCacheConfig.getCacheType() == ReuseCacheType.REUSE_FULL  
				&& !mo.isMarked());
		}
		else
			return true;
	}
	
	//---------------- CACHE SPACE MANAGEMENT METHODS -----------------
	
	private static boolean isBelowThreshold(long spaceNeeded) {
		return ((spaceNeeded + _cachesize) <= CACHE_LIMIT);
	}
	
	private static void makeSpace(long spaceNeeded) {
		// cost based eviction
		while ((spaceNeeded +_cachesize) > CACHE_LIMIT)
		{
			if (_cache.get(_end._key).isNullVal()) {
				//Must be a null function/SB placeholder entry. This 
				//function is currently being executed. Skip and continue.
				setEnd2Head(_end);
				continue;
			}
			
			if (_cache.get(_end._key).isMatrixValue()) { //spill matrix blocks only
				if (_cache.get(_end._key)._compEst > getDiskSpillEstimate() 
						&& LineageCacheConfig.isSetSpill())
					spillToLocalFS(); // If re-computation is more expensive, spill data to disk.
			}

			if (_cache.get(_end._key)._compEst == 0) {
				//Must be a function/SB/scalar entry. Move to next.
				//FIXME: Remove this logic after implementing new eviction logic.
				setEnd2Head(_end);  
				continue;
			}
			removeLastEntry();
		}
	}
	
	private static void updateSize(long space, boolean addspace) {
		if (addspace)
			_cachesize += space;
		else
			_cachesize -= space;
	}

	//---------------- COSTING RELATED METHODS -----------------

	private static double getDiskSpillEstimate() {
		// This includes sum of writing to and reading from disk
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		MatrixBlock mb = _cache.get(_end._key).getMBValue();
		long r = mb.getNumRows();
		long c = mb.getNumColumns();
		long nnz = mb.getNonZeros();
		double s = OptimizerUtils.getSparsity(r, c, nnz);
		double loadtime = CostEstimatorStaticRuntime.getFSReadTime(r, c, s);
		double writetime = CostEstimatorStaticRuntime.getFSWriteTime(r, c, s);
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementCostingTime(System.nanoTime() - t0);
		return loadtime+writetime;
	}
	
	private static double getRecomputeEstimate(Instruction inst, ExecutionContext ec) {
		if (!((ComputationCPInstruction)inst).output.isMatrix()
			|| (((ComputationCPInstruction)inst).input1 != null && !((ComputationCPInstruction)inst).input1.isMatrix()))
			return 0; //this method will be deprecated. No need to support scalar

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		double nflops = 0;
		String instop= inst.getOpcode().contains("spoof") ? "spoof" : inst.getOpcode();
		CPType cptype = CPInstructionParser.String2CPInstructionType.get(instop);
		//TODO: All other relevant instruction types.
		switch (cptype)
		{
			case MMTSJ:  //tsmm
			{
				MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				long r = mo.getNumRows();
				long c = mo.getNumColumns();
				long nnz = mo.getNnz();
				double s = OptimizerUtils.getSparsity(r, c, nnz);
				boolean sparse = MatrixBlock.evalSparseFormatInMemory(r, c, nnz);
				MMTSJType type = ((MMTSJCPInstruction)inst).getMMTSJType();
				if (type.isLeft())
					nflops = !sparse ? (r * c * s * c /2):(r * c * s * c * s /2);
				else
					nflops = !sparse ? ((double)r * c * r/2):(r*c*s + r*c*s*c*s /2);
				break;
			}
				
			case AggregateBinary:  //ba+*
			{
				MatrixObject mo1 = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				MatrixObject mo2 = ec.getMatrixObject(((ComputationCPInstruction)inst).input2);
				long r1 = mo1.getNumRows();
				long c1 = mo1.getNumColumns();
				long nnz1 = mo1.getNnz();
				double s1 = OptimizerUtils.getSparsity(r1, c1, nnz1);
				boolean lsparse = MatrixBlock.evalSparseFormatInMemory(r1, c1, nnz1);
				long r2 = mo2.getNumRows();
				long c2 = mo2.getNumColumns();
				long nnz2 = mo2.getNnz();
				double s2 = OptimizerUtils.getSparsity(r2, c2, nnz2);
				boolean rsparse = MatrixBlock.evalSparseFormatInMemory(r2, c2, nnz2);
				if( !lsparse && !rsparse )
					nflops = 2 * (r1 * c1 * ((c2>1)?s1:1.0) * c2) /2;
				else if( !lsparse && rsparse )
					nflops = 2 * (r1 * c1 * s1 * c2 * s2) /2;
				else if( lsparse && !rsparse )
					nflops = 2 * (r1 * c1 * s1 * c2) /2;
				else //lsparse && rsparse
					nflops = 2 * (r1 * c1 * s1 * c2 * s2) /2;
				break;
			}
				
			case Binary:  //*, /
			{
				MatrixObject mo1 = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				long r1 = mo1.getNumRows();
				long c1 = mo1.getNumColumns();
				if (inst.getOpcode().equalsIgnoreCase("*") || inst.getOpcode().equalsIgnoreCase("/"))
					// considering the dimensions of inputs and the output are same 
					nflops = r1 * c1; 
				else if (inst.getOpcode().equalsIgnoreCase("solve"))
					nflops = r1 * c1 * c1;
				break;
			}
			
			case MatrixIndexing:  //rightIndex
			{
				MatrixObject mo1 = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				long r1 = mo1.getNumRows();
				long c1 = mo1.getNumColumns();
				long nnz1 = mo1.getNnz();
				double s1 = OptimizerUtils.getSparsity(r1, c1, nnz1);
				boolean lsparse = MatrixBlock.evalSparseFormatInMemory(r1, c1, nnz1);
				//if (inst.getOpcode().equalsIgnoreCase("rightIndex"))
					nflops = 1.0 * (lsparse ? r1 * c1 * s1 : r1 * c1); //FIXME
				break;
			}
			
			case ParameterizedBuiltin:  //groupedagg (sum, count)
			{
				String opcode = ((ParameterizedBuiltinCPInstruction)inst).getOpcode();
				HashMap<String, String> params = ((ParameterizedBuiltinCPInstruction)inst).getParameterMap();
				long r1 = ec.getMatrixObject(params.get(Statement.GAGG_TARGET)).getNumRows();
				String fn = params.get(Statement.GAGG_FN);
				double xga = 0;
				if (opcode.equalsIgnoreCase("groupedagg")) {
					if (fn.equalsIgnoreCase("sum"))
						xga = 4;
					else if(fn.equalsIgnoreCase("count"))
						xga = 1;
					//TODO: cm, variance
				}
				//TODO: support other PBuiltin ops
				nflops = 2 * r1+xga * r1;
				break;
			}

			case Reorg:  //r'
			{
				MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				long r = mo.getNumRows();
				long c = mo.getNumColumns();
				long nnz = mo.getNnz();
				double s = OptimizerUtils.getSparsity(r, c, nnz);
				boolean sparse = MatrixBlock.evalSparseFormatInMemory(r, c, nnz);
				nflops = sparse ? r*c*s : r*c; 
				break;
			}

			case Append:  //cbind, rbind
			{
				MatrixObject mo1 = ec.getMatrixObject(((ComputationCPInstruction)inst).input1);
				MatrixObject mo2 = ec.getMatrixObject(((ComputationCPInstruction)inst).input2);
				long r1 = mo1.getNumRows();
				long c1 = mo1.getNumColumns();
				long nnz1 = mo1.getNnz();
				double s1 = OptimizerUtils.getSparsity(r1, c1, nnz1);
				boolean lsparse = MatrixBlock.evalSparseFormatInMemory(r1, c1, nnz1);
				long r2 = mo2.getNumRows();
				long c2 = mo2.getNumColumns();
				long nnz2 = mo2.getNnz();
				double s2 = OptimizerUtils.getSparsity(r2, c2, nnz2);
				boolean rsparse = MatrixBlock.evalSparseFormatInMemory(r2, c2, nnz2);
				nflops = 1.0 * ((lsparse ? r1*c1*s1 : r1*c1) + (rsparse ? r2*c2*s2 : r2*c2));
				break;
			}
			
			case SpoofFused:  //spoof
			{
				nflops = 0; //FIXME: this method will be deprecated
				break;
			}

			default:
				throw new DMLRuntimeException("Lineage Cache: unsupported instruction: "+inst.getOpcode());
		}
		
		if (DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			LineageCacheStatistics.incrementCostingTime(t1-t0);
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
			LocalFileUtils.writeMatrixBlockToLocal(outfile, _cache.get(_end._key).getMBValue());
		} catch (IOException e) {
			throw new DMLRuntimeException ("Write to " + outfile + " failed.", e);
		}
		if (DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			LineageCacheStatistics.incrementFSWriteTime(t1-t0);
			LineageCacheStatistics.incrementFSWrites();
		}

		_spillList.put(_end._key, new SpilledItem(outfile, _end._compEst));
	}
	
	private static Entry readFromLocalFS(LineageItem key) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		MatrixBlock mb = null;
		// Read from local FS
		try {
			mb = LocalFileUtils.readMatrixBlockFromLocal(_spillList.get(key)._outfile);
		} catch (IOException e) {
			throw new DMLRuntimeException ("Read from " + _spillList.get(key)._outfile + " failed.", e);
		}
		// Restore to cache
		LocalFileUtils.deleteFileIfExists(_spillList.get(key)._outfile, true);
		putIntern(key, DataType.MATRIX, mb, null, _spillList.get(key)._compEst);
		_spillList.remove(key);
		if (DMLScript.STATISTICS) {
			long t1 = System.nanoTime();
			LineageCacheStatistics.incrementFSReadTime(t1-t0);
			LineageCacheStatistics.incrementFSHits();
		}
		return _cache.get(key);
	}

	////////////////////////////////////////////
	// Cache Maintenance and Lookup Functions //
	////////////////////////////////////////////
	
	private static void removeLastEntry() {
		if (DMLScript.STATISTICS)
			_removelist.add(_end._key);
		Entry e = _cache.remove(_end._key);
		_cachesize -= e.getSize();
		delete(_end);
	}
	
	private static void removeEntry(LineageItem key) {
		// Remove the entry for key
		if (!_cache.containsKey(key))
			return;
		delete(_cache.get(key));
		_cache.remove(key);
	}
	
	private static void setEnd2Head(Entry entry) {
		delete(entry);
		setHead(entry);
	}
	
	private static void delete(Entry entry) {
		if (entry._prev != null)
			entry._prev._next = entry._next;
		else
			_head = entry._next;
		if (entry._next != null)
			entry._next._prev = entry._prev;
		else
			_end = entry._prev;
	}
	
	private static void setHead(Entry entry) {
		entry._next = _head;
		entry._prev = null;
		if (_head != null)
			_head._prev = entry;
		_head = entry;
		if (_end == null)
			_end = _head;
	}
	
	////////////////////////////////////
	// Internal Cache Data Structures //
	////////////////////////////////////
	
	private static class Entry {
		private final LineageItem _key;
		private final DataType _dt;
		private MatrixBlock _MBval;
		private ScalarObject _SOval;
		double _compEst;
		private Entry _prev;
		private Entry _next;
		private LineageItem _origItem;
		
		public Entry(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, double computecost) {
			_key = key;
			_dt = dt;
			_MBval = Mval;
			_SOval = Sval;
			_compEst = computecost;
			_origItem = null;
		}

		public synchronized MatrixBlock getMBValue() {
			try {
				//wait until other thread completes operation
				//in order to avoid redundant computation
				while( _MBval == null ) {
					wait();
				}
				return _MBval;
			}
			catch( InterruptedException ex ) {
				throw new DMLRuntimeException(ex);
			}
		}

		public synchronized ScalarObject getSOValue() {
			try {
				//wait until other thread completes operation
				//in order to avoid redundant computation
				while( _SOval == null ) {
					wait();
				}
				return _SOval;
			}
			catch( InterruptedException ex ) {
				throw new DMLRuntimeException(ex);
			}
		}
		
		public synchronized long getSize() {
			return ((_MBval != null ? _MBval.getInMemorySize() : 0) + (_SOval != null ? _SOval.getSize() : 0));
		}
		
		public boolean isNullVal() {
			return(_MBval == null && _SOval == null);
		}
		
		public boolean isMatrixValue() {
			return _dt.isMatrix();
		}
		
		public synchronized void setValue(MatrixBlock val, double compEst) {
			_MBval = val;
			_compEst = compEst;
			//resume all threads waiting for val
			notifyAll();
		}

		public synchronized void setValue(ScalarObject val, double compEst) {
			_SOval = val;
			_compEst = compEst;
			//resume all threads waiting for val
			notifyAll();
		}
	}
	
	private static class SpilledItem {
		String _outfile;
		double _compEst;

		public SpilledItem(String outfile, double computecost) {
			_outfile = outfile;
			_compEst = computecost;
		}
	}
}
