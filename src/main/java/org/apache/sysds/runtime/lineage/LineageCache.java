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

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
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
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCacheStatus;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LineageCache
{
	private static final Map<LineageItem, LineageCacheEntry> _cache = new HashMap<>();
	private static final double CACHE_FRAC = 0.05; // 5% of JVM heap size
	protected static final boolean DEBUG = false;

	static {
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		LineageCacheEviction.setCacheLimit((long)(CACHE_FRAC * maxMem));
		LineageCacheEviction.setStartTimestamp();
	}
	
	// Cache Synchronization Approach:
	//   The central static cache is only synchronized in a fine-grained manner
	//   for short get, put, or remove calls or during eviction. All blocking of
	//   threads for computing the values of placeholders is done on the individual
	//   entry objects which reduces contention and prevents deadlocks in case of
	//   function/statement block placeholders which computation itself might be
	//   a complex workflow of operations that accesses the cache as well.
	
	
	//--------------- PUBLIC CACHE API (keep it narrow) ----------------//
	
	public static boolean reuse(Instruction inst, ExecutionContext ec) {
		if (ReuseCacheType.isNone())
			return false;
		
		boolean reuse = false;
		//NOTE: the check for computation CP instructions ensures that the output
		// will always fit in memory and hence can be pinned unconditionally
		if (LineageCacheConfig.isReusable(inst, ec)) {
			ComputationCPInstruction cinst = (ComputationCPInstruction) inst;
			LineageItem instLI = cinst.getLineageItem(ec).getValue();
			List<MutablePair<LineageItem, LineageCacheEntry>> liList = null;
			if (inst instanceof MultiReturnBuiltinCPInstruction) {
				liList = new ArrayList<>();
				MultiReturnBuiltinCPInstruction mrInst = (MultiReturnBuiltinCPInstruction)inst;
				for (int i=0; i<mrInst.getNumOutputs(); i++) {
					String opcode = instLI.getOpcode() + String.valueOf(i);
					liList.add(MutablePair.of(new LineageItem(opcode, instLI.getInputs()), null));
				}
			}
			else
				liList = Arrays.asList(MutablePair.of(instLI, null));
			
			//atomic try reuse full/partial and set placeholder, without
			//obtaining value to avoid blocking in critical section
			LineageCacheEntry e = null;
			boolean reuseAll = true;
			synchronized( _cache ) {
				//try to reuse full or partial intermediates
				for (MutablePair<LineageItem,LineageCacheEntry> item : liList) {
					if (LineageCacheConfig.getCacheType().isFullReuse())
						e = LineageCache.probe(item.getKey()) ? getIntern(item.getKey()) : null;
					//TODO need to also move execution of compensation plan out of here
					//(create lazily evaluated entry)
					if (e == null && LineageCacheConfig.getCacheType().isPartialReuse())
						if( LineageRewriteReuse.executeRewrites(inst, ec) )
							e = getIntern(item.getKey());
					//TODO: MultiReturnBuiltin and partial rewrites
					reuseAll &= (e != null);
					item.setValue(e);
					
					//create a placeholder if no reuse to avoid redundancy
					//(e.g., concurrent threads that try to start the computation)
					if(e == null && isMarkedForCaching(inst, ec)) {
						putIntern(item.getKey(), cinst.output.getDataType(), null, null,  0);
						//FIXME: different o/p datatypes for MultiReturnBuiltins.
					}
				}
			}
			reuse = reuseAll;
			
			if(reuse) { //reuse
				//put reuse value into symbol table (w/ blocking on placeholders)
				for (MutablePair<LineageItem, LineageCacheEntry> entry : liList) {
					e = entry.getValue();
					String outName = null;
					if (inst instanceof MultiReturnBuiltinCPInstruction)
						outName = ((MultiReturnBuiltinCPInstruction)inst).
							getOutput(entry.getKey().getOpcode().charAt(entry.getKey().getOpcode().length()-1)-'0').getName(); 
					else
						outName = cinst.output.getName();

					if (e.isMatrixValue())
						ec.setMatrixOutput(outName, e.getMBValue());
					else
						ec.setScalarOutput(outName, e.getSOValue());
					reuse = true;
				}
				if (DMLScript.STATISTICS)
					LineageCacheStatistics.incrementInstHits();
			}
		}
		
		return reuse;
	}
	
	public static boolean reuse(List<String> outNames, List<DataIdentifier> outParams, 
			int numOutputs, LineageItem[] liInputs, String name, ExecutionContext ec)
	{
		if( !LineageCacheConfig.isMultiLevelReuse())
			return false;
		
		boolean reuse = (outParams.size() != 0);
		HashMap<String, Data> funcOutputs = new HashMap<>();
		HashMap<String, LineageItem> funcLIs = new HashMap<>();
		for (int i=0; i<numOutputs; i++) {
			String opcode = name + String.valueOf(i+1);
			LineageItem li = new LineageItem(opcode, liInputs);
			LineageCacheEntry e = null;
			synchronized(_cache) {
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
			
			if (e != null) {
				String boundVarName = outNames.get(i);
				Data boundValue = null;
				//convert to matrix object
				if (e.isMatrixValue()) {
					MetaDataFormat md = new MetaDataFormat(
						e.getMBValue().getDataCharacteristics(),FileFormat.BINARY);
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
		boolean p = _cache.containsKey(key);  // in cache or in disk
		if (!p && DMLScript.STATISTICS && LineageCacheEviction._removelist.contains(key))
			// The sought entry was in cache but removed later 
			LineageCacheStatistics.incrementDelHits();
		return p;
	}
	
	public static MatrixBlock getMatrix(LineageItem key) {
		LineageCacheEntry e = null;
		synchronized( _cache ) {
			e = getIntern(key);
		}
		return e.getMBValue();
	}
	
	//NOTE: safe to pin the object in memory as coming from CPInstruction
	//TODO why do we need both of these public put methods
	public static void putMatrix(Instruction inst, ExecutionContext ec, long computetime) {
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			LineageItem item = ((LineageTraceable) inst).getLineageItem(ec).getValue();
			//This method is called only to put matrix value
			MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction) inst).output);
			synchronized( _cache ) {
				putIntern(item, DataType.MATRIX, mo.acquireReadAndRelease(), null, computetime);
			}
		}
	}
	
	public static void putValue(Instruction inst, ExecutionContext ec, long starttime) {
		if (ReuseCacheType.isNone())
			return;
		long computetime = System.nanoTime() - starttime;
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			//if (!isMarkedForCaching(inst, ec)) return;
			List<Pair<LineageItem, Data>> liData = null;
			LineageItem instLI = ((LineageTraceable) inst).getLineageItem(ec).getValue();
			if (inst instanceof MultiReturnBuiltinCPInstruction) {
				liData = new ArrayList<>();
				MultiReturnBuiltinCPInstruction mrInst = (MultiReturnBuiltinCPInstruction)inst;
				for (int i=0; i<mrInst.getNumOutputs(); i++) {
					String opcode = instLI.getOpcode() + String.valueOf(i);
					LineageItem li = new LineageItem(opcode, instLI.getInputs());
					Data value = ec.getVariable(mrInst.getOutput(i));
					liData.add(Pair.of(li, value));
				}
			}
			else
				liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationCPInstruction) inst).output)));
			synchronized( _cache ) {
				for (Pair<LineageItem, Data> entry : liData) {
					LineageItem item = entry.getKey();
					Data data = entry.getValue();
					LineageCacheEntry centry = _cache.get(item);
					if (data instanceof MatrixObject)
						centry.setValue(((MatrixObject)data).acquireReadAndRelease(), computetime);
					else if (data instanceof ScalarObject)
						centry.setValue((ScalarObject)data, computetime);
					else {
						// Reusable instructions can return a frame (rightIndex). Remove placeholders.
						_cache.remove(item);
						continue;
					}

					long size = centry.getSize();
					//remove the entry if the entry is bigger than the cache.
					//FIXME: the resumed threads will enter into infinite wait as the entry
					//is removed. Need to add support for graceful remove (placeholder) and resume.
					if (size > LineageCacheEviction.getCacheLimit()) {
						_cache.remove(item);
						continue; 
					}

					//maintain order for eviction
					LineageCacheEviction.addEntry(centry);

					if (!LineageCacheEviction.isBelowThreshold(size))
						LineageCacheEviction.makeSpace(_cache, size);
					LineageCacheEviction.updateSize(size, true);
				}
			}
		}
	}
	
	public static void putValue(List<DataIdentifier> outputs,
		LineageItem[] liInputs, String name, ExecutionContext ec, long computetime)
	{
		if (!LineageCacheConfig.isMultiLevelReuse())
			return;

		HashMap<LineageItem, LineageItem> FuncLIMap = new HashMap<>();
		boolean AllOutputsCacheable = true;
		for (int i=0; i<outputs.size(); i++) {
			String opcode = name + String.valueOf(i+1);
			LineageItem li = new LineageItem(opcode, liInputs);
			String boundVarName = outputs.get(i).getName();
			LineageItem boundLI = ec.getLineage().get(boundVarName);
			if (boundLI != null)
				boundLI.resetVisitStatusNR();
			if (boundLI == null || !LineageCache.probe(li) || !LineageCache.probe(boundLI)) {
				AllOutputsCacheable = false;
				//FIXME: if boundLI is for a MultiReturnBuiltin instruction 
			}
			FuncLIMap.put(li, boundLI);
		}

		//cache either all the outputs, or none.
		synchronized (_cache) {
			//move or remove placeholders 
			if(AllOutputsCacheable)
				FuncLIMap.forEach((Li, boundLI) -> mvIntern(Li, boundLI, computetime));
			else
				FuncLIMap.forEach((Li, boundLI) -> _cache.remove(Li));
		}
		
		return;
	}
	
	public static void resetCache() {
		synchronized (_cache) {
			_cache.clear();
			LineageCacheEviction.resetEviction();
		}
	}
	
	//----------------- INTERNAL CACHE LOGIC IMPLEMENTATION --------------//
	
	private static void putIntern(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, long computetime) {
		if (_cache.containsKey(key))
			//can come here if reuse_partial option is enabled
			return;
		
		// Create a new entry.
		LineageCacheEntry newItem = new LineageCacheEntry(key, dt, Mval, Sval, computetime);
		
		// Make space by removing or spilling LRU entries.
		if( Mval != null || Sval != null ) {
			long size = newItem.getSize();
			if( size > LineageCacheEviction.getCacheLimit())
				return; //not applicable
			if( !LineageCacheEviction.isBelowThreshold(size) ) 
				LineageCacheEviction.makeSpace(_cache, size);
			LineageCacheEviction.updateSize(size, true);
		}
		
		// Place the entry in the weighted queue.
		LineageCacheEviction.addEntry(newItem);
		
		_cache.put(key, newItem);
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementMemWrites();
	}
	
	private static LineageCacheEntry getIntern(LineageItem key) {
		// This method is called only when entry is present either in cache or in local FS.
		LineageCacheEntry e = _cache.get(key);
		if (e != null && e.getCacheStatus() != LineageCacheStatus.SPILLED) {
			// Maintain order for eviction
			LineageCacheEviction.getEntry(e);
			if (DMLScript.STATISTICS)
				LineageCacheStatistics.incrementMemHits();
			return e;
		}
		else
			return LineageCacheEviction.readFromLocalFS(_cache, key);
	}
	
	private static void mvIntern(LineageItem item, LineageItem probeItem, long computetime) {
		if (ReuseCacheType.isNone())
			return;
		// Move the value from the cache entry with key probeItem to
		// the placeholder entry with key item.
		if (LineageCache.probe(probeItem)) {
			LineageCacheEntry oe = getIntern(probeItem);
			LineageCacheEntry e = _cache.get(item);
			boolean exists = !e.isNullVal();
			if (oe.isMatrixValue())
				e.setValue(oe.getMBValue(), computetime); 
			else
				e.setValue(oe.getSOValue(), computetime);
			e._origItem = probeItem; 
			// Add itself as original item to navigate the list.
			oe._origItem = probeItem;

			// Add the SB/func entry to the list of items pointing to the same data.
			// No cache size update is necessary.
			// Maintain _origItem as head.
			if (!exists) {
				e._nextEntry = oe._nextEntry;
				oe._nextEntry = e;
			}
			
			//maintain order for eviction
			LineageCacheEviction.addEntry(e);
		}
		else
			_cache.remove(item);    //remove the placeholder
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
	
	@Deprecated
	@SuppressWarnings("unused")
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
		
		return nflops / (2L * 1024 * 1024 * 1024);
	}
}
