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
import org.apache.sysds.runtime.controlprogram.context.MatrixObjectFuture;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.instructions.CPInstructionParser;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.BroadcastCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.PrefetchCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.fed.ComputationFEDInstruction;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.instructions.spark.ComputationSPInstruction;
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
	protected static final boolean DEBUG = false;

	static {
		LineageCacheEviction.setCacheLimit(LineageCacheConfig.CPU_CACHE_FRAC); //5%
		LineageCacheEviction.setStartTimestamp();
		LineageGPUCacheEviction.setStartTimestamp();
		// Note: GPU cache initialization is done in GPUContextPool:initializeGPU()
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
			ComputationCPInstruction cinst = inst instanceof ComputationCPInstruction ? (ComputationCPInstruction)inst : null;
			ComputationFEDInstruction cfinst = inst instanceof ComputationFEDInstruction ? (ComputationFEDInstruction)inst : null;
			ComputationSPInstruction cspinst = inst instanceof ComputationSPInstruction ? (ComputationSPInstruction)inst : null;
			GPUInstruction gpuinst = inst instanceof GPUInstruction ? (GPUInstruction)inst : null;
			//TODO: Replace with generic type
				
			LineageItem instLI = (cinst != null) ? cinst.getLineageItem(ec).getValue()
					: (cfinst != null) ? cfinst.getLineageItem(ec).getValue()
					: (cspinst != null) ? cspinst.getLineageItem(ec).getValue()
					: gpuinst.getLineageItem(ec).getValue();
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
					if (e == null && LineageCacheConfig.getCacheType().isPartialReuse() && cspinst == null)
						if( LineageRewriteReuse.executeRewrites(inst, ec) )
							e = getIntern(item.getKey());
					//TODO: Partial reuse for Spark instructions
					reuseAll &= (e != null);
					item.setValue(e);
					
					//create a placeholder if no reuse to avoid redundancy
					//(e.g., concurrent threads that try to start the computation)
					if(e == null && isMarkedForCaching(inst, ec)) {
						if (cinst != null)
							putIntern(item.getKey(), cinst.output.getDataType(), null, null,  0);
						else if (cfinst != null)
							putIntern(item.getKey(), cfinst.output.getDataType(), null, null,  0);
						else if (cspinst != null)
							putIntern(item.getKey(), cspinst.output.getDataType(), null, null,  0);
						else if (gpuinst != null)
							putIntern(item.getKey(), gpuinst._output.getDataType(), null, null,  0);
						//FIXME: different o/p datatypes for MultiReturnBuiltins.
					}
				}
			}
			reuse = reuseAll;
			
			if(reuse) { //reuse
				boolean gpuReuse = false;
				//put reuse value into symbol table (w/ blocking on placeholders)
				for (MutablePair<LineageItem, LineageCacheEntry> entry : liList) {
					e = entry.getValue();
					String outName = null;
					if (inst instanceof MultiReturnBuiltinCPInstruction)
						outName = ((MultiReturnBuiltinCPInstruction)inst).
							getOutput(entry.getKey().getOpcode().charAt(entry.getKey().getOpcode().length()-1)-'0').getName(); 
					else if (inst instanceof ComputationCPInstruction)
						outName = cinst.output.getName();
					else if (inst instanceof ComputationFEDInstruction)
						outName = cfinst.output.getName();
					else if (inst instanceof ComputationSPInstruction)
						outName = cspinst.output.getName();
					else if (inst instanceof GPUInstruction)
						outName = gpuinst._output.getName();
					
					if (e.isMatrixValue() && e._gpuObject == null) {
						MatrixBlock mb = e.getMBValue(); //wait if another thread is executing the same inst.
						if (mb == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache
						else
							ec.setMatrixOutput(outName, mb);
					}
					else if (e.isScalarValue()) {
						ScalarObject so = e.getSOValue(); //wait if another thread is executing the same inst.
						if (so == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache
						else
							ec.setScalarOutput(outName, so);
					}
					else { //TODO handle locks on gpu objects
						//shallow copy the cached GPUObj to the output MatrixObject
						ec.getMatrixObject(outName).setGPUObject(ec.getGPUContext(0), 
								ec.getGPUContext(0).shallowCopyGPUObject(e._gpuObject, ec.getMatrixObject(outName)));
						//Set dirty to true, so that it is later copied to the host for write
						ec.getMatrixObject(outName).getGPUObject(ec.getGPUContext(0)).setDirty(true);
						gpuReuse = true;
					}

					reuse = true;

					if (DMLScript.STATISTICS) //increment saved time
						LineageCacheStatistics.incrementSavedComputeTime(e._computeTime);
				}
				if (DMLScript.STATISTICS) {
					if (gpuReuse)
						LineageCacheStatistics.incrementGpuHits();
					else
						LineageCacheStatistics.incrementInstHits();
				}
			}
		}
		
		return reuse;
	}
	
	public static boolean reuse(List<String> outNames, List<DataIdentifier> outParams, 
			int numOutputs, LineageItem[] liInputs, String name, ExecutionContext ec)
	{
		if (DMLScript.LINEAGE_ESTIMATE && !name.startsWith("SB"))
			LineageEstimator.stopEstimator(outParams, liInputs, name);
		
		if( !LineageCacheConfig.isMultiLevelReuse())
			return false;
		
		boolean reuse = (outParams.size() != 0);
		long savedComputeTime = 0;
		HashMap<String, Data> funcOutputs = new HashMap<>();
		HashMap<String, LineageItem> funcLIs = new HashMap<>();
		for (int i=0; i<numOutputs; i++) {
			String opcode = name + String.valueOf(i+1);
			LineageItem li = new LineageItem(opcode, liInputs);
			// set _distLeaf2Node for this special lineage item to 1
			// to save it from early eviction if DAGHEIGHT policy is selected
			li.setHeight(1);
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
					MatrixBlock mb = e.getMBValue();
					if (mb == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
					MetaDataFormat md = new MetaDataFormat(
						e.getMBValue().getDataCharacteristics(),FileFormat.BINARY);
					boundValue = new MatrixObject(ValueType.FP64, boundVarName, md);
					((MatrixObject)boundValue).acquireModify(e.getMBValue());
					((MatrixObject)boundValue).release();
				}
				else {
					boundValue = e.getSOValue();
					if (boundValue == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
				}

				funcOutputs.put(boundVarName, boundValue);
				LineageItem orig = e._origItem;
				funcLIs.put(boundVarName, orig);
				//all the entries have the same computeTime
				savedComputeTime = e._computeTime;
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

			if (DMLScript.STATISTICS) //increment saved time
				LineageCacheStatistics.incrementSavedComputeTime(savedComputeTime);
		}
		
		return reuse;
	}
	
	//Reuse federated UDFs
	public static FederatedResponse reuse(FederatedUDF udf, ExecutionContext ec) 
	{
		if (ReuseCacheType.isNone() || udf.getOutputIds() == null)
			return new FederatedResponse(FederatedResponse.ResponseType.ERROR);
		//TODO: reuse only those UDFs which are part of reusable instructions
		
		boolean reuse = false;
		List<Long> outIds = udf.getOutputIds();
		HashMap<String, Data> udfOutputs = new HashMap<>();
		long savedComputeTime = 0;

		//TODO: support multi-return UDFs
		if (udf.getLineageItem(ec) == null)
			//TODO: trace all UDFs
			return new FederatedResponse(FederatedResponse.ResponseType.ERROR);

		LineageItem li = udf.getLineageItem(ec).getValue();
		li.setHeight(1); //to save from early eviction
		LineageCacheEntry e = null;
		synchronized(_cache) {
			if (probe(li))
				e = LineageCache.getIntern(li);
			else
				//for now allow only matrix blocks
				putIntern(li, DataType.MATRIX, null, null, 0);
		}
		
		if (e != null) {
			String outName = String.valueOf(outIds.get(0));
			Data outValue = null;
			//convert to matrix object
			if (e.isMatrixValue()) {
				MatrixBlock mb = e.getMBValue();
				if (mb == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
					//the executing thread removed this entry from cache
					return new FederatedResponse(FederatedResponse.ResponseType.ERROR);

				MetaDataFormat md = new MetaDataFormat(
					e.getMBValue().getDataCharacteristics(),FileFormat.BINARY);
				outValue = new MatrixObject(ValueType.FP64, outName, md);
				((MatrixObject)outValue).acquireModify(e.getMBValue());
				((MatrixObject)outValue).release();
			}
			else {
				outValue = e.getSOValue();
				if (outValue == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
					//the executing thread removed this entry from cache
					return new FederatedResponse(FederatedResponse.ResponseType.ERROR);
			}
			udfOutputs.put(outName, outValue);
			savedComputeTime = e._computeTime;
			reuse = true;
		}
		else
			reuse = false;
		
		if (reuse) {
			FederatedResponse res = null;
			for (Map.Entry<String, Data> entry : udfOutputs.entrySet()) {
				String var = entry.getKey();
				Data val = entry.getValue();
				//cleanup existing data bound to output name
				Data exdata = ec.removeVariable(var);
				if (exdata != val)
					ec.cleanupDataObject(exdata);
				//add or replace data in the symbol table
				ec.setVariable(var, val);
				//build and return a federated response
				res = LineageItemUtils.setUDFResponse(udf, (MatrixObject) val);
			}

			if (DMLScript.STATISTICS) {
				//TODO: dedicated stats for federated reuse
				LineageCacheStatistics.incrementInstHits();
				LineageCacheStatistics.incrementSavedComputeTime(savedComputeTime);
			}
			
			return res;
		}
		return new FederatedResponse(FederatedResponse.ResponseType.ERROR);
	}

	public static boolean reuseFedRead(String outName, DataType dataType, LineageItem li, ExecutionContext ec) {
		if (ReuseCacheType.isNone() || dataType != DataType.MATRIX)
			return false;

		LineageCacheEntry e = null;
		synchronized(_cache) {
			if(LineageCache.probe(li)) {
				e = LineageCache.getIntern(li);
			}
			else {
				putIntern(li, dataType, null, null, 0);
				return false; // direct return after placing the placeholder
			}
		}

		if(e != null && e.isMatrixValue()) {
			MatrixBlock mb = e.getMBValue(); // waiting if the value is not set yet
			if (mb == null || e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
				return false;  // the executing thread removed this entry from cache
			ec.setMatrixOutput(outName, e.getMBValue());

			if (DMLScript.STATISTICS) { //increment saved time
				FederatedStatistics.incFedReuseReadHitCount();
				FederatedStatistics.incFedReuseReadBytesCount(mb);
				LineageCacheStatistics.incrementSavedComputeTime(e._computeTime);
			}

			return true;
		}
		return false;
	}

	public static byte[] reuseSerialization(LineageItem objLI) {
		if (ReuseCacheType.isNone() || objLI == null)
			return null;

		LineageItem li = LineageItemUtils.getSerializedFedResponseLineageItem(objLI);

		LineageCacheEntry e = null;
		synchronized(_cache) {
			if(LineageCache.probe(li)) {
				e = LineageCache.getIntern(li);
			}
			else {
				putIntern(li, DataType.UNKNOWN, null, null, 0);
				return null; // direct return after placing the placeholder
			}
		}

		if(e != null && e.isSerializedBytes()) {
			byte[] sBytes = e.getSerializedBytes(); // waiting if the value is not set yet
			if (sBytes == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
				return null;  // the executing thread removed this entry from cache

			if (DMLScript.STATISTICS) { // increment statistics
				LineageCacheStatistics.incrementSavedComputeTime(e._computeTime);
				FederatedStatistics.aggFedSerializationReuse(sBytes.length);
			}

			return sBytes;
		}
		return null;
	}

	public static boolean probe(LineageItem key) {
		//TODO problematic as after probe the matrix might be kicked out of cache
		boolean p = _cache.containsKey(key);  // in cache or in disk
		if (!p && DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(key))
			// The sought entry was in cache but removed later 
			LineageCacheStatistics.incrementDelHits();
		return p;
	}
	
	//This method is for hard removal of an entry, w/o maintaining eviction data structures
	public static void removeEntry(LineageItem key) {
		boolean p = _cache.containsKey(key);
		if (!p) return;
		synchronized(_cache) {
			LineageCacheEntry e = getEntry(key);
			long size = e.getSize();
			if (e._origItem == null)
				_cache.remove(e._key);

			else {
				LineageCacheEntry h = _cache.get(e._origItem); //head
				while (h != null) {
					LineageCacheEntry tmp = h;
					h = h._nextEntry;
					_cache.remove(tmp._key);
				}
			}
			LineageCacheEviction.updateSize(size, false);
		}
	}
	
	public static MatrixBlock getMatrix(LineageItem key) {
		LineageCacheEntry e = null;
		synchronized( _cache ) {
			e = getIntern(key);
		}
		return e.getMBValue();
	}

	public static LineageCacheEntry getEntry(LineageItem key) {
		LineageCacheEntry e = null;
		synchronized( _cache ) {
			e = getIntern(key);
		}
		return e;
	}
	
	//NOTE: safe to pin the object in memory as coming from CPInstruction
	//TODO why do we need both of these public put methods
	public static void putMatrix(Instruction inst, ExecutionContext ec, long computetime) {
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			LineageItem item = ((LineageTraceable) inst).getLineageItem(ec).getValue();
			//This method is called only to put matrix value
			MatrixObject mo = null;
			if (inst instanceof ComputationCPInstruction)
				mo = ec.getMatrixObject(((ComputationCPInstruction) inst).output);
			else if (inst instanceof ComputationFEDInstruction)
				mo = ec.getMatrixObject(((ComputationFEDInstruction) inst).output);
			else if (inst instanceof ComputationSPInstruction)
				mo = ec.getMatrixObject(((ComputationSPInstruction) inst).output);

			synchronized( _cache ) {
				putIntern(item, DataType.MATRIX, mo.acquireReadAndRelease(), null, computetime);
			}
		}
	}
	
	public static void putValue(Instruction inst, ExecutionContext ec, long starttime) {
		if (DMLScript.LINEAGE_ESTIMATE)
			//forward to estimator
			LineageEstimator.processSingleInst(inst, ec, starttime);

		if (ReuseCacheType.isNone())
			return;
		long computetime = System.nanoTime() - starttime;
		if (LineageCacheConfig.isReusable(inst, ec) ) {
			//if (!isMarkedForCaching(inst, ec)) return;
			List<Pair<LineageItem, Data>> liData = null;
			GPUObject liGpuObj = null;
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
			else if (inst instanceof GPUInstruction) {
				// TODO: gpu multiretrun instructions
				Data gpudata = ec.getVariable(((GPUInstruction) inst)._output);
				liGpuObj = gpudata instanceof MatrixObject ? 
						ec.getMatrixObject(((GPUInstruction)inst)._output).getGPUObject(ec.getGPUContext(0)) : null;

				// Scalar gpu intermediates is always copied back to host. 
				// No need to cache the GPUobj for scalar intermediates.
				if (liGpuObj == null)
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((GPUInstruction)inst)._output)));
			}
			else
				if (inst instanceof ComputationCPInstruction)
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationCPInstruction) inst).output)));
				else if (inst instanceof ComputationFEDInstruction)
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationFEDInstruction) inst).output)));
				else if (inst instanceof ComputationSPInstruction)
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationSPInstruction) inst).output)));

			if (liGpuObj == null)
				putValueCPU(inst, liData, computetime);
			else
				putValueGPU(liGpuObj, instLI, computetime);
		}
	}
	
	private static void putValueCPU(Instruction inst, List<Pair<LineageItem, Data>> liData, long computetime)
	{
		synchronized( _cache ) {
			for (Pair<LineageItem, Data> entry : liData) {
				LineageItem item = entry.getKey();
				Data data = entry.getValue();

				if (!probe(item))
					continue;

				LineageCacheEntry centry = _cache.get(item);

				if (!(data instanceof MatrixObject) && !(data instanceof ScalarObject)) {
					// Reusable instructions can return a frame (rightIndex). Remove placeholders.
					removePlaceholder(item);
					continue;
				}

				if (data instanceof MatrixObjectFuture) {
					// We don't want to call get() on the future immediately after the execution
					removePlaceholder(item);
					continue;
				}

				if (inst instanceof PrefetchCPInstruction || inst instanceof BroadcastCPInstruction)
					// For the async. instructions, caching is handled separately by the tasks
					continue;

				if (data instanceof MatrixObject && ((MatrixObject) data).hasRDDHandle()) {
					// Avoid triggering pre-matured Spark instruction chains
					removePlaceholder(item);
					continue;
				}

				if (LineageCacheConfig.isOutputFederated(inst, data)) {
					// Do not cache federated outputs (in the coordinator)
					// Cannot skip putting the placeholder as the above is only known after execution
					removePlaceholder(item);
					continue;
				}

				MatrixBlock mb = (data instanceof MatrixObject) ? 
						((MatrixObject)data).acquireReadAndRelease() : null;
				long size = mb != null ? mb.getInMemorySize() : ((ScalarObject)data).getSize();

				//remove the placeholder if the entry is bigger than the cache.
				if (size > LineageCacheEviction.getCacheLimit()) {
					removePlaceholder(item);
					continue; 
				}

				//make space for the data
				if (!LineageCacheEviction.isBelowThreshold(size))
					LineageCacheEviction.makeSpace(_cache, size);
				LineageCacheEviction.updateSize(size, true);

				//place the data
				if (data instanceof MatrixObject)
					centry.setValue(mb, computetime);
				else if (data instanceof ScalarObject)
					centry.setValue((ScalarObject)data, computetime);

				if (DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(centry._key)) {
					// Add to missed compute time
					LineageCacheStatistics.incrementMissedComputeTime(centry._computeTime);
				}

				//maintain order for eviction
				LineageCacheEviction.addEntry(centry);
			}
		}
	}
	
	private static void putValueGPU(GPUObject gpuObj, LineageItem instLI, long computetime) {
		synchronized( _cache ) {
			LineageCacheEntry centry = _cache.get(instLI);
			// Update the total size of lineage cached gpu objects
			// The eviction is handled by the unified gpu memory manager
			LineageGPUCacheEviction.updateSize(gpuObj.getSizeOnDevice(), true);
			// Set the GPUOject in the cache
			centry.setGPUValue(gpuObj, computetime);
			// Maintain order for eviction
			LineageGPUCacheEviction.addEntry(centry);
		}
	}

	public static void putValueAsyncOp(LineageItem instLI, Data data, boolean prefetched, long starttime)
	{
		if (ReuseCacheType.isNone())
			return;
		if (!prefetched) //prefetching was not successful
			return;

		synchronized( _cache )
		{
			if (!probe(instLI))
				return;

			long computetime = System.nanoTime() - starttime;
			LineageCacheEntry centry = _cache.get(instLI);
			if(!(data instanceof MatrixObject) && !(data instanceof ScalarObject)) {
				// Reusable instructions can return a frame (rightIndex). Remove placeholders.
				removePlaceholder(instLI);
				return;
			}

			MatrixBlock mb = (data instanceof MatrixObject) ?
				((MatrixObject)data).acquireReadAndRelease() : null;
			long size = mb != null ? mb.getInMemorySize() : ((ScalarObject)data).getSize();

			// remove the placeholder if the entry is bigger than the cache.
			if (size > LineageCacheEviction.getCacheLimit()) {
				removePlaceholder(instLI);
				return;
			}

			// place the data
			if (data instanceof MatrixObject)
				centry.setValue(mb, computetime);
			else if (data instanceof ScalarObject)
				centry.setValue((ScalarObject)data, computetime);

			if (DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(centry._key)) {
				// Add to missed compute time
				LineageCacheStatistics.incrementMissedComputeTime(centry._computeTime);
			}

			//maintain order for eviction
			LineageCacheEviction.addEntry(centry);
		}
	}

	public static void putValue(List<DataIdentifier> outputs,
		LineageItem[] liInputs, String name, ExecutionContext ec, long computetime)
	{
		if (LineageCacheConfig.isEstimator())
			//forward to estimator
			LineageEstimator.processFunc(outputs, liInputs, name, ec, computetime);

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
				FuncLIMap.forEach((Li, boundLI) -> removePlaceholder(Li));
		}
		
		return;
	}
	
	public static void putValue(FederatedUDF udf, ExecutionContext ec, long computetime) 
	{
		if (ReuseCacheType.isNone() || udf.getOutputIds() == null)
			return;

		List<Long> outIds = udf.getOutputIds();
		if (udf.getLineageItem(ec) == null)
			//TODO: trace all UDFs
			return;
		synchronized (_cache) {
			LineageItem item = udf.getLineageItem(ec).getValue();
			if (!probe(item))
				return;
			LineageCacheEntry entry = _cache.get(item);
			Data data = ec.getVariable(String.valueOf(outIds.get(0)));
			if (!(data instanceof MatrixObject) && !(data instanceof ScalarObject)) {
				// Don't cache if the udf outputs frames
				removePlaceholder(item);
				return;
			}
			
			MatrixBlock mb = (data instanceof MatrixObject) ? 
					((MatrixObject)data).acquireReadAndRelease() : null;
			long size = mb != null ? mb.getInMemorySize() : ((ScalarObject)data).getSize();

			//remove the placeholder if the entry is bigger than the cache.
			if (size > LineageCacheEviction.getCacheLimit()) {
				removePlaceholder(item);
				return;
			}

			//make space for the data
			if (!LineageCacheEviction.isBelowThreshold(size))
				LineageCacheEviction.makeSpace(_cache, size);
			LineageCacheEviction.updateSize(size, true);

			//place the data
			if (data instanceof MatrixObject)
				entry.setValue(mb, computetime);
			else if (data instanceof ScalarObject)
				entry.setValue((ScalarObject)data, computetime);

			//TODO: maintain statistics, lineage estimate

			//maintain order for eviction
			LineageCacheEviction.addEntry(entry);
		}
	}

	public static void putFedReadObject(Data data, LineageItem li, ExecutionContext ec) {
		if(ReuseCacheType.isNone())
			return;

		LineageCacheEntry entry = _cache.get(li);
		if(entry != null && data instanceof MatrixObject) {
			long t0 = System.nanoTime();
			MatrixBlock mb = ((MatrixObject)data).acquireRead();
			long t1 = System.nanoTime();
			synchronized(_cache) {
				long size = mb != null ? mb.getInMemorySize() : 0;

				//remove the placeholder if the entry is bigger than the cache.
				if (size > LineageCacheEviction.getCacheLimit()) {
					removePlaceholder(li);
				}

				//make space for the data
				if (!LineageCacheEviction.isBelowThreshold(size))
					LineageCacheEviction.makeSpace(_cache, size);
				LineageCacheEviction.updateSize(size, true);

				entry.setValue(mb, t1 - t0);
			}
		}
		else {
			synchronized(_cache) {
				removePlaceholder(li);
			}
		}
	}

	public static void putSerializedObject(byte[] serialBytes, LineageItem objLI, long computetime) {
		if(ReuseCacheType.isNone())
			return;

		LineageItem li = LineageItemUtils.getSerializedFedResponseLineageItem(objLI);

		LineageCacheEntry entry = getIntern(li);

		if(entry != null && serialBytes != null) {
			synchronized(_cache) {
				long size = serialBytes.length;

				// remove the placeholder if the entry is bigger than the cache.
				if (size > LineageCacheEviction.getCacheLimit()) {
					removePlaceholder(li);
				}

				// make space for the data
				if (!LineageCacheEviction.isBelowThreshold(size))
					LineageCacheEviction.makeSpace(_cache, size);
				LineageCacheEviction.updateSize(size, true);

				entry.setValue(serialBytes, computetime);
			}
		}
		else {
			synchronized(_cache) {
				removePlaceholder(li);
			}
		}
	}

	public static void resetCache() {
		synchronized (_cache) {
			_cache.clear();
			LineageCacheEviction.resetEviction();
			LineageGPUCacheEviction.resetEviction();
		}
	}
	
	public static Map<LineageItem, LineageCacheEntry> getLineageCache() {
		return _cache;
	}

	
	//----------------- INTERNAL CACHE LOGIC IMPLEMENTATION --------------//
	
	private static void putIntern(LineageItem key, DataType dt, MatrixBlock Mval, ScalarObject Sval, long computetime) {
		if (_cache.containsKey(key))
			//can come here if reuse_partial option is enabled
			return;
		
		// Create a new entry.
		LineageCacheEntry newItem = new LineageCacheEntry(key, dt, Mval, Sval, computetime);
		
		// Make space by removing or spilling entries.
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
			if (DMLScript.STATISTICS)
				// Increment hit count.
				LineageCacheStatistics.incrementMemHits();

			// Maintain order for eviction
			LineageCacheEviction.getEntry(e);
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

			if (DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(e._key))
				// Add to missed compute time
				LineageCacheStatistics.incrementMissedComputeTime(e._computeTime);
			
			//maintain order for eviction
			LineageCacheEviction.addEntry(e);
		}
		else
			removePlaceholder(item);    //remove the placeholder
	}
	
	private static void removePlaceholder(LineageItem item) {
		//Caller should hold the monitor on _cache
		if (!_cache.containsKey(item))
			return;
		LineageCacheEntry centry = _cache.get(item);
		centry.removeAndNotify();
		_cache.remove(item);
	}
	
	private static boolean isMarkedForCaching (Instruction inst, ExecutionContext ec) {
		if (!LineageCacheConfig.getCompAssRW())
			return true;
		
		CPOperand output = inst instanceof ComputationCPInstruction ? ((ComputationCPInstruction)inst).output 
				: inst instanceof ComputationFEDInstruction ? ((ComputationFEDInstruction)inst).output
				: inst instanceof ComputationSPInstruction ? ((ComputationSPInstruction)inst).output
				: ((GPUInstruction)inst)._output;
		if (output.isMatrix()) {
			MatrixObject mo = inst instanceof ComputationCPInstruction ? ec.getMatrixObject(((ComputationCPInstruction)inst).output) 
				: inst instanceof ComputationFEDInstruction ? ec.getMatrixObject(((ComputationFEDInstruction)inst).output)
				: inst instanceof ComputationSPInstruction ? ec.getMatrixObject(((ComputationSPInstruction)inst).output)
				: ec.getMatrixObject(((GPUInstruction)inst)._output);
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
