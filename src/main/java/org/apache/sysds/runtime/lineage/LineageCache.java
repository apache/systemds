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

import jcuda.Pointer;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.MatrixObjectFuture;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.instructions.Instruction;
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
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
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
		if (LineageCacheConfig.isReusable(inst, ec))
		{
			List<MutablePair<LineageItem, LineageCacheEntry>> liList = getLineageItems(inst, ec);

			//atomic try reuse full/partial and set placeholder, without
			//obtaining value to avoid blocking in critical section
			LineageCacheEntry e = null;
			boolean reuseAll = true;
			synchronized( _cache ) {
				//try to reuse full or partial intermediates (CPU and FED only)
				for (MutablePair<LineageItem,LineageCacheEntry> item : liList) {
					if (LineageCacheConfig.getCacheType().isFullReuse())
						//e = LineageCache.probe(item.getKey()) ? getIntern(item.getKey()) : null;
						e = getIntern(item.getKey()); //avoid double probing (containsKey + get)
					//TODO need to also move execution of compensation plan out of here
					//(create lazily evaluated entry)
					if (e == null && LineageCacheConfig.getCacheType().isPartialReuse()
						&& !(inst instanceof ComputationSPInstruction)
						&& !(DMLScript.USE_ACCELERATOR))
						if( LineageRewriteReuse.executeRewrites(inst, ec) )
							e = getIntern(item.getKey());
					reuseAll &= (e != null);
					item.setValue(e);
					
					//create a placeholder if no reuse to avoid redundancy
					//(e.g., concurrent threads that try to start the computation)
					if(e == null && isMarkedForCaching(inst, ec))
						putInternPlaceholder(inst, item.getKey());
				}
			}
			reuse = reuseAll;
			
			if(reuse) { //reuse
				//put reused value into symbol table (w/ blocking on placeholders)
				for (MutablePair<LineageItem, LineageCacheEntry> entry : liList) {
					e = entry.getValue();
					String outName = getOutputName(inst, entry.getKey());

					if (e.isMatrixValue() && !e.isGPUObject()) {
						MatrixBlock mb = e.getMBValue(); //wait if another thread is executing the same inst.
						if (mb == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache
						if (e.getCacheStatus() == LineageCacheStatus.TOCACHE) { //not cached yet
							ec.replaceLineageItem(outName, e._key); //reuse the lineage trace
							return false;
						}
						ec.setMatrixOutput(outName, mb);
					}
					else if (e.isScalarValue()) {
						ScalarObject so = e.getSOValue(); //wait if another thread is executing the same inst.
						if (so == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache
						if (e.getCacheStatus() == LineageCacheStatus.TOCACHE) { //not cached yet
							ec.replaceLineageItem(outName, e._key); //reuse the lineage trace
							return false;
						}
						ec.setScalarOutput(outName, so);
					}
					else if (e.isRDDPersist()) {
						RDDObject rdd = e.getRDDObject();
						if (rdd == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache

						//Reuse the cached RDD (local or persisted at the executors)
						if (e.getCacheStatus() == LineageCacheStatus.TOPERSISTRDD) {  //second hit
							//Cannot reuse rdd as already garbage collected
							//putValueRDD will save the RDD and call persist
							if (DMLScript.STATISTICS) LineageCacheStatistics.incrementDelHitsRdd();
							ec.replaceLineageItem(outName, e._key); //still reuse the lineage trace
							return false;
						}
						//Reuse from third hit onwards (status == PERSISTEDRDD)
						((SparkExecutionContext) ec).setRDDHandleForVariable(outName, rdd);
						//Set the cached data characteristics to the output matrix object
						ec.getMatrixObject(outName).updateDataCharacteristics(rdd.getDataCharacteristics());
						//Safely cleanup the child RDDs if this RDD is persisted already
						//If reused 3 times and still not persisted, move to Spark asynchronously
						if (probeRDDDistributed(e))
							LineageSparkCacheEviction.cleanupChildRDDs(e);
						else
							LineageSparkCacheEviction.moveToSpark(e);
					}
					else { //TODO handle locks on gpu objects
						Pointer gpuPtr = e.getGPUPointer();
						if (gpuPtr == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
							return false;  //the executing thread removed this entry from cache
						if (e.getCacheStatus() == LineageCacheStatus.TOCACHEGPU) {  //second hit
							//Cannot reuse as already garbage collected
							if (DMLScript.STATISTICS) LineageCacheStatistics.incrementDelHitsGpu(); //increase miss count
							ec.replaceLineageItem(outName, e._key); //still reuse the lineage trace
							return false;
						}
						//Reuse from third hit onwards (status == GPUCACHED)
						//Create a GPUObject with the cached pointer
						GPUObject gpuObj = new GPUObject(ec.getGPUContext(0),
							ec.getMatrixObject(outName), gpuPtr);
						ec.getMatrixObject(outName).setGPUObject(ec.getGPUContext(0), gpuObj);
						//Set dirty to true, so that it is later copied to the host for write
						ec.getMatrixObject(outName).getGPUObject(ec.getGPUContext(0)).setDirty(true);
						//Set the cached data characteristics to the output matrix object
						ec.getMatrixObject(outName).updateDataCharacteristics(e.getDataCharacteristics());
						//Increment the live count for this pointer
						LineageGPUCacheEviction.incrementLiveCount(gpuPtr);
						//Maintain the eviction list in the free list
						LineageGPUCacheEviction.maintainOrder(e);
					}
					//Replace the live lineage trace with the cached one (if not parfor, dedup)
					ec.replaceLineageItem(outName, e._key);
				}
				maintainReuseStatistics(ec, inst, liList.get(0).getValue());
			}
		}
		
		return reuse;
	}

	// Reuse function and statement block outputs
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
		ArrayList<LineageCacheEntry> funcOutLIs = new ArrayList<>();
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
					funcOutLIs.add(e);
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
				//String fname = "target\\testTemp\\functions\\async\\LineageReuseSparkTest\\LineageReuseSpark8/target/scratch_space//_p11736_192.168.0.113//_t0/temp999";
				//fname = VariableCPInstruction.getUniqueFileName(fname);
				//convert to matrix object
				if (e.isMatrixValue()) {
					MatrixBlock mb = e.getMBValue();
					if (mb == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
					if (e.getCacheStatus() == LineageCacheStatus.TOCACHE)  //not cached yet
						return false;
					MetaDataFormat md = new MetaDataFormat(
						e.getMBValue().getDataCharacteristics(),FileFormat.BINARY);
					md.getDataCharacteristics().setBlocksize(ConfigurationManager.getBlocksize());
					boundValue = new MatrixObject(ValueType.FP64, boundVarName, md);
					((MatrixObject)boundValue).acquireModify(e.getMBValue());
					((MatrixObject)boundValue).release();
				}
				else if (e.isGPUObject()) {
					Pointer gpuPtr = e.getGPUPointer();
					if (gpuPtr == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
					MetaDataFormat md = new MetaDataFormat(e.getDataCharacteristics(), FileFormat.BINARY);
					boundValue = new MatrixObject(ValueType.FP64, boundVarName, md);
					//Create a GPUObject with the cached pointer
					GPUObject gpuObj = new GPUObject(ec.getGPUContext(0),
						((MatrixObject)boundValue), gpuPtr);
					//Set dirty to true, so that it is later copied to the host for write
					gpuObj.setDirty(true);
					((MatrixObject) boundValue).setGPUObject(ec.getGPUContext(0), gpuObj);
				}
				else if (e.isRDDPersist()) {
					RDDObject rdd = e.getRDDObject();
					if (rdd == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
					//Set the data characteristics and hdfs file to the output matrix object
					MetaDataFormat md = new MetaDataFormat(rdd.getDataCharacteristics(),FileFormat.BINARY);
					String filename = rdd.getHDFSFilename() != null ? rdd.getHDFSFilename() : boundVarName;
					boundValue = new MatrixObject(ValueType.FP64, filename, md);
					((MatrixObject) boundValue).setRDDHandle(rdd);
				}
				else if (e.isScalarValue()) {
					boundValue = e.getSOValue();
					if (boundValue == null && e.getCacheStatus() == LineageCacheStatus.NOTCACHED)
						return false;  //the executing thread removed this entry from cache
					if (e.getCacheStatus() == LineageCacheStatus.TOCACHE)  //not cached yet
						return false;
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
			//Additional maintenance for GPU pointers and RDDs
			for (LineageCacheEntry e : funcOutLIs) {
				if (e.isGPUObject()) {
					switch(e.getCacheStatus()) {
						case TOCACHEGPU:
							//Cannot reuse as already garbage collected putValue method
							// will save the pointer while caching the original instruction
							if (DMLScript.STATISTICS) LineageCacheStatistics.incrementDelHitsGpu(); //increase miss count
							return false;
						case GPUCACHED:
							//Increment the live count for this pointer
							LineageGPUCacheEviction.incrementLiveCount(e.getGPUPointer());
							//Maintain the eviction list in the free list
							LineageGPUCacheEviction.maintainOrder(e);
							if (DMLScript.STATISTICS) LineageCacheStatistics.incrementGpuHits();
							break;
						default:
							return false;
					}
				}
				else if (e.isRDDPersist()) {
					//Reuse the cached RDD (local or persisted at the executors)
					switch(e.getCacheStatus()) {
						case TOPERSISTRDD:
							//Cannot reuse rdd as already garbage collected
							//putValue method will save the RDD and call persist
							//while caching the original instruction
							if (DMLScript.STATISTICS) LineageCacheStatistics.incrementDelHitsRdd(); //increase miss count
							return false;
						case PERSISTEDRDD:
							//Reuse the persisted intermediate at the executors
							//Safely cleanup the child RDDs if this RDD is persisted already
							//If reused 3 times and still not persisted, move to Spark asynchronously
							if (probeRDDDistributed(e)) {
								LineageSparkCacheEviction.cleanupChildRDDs(e);
								if (DMLScript.STATISTICS) LineageCacheStatistics.incrementRDDPersistHits();
							}
							else {
								LineageSparkCacheEviction.moveToSpark(e);
								if (DMLScript.STATISTICS) LineageCacheStatistics.incrementRDDHits();
							}
							break;
						default:
							return false;
					}
				}
			}
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
				md.getDataCharacteristics().setBlocksize(ConfigurationManager.getBlocksize());
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
		return _cache.containsKey(key);
	}

	private static boolean probeRDDDistributed(LineageItem key) {
		if (!_cache.containsKey(key))
			return false;
		return probeRDDDistributed(_cache.get(key));
	}

	protected static boolean probeRDDDistributed(LineageCacheEntry e) {
		if (!e.isRDDPersist())
			return false;
		return SparkExecutionContext.isRDDCached(e.getRDDObject().getRDD().id());
	}
	
	//This method is for hard removal of an entry, w/o maintaining eviction data structures
	public static void removeEntry(LineageItem key) {
		boolean p = _cache.containsKey(key);
		if (!p) return;
		synchronized(_cache) {
			LineageCacheEntry e = getEntry(key);
			if (e.isRDDPersist()) {
				e.getRDDObject().getRDD().unpersist(false);
				e.getRDDObject().setCheckpointRDD(false);
				return;
			}

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

			if (e.isRDDPersist())
				LineageSparkCacheEviction.updateSize(e.getSize(), false);
			else
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
			GPUObject liGPUObj= null;
			//LineageItem instLI = ((LineageTraceable) inst).getLineageItem(ec).getValue();
			LineageItem instLI = null;
			if (inst instanceof MultiReturnBuiltinCPInstruction) {
				liData = new ArrayList<>();
				instLI = ((LineageTraceable) inst).getLineageItem(ec).getValue();
				MultiReturnBuiltinCPInstruction mrInst = (MultiReturnBuiltinCPInstruction)inst;
				for (int i=0; i<mrInst.getNumOutputs(); i++) {
					String opcode = instLI.getOpcode() + String.valueOf(i);
					LineageItem li = new LineageItem(opcode, instLI.getInputs());
					Data value = ec.getVariable(mrInst.getOutput(i));
					liData.add(Pair.of(li, value));
				}
			}
			else if (inst instanceof GPUInstruction) {
				// TODO: gpu multi-return instructions
				if (!LineageCacheConfig.isMultiBackendReuse()) {
					// Multi-backend reuse is disabled
					instLI = ec.getLineageItem(((GPUInstruction) inst)._output);
					removePlaceholder(instLI);
					return;
				}
				Data gpudata = ec.getVariable(((GPUInstruction) inst)._output);
				liGPUObj = gpudata instanceof MatrixObject ?
						ec.getMatrixObject(((GPUInstruction)inst)._output).
							getGPUObject(ec.getGPUContext(0)) : null;

				// Scalar gpu intermediates is always copied back to host. 
				// No need to cache the GPUobj for scalar intermediates.
				instLI = ec.getLineageItem(((GPUInstruction) inst)._output);
				if (liGPUObj == null)
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((GPUInstruction) inst)._output)));
			}
			else if (inst instanceof ComputationSPInstruction
				&& (ec.getVariable(((ComputationSPInstruction) inst).output) instanceof MatrixObject)
				&& (ec.getCacheableData(((ComputationSPInstruction)inst).output.getName())).hasRDDHandle()) {
				instLI = ec.getLineageItem(((ComputationSPInstruction) inst).output);
				if (!LineageCacheConfig.isMultiBackendReuse()) {
					// Multi-backend reuse is disabled
					removePlaceholder(instLI);
					return;
				}
				putValueRDD(inst, instLI, ec, computetime);
				return;
			}
			else
				if (inst instanceof ComputationCPInstruction) {
					instLI = ec.getLineageItem(((ComputationCPInstruction) inst).output);
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationCPInstruction) inst).output)));
				}
				else if (inst instanceof ComputationFEDInstruction) {
					instLI = ec.getLineageItem(((ComputationFEDInstruction) inst).output);
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationFEDInstruction) inst).output)));
				}
				else if (inst instanceof ComputationSPInstruction) { //collects or prefetches
					instLI = ec.getLineageItem(((ComputationSPInstruction) inst).output);
					if (!LineageCacheConfig.isMultiBackendReuse()) {
						// Multi-backend reuse is disabled
						removePlaceholder(instLI);
						return;
					}
					liData = Arrays.asList(Pair.of(instLI, ec.getVariable(((ComputationSPInstruction) inst).output)));
				}

			if (liGPUObj == null)
				putValueCPU(inst, liData, computetime);
			else
				putValueGPU(liGPUObj, instLI, computetime);
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

				if (data instanceof MatrixObjectFuture || inst instanceof PrefetchCPInstruction) {
					// We don't want to call get() on the future immediately after the execution
					// For the async. instructions, caching is handled separately by the tasks
					removePlaceholder(item);
					continue;
				}

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

				//delay caching of large matrix blocks if the feature is enabled
				if (centry.getCacheStatus() == LineageCacheStatus.EMPTY && LineageCacheConfig.isDelayedCaching()) {
					if (data instanceof MatrixObject  //no delayed caching for scalars
						&& !LineageCacheEviction._removelist.containsKey(centry._key) //evicted before
						&& size > 0.05 * LineageCacheEviction.getAvailableSpace()) { //size adaptive
						centry.setCacheStatus(LineageCacheStatus.TOCACHE);
						continue;
					}
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
				centry.setCacheStatus(LineageCacheStatus.CACHED);

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
			// TODO: Cache sparse pointers
			if (gpuObj.isSparse()) {
				removePlaceholder(instLI);
				return;
			}
			if(DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(centry._key))
				LineageCacheStatistics.incrementDelHitsGpu();
			switch(centry.getCacheStatus()) {
				case EMPTY:  //first hit
					// Cache right away if removed before or a heavy hitter
					if (LineageCacheConfig.isDelayedCachingGPU()  //if delayed caching enabled
						&& !LineageCacheEviction._removelist.containsKey(centry._key)
						&& !LineageCacheConfig.isComputeGPUOps(centry._key.getOpcode())) {
						// Delayed Caching: Set the GPUOject in the cache. Will be garbage collected
						centry.setGPUValue(gpuObj.getDensePointer(), gpuObj.getAllocatedSize(),
							gpuObj.getMatrixObject().getMetaData(), computetime);
						centry.setCacheStatus(LineageCacheStatus.TOCACHEGPU);
						break;
					} //else, falls through and cache
				case TOCACHEGPU:  //second hit
					// Set the GPUOject in the cache and update the status
					centry.setGPUValue(gpuObj.getDensePointer(), gpuObj.getAllocatedSize(),
						gpuObj.getMatrixObject().getMetaData(), computetime);
					centry.setCacheStatus(LineageCacheStatus.GPUCACHED);
					// Maintain order for eviction
					LineageGPUCacheEviction.addEntry(centry);
					break;
				default:
					throw new DMLRuntimeException("Execution should not reach here: "+centry._key);
			}
		}
	}

	private static void putValueRDD(Instruction inst, LineageItem instLI, ExecutionContext ec, long computetime) {
		synchronized( _cache ) {
			if (!probe(instLI))
				return;
			LineageCacheEntry centry = _cache.get(instLI);
			// Avoid reuse chkpoint, which is unnecessary
			if (inst.getOpcode().equalsIgnoreCase("chkpoint")) {
				removePlaceholder(instLI);
				return;
			}
			boolean opToPersist = LineageCacheConfig.isReusableRDDType(inst);
			// Return if the intermediate is not to be persisted in the executors
			if (!opToPersist) {
				removePlaceholder(instLI);
				return;
			}

			// Get the RDD handle of the RDD
			CacheableData<?> cd = ec.getCacheableData(((ComputationSPInstruction)inst).output.getName());
			RDDObject rddObj = cd.getRDDHandle();
			// Save the metadata and hdfs filename. Required during reuse and space management.
			rddObj.setDataCharacteristics(cd.getDataCharacteristics());
			rddObj.setHDFSFilename(cd.getFileName());
			// Set the RDD object in the cache
			switch(centry.getCacheStatus()) {
				case EMPTY:  //first hit
					// Cache right away if delayed caching is disabled
					if (LineageCacheConfig.isDelayedCachingRDD()) {
						// Do not save the child RDDS (incl. broadcast vars) on the first hit.
						// Let them be garbage collected via rmvar. Save them on the second hit
						// by disabling garbage collection on this and the child RDDs.
						centry.setRDDValue(rddObj, computetime); //rddObj will be garbage collected
						break;
					} //else, fall through and cache
				case TOPERSISTRDD:  //second hit
					// Replace the old RDD (GCed) with the new one
					centry.setRDDValue(rddObj);
					// Set the correct status to indicate the RDD is marked to be persisted
					centry.setCacheStatus(LineageCacheStatus.PERSISTEDRDD);
					// Call persist. Next collect will materialize this intermediate in Spark
					persistRDD(inst, centry, ec);
					// Mark lineage cached to prevent this and child RDDs from cleanup by rmvar
					centry.getRDDObject().setLineageCached();
					break;
				default:
					throw new DMLRuntimeException("Execution should not reach here: "+centry._key);
			}
		}
	}

	// This method is called from inside the asynchronous operators and directly put the output of
	// an asynchronous instruction into the lineage cache. As the consumers, a different operator,
	// materializes the intermediate, we skip the placeholder placing logic.
	public static void putValueAsyncOp(LineageItem instLI, Data data, MatrixBlock mb, long starttime)
	{
		if (ReuseCacheType.isNone())
			return;
		if (!ArrayUtils.contains(LineageCacheConfig.getReusableOpcodes(), instLI.getOpcode()))
			return;
		if(!(data instanceof MatrixObject) && !(data instanceof ScalarObject)) {
			return;
		}
		// No async. OP reuse if multi-backend reuse is disabled
		if (!LineageCacheConfig.isMultiBackendReuse())
			return;

		synchronized( _cache )
		{
			// If prefetching a persisted rdd, reduce the score of the persisted rdd by one hit count
			if (instLI.getOpcode().equals(Opcodes.PREFETCH.toString()) && probeRDDDistributed(instLI.getInputs()[0])) {
				LineageCacheEntry e = _cache.get(instLI.getInputs()[0]);
				if (e.getRDDObject().getNumReferences() < 1) //no other rdd consumer
					e.updateScore(false);
			}

			long computetime = System.nanoTime() - starttime;
			// Make space, place data and manage queue
			putIntern(instLI, DataType.MATRIX, mb, null, computetime);

			if (DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(instLI))
				// Add to missed compute time
				LineageCacheStatistics.incrementMissedComputeTime(computetime);
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
			if (boundLI != null) boundLI.resetVisitStatusNR();
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
			LineageSparkCacheEviction.resetEviction();
		}
	}
	
	public static Map<LineageItem, LineageCacheEntry> getLineageCache() {
		return _cache;
	}

	
	//----------------- INTERNAL CACHE LOGIC IMPLEMENTATION --------------//

	private static void putInternPlaceholder(Instruction inst, LineageItem key) {
		ComputationCPInstruction cinst = inst instanceof ComputationCPInstruction ? (ComputationCPInstruction)inst : null;
		ComputationFEDInstruction cfinst = inst instanceof ComputationFEDInstruction ? (ComputationFEDInstruction)inst : null;
		ComputationSPInstruction cspinst = inst instanceof ComputationSPInstruction ? (ComputationSPInstruction)inst : null;
		GPUInstruction gpuinst = inst instanceof GPUInstruction ? (GPUInstruction)inst : null;

		if (cinst != null)
			putIntern(key, cinst.output.getDataType(), null, null,  0);
		else if (cfinst != null)
			putIntern(key, cfinst.output.getDataType(), null, null,  0);
		else if (cspinst != null)
			putIntern(key, cspinst.output.getDataType(), null, null,  0);
		else if (gpuinst != null)
			putIntern(key, gpuinst._output.getDataType(), null, null,  0);
		//FIXME: different o/p datatypes for MultiReturnBuiltins.
	}

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
	}
	
	private static LineageCacheEntry getIntern(LineageItem key) {
		LineageCacheEntry e = _cache.get(key);
		if (e == null) {
			if(DMLScript.STATISTICS && LineageCacheEviction._removelist.containsKey(key))
				// The sought entry was in cache but removed later
				LineageCacheStatistics.incrementDelHits();
			return null;
		}

		if (e.getCacheStatus() != LineageCacheStatus.SPILLED) {
			if (DMLScript.STATISTICS)
				// Increment hit count.
				LineageCacheStatistics.incrementMemHits();

			// Maintain order for eviction
			if (e.isRDDPersist())
				LineageSparkCacheEviction.maintainOrder(e);
			else if (!e.isGPUObject())
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
		// Entries with RDDs are cached twice. First hit is GCed,
		// Second hit saves the child RDDs
		if (LineageCache.probe(probeItem)) {
			//LineageCacheEntry oe = getIntern(probeItem);
			LineageCacheEntry oe = _cache.get(probeItem);
			LineageCacheEntry e = _cache.get(item);
			boolean exists = !e.isNullVal();
			e.copyValueFrom(oe, computetime);
			//if (e.isNullVal())
			//	throw new DMLRuntimeException("Lineage Cache: Original item is empty: "+oe._key);

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
			
			// Maintain order for eviction
			if (!e.isRDDPersist() && !e.isGPUObject())
				LineageCacheEviction.addEntry(e);
			// TODO: Handling of func/SB cache entries for Spark and GPU
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

	private static boolean persistRDD(Instruction inst, LineageCacheEntry centry, ExecutionContext ec) {
		// If already persisted, change the status and return true.
		// Else, persist, change cache status and return false.
		if (probeRDDDistributed(centry)) {
			// Update status to indicate persisted in the executors
			centry.setCacheStatus(LineageCacheStatus.PERSISTEDRDD);
			return true;
		}
		CacheableData<?> cd = ec.getCacheableData(((ComputationSPInstruction)inst).output.getName());
		// Estimate worst case dense size
		long estimatedSize = MatrixBlock.estimateSizeInMemory(cd.getDataCharacteristics());
		// Skip if the entry is bigger than the total storage.
		if (estimatedSize > LineageSparkCacheEviction.getSparkStorageLimit())
			return false;
		// Mark for distributed caching and change status
		persistRDDIntern(centry, estimatedSize);
		centry.setCacheStatus(LineageCacheStatus.PERSISTEDRDD);
		//centry.getRDDObject().getRDD().count(); //eager caching (experimental)
		return false;
	}

	@SuppressWarnings("unused")
	private static boolean persistRDD(LineageCacheEntry centry, long estimatedSize) {
		// If already persisted, change the status and return true.
		// Else, persist, change cache status and return false.
		if (probeRDDDistributed(centry)) {
			// Update status to indicate persisted in the executors
			centry.setCacheStatus(LineageCacheStatus.PERSISTEDRDD);
			return true;
		}
		// Mark for distributed caching and change status
		persistRDDIntern(centry, estimatedSize);
		centry.setCacheStatus(LineageCacheStatus.PERSISTEDRDD);
		return false;
	}

	private static void persistRDDIntern(LineageCacheEntry centry, long estimatedSize) {
		// Mark the rdd for lazy checkpointing
		RDDObject rddObj = centry.getRDDObject();
		JavaPairRDD<?,?> rdd = rddObj.getRDD();
		rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK());
		//cut-off RDD lineage & broadcasts to prevent errors on
		// task closure serialization with destroyed broadcasts
		//rdd.checkpoint();
		rdd.rdd().localCheckpoint();
		rddObj.setRDD(rdd);
		rddObj.setCheckpointRDD(true);
		
		// Make space based on the estimated size
		if(!LineageSparkCacheEviction.isBelowThreshold(estimatedSize))
			LineageSparkCacheEviction.makeSpace(_cache, estimatedSize);
		LineageSparkCacheEviction.updateSize(estimatedSize, true);
		// Maintain order for eviction
		LineageSparkCacheEviction.addEntry(centry, estimatedSize);

		// Count number of RDDs marked for caching at the executors
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementRDDPersists();
	}

	@Deprecated
	@SuppressWarnings("unused")
	private static double getRecomputeEstimate(Instruction inst, ExecutionContext ec) {
		if (!((ComputationCPInstruction)inst).output.isMatrix()
			|| (((ComputationCPInstruction)inst).input1 != null && !((ComputationCPInstruction)inst).input1.isMatrix()))
			return 0; //this method will be deprecated. No need to support scalar

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		double nflops = 0;
		String instop= inst.getOpcode().contains(Opcodes.SPOOF.toString()) ? "spoof" : inst.getOpcode();
		CPType cptype = Opcodes.getCPTypeByOpcode(instop);
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
				if (inst.getOpcode().equalsIgnoreCase(Opcodes.MULT.toString()) || inst.getOpcode().equalsIgnoreCase(Opcodes.DIV.toString()))
					// considering the dimensions of inputs and the output are same 
					nflops = r1 * c1; 
				else if (inst.getOpcode().equalsIgnoreCase(Opcodes.SOLVE.toString()))
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
				if (opcode.equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
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


	//----------------- UTILITY FUNCTIONS --------------------//

	private static List<MutablePair<LineageItem, LineageCacheEntry>> getLineageItems(Instruction inst, ExecutionContext ec) {
		ComputationCPInstruction cinst = inst instanceof ComputationCPInstruction ? (ComputationCPInstruction)inst : null;
		ComputationFEDInstruction cfinst = inst instanceof ComputationFEDInstruction ? (ComputationFEDInstruction)inst : null;
		ComputationSPInstruction cspinst = inst instanceof ComputationSPInstruction ? (ComputationSPInstruction)inst : null;
		GPUInstruction gpuinst = inst instanceof GPUInstruction ? (GPUInstruction)inst : null;
		//TODO: Replace with generic type

		List<MutablePair<LineageItem, LineageCacheEntry>> liList = null;
		//FIXME: Replace getLineageItem with get/getOrCreate to avoid creating a new LI object
		LineageItem instLI = (cinst != null) ? ec.getLineageItem(cinst.output)
			: (cfinst != null) ? ec.getLineageItem(cfinst.output)
			: (cspinst != null) ? ec.getLineageItem(cspinst.output)
			: ec.getLineageItem(gpuinst._output);
		/*LineageItem instLI = (cinst != null) ? cinst.getLineageItem(ec).getValue()
			: (cfinst != null) ? cfinst.getLineageItem(ec).getValue()
			: (cspinst != null) ? cspinst.getLineageItem(ec).getValue()
			: gpuinst.getLineageItem(ec).getValue();*/
		if (inst instanceof MultiReturnBuiltinCPInstruction) {
			liList = new ArrayList<>();
			MultiReturnBuiltinCPInstruction mrInst = (MultiReturnBuiltinCPInstruction)inst;
			for (int i=0; i<mrInst.getNumOutputs(); i++) {
				String opcode = instLI.getOpcode() + String.valueOf(i);
				liList.add(MutablePair.of(new LineageItem(opcode, instLI.getInputs()), null));
			}
		}
		else
			liList = List.of(MutablePair.of(instLI, null));

		return liList;
	}

	private static String getOutputName(Instruction inst, LineageItem li) {
		ComputationCPInstruction cinst = inst instanceof ComputationCPInstruction ? (ComputationCPInstruction)inst : null;
		ComputationFEDInstruction cfinst = inst instanceof ComputationFEDInstruction ? (ComputationFEDInstruction)inst : null;
		ComputationSPInstruction cspinst = inst instanceof ComputationSPInstruction ? (ComputationSPInstruction)inst : null;
		GPUInstruction gpuinst = inst instanceof GPUInstruction ? (GPUInstruction)inst : null;

		String outName = null;
		if (inst instanceof MultiReturnBuiltinCPInstruction)
			outName = ((MultiReturnBuiltinCPInstruction)inst).
				getOutput(li.getOpcode().charAt(li.getOpcode().length()-1)-'0').getName();
		else if (inst instanceof ComputationCPInstruction)
			outName = cinst.output.getName();
		else if (inst instanceof ComputationFEDInstruction)
			outName = cfinst.output.getName();
		else if (inst instanceof ComputationSPInstruction)
			outName = cspinst.output.getName();
		else if (inst instanceof GPUInstruction)
			outName = gpuinst._output.getName();

		return outName;
	}

	@SuppressWarnings("unused")
	private static boolean allInputsSpark(Instruction inst, ExecutionContext ec) {
		CPOperand in1 = ((ComputationSPInstruction)inst).input1;
		CPOperand in2 = ((ComputationSPInstruction)inst).input2;
		CPOperand in3 = ((ComputationSPInstruction)inst).input3;

		// All inputs must be matrices
		if ((in1 != null && !in1.isMatrix()) || (in2 != null && !in2.isMatrix()) || (in3 != null && !in3.isMatrix()))
			return false;

		// Filter out if any input is local
		if (in1 != null && (!ec.getMatrixObject(in1.getName()).hasRDDHandle() ||
			ec.getMatrixObject(in1.getName()).hasBroadcastHandle()))
			return false;
		if (in2 != null && (!ec.getMatrixObject(in2.getName()).hasRDDHandle() ||
			ec.getMatrixObject(in2.getName()).hasBroadcastHandle()))
			return false;
		if (in3 != null && (!ec.getMatrixObject(in3.getName()).hasRDDHandle() ||
			ec.getMatrixObject(in3.getName()).hasBroadcastHandle()))
			return false;

		return true;
	}

	private static void maintainReuseStatistics(ExecutionContext ec, Instruction inst, LineageCacheEntry e) {
		if (!DMLScript.STATISTICS)
			return;

		LineageCacheStatistics.incrementSavedComputeTime(e._computeTime);
		if (e.isGPUObject()) LineageCacheStatistics.incrementGpuHits();
		if (inst.getOpcode().equals(Opcodes.PREFETCH.toString()) && DMLScript.USE_ACCELERATOR)
			LineageCacheStatistics.incrementGpuPrefetch();
		if (e.isRDDPersist()) {
			if (SparkExecutionContext.isRDDCached(e.getRDDObject().getRDD().id()))
				LineageCacheStatistics.incrementRDDPersistHits(); //persisted in the executors
			else
				LineageCacheStatistics.incrementRDDHits();  //only locally cached
		}
		if (e.isMatrixValue() || e.isScalarValue()) {
			if (inst instanceof ComputationSPInstruction
				|| (inst.getOpcode().equals(Opcodes.PREFETCH.toString()) && !DMLScript.USE_ACCELERATOR))
				// Single_block Spark instructions (sync/async) and prefetch
				LineageCacheStatistics.incrementSparkCollectHits();
			else
				LineageCacheStatistics.incrementInstHits();
		}
	}
}
