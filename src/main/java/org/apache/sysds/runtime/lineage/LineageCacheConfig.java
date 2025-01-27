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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixMatrixCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryScalarScalarCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FrameIndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ListIndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MatrixIndexingCPInstruction;
import org.apache.sysds.runtime.instructions.fed.ComputationFEDInstruction;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysds.runtime.instructions.spark.ComputationSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CpmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmSPInstruction;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Stream;

public class LineageCacheConfig 
{
	//-------------CACHING LOGIC RELATED CONFIGURATIONS--------------//

	private static final String[] OPCODES = new String[] {
		Opcodes.TSMM.toString(), Opcodes.MMULT.toString(), Opcodes.MULT.toString(), Opcodes.DIV.toString(), Opcodes.PLUS.toString(), Opcodes.OR.toString(), Opcodes.NROW.toString(), Opcodes.NCOL.toString(), Opcodes.ROUND.toString(), Opcodes.EXP.toString(), Opcodes.LOG.toString(),
		"rightIndex", "leftIndex", Opcodes.GROUPEDAGG.toString(), Opcodes.TRANSPOSE.toString(), Opcodes.SOLVE.toString(), Opcodes.SPOOF.toString(), Opcodes.ISNA.toString(),
		Opcodes.UAMEAN.toString(), Opcodes.MAX.toString(), Opcodes.MIN.toString(), Opcodes.IFELSE.toString(), Opcodes.MINUS.toString(), Opcodes.SQRT.toString(), Opcodes.LESS.toString(), Opcodes.GREATER.toString(), Opcodes.UAKP.toString(), Opcodes.LESSEQUAL.toString(),
		Opcodes.POW.toString(), Opcodes.UAMAX.toString(), Opcodes.UARKP.toString(), Opcodes.UACMEAN.toString(), Opcodes.EIGEN.toString(),Opcodes.CTABLE.toString(), Opcodes.CTABLEEXPAND.toString(), Opcodes.REPLACE.toString(),
		Opcodes.POW2.toString(), Opcodes.MULT2.toString(), Opcodes.UACKP.toString(), Opcodes.TAKPM.toString(), Opcodes.UACSQKP.toString(), Opcodes.UARKP.toString(), Opcodes.NP.toString(), Opcodes.UARIMAX.toString(), Opcodes.QSORT.toString(),
		Opcodes.QPICK.toString(), Opcodes.TRANSFORMAPPLY.toString(), Opcodes.UARMAX.toString(), Opcodes.NP.toString(), Opcodes.MINUSMULT.toString(), "castdtm", Opcodes.LOWERTRI.toString(), Opcodes.MINUS1_MULT.toString(),
		Opcodes.PREFETCH.toString(), "mapmm", Opcodes.CONTAINS.toString(), Opcodes.MMCHAIN.toString(), "mapmmchain", Opcodes.PM.toString(), Opcodes.EQUAL.toString(), Opcodes.RMEMPTY.toString(),
		Opcodes.CONV2D_BIAS_ADD.toString(), Opcodes.RELU_MAXPOOLING.toString(), Opcodes.MAXPOOLING.toString(), Opcodes.BATCH_NORM2D.toString(), Opcodes.AVGPOOLING.toString(),
		"softmax"
		//TODO: Reuse everything.
	};

	// Relatively expensive instructions. Most include shuffles.
	private static final String[] PERSIST_OPCODES1 = new String[] {
		"cpmm", "rmm", Opcodes.PMM.toString(), "zipmm", Opcodes.REV.toString(), Opcodes.ROLL.toString(), Opcodes.RESHAPE.toString(), Opcodes.SORT.toString(), Opcodes.MINUS.toString(), Opcodes.MULT.toString(), Opcodes.PLUS.toString(),
		Opcodes.DIV.toString(), Opcodes.MODULUS.toString(), Opcodes.INTDIV.toString(), Opcodes.MINUS1_MULT.toString(), Opcodes.POW.toString(), Opcodes.POW2.toString(), Opcodes.MULT2.toString(), Opcodes.EQUAL.toString(), Opcodes.NOTEQUAL.toString(), Opcodes.LESS.toString(), Opcodes.GREATER.toString(),
		Opcodes.LESSEQUAL.toString(), Opcodes.GREATEREQUAL.toString(), Opcodes.AND.toString(), Opcodes.OR.toString(), Opcodes.XOR.toString(), Opcodes.MAX.toString(), Opcodes.MIN.toString(), Opcodes.RMEMPTY.toString(), "rappend",
		"gappend", "galignedappend", Opcodes.RBIND.toString(), Opcodes.CBIND.toString(), Opcodes.NMIN.toString(), Opcodes.NMAX.toString(),
		Opcodes.NP.toString(), Opcodes.CTABLE.toString(), "ucumack+", "ucumac*", "ucumacmin", "ucumacmax",
		Opcodes.QSORT.toString(), Opcodes.QPICK.toString()
	};

	// Relatively inexpensive instructions.
	private static final String[] PERSIST_OPCODES2 = new String[] {
		"mapmm", "isna", "leftIndex"
	};

	private static final String[] GPU_OPCODE_HEAVY = new String[] {
		Opcodes.CONV2D_BIAS_ADD.toString(), Opcodes.RELU_MAXPOOLING.toString(), Opcodes.MAXPOOLING.toString(), Opcodes.BATCH_NORM2D.toString(), Opcodes.AVGPOOLING.toString()  //DNN OPs
	};

	private static String[] REUSE_OPCODES  = new String[] {};
	private static String[] CHKPOINT_OPCODES  = new String[] {};

	public enum ReuseCacheType {
		REUSE_FULL,
		REUSE_PARTIAL,
		REUSE_MULTILEVEL,
		REUSE_HYBRID,
		NONE;
		public boolean isFullReuse() {
			return this == REUSE_FULL || this == REUSE_MULTILEVEL || this == REUSE_HYBRID;
		}
		public boolean isPartialReuse() {
			return this == REUSE_PARTIAL || this == REUSE_HYBRID;
		}
		public boolean isMultilevelReuse() {
			return this == REUSE_MULTILEVEL || this == REUSE_HYBRID;
		}
		public static boolean isNone() {
			return DMLScript.LINEAGE_REUSE == null
				|| DMLScript.LINEAGE_REUSE == NONE;
		}
	}

	protected static final double CPU_CACHE_FRAC = 0.05; // 5% of JVM heap size
	private static ReuseCacheType _cacheType = null;
	@SuppressWarnings("unused")
	private static CachedItemHead _itemH = null;
	@SuppressWarnings("unused")
	private static CachedItemTail _itemT = null;
	private static boolean _compilerAssistedRW = false;
	private static boolean _onlyEstimate = false;
	private static boolean _reuseLineageTraces = true;
	private static boolean DELAYED_CACHING = false;

	// Delayed caching may lead to deletion and cache misses in GPU.
	// Once the GPU memory is full, the non-reusable intermediates deallocates/deletes the cached
	// entries from the free lists, leading to cache misses and high eviction overhead. Eager caching,
	// however places every intermediate in a free list, increasing recycling and reducing deletion.
	// Note, delayed caching helps in reducing lineage caching/probing overhead for use cases with
	// no reusable instructions, but is anti-productive for use cases with repeating patterns (eg. scoring).
	private static boolean DELAYED_CACHING_GPU = true;
	private static boolean DELAYED_CACHING_RDD = true;

	private static boolean MULTI_BACKEND_REUSE = true;

	//-------------DISK SPILLING RELATED CONFIGURATIONS--------------//

	//private static boolean _allowSpill = false;
	// Minimum reliable spilling estimate in milliseconds.
	public static final double MIN_SPILL_TIME_ESTIMATE = 10;
	// Minimum reliable data size for spilling estimate in MB.
	public static final double MIN_SPILL_DATA = 2;
	// Default I/O in MB per second for binary blocks
	// NOTE: These defaults are tuned according to high
	// speed disks, so that spilling starts early. These 
	// will anyway be adjusted as per the current disk.
	public static double FSREAD_DENSE = 500;
	public static double FSREAD_SPARSE = 400;
	public static double FSWRITE_DENSE = 450;
	public static double FSWRITE_SPARSE = 225;
	public static double D2HCOPYBANDWIDTH = 1500; //MB/sec
	public static double D2HMAXBANDWIDTH = 8192;
	
	private enum CachedItemHead {
		TSMM,
		ALL
	}
	
	private enum CachedItemTail {
		CBIND,
		RBIND,
		INDEX,
		ALL
	}

	//-------------EVICTION RELATED CONFIGURATIONS--------------//

	private static LineageCachePolicy _cachepolicy = null;
	// Weights for scoring components (computeTime/size, LRU timestamp, DAG height)
	protected static double[] WEIGHTS = {1, 0, 0};
	public static boolean GPU2HOSTEVICTION = false;

	protected enum LineageCacheStatus {
		EMPTY,     //Placeholder with no data. Cannot be evicted.
		NOTCACHED, //Placeholder removed from the cache
		TOCACHE,   //To be cached in memory if reoccur
		CACHED,    //General cached data. Can be evicted.
		SPILLED,   //Data is in disk. Empty value. Cannot be evicted.
		RELOADED,  //Reloaded from disk. Can be evicted.
		PINNED,    //Pinned to memory. Cannot be evicted.
		TOCACHEGPU, //To be cached in GPU if the instruction reoccur
		GPUCACHED, //Points to GPU intermediate
		PERSISTEDRDD, //Persisted at the Spark executors
		TOPERSISTRDD, //To be persisted if the instruction reoccur
		TOSPILL,   //To be spilled lazily 
		TODELETE;  //TO be removed lazily
		public boolean canEvict() {
			return this == CACHED || this == RELOADED;
		}
	}
	
	public enum LineageCachePolicy {
		LRU,
		COSTNSIZE,
		DAGHEIGHT,
	}
	
	protected static Comparator<LineageCacheEntry> LineageCacheComparator = (e1, e2) -> {
		/*return e1.score == e2.score ?
			Long.compare(e1._key.getId(), e2._key.getId()) :
			e1.score < e2.score ? -1 : 1;
		*/
		int ret = 0;
		if (e1.score == e2.score) {
			switch(_cachepolicy) {
				case LRU:
				case DAGHEIGHT:
				{
					// order entries with same score by cost, size ratio
					double e1_cs = e1.getCostNsize();
					double e2_cs = e2.getCostNsize();
					ret = e1_cs == e2_cs ?
						Long.compare(e1._key.getId(), e2._key.getId()) :
						e1_cs < e2_cs ? -1 : 1;
					break;
				}
				case COSTNSIZE:
				{
					// order entries with same score by last used time
					double e1_ts = e1.getTimestamp();
					double e2_ts = e2.getTimestamp();
					ret = e1_ts == e2_ts ?
						Long.compare(e1._key.getId(), e2._key.getId()) :
						e1_ts < e2_ts ? -1 : 1;
					break;
				}
			}
		}
		else
			ret = e1.score < e2.score ? -1 : 1;

		return ret;
	};

	protected static Comparator<LineageCacheEntry> LineageGPUCacheComparator = (e1, e2) -> {
		if (e1._key.getId() == e2._key.getId())
			return 0;
		if (e1.score == e2.score)
			return Long.compare(e1._key.getId(), e2._key.getId());
		else
			return e1.score < e2.score ? -1 : 1;
	};

	//-------------SPARK OPERATION RELATED CONFIGURATIONS--------------//

	protected static boolean ENABLE_LOCAL_ONLY_RDD_CACHING = false;

	//----------------------------------------------------------------//

	static {
		//setup static configuration parameters
		REUSE_OPCODES = OPCODES;
		CHKPOINT_OPCODES = Stream.concat(Arrays.stream(PERSIST_OPCODES1), Arrays.stream(PERSIST_OPCODES2))
			.toArray(String[]::new);
		//setSpill(true);
		setCachePolicy(LineageCachePolicy.COSTNSIZE);
		setCompAssRW(true);
	}

	public static void setReusableOpcodes(String... ops) {
		REUSE_OPCODES = ops;
	}

	public static String[] getReusableOpcodes() {
		return REUSE_OPCODES;
	}

	public static void resetReusableOpcodes() {
		REUSE_OPCODES = OPCODES;
	}

	public static boolean isReusable (Instruction inst, ExecutionContext ec) {
		boolean insttype = (inst instanceof ComputationCPInstruction 
			|| inst instanceof ComputationFEDInstruction
			|| inst instanceof GPUInstruction
			|| inst instanceof ComputationSPInstruction)
			&& !(inst instanceof ListIndexingCPInstruction)
			&& !(inst instanceof BinaryScalarScalarCPInstruction)
			&& !(inst instanceof FrameIndexingCPInstruction);
		boolean rightCPOp = (ArrayUtils.contains(REUSE_OPCODES, inst.getOpcode())
			|| (inst.getOpcode().equals(Opcodes.APPEND.toString()) && isVectorAppend(inst, ec))
			|| (inst.getOpcode().startsWith(Opcodes.SPOOF.toString()))
			|| (inst instanceof DataGenCPInstruction) && ((DataGenCPInstruction) inst).isMatrixCall());
		boolean rightSPOp = isReusableRDDType(inst);
		boolean updateInplace = (inst instanceof MatrixIndexingCPInstruction)
			&& ec.getMatrixObject(((ComputationCPInstruction)inst).input1).getUpdateType().isInPlace();
		updateInplace = updateInplace || ((inst instanceof BinaryMatrixMatrixCPInstruction)
			&& ((BinaryMatrixMatrixCPInstruction) inst).isInPlace());
		boolean federatedOutput = false;
		return insttype && (rightCPOp || rightSPOp) && !updateInplace && !federatedOutput;
	}
	
	private static boolean isVectorAppend(Instruction inst, ExecutionContext ec) {
		if (inst instanceof ComputationFEDInstruction) {
			ComputationFEDInstruction fedinst = (ComputationFEDInstruction) inst;
			if (!fedinst.input1.isMatrix() || !fedinst.input2.isMatrix())
				return false;
			long c1 = ec.getMatrixObject(fedinst.input1).getNumColumns();
			long c2 = ec.getMatrixObject(fedinst.input2).getNumColumns();
			return(c1 == 1 || c2 == 1);
		}
		else if (inst instanceof ComputationCPInstruction) { //CPInstruction
			ComputationCPInstruction cpinst = (ComputationCPInstruction) inst;
			if( !cpinst.input1.isMatrix() || !cpinst.input2.isMatrix() )
				return false;
			long c1 = ec.getMatrixObject(cpinst.input1).getNumColumns();
			long c2 = ec.getMatrixObject(cpinst.input2).getNumColumns();
			return(c1 == 1 || c2 == 1);
		}
		if (inst instanceof ComputationSPInstruction) {
			ComputationSPInstruction fedinst = (ComputationSPInstruction) inst;
			if (!fedinst.input1.isMatrix() || !fedinst.input2.isMatrix())
				return false;
			long c1 = ec.getMatrixObject(fedinst.input1).getNumColumns();
			long c2 = ec.getMatrixObject(fedinst.input2).getNumColumns();
			return(c1 == 1 || c2 == 1);
		}
		else { //GPUInstruction
			GPUInstruction gpuinst = (GPUInstruction)inst;
			if( !gpuinst._input1.isMatrix() || !gpuinst._input2.isMatrix() )
				return false;
			long c1 = ec.getMatrixObject(gpuinst._input1).getNumColumns();
			long c2 = ec.getMatrixObject(gpuinst._input2).getNumColumns();
			return(c1 == 1 || c2 == 1);
		}
	}

	protected static boolean isReusableRDDType(Instruction inst) {
		boolean insttype = inst instanceof ComputationSPInstruction;
		boolean rightOp = ArrayUtils.contains(CHKPOINT_OPCODES, inst.getOpcode());
		if (rightOp && inst instanceof MapmmSPInstruction
			&& ((MapmmSPInstruction) inst).getAggType() == AggBinaryOp.SparkAggType.SINGLE_BLOCK)
			rightOp = false;
		if (rightOp && inst instanceof CpmmSPInstruction
			&& ((CpmmSPInstruction) inst).getAggType() == AggBinaryOp.SparkAggType.SINGLE_BLOCK)
			rightOp = false;
		return insttype && rightOp;
	}

	protected static boolean isShuffleOp(String opcode) {
		return ArrayUtils.contains(PERSIST_OPCODES1, opcode);
	}

	protected static boolean isComputeGPUOps(String opcode) {
		return ArrayUtils.contains(GPU_OPCODE_HEAVY, opcode);
	}

	protected static int getComputeGroup(String opcode) {
		boolean heavy_hitter = ArrayUtils.contains(PERSIST_OPCODES1, opcode)
			|| ArrayUtils.contains(GPU_OPCODE_HEAVY, opcode);
		return heavy_hitter ? 2 : 1;
	}


	public static boolean isOutputFederated(Instruction inst, Data data) {
		if (!(inst instanceof ComputationFEDInstruction))
			return false;
		// return true if the output matrixobject is federated
		if (inst instanceof ComputationFEDInstruction)
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated())
				return true;
		return false;
	}
	
	public static void setConfigTsmmCbind(ReuseCacheType ct) {
		_cacheType = ct;
		_itemH = CachedItemHead.TSMM;
		_itemT = CachedItemTail.CBIND;
	}
	
	public static void setConfig(ReuseCacheType ct) {
		_cacheType = ct;
	}
	
	public static void setConfig(ReuseCacheType ct, CachedItemHead ith, CachedItemTail itt) {
		_cacheType = ct;
		_itemH = ith;
		_itemT = itt;
	}
	
	public static void setCompAssRW(boolean comp) {
		_compilerAssistedRW = comp;
	}
	
	public static void shutdownReuse() {
		DMLScript.LINEAGE = false;
		DMLScript.LINEAGE_REUSE = ReuseCacheType.NONE;
	}

	public static void restartReuse(ReuseCacheType rop) {
		DMLScript.LINEAGE = true;
		DMLScript.LINEAGE_REUSE = rop;
	}
	public static ReuseCacheType getCacheType() {
		return _cacheType;
	}
	
	public static boolean isMultiLevelReuse() {
		return !ReuseCacheType.isNone()
			&& _cacheType.isMultilevelReuse();
	}

	public static boolean getCompAssRW() {
		return _compilerAssistedRW;
	}

	public static void setReuseLineageTraces(boolean reuseTrace) {
		_reuseLineageTraces = reuseTrace;
	}

	public static boolean isLineageTraceReuse() {
		return _reuseLineageTraces;
	}

	public static boolean isMultiBackendReuse() {
		return MULTI_BACKEND_REUSE;
	}

	public static boolean isDelayedCaching() {
		return DELAYED_CACHING;
	}

	public static boolean isDelayedCachingGPU() {
		return DELAYED_CACHING_GPU;
	}

	public static boolean isDelayedCachingRDD() {
		return DELAYED_CACHING_RDD;
	}

	public static void setCachePolicy(LineageCachePolicy policy) {
		// TODO: Automatic tuning of weights.
		switch(policy) {
			case LRU:
				WEIGHTS[0] = 0; WEIGHTS[1] = 1; WEIGHTS[2] = 0;
				break;
			case COSTNSIZE:
				WEIGHTS[0] = 1; WEIGHTS[1] = 0; WEIGHTS[2] = 0;
				break;
			case DAGHEIGHT:
				WEIGHTS[0] = 0; WEIGHTS[1] = 0; WEIGHTS[2] = 1;
				break;
		}
		_cachepolicy = policy;
	}

	public static LineageCachePolicy getCachePolicy() {
		return _cachepolicy;
	}
	
	public static void setEstimator(boolean onlyEstimator) { 
		_onlyEstimate = onlyEstimator;
	}
	
	public static boolean isEstimator() {
		return _onlyEstimate;
	}
	
	public static boolean isTimeBased() {
		// Check the LRU component of weights array.
		return (WEIGHTS[1] > 0);
	}
	
	public static boolean isCostNsize() {
		return (WEIGHTS[0] > 0);
	}

	public static boolean isDagHeightBased() {
		// Check the DAGHEIGHT component of weights array.
		return (WEIGHTS[2] > 0);
	}

	/*public static void setSpill(boolean toSpill) {
		_allowSpill = toSpill;
		// NOTE: _allowSpill only enables/disables disk spilling, but has 
		// no control over eviction order of cached items.
	}*/
	
	public static boolean isSetSpill() {
		// Check if cachespill set in SystemDS-config (default true)
		DMLConfig conf = ConfigurationManager.getDMLConfig();
		return conf.getBooleanValue(DMLConfig.LINEAGECACHESPILL);
	}
}
