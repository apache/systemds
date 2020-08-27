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
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ListIndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MatrixIndexingCPInstruction;

import java.util.Comparator;

public class LineageCacheConfig 
{
	//-------------CACHING LOGIC RELATED CONFIGURATIONS--------------//

	private static final String[] OPCODES = new String[] {
		"tsmm", "ba+*", "*", "/", "+", "||", "nrow", "ncol", "round", "exp", "log",
		"rightIndex", "leftIndex", "groupedagg", "r'", "solve", "spoof",
		"uamean", "max", "min", "ifelse", "-", "sqrt", ">", "uak+", "<=",
		"^", "uamax", "uark+", "uacmean", "eigen", "ctableexpand", "replace",
		"^2", "uack+", "tak+*", "uacsqk+", "uark+", "n+", "uarimax", "qsort", "qpick"
		//TODO: Reuse everything. 
	};
	private static String[] REUSE_OPCODES  = new String[] {};
	
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

	private static ReuseCacheType _cacheType = null;
	private static CachedItemHead _itemH = null;
	private static CachedItemTail _itemT = null;
	private static boolean _compilerAssistedRW = false;

	//-------------DISK SPILLING RELATED CONFIGURATIONS--------------//

	private static boolean _allowSpill = false;
	// Minimum reliable spilling estimate in milliseconds.
	public static final double MIN_SPILL_TIME_ESTIMATE = 100;
	// Minimum reliable data size for spilling estimate in MB.
	public static final double MIN_SPILL_DATA = 20;
	// Default I/O in MB per second for binary blocks
	public static double FSREAD_DENSE = 200;
	public static double FSREAD_SPARSE = 100;
	public static double FSWRITE_DENSE = 150;
	public static double FSWRITE_SPARSE = 75;
	
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
	// Weights for scoring components (computeTime/size, LRU timestamp)
	protected static double[] WEIGHTS = {0, 1};

	protected enum LineageCacheStatus {
		EMPTY,     //Placeholder with no data. Cannot be evicted.
		CACHED,    //General cached data. Can be evicted.
		SPILLED,   //Data is in disk. Empty value. Cannot be evicted.
		RELOADED,  //Reloaded from disk. Can be evicted.
		PINNED,    //Pinned to memory. Cannot be evicted.
		TOSPILL,   //To be spilled lazily 
		TODELETE;  //TO be removed lazily
		public boolean canEvict() {
			return this == CACHED || this == RELOADED;
		}
	}
	
	public enum LineageCachePolicy {
		LRU,
		COSTNSIZE,
		HYBRID;
	}
	
	protected static Comparator<LineageCacheEntry> LineageCacheComparator = (e1, e2) -> {
		return e1.score == e2.score ?
			Long.compare(e1._key.getId(), e2._key.getId()) :
			e1.score < e2.score ? -1 : 1;
	};

	//----------------------------------------------------------------//

	static {
		//setup static configuration parameters
		REUSE_OPCODES = OPCODES;
		setSpill(true); 
		setCachePolicy(LineageCachePolicy.HYBRID);
		setCompAssRW(true);
	}

	public static void setReusableOpcodes(String... ops) {
		REUSE_OPCODES = ops;
	}
	
	public static void resetReusableOpcodes() {
		REUSE_OPCODES = OPCODES;
	}

	public static boolean isReusable (Instruction inst, ExecutionContext ec) {
		boolean insttype = inst instanceof ComputationCPInstruction 
			&& !(inst instanceof ListIndexingCPInstruction);
		boolean rightop = (ArrayUtils.contains(REUSE_OPCODES, inst.getOpcode())
			|| (inst.getOpcode().equals("append") && isVectorAppend(inst, ec))
			|| (inst instanceof DataGenCPInstruction) && ((DataGenCPInstruction) inst).isMatrixCall());
		boolean updateInplace = (inst instanceof MatrixIndexingCPInstruction)
			&& ec.getMatrixObject(((ComputationCPInstruction)inst).input1).getUpdateType().isInPlace();
		return insttype && rightop && !updateInplace;
	}
	
	private static boolean isVectorAppend(Instruction inst, ExecutionContext ec) {
		ComputationCPInstruction cpinst = (ComputationCPInstruction) inst;
		if( !cpinst.input1.isMatrix() || !cpinst.input2.isMatrix() )
			return false;
		long c1 = ec.getMatrixObject(cpinst.input1).getNumColumns();
		long c2 = ec.getMatrixObject(cpinst.input2).getNumColumns();
		return(c1 == 1 || c2 == 1);
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

	public static CachedItemHead getCachedItemHead() {
		return _itemH;
	}

	public static CachedItemTail getCachedItemTail() {
		return _itemT;
	}
	
	public static boolean getCompAssRW() {
		return _compilerAssistedRW;
	}

	public static void setCachePolicy(LineageCachePolicy policy) {
		switch(policy) {
			case LRU:
				WEIGHTS[0] = 0; WEIGHTS[1] = 1;
				break;
			case COSTNSIZE:
				WEIGHTS[0] = 1; WEIGHTS[1] = 0;
				break;
			case HYBRID:
				WEIGHTS[0] = 1; WEIGHTS[1] = 0.0033;
				// FIXME: Relative timestamp fix reduces the absolute
				// value of the timestamp component of the scoring function
				// to a comparatively much smaller number. W[1] needs to be
				// re-tuned accordingly.
				// TODO: Automatic tuning of weights.
				break;
		}
		_cachepolicy = policy;
	}

	public static LineageCachePolicy getCachePolicy() {
		return _cachepolicy;
	}
	
	public static boolean isTimeBased() {
		// Check the LRU component of weights array.
		return (WEIGHTS[1] > 0);
	}

	public static void setSpill(boolean toSpill) {
		_allowSpill = toSpill;
		// NOTE: _allowSpill only enables/disables disk spilling, but has 
		// no control over eviction order of cached items.
	}
	
	public static boolean isSetSpill() {
		return _allowSpill;
	}
}
