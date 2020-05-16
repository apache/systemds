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
import org.apache.sysds.runtime.instructions.cp.ListIndexingCPInstruction;

import java.util.ArrayList;

public class LineageCacheConfig {
	
	private static final String[] REUSE_OPCODES = new String[] {
		"tsmm", "ba+*", "*", "/", "+", "nrow", "ncol",
		"rightIndex", "leftIndex", "groupedagg", "r'", "solve", "spoof"
	};
	
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
	
	public enum CachedItemHead {
		TSMM,
		ALL
	}
	
	public enum CachedItemTail {
		CBIND,
		RBIND,
		INDEX,
		ALL
	}
	
	 public enum LineageCacheStatus {
		EMPTY,		//Placeholder with no data. Cannot be evicted.
		CACHED,		//General cached data. Can be evicted.
		EVICTED,	//Data is in disk. Empty value. Cannot be evicted.
		RELOADED,	//Reloaded from disk. Can be evicted.
		PINNED;		//Pinned to memory. Cannot be evicted.
		public boolean canEvict() {
			return this == CACHED || this == RELOADED;
		}
	 }
	 
	 public enum LineageCachePolicy {
		 LRU,
		 WEIGHTED;
		 public boolean isLRUcache() {
			 return this == LRU;
		 }
	 }
	
	public ArrayList<String> _MMult = new ArrayList<>();
	public static boolean _allowSpill = true;
	// Minimum reliable spilling estimate in milliseconds.
	public static final double MIN_SPILL_TIME_ESTIMATE = 100;
	// Minimum reliable data size for spilling estimate in MB.
	public static final double MIN_SPILL_DATA = 20;

	// Default I/O in MB per second for binary blocks
	public static double FSREAD_DENSE = 200;
	public static double FSREAD_SPARSE = 100;
	public static double FSWRITE_DENSE = 150;
	public static double FSWRITE_SPARSE = 75;

	private static ReuseCacheType _cacheType = null;
	private static CachedItemHead _itemH = null;
	private static CachedItemTail _itemT = null;
	private static LineageCachePolicy _cachepolicy = null;
	private static boolean _compilerAssistedRW = true;

	static {
		//setup static configuration parameters
		setSpill(true); //enable/disable disk spilling.
		setCachePolicy(LineageCachePolicy.WEIGHTED);
	}
	
	public static boolean isReusable (Instruction inst, ExecutionContext ec) {
		boolean insttype = inst instanceof ComputationCPInstruction 
			&& !(inst instanceof ListIndexingCPInstruction);
		boolean rightop = (ArrayUtils.contains(REUSE_OPCODES, inst.getOpcode())
			|| (inst.getOpcode().equals("append") && isVectorAppend(inst, ec)));
		return insttype && rightop;
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
	
	public static void setSpill(boolean toSpill) {
		_allowSpill = toSpill;
	}
	
	public static void setCachePolicy(LineageCachePolicy policy) {
		_cachepolicy = policy;
	}
	
	public static boolean isSetSpill() {
		return _allowSpill;
	}
	
	public static LineageCachePolicy getCachePolicy() {
		return _cachepolicy;
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

}
