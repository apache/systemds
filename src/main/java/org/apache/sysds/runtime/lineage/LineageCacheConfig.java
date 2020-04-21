/*
 * Copyright 2020 Graz University of Technology
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
import java.util.ArrayList;

public class LineageCacheConfig {
	
	public enum ReuseCacheType {
		REUSE_FULL,
		REUSE_PARTIAL,
		REUSE_HYBRID,
		NONE;
		public boolean isFullReuse() {
			return this == REUSE_FULL || this == REUSE_HYBRID;
		}
		public boolean isPartialReuse() {
			return this == REUSE_PARTIAL || this == REUSE_HYBRID;
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
	
	public ArrayList<String> _MMult = new ArrayList<>();
	public static boolean _allowSpill = true;

	private static ReuseCacheType _cacheType = null;
	private static CachedItemHead _itemH = null;
	private static CachedItemTail _itemT = null;
	private static boolean _compilerAssistedRW = true;
	static {
		//setup static configuration parameters
		setSpill(false); //disable spilling of cache entries to disk
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
	
	public static boolean isSetSpill() {
		return _allowSpill;
	}
	
	public static ReuseCacheType getCacheType() {
		return _cacheType;
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
