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
import java.util.ArrayList;

public class LineageCacheConfig {
	public enum CacheType {
		FULL,   // no rewrites
		PARTIAL, 
		HYBRID_FULL_PARTIAL,
		NONE;
		public boolean isFullReuse() {
			return this == FULL || this == HYBRID_FULL_PARTIAL;
		}
		public boolean isPartialReuse() {
			return this == PARTIAL || this == HYBRID_FULL_PARTIAL;
		}
	}
	
	public ArrayList<String> _MMult = new ArrayList<String>();
	
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

	private static CacheType _cacheType = null;
	private static CachedItemHead _itemH = null;
	private static CachedItemTail _itemT = null;
	
	public static void setConfigTsmmCbind(CacheType ct) {
		_cacheType = ct;
		_itemH = CachedItemHead.TSMM;
		_itemT = CachedItemTail.CBIND;
	}
	
	public static void setConfig(CacheType ct, CachedItemHead ith, CachedItemTail itt) {
		_cacheType = ct;
		_itemH = ith;
		_itemT = itt;
	}
	
	public static void shutdownReuse() {
		DMLScript.LINEAGE = false;
		DMLScript.LINEAGE_REUSE = false;
	}

	public static void restartReuse() {
		DMLScript.LINEAGE = true;
		DMLScript.LINEAGE_REUSE = true;
	}
	
	public static CacheType getCacheType() {
		return _cacheType;
	}

	public static CachedItemHead getCachedItemHead() {
		return _itemH;
	}

	public static CachedItemTail getCachedItemTail() {
		return _itemT;
	}
}
