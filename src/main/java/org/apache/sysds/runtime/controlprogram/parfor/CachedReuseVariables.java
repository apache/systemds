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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.lang.ref.SoftReference;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.spark.broadcast.Broadcast;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.cp.Data;

public class CachedReuseVariables
{
	private final HashMap<Long, SoftReference<LocalVariableMap>> _data;
	
	public CachedReuseVariables() {
		_data = new HashMap<>();
	}

	public synchronized boolean containsVars(long pfid) {
		return _data.containsKey(pfid);
	}
	
	@SuppressWarnings("unused")
	public synchronized void reuseVariables(long pfid, LocalVariableMap vars, Collection<String> excludeList, Map<String, Broadcast<CacheBlock>> _brInputs, boolean cleanCache) {

		//fetch the broadcast variables
		if (ParForProgramBlock.ALLOW_BROADCAST_INPUTS && !containsVars(pfid)) {
			loadBroadcastVariables(vars, _brInputs);
		}

		//check for existing reuse map
		LocalVariableMap tmp = null;
		if( containsVars(pfid) )
			tmp = _data.get(pfid).get();
		
		//build reuse map if not created yet or evicted
		if( tmp == null ) {
			if( cleanCache )
				_data.clear();
			tmp = new LocalVariableMap(vars);
			tmp.removeAllIn((excludeList instanceof HashSet) ?
				(HashSet<String>)excludeList : new HashSet<>(excludeList));
			_data.put(pfid, new SoftReference<>(tmp));
		}
		//reuse existing reuse map
		else {
			for( String varName : tmp.keySet() )
				vars.put(varName, tmp.get(varName));
		}
	}

	public synchronized void clearVariables(long pfid) {
		_data.remove(pfid);
	}

	@SuppressWarnings("unchecked")
	private static void loadBroadcastVariables(LocalVariableMap variables, Map<String, Broadcast<CacheBlock>> brInputs) {
		for( Entry<String, Broadcast<CacheBlock>> e : brInputs.entrySet() ) {
			Data d = variables.get(e.getKey());
			CacheableData<CacheBlock> cdcb = (CacheableData<CacheBlock>) d;
			cdcb.acquireModify(e.getValue().getValue());
			cdcb.setEmptyStatus(); // avoid eviction
			cdcb.refreshMetaData();
		}
	}
}
