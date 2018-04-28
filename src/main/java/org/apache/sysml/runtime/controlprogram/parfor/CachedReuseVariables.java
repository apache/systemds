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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.lang.ref.SoftReference;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.apache.spark.broadcast.Broadcast;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.instructions.cp.Data;

public class CachedReuseVariables
{
	private final HashMap<Long, SoftReference<LocalVariableMap>> _data;
	
	public CachedReuseVariables() {
		_data = new HashMap<>();
	}

	public synchronized boolean containsVariables(long pfid) {
		return _data.containsKey(pfid);
	}
	
	public synchronized void reuseVariables(long pfid, LocalVariableMap vars, Collection<String> blacklist, Map<String, Broadcast<CacheBlock>> _brInputs) {

		//fetch the broadcast variables
		if (ParForProgramBlock.ALLOW_BROADCAST_INPUTS && !containsVariables(pfid)) {
			loadBroadcastVariables(vars, _brInputs);
		}

		//check for existing reuse map
		LocalVariableMap tmp = null;
		if( _data.containsKey(pfid) )
			tmp = _data.get(pfid).get();
		
		//build reuse map if not created yet or evicted
		if( tmp == null ) {
			tmp = new LocalVariableMap(vars);
			tmp.removeAllIn((blacklist instanceof HashSet) ?
				(HashSet<String>)blacklist : new HashSet<>(blacklist));
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
	private void loadBroadcastVariables(LocalVariableMap variables, Map<String, Broadcast<CacheBlock>> _brInputs) {
		for (String key : variables.keySet()) {
			if (!_brInputs.containsKey(key)) {
				continue;
			}
			Data d = variables.get(key);
			CacheableData<CacheBlock> cdcb = (CacheableData<CacheBlock>) d;
			CacheBlock cb = _brInputs.get(key).getValue();
			cdcb.acquireModify(cb);
			cdcb.setEmptyStatus();// set status from modified to empty
			cdcb.refreshMetaData();
		}
	}
}
