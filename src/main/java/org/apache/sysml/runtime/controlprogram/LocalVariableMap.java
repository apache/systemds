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

package org.apache.sysml.runtime.controlprogram;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.cp.Data;

/**
 * Replaces <code>HashMap&lang;String, Data&rang;</code> as the table of
 * variable names and references.  No longer supports global consistency.
 * 
 */
public class LocalVariableMap implements Cloneable
{
	private static final String eol = System.getProperty ("line.separator");
	private static final String ELEMENT_DELIM = ProgramConverter.ELEMENT_DELIM;
	private static final IDSequence _seq = new IDSequence();
	
	//variable map data and id
	private final HashMap<String, Data> localMap;
	private final long localID;
	
	//optional set of registered outputs
	private HashSet<String> outputs = null;
	
	public LocalVariableMap() {
		localMap = new HashMap<>();
		localID = _seq.getNextID();
	}
	
	public LocalVariableMap(LocalVariableMap vars) {
		localMap = new HashMap<>(vars.localMap);
		localID = _seq.getNextID();
	}

	public Set<String> keySet() {
		return localMap.keySet();
	}
	
	public Set<Entry<String, Data>> entrySet() {
		return localMap.entrySet();
	}
	
	/**
	 * Retrieves the data object given its name.
	 * 
	 * @param name the variable name for the data object
	 * @return the direct reference to the data object
	 */
	public Data get( String name ) {
		return localMap.get( name );
	}
	
	/**
	 * Adds a new (name, value) pair to the variable map, or replaces an old pair with
	 * the same name.  Several different variable names may refer to the same value.
	 * 
	 * @param name the variable name for the data value
	 * @param val the data value object (such as envelope)
	 */
	public void put(String name, Data val) {
		localMap.put( name, val );
	}
	
	public void putAll(Map<String, Data> vals) {
		localMap.putAll(vals);
	}

	public Data remove( String name ) {
		return localMap.remove( name );
	}

	public void removeAll() {
		localMap.clear();
	}
	
	public void removeAllIn(Set<String> blacklist) {
		localMap.entrySet().removeIf(
			e -> blacklist.contains(e.getKey()));
	}
	
	public void removeAllNotIn(Set<String> blacklist) {
		localMap.entrySet().removeIf(
			e -> !blacklist.contains(e.getKey()));
	}

	public boolean hasReferences( Data d ) {
		return localMap.containsValue(d);
	}
	
	public void setRegisteredOutputs(HashSet<String> outputs) {
		this.outputs = outputs;
	}
	
	public HashSet<String> getRegisteredOutputs() {
		return outputs;
	}

	public String serialize() throws DMLRuntimeException {
		StringBuilder sb = new StringBuilder();
		int count = 0;
		for (Entry <String, Data> e : localMap.entrySet ()) {
			if (count != 0)
				sb.append (ELEMENT_DELIM);
			sb.append(ProgramConverter
				.serializeDataObject(e.getKey(), e.getValue()));
			count++;
		}
		return sb.toString();
	}

	public static LocalVariableMap deserialize(String varStr) 
		throws DMLRuntimeException
	{
		StringTokenizer st2 = new StringTokenizer (varStr, ELEMENT_DELIM );
		LocalVariableMap vars = new LocalVariableMap ();
		while( st2.hasMoreTokens() ) {
			String tmp = st2.nextToken().trim();
			Object[] tmp2 = ProgramConverter.parseDataObject (tmp);
			vars.put((String) tmp2 [0], (Data) tmp2 [1]);
		}
		return vars;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Local Variable Map ID = \"");
		sb.append(localID);
		sb.append("\":");
		sb.append(eol);
		for (Entry <String, Data> pair : localMap.entrySet()) {
			sb.append("  ");
			sb.append(pair.getKey());
			sb.append(" = ");
			sb.append(pair.getValue());
			sb.append(eol);
		}
		return sb.toString();
	}
		
	@Override
	public Object clone() {
		return new LocalVariableMap(this);
	}
}
