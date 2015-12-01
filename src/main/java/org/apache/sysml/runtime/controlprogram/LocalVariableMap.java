/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.controlprogram;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.spark.data.LineageObject;

import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Replaces <code>HashMap&lang;String, Data&rang;</code> as the table of
 * variable names and references.  No longer supports global consistency.
 * 
 */
public class LocalVariableMap implements Cloneable
{	
	private static String eol = System.getProperty ("line.separator");
	private static String ELEMENT_DELIM = org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter.ELEMENT_DELIM;
	private static IDSequence _seq = new IDSequence();
	
	private HashMap <String, Data> localMap = null;
	private final long localID;
	
	public LocalVariableMap()
	{
		localMap = new HashMap <String, Data>();
		localID = _seq.getNextID();
	}
	
	public LocalVariableMap(LocalVariableMap vars)
	{
		localMap = new HashMap <String, Data>(vars.localMap);
		localID = _seq.getNextID();
	}
	
	/**
	 * 
	 * @return
	 */
	public Set<String> keySet()
	{
		return localMap.keySet();
	}
	
	/**
	 * Retrieves the data object given its name.
	 * 
	 * @param name : the variable name for the data object
	 * @return the direct reference to the data object
	 */
	public Data get( String name )
	{
		return localMap.get( name );
	}
	
	/**
	 * Adds a new (name, value) pair to the variable map, or replaces an old pair with
	 * the same name.  Several different variable names may refer to the same value.
	 * 
	 * @param name : the variable name for the data value
	 * @param val  : the data value object (such as envelope)
	 */
	public void put(String name, Data val)
	{
		localMap.put( name, val );
	}

	/**
	 * 
	 * @param vars
	 */
	public void putAll( LocalVariableMap vars )
	{
		if( vars == this || vars == null )
			return;
		localMap.putAll (vars.localMap);
	}
	
	/**
	 * 
	 * @param name
	 */
	public Data remove( String name )
	{
		return localMap.remove( name );
	}
	
	/**
	 * 
	 */
	public void removeAll()
	{
		localMap.clear();
	}
	
	/**
	 * 
	 * @param d
	 * @return
	 */
	public boolean hasReferences( Data d )
	{
		return localMap.containsValue(d);
	}

	/**
	 * 
	 * @param bo
	 * @return
	 */
	public boolean hasReferences( LineageObject bo )
	{
		for( Data tmpdat : localMap.values() ) 
			if ( tmpdat instanceof MatrixObject ) {
				MatrixObject mo = (MatrixObject)tmpdat; 
				if( mo.getBroadcastHandle()==bo || mo.getRDDHandle()==bo )
					return true;
			}
		return false;
	}
		
	/**
	 * 
	 * @param d
	 * @param earlyAbort
	 * @return
	 */
	public int getNumReferences( Data d, boolean earlyAbort )
	{
		if ( d == null )
			return 0;
		
		int refCount = 0;		
		for( Data tmpdat : localMap.values() ) 
			if ( tmpdat == d ) 
				if( ++refCount > 1 && earlyAbort )
					return refCount;
	
		return refCount;		
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public String serialize() 
		throws DMLRuntimeException
	{
		StringBuilder sb = new StringBuilder();
		
		int count = 0;
		for (Entry <String, Data> e : localMap.entrySet ())
		{
			if (count != 0)
				sb.append (ELEMENT_DELIM);
			sb.append (ProgramConverter.serializeDataObject (e.getKey(), e.getValue()));
			count++;
		}
		
		return sb.toString();		
	}
	
	/**
	 * 
	 * @param varStr
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static LocalVariableMap deserialize(String varStr) 
		throws DMLRuntimeException
	{
		StringTokenizer st2 = new StringTokenizer (varStr, ELEMENT_DELIM );
		LocalVariableMap vars = new LocalVariableMap ();
		while( st2.hasMoreTokens() )
		{
			String tmp = st2.nextToken().trim();
			Object[] tmp2 = ProgramConverter.parseDataObject (tmp);
			vars.put ((String) tmp2 [0], (Data) tmp2 [1]);
		}
		return vars;		
	}

	@Override
	public String toString()
	{
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
	public Object clone()
	{
		return new LocalVariableMap( this );
	}
}
