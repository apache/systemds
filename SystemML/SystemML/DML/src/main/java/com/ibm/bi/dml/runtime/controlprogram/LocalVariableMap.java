/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static String eol = System.getProperty ("line.separator");
	private static String ELEMENT_DELIM = com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter.ELEMENT_DELIM;
	private static IDSequence _seq = new IDSequence();
	
	private HashMap <String, Data> localMap = null;
	private final long localID;
	
	public LocalVariableMap ()
	{
		localMap = new HashMap <String, Data> ();
		localID = _seq.getNextID();
	}
	
	public Set<String> keySet ()
	{
		return localMap.keySet ();
	}
	
	/**
	 * Retrieves the data object given its name.
	 * 
	 * @param name : the variable name for the data object
	 * @return the direct reference to the data object
	 */
	public Data get (String name)
	{
		return localMap.get (name);
	}
	
	/**
	 * Adds a new (name, value) pair to the variable map, or replaces an old pair with
	 * the same name.  Several different variable names may refer to the same value.
	 * 
	 * @param name : the variable name for the data value
	 * @param val  : the data value object (such as envelope)
	 */
	public void put (String name, Data val)
	{
		localMap.put (name, val);
	}

	public void putAll (LocalVariableMap vars)
	{
		if (vars == this || vars == null)
			return;
		localMap.putAll (vars.localMap);
	}
	
	public void remove (String name)
	{
		localMap.remove (name);
	}
	
	public void removeAll()
	{
		localMap.clear();
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
		for( Entry<String, Data> e : localMap.entrySet() ) 
			if ( e.getValue().equals(d) ) 
				if( ++refCount > 1 && earlyAbort )
					return refCount;
		
		return refCount;
	}
	
	public String serialize () throws DMLRuntimeException
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
		
		return sb.toString ();		
	}
	
	public static LocalVariableMap deserialize (String varStr) throws DMLRuntimeException
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
	public String toString ()
	{
		String output = "Local Variable Map ID = \"" + localID + "\":" + eol;
		for (Entry <String, Data> pair : localMap.entrySet())
		    output += "  " + pair.getKey() + " = " + pair.getValue() + eol;
		return output;
	}
		
	@Override
	public Object clone()
	{
		LocalVariableMap newMap = new LocalVariableMap ();
		newMap.putAll (this);
		return newMap;
	}
}
