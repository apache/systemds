package com.ibm.bi.dml.runtime.controlprogram;

import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.utils.DMLRuntimeException;

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
	private static String ELEMENT_DELIM = com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter.ELEMENT_DELIM;
	private volatile int globalCount = 0;
	
	private HashMap <String, Data> localMap = null;
	private final String localID;
	
	public LocalVariableMap ()
	{
		localMap = new HashMap <String, Data> ();
		localID = getNewID ();
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
	
	public void removeAll ()
	{
		localMap = new HashMap <String, Data> ();
	}
	
	public String serialize () throws DMLRuntimeException
	{
		StringBuffer sb = new StringBuffer ();
		
		int count = 0;
		for (Entry <String, Data> e : localMap.entrySet ())
		{
			if (count != 0)
				sb.append (ELEMENT_DELIM);
			sb.append (ProgramConverter.serializeDataObject (e.getKey (), e.getValue ()));
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
	public Object clone() throws CloneNotSupportedException
	{
		LocalVariableMap newMap = new LocalVariableMap ();
		newMap.putAll (this);
		return newMap;
	}

	@Override
	protected void finalize ()
	{
		localMap = null;
	}
	
	private String getNewID ()
	{
        char[] out = {'0', '0', '0', '0', '0', '0'};
        String src = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        int srcIndex = out.length - 1;
        for (int newID = globalCount ++; newID != 0; newID /= src.length ())
        {
            if (newID < 0 || srcIndex < 0)
            	throw new RuntimeException ("Too many variable-maps");
            out [srcIndex] = src.charAt (newID % src.length ());
            srcIndex--;
        }
        return new String (out);
	}
}
