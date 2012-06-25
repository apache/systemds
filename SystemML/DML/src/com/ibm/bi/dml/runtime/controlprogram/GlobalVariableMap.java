package com.ibm.bi.dml.runtime.controlprogram;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class GlobalVariableMap
{
	private static String eol = System.getProperty ("line.separator");  
	private Map <String, Data> varMap;
	private Map <String, Integer> referenceCountMap;
	private volatile int globalCount = 0;

	public GlobalVariableMap ()
	{
		varMap = Collections.synchronizedMap (new HashMap <String, Data> ());
		referenceCountMap = Collections.synchronizedMap (new HashMap <String, Integer> ());
	}
	
	public synchronized String getNewID ()
	{
        char[] out = {'0', '0', '0', '0', '0', '0', '.'};
        String src = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        int id = globalCount ++;
        if (id < 0)
        	throw new RuntimeException ("Too many variable-maps.");
        for (int i = 5; i >= 0; i--)
        {
            out [i] = src.charAt (id % 36);
            id /= 36;
            if (id == 0)
                break;
        }
        return new String (out);
	}
	
	/**
	 * Returns the {@link Data} object for the given variable name.
	 * 
	 * @param name : the variable name
	 * @return the {@link Data} object for this name
	 */
	public synchronized Data get (String localID, String name)
	{
		String newName = localID + name;
		return varMap.get (newName);
	}

	/**
	 * Adds a new (name, value) pair into the global variables table, or adds 1
	 * to its reference count if the pair already exists in the table.  The value
	 * here is a {@link Data} object, which may be a cache envelope.  If the name
	 * is already associated with a different data object, the old pair is removed
	 * and its reference count is set to zero.  If the value is <code>null</code>,
	 * an exception is thrown.  Keep in mind that the "new" data object may
	 * already be associated with some other variable name.
	 * 
	 * @param name : the variable name
	 * @param val : the {@link Data} object
	 * @return the new reference count for the pair
	 */
	public synchronized int addOneCount (String localID, String name, Data val)
	{
		String newName = localID + name;
		int newValue = 0;
		if (val == null)
			throw new NullPointerException ("GlobalVariableMap.addOneCount (\"" + name + "\", null)" + eol);
		if (val == varMap.get (newName))
		{
			newValue = referenceCountMap.remove (newName).intValue () + 1;
			referenceCountMap.put (newName, new Integer (newValue));
		}
		else
		{
			newValue = 1;
			varMap.put (newName, val);
			referenceCountMap.put (newName, new Integer (1));
		}
		return newValue;
	}
	
	/**
	 * Decrements the reference count for the (name, value) pair that corresponds
	 * to the specified variable name.  If that results in the count of zero, removes
	 * the pair.  If there is no such pair, does nothing and returns <code>-1</code>.
	 * 
	 * @param name : the variable name
	 * @return the new reference count for the pair; <code>0</code> if the pair
	 *     is removed, <code>-1</code> if there was no such pair at the start.
	 */
	public synchronized int removeOneCount (String localID, String name)
	{
		String newName = localID + name;
		Integer countObject = referenceCountMap.remove (newName);
		if (countObject == null)
			return -1;
		int new_value = countObject.intValue () - 1;
		if (new_value > 0)
			referenceCountMap.put (newName, new Integer (new_value));
		else
		{
			new_value = 0;
			varMap.remove (newName);
			referenceCountMap.remove (newName);
		}
		return new_value;
	}
	
	public synchronized int getReferenceCount (String localID, String name)
	{
		String newName = localID + name;
		Integer countObject = referenceCountMap.get (newName);
		if (countObject == null)
			return 0;
		return countObject.intValue ();
	}
}
