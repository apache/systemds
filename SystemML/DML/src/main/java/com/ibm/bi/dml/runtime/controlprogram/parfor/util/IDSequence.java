/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

/**
 * ID sequence for generating unique long identifiers with start 0 and increment 1.
 * 
 */
public class IDSequence 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _current = -1;
	private boolean wrapAround = false;
	
	public IDSequence()
	{
		reset();
	}
	
	public IDSequence(boolean wrapAround)
	{
		reset();
		this.wrapAround = wrapAround;
	}
	
	/**
	 * Creates the next ID, if overflow a RuntimeException is thrown.
	 * 
	 * @return ID
	 */
	public synchronized long getNextID()
	{
		_current++;
		
		if( _current == Long.MAX_VALUE ) {
			if (wrapAround)
				reset();
			else
				throw new RuntimeException("WARNING: IDSequence will produced numeric overflow.");
		}
		
		return _current;
	}
	
	public synchronized void reset()
	{
		_current = 0;
	}
	
	/*
	private AtomicLong _seq = new AtomicLong(0);
	
	public long getNextID()
	{
		return _seq.getAndIncrement();
	}
	
	public void reset()
	{
		_seq = new AtomicLong( 0 );
	}
	*/
}
