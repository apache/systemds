/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

/**
 * Wrapper for WriteBuffer byte array per matrix in order to
 * support matrix serialization outside global lock.
 * 
 */
public class ByteBuffer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected byte[] data;
	private boolean serialized;
	
	public ByteBuffer( byte[] idata )
	{
		data = idata;
		serialized = false;
	}
	
	/**
	 * 
	 */
	public void markSerialized()
	{
		serialized = true;
	}
	
	/**
	 * 
	 */
	public void checkSerialized()
	{
		if( serialized )
			return;
		
		while( !serialized )
			try{Thread.sleep(5);} catch(Exception e) {}
	}
}
