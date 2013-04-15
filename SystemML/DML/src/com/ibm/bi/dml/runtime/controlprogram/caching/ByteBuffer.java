package com.ibm.bi.dml.runtime.controlprogram.caching;

/**
 * Wrapper for WriteBuffer byte array per matrix in order to
 * support matrix serialization outside global lock.
 * 
 */
public class ByteBuffer
{
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
