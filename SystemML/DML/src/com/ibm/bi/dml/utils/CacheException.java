package com.ibm.bi.dml.utils;

public class CacheException extends DMLRuntimeException
{

	public CacheException ()
	{
		super ("Cache Exception");
	}

	public CacheException (String message)
	{
		super (message);
	}

	public CacheException (Exception cause)
	{
		super (cause);
	}

	public CacheException (String message, Exception cause)
	{
		super (message, cause);
	}

}
