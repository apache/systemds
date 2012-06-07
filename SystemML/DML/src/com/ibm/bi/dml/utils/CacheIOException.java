package com.ibm.bi.dml.utils;

public class CacheIOException extends CacheException
{
	public CacheIOException ()
	{
		super ();
	}

	public CacheIOException (String message)
	{
		super (message);
	}

	public CacheIOException (Exception cause)
	{
		super (cause);
	}

	public CacheIOException (String message, Exception cause)
	{
		super (message, cause);
	}

}
