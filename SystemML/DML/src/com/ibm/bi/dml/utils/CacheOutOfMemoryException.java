package com.ibm.bi.dml.utils;

public class CacheOutOfMemoryException extends CacheException
{

	public CacheOutOfMemoryException ()
	{
		super ();
	}

	public CacheOutOfMemoryException (String message)
	{
		super (message);
	}

	public CacheOutOfMemoryException (Exception cause)
	{
		super (cause);
	}

	public CacheOutOfMemoryException (String message, Exception cause)
	{
		super (message, cause);
	}

}
