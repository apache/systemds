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

	public CacheOutOfMemoryException (Throwable cause)
	{
		super (cause);
	}

	public CacheOutOfMemoryException (String message, Throwable cause)
	{
		super (message, cause);
	}

}
