package com.ibm.bi.dml.utils;

public class CacheStatusException extends CacheException
{
	private static final long serialVersionUID = 1L;

	public CacheStatusException (String message)
	{
		super (message);
	}

	public CacheStatusException (Exception cause)
	{
		super (cause);
	}

	public CacheStatusException (String message, Exception cause)
	{
		super (message, cause);
	}

}
