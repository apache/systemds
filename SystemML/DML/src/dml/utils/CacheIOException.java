package dml.utils;

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

	public CacheIOException (Throwable cause)
	{
		super (cause);
	}

	public CacheIOException (String message, Throwable cause)
	{
		super (message, cause);
	}

}
