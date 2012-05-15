package dml.runtime.controlprogram.parfor.util;

/**
 * ID sequence for generating unique long identifiers with start 0 and increment 1.
 * 
 * @author mboehm
 */
public class IDSequence 
{
	private long _current = -1;
	
	public IDSequence()
	{
		reset();
	}
	
	/**
	 * Creates the next ID, if overflow a RuntimeException is thrown.
	 * 
	 * @return ID
	 */
	public synchronized long getNextID()
	{
		_current++;
		
		if( _current == Long.MAX_VALUE )
			throw new RuntimeException("WARNING: IDSequence will produced numeric overflow.");
		
		return _current;
	}
	
	public synchronized void reset()
	{
		_current = 0;
	}
}
