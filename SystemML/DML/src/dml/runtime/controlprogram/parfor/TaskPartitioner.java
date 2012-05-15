package dml.runtime.controlprogram.parfor;

import java.util.Collection;

import dml.runtime.instructions.CPInstructions.IntObject;
import dml.utils.DMLRuntimeException;

/**
 * This is the base class for all task partitioner. For this purpose it stores relevant information such as
 * the loop specification (FROM, TO, INCR), the index variable and the task size. Furthermore, it declares two
 * prototypes: (1) full task creation, (2) streaming task creation.
 * 
 * Known implementation classes: TaskPartitionerFixedsize, TaskPartitionerFactoring
 * 
 * @author mboehm
 */
public abstract class TaskPartitioner 
{
	protected int            _taskSize     = -1;
	
	protected String  		 _iterVarName  = null;
	protected IntObject      _fromVal      = null;
	protected IntObject      _toVal        = null;
	protected IntObject      _incrVal      = null;
	
	protected int            _numIter      = -1;
	
	
	protected TaskPartitioner( int taskSize, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		_taskSize    = taskSize;
		
		_iterVarName = iterVarName;
		_fromVal     = fromVal;
		_toVal       = toVal;
		_incrVal     = incrVal;
		
		_numIter     = (_toVal.getIntValue()-_fromVal.getIntValue()+1 ) / _incrVal.getIntValue(); 
	}
	
	/**
	 * Creates and returns set of all tasks for given problem at once.
	 * 
	 * @return
	 */
	public abstract Collection<Task> createTasks()
		throws DMLRuntimeException;
	
	/**
	 * Creates set of all tasks for given problem, but streams them directly
	 * into task queue. This allows for more tasks than fitting in main memory.
	 * 
	 * @return
	 */
	public abstract int createTasks( LocalTaskQueue queue )
		throws DMLRuntimeException;

	
	/**
	 * 
	 * @return
	 */
	public int getNumIterations()
	{
		return _numIter;
	}
}
