/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import com.ibm.bi.dml.runtime.instructions.cp.IntObject;

/**
 * A task is a logical group of one or multiple iterations (each iteration is assigned to exactly one task).
 * There, each single task is executed sequentially. See TaskPartitioner for how tasks are created and
 * ParWorker for how those tasks are eventually executed.
 * 
 * NOTE: (Extension possibility: group of statements) 
 * 
 */
public class Task implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 2815832451487164284L;
	
	public enum TaskType {
		RANGE, 
		SET
	}
	
	public static final int MAX_VARNAME_SIZE  = 256;
	public static final int MAX_TASK_SIZE     = Integer.MAX_VALUE-1; 
	
	private TaskType           	  _type;
	private LinkedList<IntObject> _iterations; //each iteration is specified as an ordered set of index values
	
	public Task() {
		//default constructor for serialize
	}
	
	public Task( TaskType type )
	{
		_type = type;
		
		_iterations = new LinkedList<IntObject>();
	}
	
	public void addIteration( IntObject indexVal ) 
	{
		if( indexVal.getName().length() > MAX_VARNAME_SIZE )
			throw new RuntimeException("Cannot add iteration, MAX_VARNAME_SIZE exceeded.");
		
		if( size() >= MAX_TASK_SIZE )
			throw new RuntimeException("Cannot add iteration, MAX_TASK_SIZE reached.");
			
		_iterations.addLast( indexVal );
	}
	
	public List<IntObject> getIterations()
	{
		return _iterations;
	}
	
	public TaskType getType()
	{
		return _type;
	}
	
	public int size()
	{
		return _iterations.size();
	}
	
	/**
	 * 
	 * @param task
	 */
	public void mergeTask( Task task )
	{
		//check for set iteration type
		if( _type==TaskType.RANGE )
			throw new RuntimeException("Task Merging not supported for tasks of type ITERATION_RANGE.");
		
		//check for same iteration name
		String var1 = _iterations.getFirst().getName();
		String var2 = task._iterations.getFirst().getName();
		if( !var1.equals(var2) )
			throw new RuntimeException("Task Merging not supported for tasks with different variable names");
	
		//merge tasks
		for( IntObject o : task._iterations )
			_iterations.addLast( o );
	}
	

	@Override
	public String toString() 
	{
		return toFormatedString();
	}
	
	/**
	 * 
	 * @return
	 */
	public String toFormatedString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("task (type=");
		sb.append(_type);
		sb.append(", iterations={");
		int count=0;
		for( IntObject dat : _iterations )
		{
			if( count!=0 ) 
				sb.append(";");
			sb.append("[");
			sb.append(dat.getName());
			sb.append("=");
			sb.append(dat.getLongValue());
			sb.append("]");
			
			count++;
		}
		sb.append("})");
		return sb.toString();
	}
	
	/**
	 * 
	 * @return
	 */
	public String toCompactString()
	{
		StringBuilder sb = new StringBuilder( );
		sb.append(_type);
		
		if( size() > 0 )
		{
			sb.append(".");
			IntObject dat0 = _iterations.getFirst();
			sb.append(dat0.getName());
			sb.append(".{");
		
			int count = 0;
			for( IntObject dat : _iterations )
			{
				if( count!=0 ) 
					sb.append(",");
				sb.append(dat.getLongValue());
				count++;
			}
			
			sb.append("}");
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @return
	 */
	public String toCompactString( int maxDigits )
	{
		StringBuilder sb = new StringBuilder( );
		sb.append(_type);
		
		if( size() > 0 )
		{
			sb.append(".");
			IntObject dat0 = _iterations.getFirst();
			sb.append(dat0.getName());
			sb.append(".{");
		
			int count = 0;
			for( IntObject dat : _iterations )
			{
				if( count!=0 ) 
					sb.append(",");
				
				String tmp = String.valueOf(dat.getLongValue());
				for( int k=tmp.length(); k<maxDigits; k++ )
					sb.append("0");
				sb.append(tmp);
				count++;
			}
			
			sb.append("}");
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param stask
	 * @return
	 */
	public static Task parseCompactString( String stask )
	{
		StringTokenizer st = new StringTokenizer( stask.trim(), "." );		
		
		Task newTask = new Task( TaskType.valueOf(st.nextToken()) );
		String meta = st.nextToken();
		
		//iteration data
		String sdata = st.nextToken();
		sdata = sdata.substring(1,sdata.length()-1); // remove brackets
		StringTokenizer st2 = new StringTokenizer(sdata, ",");
		while( st2.hasMoreTokens() )
		{
			//create new iteration
			String lsdata = st2.nextToken();
			IntObject ldata = new IntObject(meta,Integer.parseInt( lsdata ) );
			newTask.addIteration(ldata);
		}
		
		return newTask;
	}
}
