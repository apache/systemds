/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import org.apache.spark.storage.StorageLevel;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Lop for checkpoint operations. For example, on Spark, the semantic of a checkpoint 
 * is to persist an intermediate result into a specific storage level (e.g., mem_only). 
 * 
 * We use the name checkpoint in terms of cache/persist in Spark (not in terms of checkpoint
 * in Spark streaming) in order to differentiate from CP caching.
 * 
 * NOTE: since this class uses spark apis, it should only be instantiated if we are
 * running in execution mode spark (whenever all spark libraries are available)
 */
public class Checkpoint extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final String OPCODE = "chkpoint"; 
	
	private static final StorageLevel DEFAULT_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK();
	public static final String STORAGE_LEVEL = "storage.level"; 

	private StorageLevel _storageLevel;
	

	/**
	 * TODO change string parameter storage.level to StorageLevel as soon as we can assume
	 * that Spark libraries are always available.
	 * 
	 * @param input
	 * @param dt
	 * @param vt
	 * @param level
	 * @param et
	 * @throws LopsException
	 */
	public Checkpoint(Lop input, DataType dt, ValueType vt, String level, ExecType et) 
		throws LopsException
	{
		super(Lop.Type.Checkpoint, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		_storageLevel = StorageLevel.fromString(level);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	public StorageLevel getStorageLevel()
	{
		return _storageLevel;
	}
	
	public void setStorageLevel(StorageLevel level)
	{
		_storageLevel = level;
	}
	
	@Override
	public String toString() {
		return "Checkpoint - storage.level = " + _storageLevel.toString();
	}
	
	@Override
	public String getInstructions(String input1, String output) 
		throws LopsException 
	{
		//valid execution type
		if(getExecType() != ExecType.SPARK) {
			throw new LopsException("The method getInstructions(String,String) for Checkpoint should be called only for Spark execution type.");
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getStorageLevelString(_storageLevel) );
		
		return sb.toString();

	}
	
	/**
	 * This is a utility method because Sparks StorageLevel.toString() is incompatible with its own
	 * fromString() method.
	 * 
	 * @param level
	 * @return
	 */
	public static String getStorageLevelString( StorageLevel level)
	{
		if( StorageLevel.NONE().equals(level) )
			return "NONE";
		else if( StorageLevel.MEMORY_ONLY().equals(level) )
			return "MEMORY_ONLY";
		else if( StorageLevel.MEMORY_ONLY_2().equals(level) )
			return "MEMORY_ONLY_2";
		else if( StorageLevel.MEMORY_ONLY_SER().equals(level) )
			return "MEMORY_ONLY_SER";
		else if( StorageLevel.MEMORY_ONLY_SER_2().equals(level) )
			return "MEMORY_ONLY_SER_2";
		else if( StorageLevel.MEMORY_AND_DISK().equals(level) )
			return "MEMORY_AND_DISK";
		else if( StorageLevel.MEMORY_AND_DISK_2().equals(level) )
			return "MEMORY_AND_DISK_2";
		else if( StorageLevel.MEMORY_AND_DISK_SER().equals(level) )
			return "MEMORY_AND_DISK_SER";
		else if( StorageLevel.MEMORY_AND_DISK_SER_2().equals(level) )
			return "MEMORY_AND_DISK_SER_2";
		else if( StorageLevel.DISK_ONLY().equals(level) )
			return "DISK_ONLY";
		else if( StorageLevel.DISK_ONLY_2().equals(level) )
			return "DISK_ONLY_2";
		
		return "INVALID";
	}
	
	/**
	 * 
	 * @return
	 */
	public static String getDefaultStorageLevelString()
	{
		return getStorageLevelString( DEFAULT_STORAGE_LEVEL );
	}
}