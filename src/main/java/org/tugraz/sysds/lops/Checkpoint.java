/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.tugraz.sysds.lops;

import org.apache.spark.storage.StorageLevel;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


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
	public static final String OPCODE = "chkpoint"; 
	 
	public static final StorageLevel DEFAULT_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK();
	public static final StorageLevel SER_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK_SER();
	public static final boolean CHECKPOINT_SPARSE_CSR = true; 

	private StorageLevel _storageLevel;
	

	/**
	 * TODO change string parameter storage.level to StorageLevel as soon as we can assume
	 * that Spark libraries are always available.
	 * 
	 * @param input low-level operator
	 * @param dt data type
	 * @param vt value type
	 * @param level storage level
	 */
	public Checkpoint(Lop input, DataType dt, ValueType vt, String level)  {
		super(Lop.Type.Checkpoint, dt, vt);
		this.addInput(input);
		input.addOutput(this);
		
		_storageLevel = StorageLevel.fromString(level);
		lps.setProperties(inputs, ExecType.SPARK);
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
	public String getInstructions(String input1, String output) {
		//valid execution type
		if(getExecType() != ExecType.SPARK) {
			throw new LopsException("Wrong execution type for Checkpoint.getInstructions (expected: SPARK, found: "+getExecType()+").");
		}
		
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output),
			getStorageLevelString(_storageLevel));
	}
	
	/**
	 * This is a utility method because Sparks StorageLevel.toString() is incompatible with its own
	 * fromString() method.
	 * 
	 * @param level RDD storage level
	 * @return storage level as a string
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
	
	public static String getDefaultStorageLevelString() {
		return getStorageLevelString( DEFAULT_STORAGE_LEVEL );
	}
	
	public static String getSerializeStorageLevelString() {
		return getStorageLevelString( SER_STORAGE_LEVEL );
	}
}