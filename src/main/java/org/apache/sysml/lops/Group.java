/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;

/**
 * Lop to represent a grouping operation.
 */

public class Group extends Lop  
{

	
	
	public enum OperationTypes {Sort};
	
	OperationTypes operation;
	
	/**
	 * Constructor to create a grouping operation.
	 * @param input
	 * @param op
	 */

	public Group(Lop input, Group.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Grouping, dt, vt);		
		operation = op;
		this.addInput(input);
		input.addOutput(this);
		
		/*
		 *  This lop can be executed in only in GMR and RAND.
		 *  MMCJ, REBLOCK, and PARTITION themselves has location MapAndReduce.
		 */
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		
		boolean breaksAlignment = false;
		boolean aligner = true;
		boolean definesMRJob = true;
		
		this.lps.setProperties ( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() 
	{
		//return "Group " + "Operation: " + operation;
		return "Operation: " + operation;
	
	}

}
