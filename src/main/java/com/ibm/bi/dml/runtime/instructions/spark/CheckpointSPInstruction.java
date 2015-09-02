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

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.storage.StorageLevel;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyBlockFunction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class CheckpointSPInstruction extends UnarySPInstruction
{
	
	private StorageLevel _level = null;
	
	public CheckpointSPInstruction(Operator op, CPOperand in, CPOperand out, StorageLevel level, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Reorg;
		
		_level = level;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		InstructionUtils.checkNumFields(str, 3);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		String opcode = parts[0];
		in.split(parts[1]);
		out.split(parts[2]);

		StorageLevel level = StorageLevel.fromString(parts[3]);

		return new CheckpointSPInstruction(null, in, out, level, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//early abort on non-existing inputs (checkpoints are generated for all read only variables
		//in loops; due to unbounded scoping and conditional control flow they to not necessarily 
		//exist in the symbol table during runtime - this is valid if relevant branches are never entered)
		if( sec.getVariable( input1.getName() ) == null ) {
			//add a dummy entry to the input, which will be immediately overwritten by the null output.
			sec.setVariable( input1.getName(), new BooleanObject(false));
			return;
		}
		
		//get input rdd handle
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );

		// Step 1: Checkpoint given rdd (only if currently in different storage level to prevent redundancy)
		// -------
		// Note that persist is an transformation which will be triggered on-demand with the next rdd operations
		// This prevents unnecessary overhead if the dataset is only consumed by cp operations.
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( !in.getStorageLevel().equals(_level) ) {
			//since persist is an in-place marker for a storage level, we 
			//apply a narrow shallow copy to allow for short-circuit collects 
			out = in.mapValues(new CopyBlockFunction(false))
					.persist( _level );
		}
		else {
			out = in;
		}
			
		// Step 2: In-place update of input matrix rdd handle and set as output
		// -------
		// We use this in-place approach for two reasons. First, it is correct because our checkpoint 
		// injection rewrites guarantee that after checkpoint instructions there are no consumers on the 
		// given input. Second, it is beneficial because otherwise we need to pass in-memory objects and
		// filenames to the new matrix object in order to prevent repeated reads from hdfs and unnecessary
		// caching and subsequent collects. Note that in-place update requires us to explicitly handle
		// lineage information in order to prevent cycles on cleanup. 
		
		MatrixObject mo = sec.getMatrixObject( input1.getName() );
		if( out != in ) {                         //prevent unnecessary lineage info
			RDDObject inro =  mo.getRDDHandle();  //guaranteed to exist (see above)
			RDDObject outro = new RDDObject(out, output.getName()); //create new rdd object
			outro.setCheckpointRDD(true);         //mark as checkpointed
			outro.addLineageChild(inro);          //keep lineage to prevent cycles on cleanup
			mo.setRDDHandle(outro);
		}
		sec.setVariable( output.getName(), mo);
	}
}

