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

package org.apache.sysml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyFrameBlockFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CreateSparseBlockFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class CheckpointSPInstruction extends UnarySPInstruction
{
	//default storage level
	private StorageLevel _level = null;
	
	public CheckpointSPInstruction(Operator op, CPOperand in, CPOperand out, StorageLevel level, String opcode, String istr) {
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Reorg;
		
		_level = level;
	}
	
	public static CheckpointSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		StorageLevel level = StorageLevel.fromString(parts[3]);

		return new CheckpointSPInstruction(null, in, out, level, opcode, str);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		// Step 1: early abort on non-existing inputs 
		// -------
		// (checkpoints are generated for all read only variables in loops; due to unbounded scoping and 
		// conditional control flow they to not necessarily exist in the symbol table during runtime - 
		// this is valid if relevant branches are never entered)
		if( sec.getVariable( input1.getName() ) == null || sec.getVariable( input1.getName() ) instanceof BooleanObject) {
			//add a dummy entry to the input, which will be immediately overwritten by the null output.
			sec.setVariable( input1.getName(), new BooleanObject(false));
			sec.setVariable( output.getName(), new BooleanObject(false));
			return;
		}
		
		//get input rdd handle (for matrix or frame)
		JavaPairRDD<?,?> in = sec.getRDDHandleForVariable( input1.getName(), InputInfo.BinaryBlockInputInfo );
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics( input1.getName() );
		
		// Step 2: Checkpoint given rdd (only if currently in different storage level to prevent redundancy)
		// -------
		// Note that persist is an transformation which will be triggered on-demand with the next rdd operations
		// This prevents unnecessary overhead if the dataset is only consumed by cp operations.

		JavaPairRDD<?,?> out = null;
		if( !in.getStorageLevel().equals( _level ) ) 
		{
			//investigate issue of unnecessarily large number of partitions
			int numPartitions = getNumCoalescePartitions(mcIn, in);
			boolean coalesce = ( numPartitions < in.partitions().size() );
			
			//checkpoint pre-processing rdd operations
			if( coalesce ) {
				//merge partitions without shuffle if too many partitions
				out = in.coalesce( numPartitions );
			}
			else {
				//since persist is an in-place marker for a storage level, we 
				//apply a narrow shallow copy to allow for short-circuit collects 
				if( input1.getDataType() == DataType.MATRIX )
					out = ((JavaPairRDD<MatrixIndexes,MatrixBlock>)in)
						.mapValues(new CopyBlockFunction(false));
				else if( input1.getDataType() == DataType.FRAME)
					out = ((JavaPairRDD<Long,FrameBlock>)in)
						.mapValues(new CopyFrameBlockFunction(false));	
			}
		
			//convert mcsr into memory-efficient csr if potentially sparse
			if( input1.getDataType()==DataType.MATRIX 
				&& OptimizerUtils.checkSparseBlockCSRConversion(mcIn) ) 
			{				
				out = ((JavaPairRDD<MatrixIndexes,MatrixBlock>)out)
					.mapValues(new CreateSparseBlockFunction(SparseBlock.Type.CSR));
			}
			
			//actual checkpoint into given storage level
			out = out.persist( _level );
		}
		else {
			out = in; //pass-through
		}
			
		// Step 3: In-place update of input matrix/frame rdd handle and set as output
		// -------
		// We use this in-place approach for two reasons. First, it is correct because our checkpoint 
		// injection rewrites guarantee that after checkpoint instructions there are no consumers on the 
		// given input. Second, it is beneficial because otherwise we need to pass in-memory objects and
		// filenames to the new matrix object in order to prevent repeated reads from hdfs and unnecessary
		// caching and subsequent collects. Note that in-place update requires us to explicitly handle
		// lineage information in order to prevent cycles on cleanup. 
		
		CacheableData<?> cd = sec.getCacheableData( input1.getName() );
		if( out != in ) {                         //prevent unnecessary lineage info
			RDDObject inro =  cd.getRDDHandle();  //guaranteed to exist (see above)
			RDDObject outro = new RDDObject(out, output.getName()); //create new rdd object
			outro.setCheckpointRDD(true);         //mark as checkpointed
			outro.addLineageChild(inro);          //keep lineage to prevent cycles on cleanup
			cd.setRDDHandle(outro);
		}
		sec.setVariable( output.getName(), cd);
	}
	
	/**
	 * 
	 * @param mc
	 * @param in
	 * @return
	 */
	public static int getNumCoalescePartitions(MatrixCharacteristics mc, JavaPairRDD<?,?> in)
	{
		if( mc.dimsKnown(true) ) {
			double hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
			double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc);
			return (int) Math.max(Math.ceil(matrixPSize/hdfsBlockSize), 1);
		}
		else {
			return in.partitions().size();
		}
	}
}

