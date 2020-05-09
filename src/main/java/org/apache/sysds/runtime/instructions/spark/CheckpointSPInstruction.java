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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.instructions.spark.functions.CopyFrameBlockFunction;
import org.apache.sysds.runtime.instructions.spark.functions.CreateSparseBlockFunction;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.Statistics;

public class CheckpointSPInstruction extends UnarySPInstruction {
	// default storage level
	private StorageLevel _level = null;

	private CheckpointSPInstruction(Operator op, CPOperand in, CPOperand out, StorageLevel level, String opcode, String istr) {
		super(SPType.Checkpoint, op, in, out, opcode, istr);
		_level = level;
	}

	public static CheckpointSPInstruction parseInstruction ( String str ) {
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
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		// Step 1: early abort on non-existing or in-memory (cached) inputs
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
		//-------
		//(for csv input files with unknown dimensions, we might have generated a checkpoint after
		//csvreblock although not necessary because the csvreblock was subject to in-memory reblock)
		// and for federated matrices
		CacheableData<?> obj = sec.getCacheableData(input1.getName());
		DataCharacteristics mcIn = sec.getDataCharacteristics( input1.getName() );
		if (obj.isCached(true) || Recompiler.checkCPCheckpoint(mcIn) 
			|| (sec.getCacheableData(input1.getName()) instanceof MatrixObject 
				&& sec.getMatrixObject(input1.getName()).isFederated())) {
			//available in memory
			sec.setVariable(output.getName(), obj);
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}
		
		//get input rdd handle (for matrix or frame)
		JavaPairRDD<?,?> in = sec.getRDDHandleForVariable(input1.getName(), FileFormat.BINARY, -1, true);
		
		// Step 2: Checkpoint given rdd (only if currently in different storage level to prevent redundancy)
		// -------
		// Note that persist is an transformation which will be triggered on-demand with the next rdd operations
		// This prevents unnecessary overhead if the dataset is only consumed by cp operations.

		JavaPairRDD<?,?> out = null;
		if( !in.getStorageLevel().equals( _level ) ) 
		{
			//determine need for coalesce or repartition, and csr conversion
			int numPartitions = SparkUtils.getNumPreferredPartitions(mcIn, in);
			boolean coalesce = ( 1.2*numPartitions < in.getNumPartitions()
				&& !SparkUtils.isHashPartitioned(in) && in.getNumPartitions()
				> SparkExecutionContext.getDefaultParallelism(true));
			boolean repartition = mcIn.dimsKnown(true) && mcIn.isUltraSparse()
				&& numPartitions > in.getNumPartitions();
			boolean mcsr2csr = input1.getDataType()==DataType.MATRIX 
				&& OptimizerUtils.checkSparseBlockCSRConversion(mcIn)
				&& !_level.equals(Checkpoint.SER_STORAGE_LEVEL);
			
			//checkpoint pre-processing rdd operations
			if( coalesce ) {
				//merge partitions without shuffle if too many partitions
				out = in.coalesce( numPartitions );
			}
			else if( repartition ) {
				//repartition to preferred size as multiple of default parallelism
				out = in.repartition(UtilFunctions.roundToNext(numPartitions,
					SparkExecutionContext.getDefaultParallelism(true)));
			}
			else if( !mcsr2csr ) {
				//since persist is an in-place marker for a storage level, we 
				//apply a narrow shallow copy to allow for short-circuit collects 
				if( input1.getDataType() == DataType.MATRIX )
					out = SparkUtils.copyBinaryBlockMatrix(
						(JavaPairRDD<MatrixIndexes,MatrixBlock>)in, false);
				else if( input1.getDataType() == DataType.FRAME)
					out = ((JavaPairRDD<Long,FrameBlock>)in)
						.mapValues(new CopyFrameBlockFunction(false));
			}
			else {
				out = in;
			}
			
			//convert mcsr into memory-efficient csr if potentially sparse
			if( mcsr2csr ) {
				out = ((JavaPairRDD<MatrixIndexes,MatrixBlock>)out)
					.mapValues(new CreateSparseBlockFunction(SparseBlock.Type.CSR));
			}
			
			//actual checkpoint into given storage level
			out = out.persist( _level );
			
			//trigger nnz computation for datasets that are forced to spark by their dimensions
			//(larger than MAX_INT) to handle ultra-sparse data sets during recompilation because
			//otherwise these their nnz would never be evaluated due to lazy evaluation in spark
			if( input1.isMatrix() && mcIn.dimsKnown() 
				&& !mcIn.dimsKnown(true) && !OptimizerUtils.isValidCPDimensions(mcIn) ) {
				mcIn.setNonZeros(SparkUtils.getNonZeros((JavaPairRDD<MatrixIndexes,MatrixBlock>)out));
			}
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
			RDDObject outro = new RDDObject(out); //create new rdd object
			outro.setCheckpointRDD(true);         //mark as checkpointed
			outro.addLineageChild(inro);          //keep lineage to prevent cycles on cleanup
			cd.setRDDHandle(outro);
		}
		sec.setVariable( output.getName(), cd);
	}
}
