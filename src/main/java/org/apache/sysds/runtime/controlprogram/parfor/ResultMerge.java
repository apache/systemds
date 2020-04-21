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

package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

import java.io.Serializable;
import java.util.List;

/**
 * Due to independence of all iterations, any result has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 */
public abstract class ResultMerge implements Serializable
{
	//note: this class needs to be serializable to ensure that all attributes of
	//ResultMergeRemoteSparkWCompare are included in the task closure
	private static final long serialVersionUID = 2620430969346516677L;
	
	protected static final Log LOG = LogFactory.getLog(ResultMerge.class.getName());
	protected static final String NAME_SUFFIX = "_rm";
	protected static final BinaryOperator PLUS = InstructionUtils.parseBinaryOperator("+");
	
	//inputs to result merge
	protected MatrixObject   _output      = null;
	protected MatrixObject[] _inputs      = null; 
	protected String         _outputFName = null;
	protected boolean        _isAccum     = false;
	
	protected ResultMerge( ) {
		//do nothing
	}
	
	public ResultMerge( MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum ) {
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
		_isAccum = accum;
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 */
	public abstract MatrixObject executeSerialMerge();
	
	/**
	 * Merge all given input matrices in parallel into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of all input matrices.
	 * 
	 * @param par degree of parallelism
	 * @return output (merged) matrix
	 */
	public abstract MatrixObject executeParallelMerge( int par );
	
	protected void mergeWithoutComp( MatrixBlock out, MatrixBlock in, boolean appendOnly ) {
		mergeWithoutComp(out, in, appendOnly, false);
	}
	
	protected void mergeWithoutComp( MatrixBlock out, MatrixBlock in, boolean appendOnly, boolean par ) {
		//pass through to matrix block operations
		if( _isAccum )
			out.binaryOperationsInPlace(PLUS, in);
		else
			out.merge(in, appendOnly, par);
	}

	/**
	 * NOTE: append only not applicable for wiht compare because output must be populated with
	 * initial state of matrix - with append, this would result in duplicates.
	 * 
	 * @param out output matrix block
	 * @param in input matrix block
	 * @param compare ?
	 */
	protected void mergeWithComp( MatrixBlock out, MatrixBlock in, DenseBlock compare ) 
	{
		//Notes for result correctness:
		// * Always iterate over entire block in order to compare all values 
		//   (using sparse iterator would miss values set to 0) 
		// * Explicit NaN awareness because for cases were original matrix contains
		//   NaNs, since NaN != NaN, otherwise we would potentially overwrite results
		// * For the case of accumulation, we add out += (new-old) to ensure correct results
		//   because all inputs have the old values replicated
		
		if( in.isEmptyBlock(false) ) {
			if( _isAccum ) return; //nothing to do
			for( int i=0; i<in.getNumRows(); i++ )
				for( int j=0; j<in.getNumColumns(); j++ )
					if( compare.get(i, j) != 0 )
						out.quickSetValue(i, j, 0);
		}
		else { //SPARSE/DENSE
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ ) {
					double valOld = compare.get(i,j);
					double valNew = in.quickGetValue(i,j); //input value
					if( (valNew != valOld && !Double.isNaN(valNew) )      //for changed values 
						|| Double.isNaN(valNew) != Double.isNaN(valOld) ) //NaN awareness 
					{
						double value = !_isAccum ? valNew :
							(out.quickGetValue(i, j) + (valNew - valOld));
						out.quickSetValue(i, j, value);
					}
				}
		}
	}

	protected long computeNonZeros( MatrixObject out, List<MatrixObject> in ) {
		//sum of nnz of input (worker result) - output var existing nnz
		long outNNZ = out.getDataCharacteristics().getNonZeros();
		return outNNZ - in.size() * outNNZ + in.stream()
			.mapToLong(m -> m.getDataCharacteristics().getNonZeros()).sum();
	}
}
