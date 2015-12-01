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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Due to independence of all iterations, any result has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 */
public abstract class ResultMerge 
{
	
	protected static final Log LOG = LogFactory.getLog(ResultMerge.class.getName());
	
	protected static final String NAME_SUFFIX = "_rm";
	
	//inputs to result merge
	protected MatrixObject   _output      = null;
	protected MatrixObject[] _inputs      = null; 
	protected String         _outputFName = null;
	
	protected ResultMerge( )
	{
		
	}
	
	public ResultMerge( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObject executeSerialMerge() 
		throws DMLRuntimeException;
	
	/**
	 * Merge all given input matrices in parallel into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of all input matrices.
	 * 
	 * @param par degree of parallelism
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObject executeParallelMerge( int par ) 
		throws DMLRuntimeException;
	
	/**
	 * 
	 * @param out initially empty block
	 * @param in 
	 * @param appendOnly 
	 * @throws DMLRuntimeException 
	 */
	protected void mergeWithoutComp( MatrixBlock out, MatrixBlock in, boolean appendOnly ) 
		throws DMLRuntimeException
	{
		//pass through to matrix block operations
		out.merge(in, appendOnly);	
	}

	/**
	 * NOTE: append only not applicable for wiht compare because output must be populated with
	 * initial state of matrix - with append, this would result in duplicates.
	 * 
	 * @param out
	 * @param in
	 * @throws DMLRuntimeException 
	 */
	protected void mergeWithComp( MatrixBlock out, MatrixBlock in, double[][] compare ) 
		throws DMLRuntimeException
	{
		//Notes for result correctness:
		// * Always iterate over entire block in order to compare all values 
		//   (using sparse iterator would miss values set to 0) 
		// * Explicit NaN awareness because for cases were original matrix contains
		//   NaNs, since NaN != NaN, otherwise we would potentially overwrite results
		
		if( in.isInSparseFormat() ) //sparse input format
		{
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{	
				    double value = in.getValueSparseUnsafe(i,j);  //input value
					if(   (value != compare[i][j] && !Double.isNaN(value) )     //for new values only (div)
						|| Double.isNaN(value) != Double.isNaN(compare[i][j]) ) //NaN awareness 
					{
				    	out.quickSetValue( i, j, value );	
					}
				}
		}
		else //dense input format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
				    double value = in.getValueDenseUnsafe(i,j);  //input value
				    if(    (value != compare[i][j] && !Double.isNaN(value) )    //for new values only (div)
				    	|| Double.isNaN(value) != Double.isNaN(compare[i][j]) ) //NaN awareness
				    {
				    	out.quickSetValue( i, j, value );	
				    }
				}
		}	
	}

	protected long computeNonZeros( MatrixObject out, ArrayList<MatrixObject> in )
	{
		MatrixCharacteristics mc = out.getMatrixCharacteristics();
		long outNNZ = mc.getNonZeros();	
		long ret = outNNZ;
		for( MatrixObject tmp : in )
		{
			MatrixCharacteristics tmpmc = tmp.getMatrixCharacteristics();
			long inNNZ = tmpmc.getNonZeros();
			ret +=  (inNNZ - outNNZ);			
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	protected ArrayList<MatrixObject> convertToList(MatrixObject[] in)
	{
		ArrayList<MatrixObject> ret = new ArrayList<MatrixObject>();
		for( MatrixObject mo : in )
			ret.add( mo );
		
		return ret;
	}
}
