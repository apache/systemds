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

package com.ibm.bi.dml.runtime.instructions.spark.utils;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.DataConverter;

public class RDDConverterUtils 
{
	

	/**
	 * Converter from binary block rdd to rdd of labeled points. Note that the input needs to be 
	 * reblocked to satisfy the 'clen <= bclen' constraint.
	 * 
	 * @param in
	 * @return
	 */
	public static JavaRDD<LabeledPoint> convert2LabeledPoints(JavaPairRDD<MatrixIndexes, MatrixBlock> in) 
	{
		//convert indexed binary block input to collection of labeled points
		JavaRDD<LabeledPoint> pointrdd = in
				.values()
				.flatMap(new PrepareBinaryBlockFunction());
		
		return pointrdd;
	}
	
	/**
	 * This function converts a binary block input (<X,y>) into mllib's labeled points. Note that
	 * this function requires prior reblocking if the number of columns is larger than the column
	 * block size. 
	 */
	private static class PrepareBinaryBlockFunction implements FlatMapFunction<MatrixBlock, LabeledPoint> 
	{
		private static final long serialVersionUID = -6590259914203201585L;

		@Override
		public Iterable<LabeledPoint> call(MatrixBlock arg0) 
			throws Exception 
		{
			ArrayList<LabeledPoint> ret = new ArrayList<LabeledPoint>();
			for( int i=0; i<arg0.getNumRows(); i++ )
			{
				MatrixBlock tmp = arg0.sliceOperations(i+1, i+1, 1, arg0.getNumColumns()-1, new MatrixBlock());
				double[] data = DataConverter.convertToDoubleVector(tmp);
				if( tmp.isEmptyBlock(false) ) //EMPTY SPARSE ROW
				{
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.sparse(0, new int[0], new double[0])));
				}
				else if( tmp.isInSparseFormat() ) //SPARSE ROW
				{
					SparseRow row = tmp.getSparseRows()[0]; 
					int rlen = row.size();
					int[] rix = row.getIndexContainer();
					double[] rvals = row.getValueContainer();
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.sparse(rlen, rix, rvals)));
				}
				else // DENSE ROW
				{
					ret.add(new LabeledPoint(arg0.getValue(i, arg0.getNumColumns()-1), Vectors.dense(data)));
				}
			}
			
			return ret;
		}
	}
}
