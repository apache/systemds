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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.instructions.spark.data.DatasetObject;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils.DataFrameExtractIDFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.Statistics;

/**
 * TODO heavy hitter maintenance
 * TODO data partitioning with binarycell
 *
 */
public class RemoteDPParForSpark
{
	
	protected static final Log LOG = LogFactory.getLog(RemoteDPParForSpark.class.getName());

	public static RemoteParForJobReturn runJob(long pfid, String itervar, String matrixvar, String program, HashMap<String, byte[]> clsMap,
			String resultFile, MatrixObject input, ExecutionContext ec, PartitionFormat dpf, OutputInfo oi, 
			boolean tSparseCol, boolean enableCPCaching, int numReducers ) 
		throws DMLRuntimeException
	{
		String jobname = "ParFor-DPESP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//prepare input parameters
		MatrixObject mo = sec.getMatrixObject(matrixvar);
		MatrixCharacteristics mc = mo.getMatrixCharacteristics();

		//initialize accumulators for tasks/iterations, and inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
		LongAccumulator aTasks = sc.sc().longAccumulator("tasks");
		LongAccumulator aIters = sc.sc().longAccumulator("iterations");

		//compute number of reducers (to avoid OOMs and reduce memory pressure)
		int numParts = SparkUtils.getNumPreferredPartitions(mc, in);
		int numReducers2 = Math.max(numReducers, Math.min(numParts, (int)dpf.getNumParts(mc)));
		
		//core parfor datapartition-execute (w/ or w/o shuffle, depending on data characteristics)
		RemoteDPParForSparkWorker efun = new RemoteDPParForSparkWorker(program, clsMap, 
				matrixvar, itervar, enableCPCaching, mc, tSparseCol, dpf, oi, aTasks, aIters);
		JavaPairRDD<Long,Writable> tmp = getPartitionedInput(sec, matrixvar, oi, dpf);
		List<Tuple2<Long,String>> out = (requiresGrouping(dpf, mo) ?
				tmp.groupByKey(numReducers2) : tmp.map(new PseudoGrouping()) )
				   .mapPartitionsToPair(efun)  //execute parfor tasks, incl cleanup
		           .collect();                 //get output handles 
		
		//de-serialize results
		LocalVariableMap[] results = RemoteParForUtils.getResults(out, LOG);
		int numTasks = aTasks.value().intValue(); //get accumulator value
		int numIters = aIters.value().intValue(); //get accumulator value
		
		//create output symbol table entries
		RemoteParForJobReturn ret = new RemoteParForJobReturn(true, numTasks, numIters, results);
		
		//maintain statistics
	    Statistics.incrementNoOfCompiledSPInst();
	    Statistics.incrementNoOfExecutedSPInst();
	    if( DMLScript.STATISTICS ){
			Statistics.maintainCPHeavyHitters(jobname, System.nanoTime()-t0);
		}
		
		return ret;
	}
	
	@SuppressWarnings("unchecked")
	private static JavaPairRDD<Long, Writable> getPartitionedInput(SparkExecutionContext sec, 
			String matrixvar, OutputInfo oi, PartitionFormat dpf) 
		throws DMLRuntimeException 
	{
		InputInfo ii = InputInfo.BinaryBlockInputInfo;
		MatrixObject mo = sec.getMatrixObject(matrixvar);
		MatrixCharacteristics mc = mo.getMatrixCharacteristics();

		//leverage existing dataset (w/o shuffling for reblock and data partitioning) 
		//NOTE: there will always be a checkpoint rdd on top of the input rdd and the dataset
		if( hasInputDataSet(dpf, mo) )
		{
			DatasetObject dsObj = (DatasetObject)mo.getRDDHandle()
					.getLineageChilds().get(0).getLineageChilds().get(0);
			Dataset<Row> in = dsObj.getDataset();
			
			//construct or reuse row ids
			JavaPairRDD<Row, Long> prepinput = dsObj.containsID() ?
					in.javaRDD().mapToPair(new DataFrameExtractIDFunction(
						in.schema().fieldIndex(RDDConverterUtils.DF_ID_COLUMN))) :
					in.javaRDD().zipWithIndex(); //zip row index
			
			//convert row to row in matrix block format 
			return prepinput.mapToPair(new DataFrameToRowBinaryBlockFunction(
					mc.getCols(), dsObj.isVectorBased(), dsObj.containsID()));
		}
		//binary block input rdd without grouping
		else if( !requiresGrouping(dpf, mo) ) 
		{
			//get input rdd and data partitioning 
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
			DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(mc, ii, oi, dpf._dpf, dpf._N);
			return in.flatMapToPair(dpfun);
		}
		//default binary block input rdd with grouping
		else
		{
			//get input rdd, avoid unnecessary caching if input is checkpoint and not cached yet
			//to reduce memory pressure for shuffle and subsequent 
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
			if( mo.getRDDHandle().isCheckpointRDD() && !sec.isRDDCached(in.id()) )
				in = (JavaPairRDD<MatrixIndexes,MatrixBlock>)((RDDObject)
						mo.getRDDHandle().getLineageChilds().get(0)).getRDD();
			
			//data partitioning of input rdd 
			DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(mc, ii, oi, dpf._dpf, dpf._N);
			return in.flatMapToPair(dpfun);
		}
	} 
	
	//determines if given input matrix requires grouping of partial partition slices
	private static boolean requiresGrouping(PartitionFormat dpf, MatrixObject mo) {
		MatrixCharacteristics mc = mo.getMatrixCharacteristics();
		return ((dpf == PartitionFormat.ROW_WISE && mc.getNumColBlocks() > 1)
				|| (dpf == PartitionFormat.COLUMN_WISE && mc.getNumRowBlocks() > 1)
				|| (dpf._dpf == PDataPartitionFormat.ROW_BLOCK_WISE_N && mc.getNumColBlocks() > 1)
				|| (dpf._dpf == PDataPartitionFormat.COLUMN_BLOCK_WISE_N && mc.getNumRowBlocks() > 1))
			&& !hasInputDataSet(dpf, mo);
	}
	
	//determines if given input matrix wraps input data set applicable to direct processing
	private static boolean hasInputDataSet(PartitionFormat dpf, MatrixObject mo) {
		return (dpf == PartitionFormat.ROW_WISE 
			&& mo.getRDDHandle().isCheckpointRDD()
			&& mo.getRDDHandle().getLineageChilds().size()==1
			&& mo.getRDDHandle().getLineageChilds().get(0).getLineageChilds().size()==1
			&& mo.getRDDHandle().getLineageChilds().get(0).getLineageChilds().get(0) instanceof DatasetObject);
	}
	
	//function to map data partition output to parfor input signature without grouping
	private static class PseudoGrouping implements Function<Tuple2<Long, Writable>, Tuple2<Long, Iterable<Writable>>>  {
		private static final long serialVersionUID = 2016614593596923995L;

		@Override
		public Tuple2<Long, Iterable<Writable>> call(Tuple2<Long, Writable> arg0) throws Exception {
			return new Tuple2<Long, Iterable<Writable>>(arg0._1(), Collections.singletonList(arg0._2()));
		}
	}
	
	//function to map dataset rows to rows in binary block representation
	private static class DataFrameToRowBinaryBlockFunction implements PairFunction<Tuple2<Row,Long>,Long,Writable> 
	{
		private static final long serialVersionUID = -3162404379379461523L;
		
		private final long _clen;
		private final boolean _containsID;
		private final boolean _isVector;
		
		public DataFrameToRowBinaryBlockFunction(long clen, boolean containsID, boolean isVector) {
			_clen = clen;
			_containsID = containsID;
			_isVector = isVector;
		}
		
		@Override
		public Tuple2<Long, Writable> call(Tuple2<Row, Long> arg0) 
			throws Exception 
		{
			long rowix = arg0._2() + 1;
			
			//process row data
			int off = _containsID ? 1: 0;
			Object obj = _isVector ? arg0._1().get(off) : arg0._1();
			boolean sparse = (obj instanceof SparseVector);
			MatrixBlock mb = new MatrixBlock(1, (int)_clen, sparse);
			
			if( _isVector ) {
				Vector vect = (Vector) obj;
				if( vect instanceof SparseVector ) {
					SparseVector svect = (SparseVector) vect;
					int lnnz = svect.numNonzeros();
					for( int k=0; k<lnnz; k++ )
						mb.appendValue(0, svect.indices()[k], svect.values()[k]);
				}
				else { //dense
					for( int j=0; j<_clen; j++ )
						mb.appendValue(0, j, vect.apply(j));	
				}
			}
			else { //row
				Row row = (Row) obj;
				for( int j=off; j<off+_clen; j++ )
					mb.appendValue(0, j-off, UtilFunctions.getDouble(row.get(j)));
			}
			mb.examSparsity();
			
			return new Tuple2<Long, Writable>(rowix, 
					new PairWritableBlock(new MatrixIndexes(1,1),mb));
		}
	}
}
