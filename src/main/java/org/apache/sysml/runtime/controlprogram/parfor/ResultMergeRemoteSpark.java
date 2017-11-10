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


import java.util.Arrays;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockPairFunction;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Statistics;

/**
 * MR job class for submitting parfor result merge MR jobs.
 * 
 */
public class ResultMergeRemoteSpark extends ResultMerge
{	
	
	private ExecutionContext _ec = null;
	private int  _numMappers = -1;
	private int  _numReducers = -1;
	
	public ResultMergeRemoteSpark(MatrixObject out, MatrixObject[] in, String outputFilename, ExecutionContext ec, int numMappers, int numReducers) 
	{
		super(out, in, outputFilename);
		
		_ec = ec;
		_numMappers = numMappers;
		_numReducers = numReducers;
	}

	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		//graceful degradation to parallel merge
		return executeParallelMerge( _numMappers );
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)

		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (remote, spark): Execute serial merge for output "
				+_output.hashCode()+" (fname="+_output.getFileName()+")");

		try
		{
			if( _inputs != null && _inputs.length>0 )
			{
				//prepare compare
				MetaDataFormat metadata = (MetaDataFormat) _output.getMetaData();
				MatrixCharacteristics mcOld = metadata.getMatrixCharacteristics();
				MatrixObject compare = (mcOld.getNonZeros()==0) ? null : _output;
				
				//actual merge
				RDDObject ro = executeMerge(compare, _inputs, mcOld.getRows(), mcOld.getCols(), mcOld.getRowsPerBlock(), mcOld.getColsPerBlock());
				
				//create new output matrix (e.g., to prevent potential export<->read file access conflict
				moNew = new MatrixObject(_output.getValueType(), _outputFName);
				OutputInfo oiOld = metadata.getOutputInfo();
				InputInfo iiOld = metadata.getInputInfo();
				MatrixCharacteristics mc = new MatrixCharacteristics(mcOld.getRows(),mcOld.getCols(),
						                                             mcOld.getRowsPerBlock(),mcOld.getColsPerBlock());
				mc.setNonZeros( computeNonZeros(_output, Arrays.asList(_inputs)) );
				MetaDataFormat meta = new MetaDataFormat(mc,oiOld,iiOld);
				moNew.setMetaData( meta );
				moNew.setRDDHandle( ro );
			}
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return moNew;		
	}

	@SuppressWarnings("unchecked")
	protected RDDObject executeMerge(MatrixObject compare, MatrixObject[] inputs, long rlen, long clen, int brlen, int bclen)
		throws DMLRuntimeException 
	{
		String jobname = "ParFor-RMSP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)_ec;
		boolean withCompare = (compare!=null);

		RDDObject ret = null;
		
		//determine degree of parallelism
		int numRed = (int)determineNumReducers(rlen, clen, brlen, bclen, _numReducers);
		
		//sanity check for empty src files
		if( inputs == null || inputs.length==0  )
			throw new DMLRuntimeException("Execute merge should never be called with no inputs.");
		
		try
		{
			//note: initial implementation via union over all result rdds discarded due to 
			//stack overflow errors with many parfor tasks, and thus many rdds
			
			//Step 1: construct input rdd from all result files of parfor workers
			//a) construct job conf with all files
			InputInfo ii = InputInfo.BinaryBlockInputInfo;
			JobConf job = new JobConf( ResultMergeRemoteMR.class );
			job.setJobName(jobname);
			job.setInputFormat(ii.inputFormatClass);
			Path[] paths = new Path[ inputs.length ];
			for(int i=0; i<paths.length; i++) {
				//ensure input exists on hdfs (e.g., if in-memory or RDD)
				inputs[i].exportData();
				paths[i] = new Path( inputs[i].getFileName() );
				//update rdd handle to allow lazy evaluation by guarding 
				//against cleanup of temporary result files
				setRDDHandleForMerge(inputs[i], sec);
			}
			FileInputFormat.setInputPaths(job, paths);
			
			//b) create rdd from input files w/ deep copy of keys and blocks
			JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = sec.getSparkContext()
					.hadoopRDD(job, ii.inputFormatClass, ii.inputKeyClass, ii.inputValueClass)
					.mapPartitionsToPair(new CopyBlockPairFunction(true), true);
			
			//Step 2a: merge with compare
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
			if( withCompare )
			{
				JavaPairRDD<MatrixIndexes, MatrixBlock> compareRdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) 
						sec.getRDDHandleForMatrixObject(compare, InputInfo.BinaryBlockInputInfo);
		    	
				//merge values which differ from compare values
				ResultMergeRemoteSparkWCompare cfun = new ResultMergeRemoteSparkWCompare();
				out = rdd.groupByKey(numRed) //group all result blocks per key
		    			.join(compareRdd)   //join compare block and result blocks 
		    			.mapToPair(cfun);   //merge result blocks w/ compare
			}
			//Step 2b: merge without compare
			else {
				//direct merge in any order (disjointness guaranteed)
				out = RDDAggregateUtils.mergeByKey(rdd, false);
			}
		    
			//Step 3: create output rdd handle w/ lineage
			ret = new RDDObject(out);
			for(int i=0; i<paths.length; i++)
				ret.addLineageChild(inputs[i].getRDDHandle());
			if( withCompare )
				ret.addLineageChild(compare.getRDDHandle());
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException(ex);
		}	    
		
		//maintain statistics
		Statistics.incrementNoOfCompiledSPInst();
		Statistics.incrementNoOfExecutedSPInst();
		if( DMLScript.STATISTICS ){
			Statistics.maintainCPHeavyHitters(jobname, System.nanoTime()-t0);
		}
		
		return ret;
	}

	private static int determineNumReducers(long rlen, long clen, int brlen, int bclen, long numRed) {
		//set the number of mappers and reducers 
		long reducerGroups = Math.max(rlen/brlen,1) * Math.max(clen/bclen, 1);
		int ret = (int)Math.min( numRed, reducerGroups );
		
		return ret; 	
	}
	
	@SuppressWarnings("unchecked")
	private static void setRDDHandleForMerge(MatrixObject mo, SparkExecutionContext sec) {
		InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
		JavaSparkContext sc = sec.getSparkContext();
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = (JavaPairRDD<MatrixIndexes,MatrixBlock>) 
			sc.hadoopFile( mo.getFileName(), iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
		RDDObject rddhandle = new RDDObject(rdd);
		rddhandle.setHDFSFile(true);
		mo.setRDDHandle(rddhandle);
	}
}
