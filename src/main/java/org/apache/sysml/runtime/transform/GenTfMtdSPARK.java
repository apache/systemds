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

package org.apache.sysml.runtime.transform;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import scala.Tuple2;

import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.Pair;

public class GenTfMtdSPARK 
{
	/**
	 * Spark code to Generate Transform Metadata based on the given transformation
	 * specification file (JSON format).
	 * 
	 */
	public static long runSparkJob(SparkExecutionContext sec, JavaRDD<Tuple2<LongWritable, Text>> inputRDD, 
									String tfMtdPath, String spec, String partOffsetsFile, 
									CSVFileFormatProperties prop, long numCols, String headerLine) 
		throws IOException, ClassNotFoundException, InterruptedException, IllegalArgumentException, JSONException 
	{	
		// Construct transformation metadata (map-side)
		// Note: logic is similar to GTFMTDMapper
		JavaRDD<Tuple2<Integer,DistinctValue>> tfMapOutput 
			= inputRDD.mapPartitionsWithIndex(
					new GenTfMtdMap(prop.hasHeader(), prop.getDelim(), prop.getNAStrings(), 
									spec, numCols, headerLine), 
					true );
		
		// Shuffle to group by DistinctValue
		JavaPairRDD<Integer,Iterable<DistinctValue>> rdd = JavaPairRDD.fromJavaRDD(tfMapOutput).groupByKey();
		
		// Construct transformation metadata (Reduce-side)
		// Note: logic is similar to GTFMTDReducer
		JavaRDD<Long> out 
			= rdd.flatMap(new GenTfMtdReduce(prop.hasHeader(), prop.getDelim(), prop.getNAStrings(), 
									headerLine, tfMtdPath, partOffsetsFile, spec, numCols)  );
		
		// Compute the total number of transformed rows
		long numRows = out.reduce(new Function2<Long,Long,Long>() {
			private static final long serialVersionUID = 1263336168859959795L;

			@Override
			public Long call(Long v1, Long v2) throws Exception {
				return v1+v2;
			}			
		});
		
		return numRows;
	}
	
	// ----------------------------------------------------------------------------------------------------------------------
	
	private  static class GenTfMtdMap implements Function2<Integer, Iterator<Tuple2<LongWritable, Text>>, Iterator<Tuple2<Integer,DistinctValue>>> 
	{
		private static final long serialVersionUID = -5622745445470598215L;
		
		private TfUtils _agents = null;
		
		public GenTfMtdMap(boolean hasHeader, String delim, String naStrings, String spec, long numCols, String headerLine) throws IllegalArgumentException, IOException, JSONException {
			
			// Setup Transformation Agents
			String[] nas = TfUtils.parseNAStrings(naStrings);
			JSONObject jspec = new JSONObject(spec);
			_agents = new TfUtils(headerLine, hasHeader, delim, nas, jspec, numCols, null, null, null);

		}
		
		@Override
		public Iterator<Tuple2<Integer,DistinctValue>> call(Integer partitionID,
				Iterator<Tuple2<LongWritable, Text>> csvLines) throws Exception {
			
			// Construct transformation metadata by looping through csvLines
			// Note: logic is similar to GTFMTDMapper
			
			boolean first = true;
			Tuple2<LongWritable, Text> rec = null;
			long _offsetInPartFile = -1;
			
			while(csvLines.hasNext()) {
				rec = csvLines.next();
				
				if (first) {
					first = false;
					_offsetInPartFile = rec._1().get();
					
					if (partitionID == 0 && _agents.hasHeader() && _offsetInPartFile == 0 )
						continue; // skip the header line
				}
				
				_agents.prepareTfMtd(rec._2().toString());
			}
			
			// Prepare the output in the form of DistinctValues, which subsequently need to be grouped and aggregated. 
			
			ArrayList<Pair<Integer,DistinctValue>> outList = new ArrayList<Pair<Integer,DistinctValue>>();
			
			_agents.getMVImputeAgent().mapOutputTransformationMetadata(partitionID, outList, _agents);
			_agents.getRecodeAgent().mapOutputTransformationMetadata(partitionID, outList, _agents);
			_agents.getBinAgent().mapOutputTransformationMetadata(partitionID, outList, _agents);
			
			DistinctValue dv = new DistinctValue(new OffsetCount("Partition"+partitionID, _offsetInPartFile, _agents.getTotal()));
			Pair<Integer, DistinctValue> tuple = new Pair<Integer, DistinctValue>((int) (_agents.getNumCols()+1), dv); 
			outList.add(tuple);

			return toTuple2List(outList).iterator();
		}
		
	}
	
	// ------------------------------------------------------------------------------------------------
	
	private static class GenTfMtdReduce implements FlatMapFunction<Tuple2<Integer, Iterable<DistinctValue>>, Long> 
	{	
		private static final long serialVersionUID = -2733233671193035242L;
		private TfUtils _agents = null;
		
		public GenTfMtdReduce(boolean hasHeader, String delim, String naStrings, String headerLine, String tfMtdDir, String offsetFile, String spec, long numCols) throws IOException, JSONException {
			String[] nas = TfUtils.parseNAStrings(naStrings); 
			JSONObject jspec = new JSONObject(spec);
			_agents = new TfUtils(headerLine, hasHeader, delim, nas, jspec, numCols, tfMtdDir, offsetFile, null);
		}

		@SuppressWarnings("unchecked")
		@Override
		public Iterator<Long> call(Tuple2<Integer, Iterable<DistinctValue>> t)
				throws Exception {
			
			int colID = t._1();
			Iterator<DistinctValue> iterDV = t._2().iterator();

			JobConf job = new JobConf();
			FileSystem fs = FileSystem.get(job);
			
			ArrayList<Long> numRows = new ArrayList<Long>();
			
			if(colID < 0) 
			{
				// process mapper output for MV and Bin agents
				colID = colID*-1;
				_agents.getMVImputeAgent().mergeAndOutputTransformationMetadata(iterDV, _agents.getTfMtdDir(), colID, fs, _agents);
				numRows.add(0L);
			}
			else if ( colID == _agents.getNumCols() + 1)
			{
				// process mapper output for OFFSET_FILE
				ArrayList<OffsetCount> list = new ArrayList<OffsetCount>();
				while(iterDV.hasNext())
					list.add(new OffsetCount(iterDV.next().getOffsetCount()));
				Collections.sort(list);
				
				@SuppressWarnings("deprecation")
				SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, new Path(_agents.getOffsetFile()+"/part-00000"), ByteWritable.class, OffsetCount.class);
				
				long lineOffset=0;
				for(OffsetCount oc: list)
				{
					long count=oc.count;
					oc.count=lineOffset;
					writer.append(new ByteWritable((byte)0), oc);
					lineOffset+=count;
				}
				writer.close();
				list.clear();
				
				numRows.add(lineOffset);
			}
			else 
			{
				// process mapper output for Recode agent
				_agents.getRecodeAgent().mergeAndOutputTransformationMetadata(iterDV, _agents.getTfMtdDir(), colID, fs, _agents);
				numRows.add(0L);
			}
			
			return numRows.iterator();
		}
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static List<Tuple2<Integer,DistinctValue>> toTuple2List(List<Pair<Integer,DistinctValue>> in) {
		ArrayList<Tuple2<Integer,DistinctValue>> ret = new ArrayList<Tuple2<Integer,DistinctValue>>();
		for( Pair<Integer,DistinctValue> e : in )
			ret.add(new Tuple2<Integer,DistinctValue>(e.getKey(), e.getValue()));
		return ret;
	}
}
