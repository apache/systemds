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
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import scala.Tuple2;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;


public class ApplyTfCSVSPARK {
	
	/**
	 * Apply transformation metadata and generate the result in CSV format, as a
	 * JavaRDD of Strings.
	 */

	public static JavaPairRDD<Long, String> runSparkJob(
			SparkExecutionContext sec, JavaRDD<Tuple2<LongWritable, Text>> inputRDD, 
			String tfMtdPath, String spec, String tmpPath, CSVFileFormatProperties prop, 
			int numCols, String headerLine) 
		throws IOException, ClassNotFoundException, InterruptedException, IllegalArgumentException, JSONException 
	{
		// Load transformation metadata and broadcast it
		String[] naStrings = TfUtils.parseNAStrings(prop.getNAStrings());
		JSONObject jspec = new JSONObject(spec);
		TfUtils _tfmapper = new TfUtils(headerLine, prop.hasHeader(), prop.getDelim(), naStrings, jspec, numCols, tfMtdPath, null, tmpPath);
		
		_tfmapper.loadTfMetadata();

		Broadcast<TfUtils> bcast_tf = sec.getSparkContext().broadcast(_tfmapper);
		
		/*
		 * Construct transformation metadata (map-side) -- the logic is similar
		 * to GTFMTDMapper
		 * 
		 * Note: The result of mapPartitionsWithIndex is cached so that the
		 * transformed data is not redundantly computed multiple times
		 */
		JavaPairRDD<Long, String> applyRDD = inputRDD
				.mapPartitionsWithIndex( new ApplyTfCSVMap(bcast_tf),  true)
				.mapToPair(
						new PairFunction<String,Long,String>(){
							private static final long serialVersionUID = 3868143093999082931L;
							@Override
							public Tuple2<Long, String> call(String t) throws Exception {
								return new Tuple2<Long, String>(new Long(1), t);
							}
						}
				).cache();

		/*
		 * An action to force execution of apply()
		 * 
		 * We need to trigger the execution of this RDD so as to ensure the
		 * creation of a few metadata files (headers, dummycoded information,
		 * etc.), which are referenced in the caller function.
		 */
		applyRDD.count();
		
		return applyRDD;
	}

	public static class ApplyTfCSVMap implements Function2<Integer, Iterator<Tuple2<LongWritable, Text>>, Iterator<String>> {

		private static final long serialVersionUID = 1496686437276906911L;

		TfUtils _tfmapper = null;
		
		ApplyTfCSVMap(boolean hasHeader, String delim, String naStrings, String specFile, String tmpPath, String tfMtdPath, long numCols, String headerLine, Broadcast<TfUtils> tf) throws IllegalArgumentException, IOException, JSONException {
			_tfmapper = tf.getValue();
		}
		
		ApplyTfCSVMap(Broadcast<TfUtils> tf) throws IllegalArgumentException, IOException, JSONException {
			_tfmapper = tf.getValue();
		}
		
		@Override
		public Iterator<String> call(Integer partitionID,
				Iterator<Tuple2<LongWritable, Text>> csvLines) throws Exception {
			
			boolean first = true;
			Tuple2<LongWritable, Text> rec = null;
			ArrayList<String> outLines = new ArrayList<String>();
			
			while(csvLines.hasNext()) {
				rec = csvLines.next();
				
				if (first && partitionID == 0) {
					first = false;
					
					_tfmapper.processHeaderLine();
					
					if (_tfmapper.hasHeader() ) {
						//outLines.add(dcdHeader); // if the header needs to be preserved in the output file
						continue; 
					}
				}
				
				// parse the input line and apply transformation
			
				String[] words = _tfmapper.getWords(rec._2());
				
				if(!_tfmapper.omit(words))
				{
					try {
						words = _tfmapper.apply(words);
						String outStr = _tfmapper.checkAndPrepOutputString(words);
						outLines.add(outStr);
					} 
					catch(DMLRuntimeException e) {
						throw new RuntimeException(e.getMessage() + ": " + rec._2().toString());
					}
				}
			}
			
			return outLines.iterator();
		}
		
	}

	
}
