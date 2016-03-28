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
import java.io.Serializable;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

public abstract class TransformationAgent implements Serializable {
	
	private static final long serialVersionUID = -2995384194257356337L;
	
	public static enum TX_METHOD { 
		IMPUTE ("impute"), 
		RECODE ("recode"), 
		BIN ("bin"), 
		DUMMYCODE ("dummycode"), 
		SCALE ("scale"),
		OMIT ("omit"),
		MVRCD ("mvrcd");
		
		private String _name;
		
		TX_METHOD(String name) { _name = name; }
		
		public String toString() {
			return _name;
		}
	}
	
	public static final String JSON_ATTRS 	= "attributes"; 
	public static final String JSON_MTHD 	= "methods"; 
	public static final String JSON_CONSTS = "constants"; 
	public static final String JSON_NBINS 	= "numbins"; 
	
	protected static final String MV_FILE_SUFFIX 		= ".impute";
	protected static final String RCD_MAP_FILE_SUFFIX 	= ".map";
	protected static final String NDISTINCT_FILE_SUFFIX = ".ndistinct";
	protected static final String MODE_FILE_SUFFIX 		= ".mode";
	protected static final String BIN_FILE_SUFFIX 		= ".bin";
	protected static final String SCALE_FILE_SUFFIX		= ".scale";
	protected static final String DCD_FILE_NAME 		= "dummyCodeMaps.csv";
	protected static final String COLTYPES_FILE_NAME 	= "coltypes.csv";
	
	protected static final String TXMTD_SEP 	= ",";
	protected static final String DCD_NAME_SEP 	= "_";
	
	protected static final String OUT_HEADER = "column.names";
	protected static final String OUT_DCD_HEADER = "dummycoded.column.names";
	
	abstract public void print();
	abstract public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TfUtils agents) throws IOException;
	abstract public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, FileSystem fs, TfUtils agents) throws IOException;
	
	abstract public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents) throws IOException;
	abstract public String[] apply(String[] words, TfUtils agents);
	
	protected enum ColumnTypes { SCALE, NOMINAL, ORDINAL, DUMMYCODED, INVALID }
	protected byte columnTypeToID(ColumnTypes type) throws IOException { 
		switch(type) 
		{
		case SCALE: return 1;
		case NOMINAL: return 2;
		case ORDINAL: return 3;
		case DUMMYCODED: return 1; // Ideally, dummycoded columns should be of a different type. Treating them as SCALE is incorrect, semantically.
		default:
			throw new IOException("Invalid Column Type: " + type);
		}
	}
}
