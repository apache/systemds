/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;

import java.io.EOFException;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public abstract class TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	
	protected static String JSON_ATTRS 	= "attributes"; 
	protected static String JSON_MTHD 	= "methods"; 
	protected static String JSON_CONSTS = "constants"; 
	protected static String JSON_NBINS 	= "numbins"; 
	
	protected static String[] NAstrings = null;
	protected static String[] outputColumnNames = null;
	
	public static void init(String[] nastrings, String headerLine, String delim) {
		NAstrings = nastrings;
		outputColumnNames = headerLine.split(delim);
		for(int i=0; i < outputColumnNames.length; i++)
			outputColumnNames[i] = UtilFunctions.unquote(outputColumnNames[i]);
	}
	
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
	
	protected static long _numRecordsInPartFile = 0;	// Total number of records in the data file
	protected static long _numValidRecords = 0;			// (_numRecordsInPartFile - #of omitted records)

	abstract public void print();
	abstract public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TransformationAgent agent) throws IOException;
	abstract public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, JobConf job, TfAgents agents) throws IOException;
	
	abstract public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir) throws IOException;
	abstract public String[] apply(String[] words);
	
	protected static boolean checkValidInputFile(FileSystem fs, Path path, boolean err)
			throws IOException {
		// check non-existing file
		if (!fs.exists(path))
			if ( err )
				throw new IOException("File " + path.toString() + " does not exist on HDFS/LFS.");
			else
				return false;

		// check for empty file
		if (MapReduceTool.isFileEmpty(fs, path.toString()))
			if ( err )
			throw new EOFException("Empty input file " + path.toString() + ".");
			else
				return false;
		
		return true;
	}
	
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
