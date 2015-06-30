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
	public static enum TX_METHOD { 
		IMPUTE ("impute"), 
		RECODE ("recode"), 
		BIN ("bin"), 
		DUMMYCODE ("dummycode"), 
		SCALE ("scale");
		
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
	protected static String[] columnNames = null;
	
	public static void init(String[] nastrings, String headerLine, String delim) {
		NAstrings = nastrings;
		columnNames = headerLine.split(delim);
		for(int i=0; i < columnNames.length; i++)
			columnNames[i] = UtilFunctions.unquote(columnNames[i]);
	}
	
	protected static final String MV_FILE_SUFFIX 		= ".impute";
	protected static final String RCD_MAP_FILE_SUFFIX 	= ".map";
	protected static final String NDISTINCT_FILE_SUFFIX = ".ndistinct";
	protected static final String MODE_FILE_SUFFIX 		= ".mode";
	protected static final String BIN_FILE_SUFFIX 		= ".bin";
	protected static final String DCD_FILE_NAME 		= "dummyCodeMaps.csv";
	
	protected static final String TXMTD_SEP 	= ",";
	protected static final String DCD_NAME_SEP 	= "_";
	
	abstract public void print();
	abstract public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID) throws IOException;
	abstract public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, JobConf job) throws IOException;
	
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
	

}
