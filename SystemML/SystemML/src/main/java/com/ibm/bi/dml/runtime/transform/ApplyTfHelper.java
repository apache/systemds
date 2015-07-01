/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.regex.Pattern;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONObject;


@SuppressWarnings("deprecation")
public class ApplyTfHelper {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	boolean _hasHeader = false;
	String _delimString = null;
	Pattern _delim = null;
	String[] _naStrings = null;
	String _specFile = null;
	long _numCols = 0;
	JobConf _rJob = null;
	String _tmpPath = null;
	
	boolean _partFileWithHeader = false;
	
	MVImputeAgent _mia = null;
	RecodeAgent _ra = null;	
	BinAgent _ba = null;
	DummycodeAgent _da = null;

	public ApplyTfHelper(JobConf job) throws IllegalArgumentException, IOException {
		_hasHeader = Boolean.parseBoolean(job.get(MRJobConfiguration.TF_HAS_HEADER));
		
		_delimString = job.get(MRJobConfiguration.TF_DELIM);
		_delim = Pattern.compile(Pattern.quote(_delimString));
		
		if ( job.get(MRJobConfiguration.TF_NA_STRINGS) == null)
			_naStrings = null;
		else
			_naStrings = Pattern.compile(Pattern.quote(DataExpression.DELIM_NA_STRING_SEP)).split(job.get(MRJobConfiguration.TF_NA_STRINGS), -1);
		
		_numCols = UtilFunctions.parseToLong( job.get(MRJobConfiguration.TF_NUM_COLS) );		// #of columns in input data
		_tmpPath = job.get(MRJobConfiguration.TF_TMP_LOC);
		
		_specFile = job.get(MRJobConfiguration.TF_SPEC_FILE);
		
		_rJob = job;
	}
	
	public JSONObject parseSpec() throws IOException
	{
		FileSystem fs = FileSystem.get(_rJob);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(_specFile))));
		JSONObject obj = JSONObject.parse(br);
		br.close();
		return obj;
	}
	
	public void setupTfAgents(JSONObject spec) 
	{
		// Set up transformation agents
		TransformationAgent.init(_naStrings, _rJob.get(MRJobConfiguration.TF_HEADER), _delimString);
		_mia = new MVImputeAgent(spec);
		_ra = new RecodeAgent(spec);
		_ba = new BinAgent(spec);
		_da = new DummycodeAgent(spec, _numCols);
	}

	public void loadTfMetadata (JSONObject spec) throws IOException
	{
		Path txMtdDir = (DistributedCache.getLocalCacheFiles(_rJob))[0];
		FileSystem localFS = FileSystem.getLocal(_rJob);
		
		// load transformation metadata 
		_mia.loadTxMtd(_rJob, localFS, txMtdDir);
		_ra.loadTxMtd(_rJob, localFS, txMtdDir);
		_ba.loadTxMtd(_rJob, localFS, txMtdDir);
		
		// associate recode maps and bin definitions with dummycoding agent,
		// as recoded and binned columns are typically dummycoded
		_da.setRecodeMaps( _ra.getRecodeMaps() );
		_da.setNumBins(_ba.getBinList(), _ba.getNumBins());
		_da.loadTxMtd(_rJob, localFS, txMtdDir);

		FileSystem fs;
		fs = FileSystem.get(_rJob);
		Path thisPath=new Path(_rJob.get("map.input.file")).makeQualified(fs);
		String thisfile=thisPath.toString();
			
		Path smallestFilePath=new Path(_rJob.get(MRJobConfiguration.TF_SMALLEST_FILE)).makeQualified(fs);
		if(thisfile.toString().equals(smallestFilePath.toString()))
			_partFileWithHeader=true;
		else
			_partFileWithHeader = false;
			
	}

	public int processHeaderLine(Text rawValue) throws IOException 
	{
		String header = null;
		if(_hasHeader)
			header = rawValue.toString();
		else
			header = _rJob.get(MRJobConfiguration.TF_HEADER);
		
		String dcdHeader = _da.constructDummycodedHeader(header, _delimString);
		_da.generateDummycodeMaps(FileSystem.get(_rJob), _tmpPath, (int) _numCols, header, _delimString);
		
		// write header information (before and after transformation) to temporary path
		// these files are copied into txMtdPath, once the ApplyTf job is complete.
		DataTransform.generateHeaderFiles(FileSystem.get(_rJob), _tmpPath, header, dcdHeader);

		return _delim.split(dcdHeader, -1).length;
	}
	
	public String[] getWords(Text line)
	{
		return _delim.split(line.toString(), -1);
	}
	
	public String[] apply ( String[] words ) 
	{
		words = _mia.apply(words);
		words = _ra.apply(words);
		words = _ba.apply(words);
		words = _da.apply(words);
		return words;
	}
	
	public void check(String []words) throws DMLRuntimeException 
	{
		boolean checkEmptyString = false;
		if ( TransformationAgent.NAstrings != null && !MVImputeAgent.isNA("", TransformationAgent.NAstrings) ) {
			checkEmptyString = true;
		}
		
		if ( checkEmptyString ) 
		{
			final String msg = "When na.strings are provided, empty string \"\" is considered as a missing value, and it must be imputed appropriately. Encountered an unhandled empty string in line: ";
			for(int i=0; i<words.length; i++) 
				if ( words[i] != null && words[i].equals(""))
					throw new DMLRuntimeException(msg);
		}
	}
	
	public String checkAndPrepOutputString(String []words) throws DMLRuntimeException 
	{
		/*
		 * Check if empty strings ("") have to be handled.
		 * 
		 * Unless na.strings are provided, empty strings are (implicitly) considered as value zero.
		 * When na.strings are provided, then "" is considered a missing value indicator, and the 
		 * user is expected to provide an appropriate imputation method. Therefore, when na.strings 
		 * are provided, "" encountered in any column (after all transformations are applied) 
		 * denotes an erroneous condition.  
		 */
		boolean checkEmptyString = false;
		if ( TransformationAgent.NAstrings != null && !MVImputeAgent.isNA("", TransformationAgent.NAstrings) ) {
			checkEmptyString = true;
		}
		
		StringBuilder sb = new StringBuilder();
		int i =0;
		
		if ( checkEmptyString ) 
		{
			final String msg = "When na.strings are provided, empty string \"\" is considered as a missing value, and it must be imputed appropriately. Encountered an unhandled empty string in line: ";
			if ( words[0] != null ) 
				if ( words[0].equals("") )
					throw new DMLRuntimeException( msg );
				else 
					sb.append(words[0]);
			else
				sb.append("");
			
			for(i=1; i<words.length; i++) 
			{
				sb.append(_delimString);
				
				if ( words[i] != null ) 
					if ( words[i].equals("") )
						throw new DMLRuntimeException(msg);
					else 
						sb.append(words[i]);
				else
					sb.append("");
			}
		}
		else 
		{
			sb.append(words[0] != null ? words[0] : "");
			for(i=1; i<words.length; i++) 
			{
				sb.append(_delimString);
				sb.append(words[i] != null ? words[i] : "");
			}
		}
		
		return sb.toString();
	}
	
}
