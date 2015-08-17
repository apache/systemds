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

package com.ibm.bi.dml.runtime.transform;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

import com.ibm.bi.dml.runtime.transform.MVImputeAgent.MVMethod;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.JSONHelper;

public class BinAgent extends TransformationAgent {
	
	
	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	private int[] _binList = null;
	//private byte[] _binMethodList = null;	// Not used, since only equi-width is supported for now. 
	private int[] _numBins = null;

	private double[] _min=null, _max=null;	// min and max among non-missing values

	private double[] _binWidths = null;		// width of a bin for each attribute
	
	BinAgent() { }
	
	BinAgent(JSONObject parsedSpec) {
		JSONObject obj = (JSONObject) JSONHelper.get(parsedSpec,TX_METHOD.BIN.toString());
		
		if(obj == null) {
			return;
		}
		else {
			JSONArray attrs = (JSONArray) JSONHelper.get(obj,JSON_ATTRS);
			//JSONArray mthds = (JSONArray) JSONHelper.get(obj,JSON_MTHD);
			JSONArray nbins = (JSONArray) JSONHelper.get(obj,JSON_NBINS);
			
			assert(attrs.size() == nbins.size());
			
			_binList = new int[attrs.size()];
			_numBins = new int[attrs.size()];
			for(int i=0; i < _binList.length; i++) {
				_binList[i] = ((Long) attrs.get(i)).intValue();
				_numBins[i] = ((Long) nbins.get(i)).intValue(); 
			}
			
			// initialize internal transformation metadata
			_min = new double[_binList.length];
			Arrays.fill(_min, Double.MAX_VALUE);
			_max = new double[_binList.length];
			Arrays.fill(_max, Double.MIN_VALUE);
			
			_binWidths = new double[_binList.length];
		}
	}
	
	public void prepare(String[] words) {
		if ( _binList == null )
			return;
		
		for(int i=0; i <_binList.length; i++) {
			int colID = _binList[i];
			
			String w = null;
			double d = 0;
				
			// equi-width
			w = UtilFunctions.unquote(words[colID-1].trim());
			if(!MVImputeAgent.isNA(w, NAstrings)) {
				d = UtilFunctions.parseToDouble(w);
				if(d < _min[i])
					_min[i] = d;
				if(d > _max[i])
					_max[i] = d;
			}
		}
	}
	
	/**
	 * Method to output transformation metadata from the mappers. 
	 * This information is collected and merged by the reducers.
	 * 
	 * @param out
	 * @throws IOException
	 */
	@Override
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TransformationAgent agent) throws IOException {
		if ( _binList == null )
			return;
		
		try { 
			for(int i=0; i < _binList.length; i++) {
				int colID = _binList[i];
				int nbins = _numBins[i];
				
				IntWritable iw = new IntWritable(-colID);
				
				String s = null;
				s = MIN_PREFIX + Double.toString(_min[i]);
				out.collect(iw, new DistinctValue(s, -1L));
				s = MAX_PREFIX + Double.toString(_max[i]);
				out.collect(iw, new DistinctValue(s, -1L));
				s = NBINS_PREFIX + Long.toString((long)nbins);
				out.collect(iw, new DistinctValue(s, -1L));
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	private void writeTfMtd(int colID, String min, String max, String binwidth, String nbins, String tfMtdDir, FileSystem fs) throws IOException 
	{
		Path pt = new Path(tfMtdDir+"/Bin/"+ outputColumnNames[colID-1] + BIN_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(colID + TXMTD_SEP + min + TXMTD_SEP + max + TXMTD_SEP + binwidth + TXMTD_SEP + nbins + "\n");
		br.close();
	}

	/** 
	 * Method to merge map output transformation metadata.
	 * 
	 * @param values
	 * @return
	 * @throws IOException 
	 */
	@Override
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, JobConf job, TfAgents agents) throws IOException {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		int nbins = 0;
		
		DistinctValue val = new DistinctValue();
		String w = null;
		double d;
		while(values.hasNext()) {
			val.reset();
			val = values.next();
			w = val.getWord();
			
			if(w.startsWith(MIN_PREFIX)) {
				d = UtilFunctions.parseToDouble(w.substring( MIN_PREFIX.length() ));
				if ( d < min )
					min = d;
			}
			else if(w.startsWith(MAX_PREFIX)) {
				d = UtilFunctions.parseToDouble(w.substring( MAX_PREFIX.length() ));
				if ( d > max )
					max = d;
			}
			else if (w.startsWith(NBINS_PREFIX)) {
				nbins = (int) UtilFunctions.parseToLong( w.substring(NBINS_PREFIX.length() ) );
			}
			else
				throw new RuntimeException("MVImputeAgent: Invalid prefix while merging map output: " + w);
		}
		
		// write merged metadata
		FileSystem fs = FileSystem.get(job);
		double binwidth = (max-min)/nbins;
		writeTfMtd(colID, Double.toString(min), Double.toString(max), Double.toString(binwidth), Integer.toString(nbins), outputDir, fs);
	}
	
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs, MVImputeAgent mvagent) throws IOException {
		if(_binList == null)
			return;
		
		for(int i=0; i < _binList.length; i++) {
			int colID = _binList[i];
			
			// If the column is imputed with a constant, then adjust min and max based the value of the constant.
			if ( mvagent.isImputed(colID) != -1 && mvagent.getMethod(colID) == MVMethod.CONSTANT ) 
			{
				double cst = UtilFunctions.parseToDouble( mvagent.getReplacement(colID) );
				if ( cst < _min[i])
					_min[i] = cst;
				if ( cst > _max[i])
					_max[i] = cst;
			}
			
			double binwidth = (_max[i] - _min[i])/_numBins[i];
			writeTfMtd(colID, Double.toString(_min[i]), Double.toString(_max[i]), Double.toString(binwidth), Integer.toString(_numBins[i]), outputDir, fs);
		}
	}
	
	// ------------------------------------------------------------------------------------------------

	public int[] getBinList() { return _binList; }
	public int[] getNumBins() { return _numBins; }
	
	/**
	 * Method to load transform metadata for all attributes
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir) throws IOException {
		if ( _binList == null )
			return;
		
		//Path txMtdDir = (DistributedCache.getLocalCacheFiles(job))[0];
		//FileSystem fs = FileSystem.getLocal(job);
		
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_binList.length;i++) {
				int colID = _binList[i];
				
				Path path = new Path( txMtdDir + "/Bin/" + outputColumnNames[colID-1] + BIN_FILE_SUFFIX);
				TransformationAgent.checkValidInputFile(fs, path, true); 
					
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
				// format: colID,min,max,nbins
				String[] fields = br.readLine().split(TXMTD_SEP);
				double min = UtilFunctions.parseToDouble(fields[1]);
				//double max = UtilFunctions.parseToDouble(fields[2]);
				double binwidth = UtilFunctions.parseToDouble(fields[3]);
				int nbins = UtilFunctions.parseToInt(fields[4]);
				
				_numBins[i] = nbins;
				_min[i] = min;
				_binWidths[i] = binwidth; // (max-min)/nbins;
				
				br.close();
			}
		}
		else {
			fs.close();
			throw new RuntimeException("Path to recode maps must be a directory: " + txMtdDir);
		}
	}
	
	/**
	 * Method to apply transformations.
	 * 
	 * @param words
	 * @return
	 */
	@Override
	public String[] apply(String[] words) {
		if ( _binList == null )
			return words;
	
		for(int i=0; i < _binList.length; i++) {
			int colID = _binList[i];
			
			try {
			double val = UtilFunctions.parseToDouble(words[colID-1]);
			int binid = 1;
			double tmp = _min[i] + _binWidths[i];
			while(val > tmp && binid < _numBins[i]) {
				tmp += _binWidths[i];
				binid++;
			}
			words[colID-1] = Integer.toString(binid);
			} catch(NumberFormatException e)
			{
				throw new RuntimeException("Encountered \"" + words[colID-1] + "\" in column ID \"" + colID + "\", when expecting a numeric value. Consider adding \"" + words[colID-1] + "\" to na.strings, along with an appropriate imputation method.");
			}
			//long binid = Math.min(Math.round((val-_min[i])/_binWidths[i] - 0.5) + 1, _numBins[i]);
			//words[colID-1] = Long.toString(binid);
		}
		
		return words;
	}
	
	/**
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	public int isBinned(int colID)
	{
		if(_binList == null)
			return -1;
		
		int idx = Arrays.binarySearch(_binList, colID);
		return ( idx >= 0 ? idx : -1);
	}


	@Override
	public void print() {
		System.out.print("Binning List (Equi-width): \n    ");
		for(int i : _binList) {
			System.out.print(i + " ");
		}
		System.out.print("\n    ");
		for(int b : _numBins) {
			System.out.print(b + " ");
		}
		System.out.println();
	}

}
