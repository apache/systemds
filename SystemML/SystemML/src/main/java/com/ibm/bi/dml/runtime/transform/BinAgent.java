/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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

import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class BinAgent extends TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		JSONObject obj = (JSONObject) parsedSpec.get(TX_METHOD.BIN.toString());
		
		if(obj == null) {
			return;
		}
		else {
			JSONArray attrs = (JSONArray) obj.get(JSON_ATTRS);
			//JSONArray mthds = (JSONArray) obj.get(JSON_MTHD);
			JSONArray nbins = (JSONArray) obj.get(JSON_NBINS);
			
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
	
	/** 
	 * Method to merge map output transformation metadata.
	 * 
	 * @param values
	 * @return
	 * @throws IOException 
	 */
	@Override
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, JobConf job) throws IOException {
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
		Path pt=new Path(outputDir+"/Bin/"+ columnNames[colID-1] + BIN_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		double binwidth = (max-min)/nbins;
		br.write(colID + TXMTD_SEP + Double.toString(min) + TXMTD_SEP + Double.toString(max) + TXMTD_SEP + Double.toString(binwidth) + TXMTD_SEP + Integer.toString(nbins) + "\n");
		br.close();
	}
	
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs) throws IOException {
		if(_binList == null)
			return;
		
		for(int i=0; i < _binList.length; i++) {
			int colID = _binList[i];
			
			Path pt=new Path(outputDir+"/Bin/"+ columnNames[colID-1] + BIN_FILE_SUFFIX);
			BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			double binwidth = (_max[i] - _min[i])/_numBins[i];
			br.write(colID + TXMTD_SEP + Double.toString(_min[i]) + TXMTD_SEP + Double.toString(_max[i]) + TXMTD_SEP + Double.toString(binwidth) + TXMTD_SEP + Integer.toString(_numBins[i]) + "\n");
			br.close();
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
				
				Path path = new Path( txMtdDir + "/Bin/" + columnNames[colID-1] + BIN_FILE_SUFFIX);
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
			
			double val = UtilFunctions.parseToDouble(words[colID-1]);
			int binid = 1;
			double tmp = _min[i] + _binWidths[i];
			while(val > tmp && binid <= _numBins[i]) {
				tmp += _binWidths[i];
				binid++;
			}
			words[colID-1] = Integer.toString(binid);
			
			//long binid = Math.min(Math.round((val-_min[i])/_binWidths[i] - 0.5) + 1, _numBins[i]);
			//words[colID-1] = Long.toString(binid);
		}
		
		return words;
	}
	
	/**
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	@Override
	public int isTransformed(int colID)
	{
		if(_binList == null)
			return -1;
		
		for(int i=0; i < _binList.length; i++)
			if( _binList[i] == colID )
				return i;
		
		return -1;
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
