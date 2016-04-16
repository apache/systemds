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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.CharacterCodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.MVImputeAgent.MVMethod;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.util.UtilFunctions;

public class BinAgent extends Encoder 
{	
	private static final long serialVersionUID = 1917445005206076078L;

	public static final String MIN_PREFIX = "min";
	public static final String MAX_PREFIX = "max";
	public static final String NBINS_PREFIX = "nbins";

	private int[] _numBins = null;
	private double[] _min=null, _max=null;	// min and max among non-missing values
	private double[] _binWidths = null;		// width of a bin for each attribute
	
	public BinAgent() {
		super( null );
	}
	
	public BinAgent(JSONObject parsedSpec) 
		throws JSONException 
	{
		super( null );		
		if ( !parsedSpec.containsKey(TfUtils.TXMETHOD_BIN) )
			return;
		
		JSONObject obj = (JSONObject) parsedSpec.get(TfUtils.TXMETHOD_BIN);		
		JSONArray attrs = (JSONArray) obj.get(TfUtils.JSON_ATTRS);
		JSONArray nbins = (JSONArray) obj.get(TfUtils.JSON_NBINS);
			
		initColList(attrs);
		_numBins = new int[attrs.size()];
		for(int i=0; i < _numBins.length; i++)
			_numBins[i] = UtilFunctions.toInt(nbins.get(i)); 
		
		// initialize internal transformation metadata
		_min = new double[_colList.length];
		Arrays.fill(_min, Double.MAX_VALUE);
		_max = new double[_colList.length];
		Arrays.fill(_max, -Double.MAX_VALUE);
		
		_binWidths = new double[_colList.length];
	}

	public int[] getNumBins() { return _numBins; }
	public double[] getMin()  { return _min; }
	public double[] getBinWidths() { return _binWidths; }
	
	public void prepare(String[] words, TfUtils agents) {
		if ( !isApplicable() )
			return;
		
		for(int i=0; i <_colList.length; i++) {
			int colID = _colList[i];
			
			String w = null;
			double d = 0;
				
			// equi-width
			w = UtilFunctions.unquote(words[colID-1].trim());
			if(!TfUtils.isNA(agents.getNAStrings(),w)) {
				d = UtilFunctions.parseToDouble(w);
				if(d < _min[i])
					_min[i] = d;
				if(d > _max[i])
					_max[i] = d;
			}
		}
	}
	
	private DistinctValue prepMinOutput(int idx) throws CharacterCodingException {
		String s =  MIN_PREFIX + Double.toString(_min[idx]);
		return  new DistinctValue(s, -1L);
	}
	
	private DistinctValue prepMaxOutput(int idx) throws CharacterCodingException {
		String s =  MAX_PREFIX + Double.toString(_max[idx]);
		return  new DistinctValue(s, -1L);
	}
	
	private DistinctValue prepNBinsOutput(int idx) throws CharacterCodingException {
		String s =  NBINS_PREFIX + Double.toString(_numBins[idx]);
		return  new DistinctValue(s, -1L);
	}
	
	/**
	 * Method to output transformation metadata from the mappers. 
	 * This information is collected and merged by the reducers.
	 * 
	 * @param out
	 * @throws IOException
	 */
	@Override
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TfUtils agents) throws IOException {
		if( !isApplicable() )
			return;
		
		try { 
			for(int i=0; i < _colList.length; i++) {
				int colID = _colList[i];
				IntWritable iw = new IntWritable(-colID);
				
				out.collect(iw,  prepMinOutput(i));
				out.collect(iw,  prepMaxOutput(i));
				out.collect(iw,  prepNBinsOutput(i));
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	public ArrayList<Pair<Integer, DistinctValue>> mapOutputTransformationMetadata(int taskID, ArrayList<Pair<Integer, DistinctValue>> list, TfUtils agents) throws IOException {
		if ( !isApplicable() )
			return list;
		
		try { 
			for(int i=0; i < _colList.length; i++) {
				int colID = _colList[i];
				Integer iw = -colID;
				
				list.add( new Pair<Integer,DistinctValue>(iw, prepMinOutput(i)) );
				list.add( new Pair<Integer,DistinctValue>(iw, prepMaxOutput(i)) );
				list.add( new Pair<Integer,DistinctValue>(iw, prepNBinsOutput(i)) );
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
		return list;
	}

	private void writeTfMtd(int colID, String min, String max, String binwidth, String nbins, String tfMtdDir, FileSystem fs, TfUtils agents) throws IOException 
	{
		Path pt = new Path(tfMtdDir+"/Bin/"+ agents.getName(colID) + TfUtils.BIN_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(colID + TfUtils.TXMTD_SEP + min + TfUtils.TXMTD_SEP + max + TfUtils.TXMTD_SEP + binwidth + TfUtils.TXMTD_SEP + nbins + "\n");
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
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, FileSystem fs, TfUtils agents) throws IOException {
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
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
		double binwidth = (max-min)/nbins;
		writeTfMtd(colID, Double.toString(min), Double.toString(max), Double.toString(binwidth), Integer.toString(nbins), outputDir, fs, agents);
	}
	
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs, TfUtils agents) throws IOException {
		if( !isApplicable() )
			return;
		
		MVImputeAgent mvagent = agents.getMVImputeAgent();
		for(int i=0; i < _colList.length; i++) {
			int colID = _colList[i];
			
			// If the column is imputed with a constant, then adjust min and max based the value of the constant.
			if ( mvagent.isApplicable(colID) != -1 && mvagent.getMethod(colID) == MVMethod.CONSTANT ) 
			{
				double cst = UtilFunctions.parseToDouble( mvagent.getReplacement(colID) );
				if ( cst < _min[i])
					_min[i] = cst;
				if ( cst > _max[i])
					_max[i] = cst;
			}
			
			double binwidth = (_max[i] - _min[i])/_numBins[i];
			writeTfMtd(colID, Double.toString(_min[i]), Double.toString(_max[i]), Double.toString(binwidth), Integer.toString(_numBins[i]), outputDir, fs, agents);
		}
	}
	
	// ------------------------------------------------------------------------------------------------

	/**
	 * Method to load transform metadata for all attributes
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents) throws IOException {
		if( !isApplicable() )
			return;
		
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_colList.length;i++) {
				int colID = _colList[i];
				
				Path path = new Path( txMtdDir + "/Bin/" + agents.getName(colID) + TfUtils.BIN_FILE_SUFFIX);
				TfUtils.checkValidInputFile(fs, path, true); 
					
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
				// format: colID,min,max,nbins
				String[] fields = br.readLine().split(TfUtils.TXMTD_SEP);
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
		if( !isApplicable() )
			return words;
	
		for(int i=0; i < _colList.length; i++) {
			int colID = _colList[i];
			try {
				double val = UtilFunctions.parseToDouble(words[colID-1]);
				int binid = 1;
				double tmp = _min[i] + _binWidths[i];
				while(val > tmp && binid < _numBins[i]) {
					tmp += _binWidths[i];
					binid++;
				}
				words[colID-1] = Integer.toString(binid);
			} 
			catch(NumberFormatException e) {
				throw new RuntimeException("Encountered \"" + words[colID-1] + "\" in column ID \"" + colID + "\", when expecting a numeric value. Consider adding \"" + words[colID-1] + "\" to na.strings, along with an appropriate imputation method.");
			}
		}
		
		return words;
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		return null;
	}
	
	@Override
	public double[] encode(String[] in, double[] out) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void build(String[] in) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void build(FrameBlock in) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		// TODO Auto-generated method stub
		return null;
	}
}
