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
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Mean;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class MVImputeAgent extends TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String MEAN_PREFIX = "mean";
	public static final String CORRECTION_PREFIX = "correction";
	public static final String COUNT_PREFIX = "count";
	
	private int[] _mvList = null;
	
	/* 
	 * Imputation Methods:
	 * 1 - global_mean
	 * 2 - global_mode
	 * 3 - constant
	 * 
	 */
	private byte[] _mvMethodList = null;
	
	private Mean _meanFn = Mean.getMeanFnObject();
	private KahanObject[] _meanList = null; 
	private long[] _countList = null;	// number of non-missing values
	
	private String[] _replacementList = null;	// replacements: for global_mean, mean; and for global_mode, recode id of mode category
	private String[] _modeList = null;
	
	MVImputeAgent() {
		
	}
	
	MVImputeAgent(JSONObject parsedSpec) {
		JSONObject obj = (JSONObject) parsedSpec.get(TX_METHOD.IMPUTE.toString());
		if(obj == null) {
			// MV Impute is not applicable
			_mvList = null;
			_mvMethodList = null;
			_meanList = null;
			_countList = null;
			_replacementList = null;
		}
		else {
			JSONArray attrs = (JSONArray) obj.get(JSON_ATTRS);
			JSONArray mthds = (JSONArray) obj.get(JSON_MTHD);
			
			assert(attrs.size() == mthds.size());
			assert(attrs.size() == _mvList.length);
			
			_mvList = new int[attrs.size()];
			_mvMethodList = new byte[attrs.size()];
			
			_meanList = new KahanObject[attrs.size()];
			_countList = new long[attrs.size()];
			
			for(int i=0; i < _mvList.length; i++) {
				_mvList[i] = ((Long) attrs.get(i)).intValue();
				_mvMethodList[i] = ((Long) mthds.get(i)).byteValue(); 
				_meanList[i] = new KahanObject(0, 0);
			}
			
			_modeList = new String[attrs.size()];			// contains replacements for "categorical" columns
			_replacementList = new String[attrs.size()]; 	// contains replacements for "scale" columns, including computed means as well constants
			
			JSONArray constants = (JSONArray)obj.get(JSON_CONSTS);
			for(int i=0; i < constants.size(); i++) {
				if ( constants.get(i) == null )
					_replacementList[i] = "NaN";
				else
					_replacementList[i] = constants.get(i).toString();
			}
		}
	}
	
	public static boolean isNA(String w, String[] naStrings) {
		if(naStrings == null)
			return false;
		
		for(String na : naStrings) {
			if(w.equals(na))
				return true;
		}
		return false;
	}
	
	public boolean isNA(String w) {
		return MVImputeAgent.isNA(w, NAstrings);
	}
	
	public void prepare(String[] words) throws IOException {
		if ( _mvList == null )
			return;
		
		try {
			for(int i=0; i <_mvList.length; i++) {
				int colID = _mvList[i];
				
				String w = null;
				double d = 0;
				if(_mvMethodList[i] == 1) {
					// global_mean
					w = UtilFunctions.unquote(words[colID-1].trim());
					if(!isNA(w)) {
						d = UtilFunctions.parseToDouble(w);
						_countList[i]++;
						_meanFn.execute2(_meanList[i], d, _countList[i]);
					}
				}
				else {
					// global_mode or constant
					// Nothing to do here. Mode is computed using recode maps.
				}
			}
		} catch(Exception e) {
			throw new IOException(e);
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
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID) throws IOException {
		if ( _mvList == null )
			return;
		try { 
			for(int i=0; i < _mvList.length; i++) {
				int colID = _mvList[i];
				byte mthd = _mvMethodList[i];
				
				IntWritable iw = new IntWritable(-colID);
				if ( mthd == 1 ) {
					String s = null;
					s = MEAN_PREFIX + "_" + taskID + "_" + Double.toString(_meanList[i]._sum);
					out.collect(iw, new DistinctValue(s, -1L));
					s = CORRECTION_PREFIX + "_" + taskID + "_" + Double.toString(_meanList[i]._correction);
					out.collect(iw, new DistinctValue(s, -1L));
					s = COUNT_PREFIX + "_" + taskID + "_" + Long.toString(_countList[i]);
					out.collect(iw, new DistinctValue(s, -1L));
				}
				else {
					// nothing to do here
				}
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs) throws IOException {
		if(_mvList == null)
			return;
		
		for(int i=0; i < _mvList.length; i++) {
			int colID = _mvList[i];
			
			Path pt=new Path(outputDir+"/Impute/"+columnNames[colID-1]+ MV_FILE_SUFFIX);
			BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			if ( _countList[i] == 0 ) 
				br.write(colID + TXMTD_SEP + Double.toString(0.0) + "\n");
			else
				br.write(colID + TXMTD_SEP + Double.toString(_meanList[i]._sum) + "\n");
			br.close();
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
		double d;

		DistinctValue val = new DistinctValue();
		String w = null;
		
		class MeanObject {
			double mean, correction;
			long count;
			
			MeanObject() { }
			public String toString() {
				return mean + "," + correction + "," + count;
			}
		};
		HashMap<Integer, MeanObject> mapMeans = new HashMap<Integer, MeanObject>();
		
		while(values.hasNext()) {
			val.reset();
			val = values.next();
			w = val.getWord();
			
			if(w.startsWith(MEAN_PREFIX)) {
				String[] parts = w.split("_");
				int taskID = UtilFunctions.parseToInt(parts[1]);
				MeanObject mo = mapMeans.get(taskID);
				if ( mo==null ) 
					mo = new MeanObject();
				mo.mean = UtilFunctions.parseToDouble(parts[2]);
				mapMeans.put(taskID, mo);
			}
			else if (w.startsWith(CORRECTION_PREFIX)) {
				String[] parts = w.split("_");
				int taskID = UtilFunctions.parseToInt(parts[1]);
				MeanObject mo = mapMeans.get(taskID);
				if ( mo==null ) 
					mo = new MeanObject();
				mo.correction = UtilFunctions.parseToDouble(parts[2]);
				mapMeans.put(taskID, mo);
			}
			else if (w.startsWith(COUNT_PREFIX)) {
				String[] parts = w.split("_");
				int taskID = UtilFunctions.parseToInt(parts[1]);
				MeanObject mo = mapMeans.get(taskID);
				if ( mo==null ) 
					mo = new MeanObject();
				mo.count = UtilFunctions.parseToLong(parts[2]);
				mapMeans.put(taskID, mo);
			}
			else if(w.startsWith(BinAgent.MIN_PREFIX)) {
				d = UtilFunctions.parseToDouble( w.substring( BinAgent.MIN_PREFIX.length() ) );
				if ( d < min )
					min = d;
			}
			else if(w.startsWith(BinAgent.MAX_PREFIX)) {
				d = UtilFunctions.parseToDouble( w.substring( BinAgent.MAX_PREFIX.length() ) );
				if ( d > max )
					max = d;
			}
			else if (w.startsWith(BinAgent.NBINS_PREFIX)) {
				nbins = (int) UtilFunctions.parseToLong( w.substring(BinAgent.NBINS_PREFIX.length() ) );
			}
			else
				throw new RuntimeException("MVImputeAgent: Invalid prefix while merging map output: " + w);
		}
		
		// compute global mean across all map outputs
		KahanObject gmean = new KahanObject(0, 0);
		KahanPlus kp = KahanPlus.getKahanPlusFnObject();
		long gcount = 0;
		for(MeanObject mo : mapMeans.values()) {
			gcount = gcount + mo.count;
			if ( gcount > 0) {
				double delta = mo.mean - gmean._sum;
				kp.execute2(gmean, delta*mo.count/gcount);
				//_meanFn.execute2(gmean, mo.mean*mo.count, gcount);
			}
		}

		// write merged metadata
		FileSystem fs = FileSystem.get(job);
		Path pt=new Path(outputDir+"/Impute/"+columnNames[colID-1]+ MV_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		if ( gcount == 0 ) {
			br.write(colID + TXMTD_SEP + Double.toString(0.0) + "\n");
		}
		else
			br.write(colID + TXMTD_SEP + Double.toString(gmean._sum) + "\n");
		br.close();

		if ( min != Double.MAX_VALUE && max != Double.MIN_VALUE ) {
			pt=new Path(outputDir+"/Bin/"+ columnNames[colID-1] + BIN_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			double binwidth = (max-min)/nbins;
			br.write(colID + TXMTD_SEP + Double.toString(min) + TXMTD_SEP + Double.toString(max) + TXMTD_SEP + Double.toString(binwidth) + TXMTD_SEP + Integer.toString(nbins) + "\n");
			br.close();
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
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir) throws IOException {
		if ( _mvList == null )
			return;
		
		//Path txMtdDir = (DistributedCache.getLocalCacheFiles(job))[0];
		//FileSystem fs = FileSystem.getLocal(job);
		
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_mvList.length;i++) {
				int colID = _mvList[i];
				
				if ( _mvMethodList[i] == 1 ) {
					// global_mean
					Path path = new Path( txMtdDir + "/Impute/" + columnNames[colID-1] + MV_FILE_SUFFIX);
					TransformationAgent.checkValidInputFile(fs, path, true); 
					
					BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
					String line = br.readLine();
					_replacementList[i] = line.split(TXMTD_SEP)[1];
					br.close();
				}
				else if ( _mvMethodList[i] == 2 ) {
					// global_mode: located along with recode maps
					Path path = new Path( txMtdDir + "/Recode/" + columnNames[colID-1] + MODE_FILE_SUFFIX);
					TransformationAgent.checkValidInputFile(fs, path, true); 
					
					BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
					String line = br.readLine();

					int idxQuote = line.lastIndexOf('"');
					_modeList[i] = UtilFunctions.unquote(line.substring(0,idxQuote+1));	// mode in string form
					
					int idx = idxQuote+2;
					while(line.charAt(idx) != TXMTD_SEP.charAt(0))
						idx++;
					_replacementList[i] = line.substring(idxQuote+2,idx); // recode id of mode (unused)
					
					br.close();
				}
				else if ( _mvMethodList[i] == 3 ) {
					// constant: replace a missing value by a given constant
					// nothing to do. The constant values are loaded already during configure 
				}
				else {
					throw new RuntimeException("Invalid Missing Value Imputation methods: " + _mvMethodList[i]);
				}
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
		
		if ( _mvList == null )
			return words;
		
		for(int i=0; i < _mvList.length; i++) {
			int colID = _mvList[i];
			
			if(isNA(words[colID-1]))
				words[colID-1] = (_mvMethodList[i] == 2 ? _modeList[i] : _replacementList[i] );
		}
			
		return words;
	}
	
	public void print() {
		System.out.print("MV Imputation List: \n    ");
		for(int i : _mvList) {
			System.out.print(i + " ");
		}
		System.out.print("\n    ");
		for(byte b : _mvMethodList) {
			System.out.print(b + " ");
		}
		System.out.println();
	}

}
