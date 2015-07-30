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
import java.nio.charset.CharacterCodingException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Mean;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.KahanObject;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class MVImputeAgent extends TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String MEAN_PREFIX = "mean";
	public static final String VARIANCE_PREFIX = "var";
	public static final String CORRECTION_PREFIX = "correction";
	public static final String COUNT_PREFIX = "validcount";		// #of valid or non-missing values in a column
	public static final String TOTAL_COUNT_PREFIX = "totalcount";	// #of total records processed by a mapper
	public static final String CONSTANT_PREFIX = "constant";
	
	public enum MVMethod { INVALID, GLOBAL_MEAN, GLOBAL_MODE, CONSTANT };
	
	private int[] _mvList = null;
	/* 
	 * Imputation Methods:
	 * 1 - global_mean
	 * 2 - global_mode
	 * 3 - constant
	 * 
	 */
	private byte[] _mvMethodList = null;
	private byte[] _mvscMethodList = null;	// scaling methods for attributes that are imputed and also scaled
	
	private BitSet _isMVScaled = null;
	private CM _varFn = CM.getCMFnObject(AggregateOperationTypes.VARIANCE);		// function object that understands variance computation
	
	// objects required to compute mean and variance of all non-missing entries 
	private Mean _meanFn = Mean.getMeanFnObject();	// function object that understands mean computation
	private KahanObject[] _meanList = null; 		// column-level means, computed so far
	private long[] _countList = null;				// #of non-missing values
	
	private CM_COV_Object[] _varList = null;		// column-level variances, computed so far (for scaling)
	

	private int[] 			_scnomvList = null;			// List of attributes that are scaled but not imputed
	private byte[]			_scnomvMethodList = null;	// scaling methods: 0 for invalid; 1 for mean-subtraction; 2 for z-scoring
	private KahanObject[] 	_scnomvMeanList = null;		// column-level means, for attributes scaled but not imputed
	private long[] 			_scnomvCountList = null;	// #of non-missing values, for attributes scaled but not imputed
	private CM_COV_Object[] _scnomvVarList = null;		// column-level variances, computed so far
	
	
	private String[] _replacementList = null;		// replacements: for global_mean, mean; and for global_mode, recode id of mode category
	private String[] _modeList = null;
	
	MVImputeAgent() {
		
	}
	
	MVImputeAgent(JSONObject parsedSpec) {
		JSONObject mvobj = (JSONObject) parsedSpec.get(TX_METHOD.IMPUTE.toString());
		JSONObject scobj = (JSONObject) parsedSpec.get(TX_METHOD.SCALE.toString());
		
		if(mvobj == null) {
			// MV Impute is not applicable
			_mvList = null;
			_mvMethodList = null;
			_meanList = null;
			_countList = null;
			_replacementList = null;
		}
		else {
			JSONArray mvattrs = (JSONArray) mvobj.get(JSON_ATTRS);
			JSONArray mvmthds = (JSONArray) mvobj.get(JSON_MTHD);
			int mvLength = mvattrs.size();
			
			assert(mvLength == mvmthds.size());
			
			_mvList = new int[mvLength];
			_mvMethodList = new byte[mvLength];
			
			_meanList = new KahanObject[mvLength];
			_countList = new long[mvLength];
			_varList = new CM_COV_Object[mvLength];
			
			_isMVScaled = new BitSet(_mvList.length);
			_isMVScaled.clear();
			
			for(int i=0; i < _mvList.length; i++) {
				_mvList[i] = ((Long) mvattrs.get(i)).intValue();
				_mvMethodList[i] = ((Long) mvmthds.get(i)).byteValue(); 
				_meanList[i] = new KahanObject(0, 0);
			}
			
			_modeList = new String[mvLength];			// contains replacements for "categorical" columns
			_replacementList = new String[mvLength]; 	// contains replacements for "scale" columns, including computed means as well constants
			
			JSONArray constants = (JSONArray)mvobj.get(JSON_CONSTS);
			for(int i=0; i < constants.size(); i++) {
				if ( constants.get(i) == null )
					_replacementList[i] = "NaN";
				else
					_replacementList[i] = constants.get(i).toString();
			}
		}
		
		// Handle scaled attributes
		if ( scobj == null )
		{
			// scaling is not applicable
			_scnomvCountList = null;
			_scnomvMeanList = null;
			_scnomvVarList = null;
		}
		else
		{
			if ( _mvList != null ) 
				_mvscMethodList = new byte[_mvList.length];
			
			JSONArray scattrs = (JSONArray) scobj.get(JSON_ATTRS);
			JSONArray scmthds = (JSONArray) scobj.get(JSON_MTHD);
			int scLength = scattrs.size();
			
			int[] _allscaled = new int[scLength];
			int scnomv = 0, colID;
			byte mthd;
			for(int i=0; i < scLength; i++)
			{
				colID = ((Long) scattrs.get(i)).intValue();
				mthd = ((Long) scmthds.get(i)).byteValue(); 
						
				_allscaled[i] = colID;
				
				// check if the attribute is also MV imputed
				int mvidx = isImputed(colID);
				if(mvidx != -1)
				{
					_isMVScaled.set(mvidx);
					_mvscMethodList[mvidx] = mthd;
					_varList[mvidx] = new CM_COV_Object();
				}
				else
					scnomv++;	// count of scaled but not imputed 
			}
			
			if(scnomv > 0)
			{
				_scnomvList = new int[scnomv];			
				_scnomvMethodList = new byte[scnomv];	
	
				_scnomvMeanList = new KahanObject[scnomv];
				_scnomvCountList = new long[scnomv];
				_scnomvVarList = new CM_COV_Object[scnomv];
				
				for(int i=0, idx=0; i < scLength; i++)
				{
					colID = ((Long) scattrs.get(i)).intValue();
					mthd = ((Long) scmthds.get(i)).byteValue(); 
							
					if(isImputed(colID) == -1)
					{	// scaled but not imputed
						_scnomvList[idx] = colID;
						_scnomvMethodList[idx] = mthd;
						_scnomvMeanList[idx] = new KahanObject(0, 0);
						_scnomvVarList[idx] = new CM_COV_Object();
						idx++;
					}
				}
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
		
		try {
			String w = null;
			if(_mvList != null)
			for(int i=0; i <_mvList.length; i++) {
				int colID = _mvList[i];
				w = UtilFunctions.unquote(words[colID-1].trim());
				
				try {
				if(!isNA(w)) {
					_countList[i]++;
					
					boolean computeMean = (_mvMethodList[i] == 1 || _isMVScaled.get(i) );
					if(computeMean) {
						// global_mean
						double d = UtilFunctions.parseToDouble(w);
						_meanFn.execute2(_meanList[i], d, _countList[i]);
						
						if (_isMVScaled.get(i) && _mvscMethodList[i] == 2)
							_varFn.execute(_varList[i], d);
					}
					else {
						// global_mode or constant
						// Nothing to do here. Mode is computed using recode maps.
					}
				}
				} catch (NumberFormatException e) 
				{
					throw new RuntimeException("Encountered \"" + w + "\" in column ID \"" + colID + "\", when expecting a numeric value. Consider adding \"" + w + "\" to na.strings, along with an appropriate imputation method.");
				}
			}
			
			// Compute mean and variance for attributes that are scaled but not imputed
			if(_scnomvList != null)
			for(int i=0; i < _scnomvList.length; i++) 
			{
				int colID = _scnomvList[i];
				w = UtilFunctions.unquote(words[colID-1].trim());
				double d = UtilFunctions.parseToDouble(w);
				_scnomvCountList[i]++; 		// not required, this is always equal to total #records processed
				_meanFn.execute2(_scnomvMeanList[i], d, _scnomvCountList[i]);
				if(_scnomvMethodList[i] == 2)
					_varFn.execute(_scnomvVarList[i], d);
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	private String encodeCMObj(CM_COV_Object obj)
	{
		StringBuilder sb = new StringBuilder();
		sb.append(obj.w);
		sb.append(",");
		sb.append(obj.mean._sum);
		sb.append(",");
		sb.append(obj.mean._correction);
		sb.append(",");
		sb.append(obj.m2._sum);
		sb.append(",");
		sb.append(obj.m2._correction);
		return sb.toString();
	}
	
	private CM_COV_Object decodeCMObj(String s) 
	{
		CM_COV_Object obj = new CM_COV_Object();
		String[] parts = s.split(",");
		obj.w = UtilFunctions.parseToDouble(parts[0]);
		obj.mean._sum = UtilFunctions.parseToDouble(parts[1]);
		obj.mean._correction = UtilFunctions.parseToDouble(parts[2]);
		obj.m2._sum = UtilFunctions.parseToDouble(parts[3]);
		obj.m2._correction = UtilFunctions.parseToDouble(parts[4]);
		
		return obj;
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
		try { 
			if(_mvList != null)
			for(int i=0; i < _mvList.length; i++) {
				int colID = _mvList[i];
				byte mthd = _mvMethodList[i];
				
				IntWritable iw = new IntWritable(-colID);
				
				if ( mthd == 1 || _isMVScaled.get(i) ) {
					String suffix = null;
					if ( mthd ==1 && _isMVScaled.get(i) )
						suffix = "scmv"; 	// both scaled and mv imputed
					else if ( mthd == 1 )
						suffix = "noscmv";
					else
						suffix = "scnomv";
					
					writeMean(iw, taskID, _meanList[i], suffix, _countList[i], out);
				}
				else if (mthd == 3){
					String s = CONSTANT_PREFIX + "_" + _replacementList[i];
					out.collect(iw, new DistinctValue(s, -1L));
				}
				
				// output variance information relevant to scaling
				if(_isMVScaled.get(i) && _mvscMethodList[i] == 2) 
					writeVar(iw, taskID, _varList[i], out);
			}
			
			// handle attributes that are scaled but not imputed
			if(_scnomvList != null)
			for(int i=0; i < _scnomvList.length; i++)
			{
				int colID = _scnomvList[i];
				IntWritable iw = new IntWritable(-colID);
				byte mthd = _scnomvMethodList[i];
				
				writeMean(iw, taskID, _scnomvMeanList[i], "scnomv", _scnomvCountList[i], out);
				
				if(mthd==2)
					writeVar(iw, taskID, _scnomvVarList[i], out);
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	private void writeVar(IntWritable iw, int taskID, CM_COV_Object var, OutputCollector<IntWritable, DistinctValue> out) throws CharacterCodingException, IOException
	{
		StringBuilder sb = new StringBuilder();
		sb.append(VARIANCE_PREFIX);
		sb.append("_");
		sb.append(taskID);
		sb.append("_");
		sb.append(encodeCMObj(var));
		out.collect(iw, new DistinctValue(sb.toString(), -1L));
	}
	
	private void writeMean(IntWritable iw, int taskID, KahanObject mean, String suffix, long count, OutputCollector<IntWritable, DistinctValue> out) throws CharacterCodingException, IOException
	{
		String s = null;
		s = MEAN_PREFIX + "_" + taskID + "_" + Double.toString(mean._sum) + "," + suffix;
		out.collect(iw, new DistinctValue(s, -1L));
		s = CORRECTION_PREFIX + "_" + taskID + "_" + Double.toString(mean._correction);
		out.collect(iw, new DistinctValue(s, -1L));
		s = COUNT_PREFIX + "_" + taskID + "_" + Long.toString(count);
		out.collect(iw, new DistinctValue(s, -1L));
		s = TOTAL_COUNT_PREFIX + "_" + taskID + "_" + Long.toString(TransformationAgent._numValidRecords);
		out.collect(iw, new DistinctValue(s, -1L));

	}
	
	private void writeTfMtd(int colID, String mean, String tfMtdDir, FileSystem fs) throws IOException 
	{
		Path pt=new Path(tfMtdDir+"/Impute/"+outputColumnNames[colID-1]+ MV_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(colID + TXMTD_SEP + mean + "\n");
		br.close();
	}
	
	private void writeTfMtd(int colID, String mean, String sdev, String tfMtdDir, FileSystem fs) throws IOException 
	{
		Path pt=new Path(tfMtdDir+"/Scale/"+outputColumnNames[colID-1]+ SCALE_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(colID + TXMTD_SEP + mean + TXMTD_SEP + sdev + "\n");
		br.close();
	}
	
	private void writeTfMtd(int colID, String min, String max, String binwidth, String nbins, String tfMtdDir, FileSystem fs) throws IOException 
	{
		Path pt = new Path(tfMtdDir+"/Bin/"+ outputColumnNames[colID-1] + BIN_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(colID + TXMTD_SEP + min + TXMTD_SEP + max + TXMTD_SEP + binwidth + TXMTD_SEP + nbins + "\n");
		br.close();
	}
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs) throws IOException {
		
		try{
			if (_mvList != null)
				for(int i=0; i < _mvList.length; i++) {
					int colID = _mvList[i];
					
					double imputedValue = Double.NaN;
					KahanObject gmean = null;
					if ( _mvMethodList[i] == 1 ) 
					{
						gmean = _meanList[i];
						imputedValue = _meanList[i]._sum;
						
						double mean = ( _countList[i] == 0 ? 0.0 : _meanList[i]._sum); 
						writeTfMtd(colID, Double.toString(mean), outputDir, fs);
					}
					else if ( _mvMethodList[i] == 3 ) 
					{
						writeTfMtd(colID, _replacementList[i], outputDir, fs);
						
						if (_isMVScaled.get(i) )
						{
							imputedValue = UtilFunctions.parseToDouble(_replacementList[i]);
							// adjust the global mean, by combining gmean with "replacement" (weight = #missing values)
							gmean = new KahanObject(_meanList[i]._sum, _meanList[i]._correction);
							_meanFn.execute(gmean, imputedValue, TransformationAgent._numValidRecords);
						}
					}
						
					if ( _isMVScaled.get(i) ) 
					{
						double sdev = -1.0;
						if ( _mvscMethodList[i] == 2 ) {
							// Adjust variance with missing values
							long totalMissingCount = (TransformationAgent._numValidRecords - _countList[i]);
							_varFn.execute(_varList[i], imputedValue, totalMissingCount);
							double var = _varList[i].getRequiredResult(new CMOperator(_varFn, AggregateOperationTypes.VARIANCE));
							sdev = Math.sqrt(var);
						}
						writeTfMtd(colID, Double.toString(gmean._sum), Double.toString(sdev), outputDir, fs);
					}
				}
		
			if(_scnomvList != null)
				for(int i=0; i < _scnomvList.length; i++ )
				{
					int colID = _scnomvList[i];
					double mean = (_scnomvCountList[i] == 0 ? 0.0 : _scnomvMeanList[i]._sum);
					double sdev = -1.0;
					if ( _scnomvMethodList[i] == 2 ) 
					{
						double var = _scnomvVarList[i].getRequiredResult(new CMOperator(_varFn, AggregateOperationTypes.VARIANCE));
						sdev = Math.sqrt(var);
					}
					writeTfMtd(colID, Double.toString(mean), Double.toString(sdev), outputDir, fs);
				}
			
		} catch(DMLRuntimeException e) {
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
		double d;
		long totalRecordCount = 0, totalValidCount=0;
		String mvConstReplacement = null;
		
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
		HashMap<Integer, CM_COV_Object> mapVars = new HashMap<Integer, CM_COV_Object>();
		boolean isImputed = false;
		boolean isScaled = false;
		boolean isBinned = false;
		
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
				
				mo.mean = UtilFunctions.parseToDouble(parts[2].split(",")[0]);
				
				// check if this attribute is scaled
				String s = parts[2].split(",")[1]; 
				if(s.equalsIgnoreCase("scmv"))
					isScaled = isImputed = true;
				else if ( s.equalsIgnoreCase("scnomv") )
					isScaled = true;
				else
					isImputed = true;
				
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
			else if ( w.startsWith(CONSTANT_PREFIX) )
			{
				String[] parts = w.split("_");
				mvConstReplacement = parts[1];
			}
			else if (w.startsWith(COUNT_PREFIX)) {
				String[] parts = w.split("_");
				int taskID = UtilFunctions.parseToInt(parts[1]);
				MeanObject mo = mapMeans.get(taskID);
				if ( mo==null ) 
					mo = new MeanObject();
				mo.count = UtilFunctions.parseToLong(parts[2]);
				totalValidCount += mo.count;
				mapMeans.put(taskID, mo);
			}
			else if (w.startsWith(TOTAL_COUNT_PREFIX)) {
				String[] parts = w.split("_");
				//int taskID = UtilFunctions.parseToInt(parts[1]);
				totalRecordCount += UtilFunctions.parseToLong(parts[2]);
			}
			else if (w.startsWith(VARIANCE_PREFIX)) {
				String[] parts = w.split("_");
				int taskID = UtilFunctions.parseToInt(parts[1]);
				CM_COV_Object cm = decodeCMObj(parts[2]);
				mapVars.put(taskID, cm);
			}
			else if(w.startsWith(BinAgent.MIN_PREFIX)) {
				isBinned = true;
				d = UtilFunctions.parseToDouble( w.substring( BinAgent.MIN_PREFIX.length() ) );
				if ( d < min )
					min = d;
			}
			else if(w.startsWith(BinAgent.MAX_PREFIX)) {
				isBinned = true;
				d = UtilFunctions.parseToDouble( w.substring( BinAgent.MAX_PREFIX.length() ) );
				if ( d > max )
					max = d;
			}
			else if (w.startsWith(BinAgent.NBINS_PREFIX)) {
				isBinned = true;
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
		
		// compute global variance across all map outputs
		CM_COV_Object gcm = new CM_COV_Object();
		try {
			for(CM_COV_Object cm : mapVars.values())
				gcm = (CM_COV_Object) _varFn.execute(gcm, cm);
		} catch (DMLRuntimeException e) {
			throw new IOException(e);
		}
		
		// If the column is imputed with a constant, then adjust min and max based the value of the constant.
		if(isImputed && isBinned && mvConstReplacement != null)
		{
			double cst = UtilFunctions.parseToDouble(mvConstReplacement);
			if ( cst < min)
				min = cst;
			if ( cst > max)
				max = cst;
		}

		// write merged metadata
		FileSystem fs = FileSystem.get(job);
		if( isImputed ) 
		{
			String imputedValue = null;
			if ( mvConstReplacement != null )
				imputedValue = mvConstReplacement;
			else 
				imputedValue = Double.toString(gcount == 0 ? 0.0 : gmean._sum);
			
			writeTfMtd(colID, imputedValue, outputDir, fs);
		}
		
		if ( min != Double.MAX_VALUE && max != Double.MIN_VALUE ) {
			double binwidth = (max-min)/nbins;
			writeTfMtd(colID, Double.toString(min), Double.toString(max), Double.toString(binwidth), Integer.toString(nbins), outputDir, fs);
		}
		
		if ( isScaled ) 
		{
			try {
				if( totalValidCount != totalRecordCount) {
					// In the presense of missing values, the variance needs to be adjusted.
					// The mean does not need to be adjusted, when mv impute method is global_mean, 
					// since missing values themselves are replaced with gmean.
					long totalMissingCount = (totalRecordCount-totalValidCount);
					int idx = isImputed(colID);
					if(idx != -1 && _mvMethodList[idx] == 3) 
						_meanFn.execute(gmean, UtilFunctions.parseToDouble(_replacementList[idx]), totalRecordCount);
					_varFn.execute(gcm, gmean._sum, totalMissingCount);
				}
				
				double mean = (gcount == 0 ? 0.0 : gmean._sum);
				double var = gcm.getRequiredResult(new CMOperator(_varFn, AggregateOperationTypes.VARIANCE));
				double sdev = (mapVars.size() > 0 ? Math.sqrt(var) : -1.0 );
				
				writeTfMtd(colID, Double.toString(mean), Double.toString(sdev), outputDir, fs);
				
				
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}
	}
	
	// ------------------------------------------------------------------------------------------------

	private String readReplacement(int colID, FileSystem fs, Path  txMtdDir) throws IOException
	{
		Path path = new Path( txMtdDir + "/Impute/" + outputColumnNames[colID-1] + MV_FILE_SUFFIX);
		TransformationAgent.checkValidInputFile(fs, path, true); 
		
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
		String line = br.readLine();
		String replacement = line.split(TXMTD_SEP)[1];
		br.close();
		
		return replacement;
	}
	
	private String readMode(int colID, FileSystem fs, Path txMtdDir) throws IOException
	{
		Path path = new Path( txMtdDir + "/Recode/" + outputColumnNames[colID-1] + MODE_FILE_SUFFIX);
		TransformationAgent.checkValidInputFile(fs, path, true); 
		
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
		String line = br.readLine();

		int idxQuote = line.lastIndexOf('"');
		String mode = UtilFunctions.unquote(line.substring(0,idxQuote+1));	// mode in string form
		br.close();
		
		/*int idx = idxQuote+2;
		while(line.charAt(idx) != TXMTD_SEP.charAt(0))
			idx++;
		_replacementList[i] = line.substring(idxQuote+2,idx); // recode id of mode (unused)
		*/
		
		return mode;
	}
	
	public String readScaleLine(int colID, FileSystem fs, Path txMtdDir) throws IOException
	{
		Path path = new Path( txMtdDir + "/Scale/" + outputColumnNames[colID-1] + SCALE_FILE_SUFFIX);
		TransformationAgent.checkValidInputFile(fs, path, true); 
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
		String line = br.readLine();
		br.close();
		
		return line;
	}
	
	private void processScalingFile(int i, int[] list, KahanObject[] meanList, CM_COV_Object[] varList, FileSystem fs, Path tfMtdDir ) throws IOException
	{
		int colID = list[i];
		
		String line = readScaleLine(colID, fs, tfMtdDir);
		String[] parts = line.split(",");
		double mean = UtilFunctions.parseToDouble(parts[1]);
		double sd = UtilFunctions.parseToDouble(parts[2]);
		
		meanList[i]._sum = mean;
		varList[i].mean._sum = sd;
	}
	
	// ------------------------------------------------------------------------------------------------

	/**
	 * Method to load transform metadata for all attributes
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path tfMtdDir) throws IOException {
		
		if(fs.isDirectory(tfMtdDir)) {
			
			// Load information about missing value imputation
			if (_mvList != null)
				for(int i=0; i<_mvList.length;i++) {
					int colID = _mvList[i];
					
					if ( _mvMethodList[i] == 1 )
						// global_mean
						_replacementList[i] = readReplacement(colID, fs, tfMtdDir);
					else if ( _mvMethodList[i] == 2 )
						// global_mode: located along with recode maps
						_modeList[i] = readMode(colID, fs, tfMtdDir);
					else if ( _mvMethodList[i] == 3 ) {
						// constant: replace a missing value by a given constant
						// nothing to do. The constant values are loaded already during configure 
					}
					else
						throw new RuntimeException("Invalid Missing Value Imputation methods: " + _mvMethodList[i]);
				}
			
			// Load scaling information
			if(_mvList != null)
				for(int i=0; i < _mvList.length; i++)
					if ( _isMVScaled.get(i) ) 
						processScalingFile(i, _mvList, _meanList, _varList, fs, tfMtdDir);
			
			if(_scnomvList != null)
				for(int i=0; i < _scnomvList.length; i++)
					processScalingFile(i, _scnomvList, _scnomvMeanList, _scnomvVarList, fs, tfMtdDir);
		}
		else {
			fs.close();
			throw new RuntimeException("Path to recode maps must be a directory: " + tfMtdDir);
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
		
		if ( _mvList != null)
		for(int i=0; i < _mvList.length; i++) {
			int colID = _mvList[i];
			String w = UtilFunctions.unquote(words[colID-1]);
			if(isNA(w))
				words[colID-1] = (_mvMethodList[i] == 2 ? _modeList[i] : _replacementList[i] );
			
			if ( _isMVScaled.get(i) )
				if ( _mvscMethodList[i] == 1 )
					words[colID-1] = Double.toString( UtilFunctions.parseToDouble(w) - _meanList[i]._sum );
				else
					words[colID-1] = Double.toString( (UtilFunctions.parseToDouble(w) - _meanList[i]._sum) / _varList[i].mean._sum );
		}
		
		if(_scnomvList != null)
		for(int i=0; i < _scnomvList.length; i++)
		{
			int colID = _scnomvList[i];
			if ( _scnomvMethodList[i] == 1 )
				words[colID-1] = Double.toString( UtilFunctions.parseToDouble(words[colID-1]) - _scnomvMeanList[i]._sum );
			else
				words[colID-1] = Double.toString( (UtilFunctions.parseToDouble(words[colID-1]) - _scnomvMeanList[i]._sum) / _scnomvVarList[i].mean._sum );
		}
			
		return words;
	}
	
	/**
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	public int isImputed(int colID)
	{
		if(_mvList == null)
			return -1;
		
		int idx = Arrays.binarySearch(_mvList, colID);
		return ( idx >= 0 ? idx : -1);
	}
	
	public MVMethod getMethod(int colID) 
	{
		int idx = isImputed(colID);
		
		if(idx == -1)
			return MVMethod.INVALID;
		
		switch(_mvMethodList[idx])
		{
			case 1: return MVMethod.GLOBAL_MEAN;
			case 2: return MVMethod.GLOBAL_MODE;
			case 3: return MVMethod.CONSTANT;
			default: return MVMethod.INVALID;
		}
		
	}
	
	public long getNonMVCount(int colID) 
	{
		int idx = isImputed(colID);
		if(idx == -1)
			return 0;
		else
			return _countList[idx];
	}
	
	public String getReplacement(int colID) 
	{
		int idx = isImputed(colID);
		
		if(idx == -1)
			return null;
		else
			return _replacementList[idx];
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
