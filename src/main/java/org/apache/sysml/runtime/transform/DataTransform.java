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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import scala.Tuple2;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.lops.CSVReBlock;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionParser;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.instructions.spark.ParameterizedBuiltinSPInstruction;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.CSVReblockMR.AssignRowIDMRReturn;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.transform.encode.EncoderFactory;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.JSONHelper;

public class DataTransform 
{
	private static final String ERROR_MSG_ZERO_ROWS = "Number of rows in the transformed output (potentially, after ommitting the ones with missing values) is zero. Cannot proceed.";

	
	/**
	 * Method to read the header line from the input data file.
	 * 
	 * @param fs
	 * @param prop
	 * @param smallestFile
	 * @return
	 * @throws IOException
	 */
	private static String readHeaderLine(FileSystem fs, CSVFileFormatProperties prop, String smallestFile) throws IOException {
		String line = null;
		
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(smallestFile))));
		line = br.readLine();
		br.close();
		if(prop.hasHeader()) {
			; // nothing here
		}
		else 
		{
			// construct header with default column names, V1, V2, etc.
			int ncol = Pattern.compile( Pattern.quote(prop.getDelim()) ).split(line, -1).length;
			line = null;
			
			StringBuilder sb = new StringBuilder();
			sb.append("V1");
			for(int i=2; i <= ncol; i++)
				sb.append(prop.getDelim() + "V" + i);
			line = sb.toString();
		}
		return line;
	}
	
	/**
	 * Method to construct a mapping between column names and their
	 * corresponding column IDs. The mapping is used to prepare the
	 * specification file in <code>processSpecFile()</code>.
	 * 
	 * @param fs
	 * @param prop
	 * @param headerLine
	 * @param smallestFile
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IOException
	 */
	private static HashMap<String, Integer> processColumnNames(FileSystem fs, CSVFileFormatProperties prop, String headerLine, String smallestFile) throws IllegalArgumentException, IOException {
		HashMap<String, Integer> colNames = new HashMap<String,Integer>();
		
		String escapedDelim = Pattern.quote(prop.getDelim());
		Pattern compiledDelim = Pattern.compile(escapedDelim);
		String[] names = compiledDelim.split(headerLine, -1);
			
		for(int i=0; i< names.length; i++)
			colNames.put(UtilFunctions.unquote(names[i].trim()), i+1);

		return colNames;
	}
	
	/**
	 * In-place permutation of list, mthd, and cst arrays based on indices,
	 * by navigating through cycles in the permutation. 
	 * 
	 * @param list
	 * @param mthd
	 * @param cst
	 * @param indices
	 */
	private static void inplacePermute(int[] list, byte[] mthd, Object[] cst, Integer[] indices) 
	{
		int x;
		byte xb = 0;
		Object xo = null;
		
		int j, k;
		for(int i=0; i < list.length; i++) 
		{
		    x = list[i];
		    xb = mthd[i];
		    if ( cst != null )  xo = cst[i];
		    
		    j = i;
		    while(true) {
		        k = indices[j];
		        indices[j] = j;
		        
		        if (k == i)
		            break;
		        
		        list[j] = list[k];
		        mthd[j] = mthd[k]; 
		        if ( cst != null )  cst[j] = cst[k]; 
		        j = k;
		    }
		    list[j] = x;
	        mthd[j] = xb; 
	        if ( cst != null )  cst[j] = xo; 
		}

	}
	
	/**
	 * Convert input transformation specification file with column names into a
	 * specification with corresponding column Ids. This file is sent to all the
	 * relevant MR jobs.
	 * 
	 * @param fs
	 * @param inputPath
	 * @param smallestFile
	 * @param colNames
	 * @param prop
	 * @param specFileWithNames
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IOException
	 * @throws JSONException 
	 */
	private static String processSpecFile(FileSystem fs, String inputPath, String smallestFile, HashMap<String,Integer> colNames, CSVFileFormatProperties prop, String specWithNames) throws IllegalArgumentException, IOException, JSONException {
		JSONObject inputSpec = new JSONObject(specWithNames);
		
		final String NAME = "name";
		final String ID = "id";
		final String METHOD = "method";
		final String VALUE = "value";
		final String MV_METHOD_MEAN = "global_mean";
		final String MV_METHOD_MODE = "global_mode";
		final String MV_METHOD_CONSTANT = "constant";
		final String BIN_METHOD_WIDTH = "equi-width";
		final String BIN_METHOD_HEIGHT = "equi-height";
		final String SCALE_METHOD_Z = "z-score";
		final String SCALE_METHOD_M = "mean-subtraction";
		final String JSON_BYPOS = "ids";
		
		String stmp = null;
		JSONObject entry = null;
		byte btmp = 0;
		
		final int[] mvList;
		int[] rcdList, dcdList, omitList;
		final int[] binList;
		final int[] scaleList;
		byte[] mvMethods = null, binMethods=null, scaleMethods=null;
		Object[] numBins = null;
		Object[] mvConstants = null;
		
		boolean byPositions = (inputSpec.containsKey(JSON_BYPOS) && ((Boolean)inputSpec.get(JSON_BYPOS)).booleanValue() == true);
		
		// --------------------------------------------------------------------------
		// Omit
		if( inputSpec.containsKey(TfUtils.TXMETHOD_OMIT) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_OMIT);
			omitList = new int[arrtmp.size()];
			for(int i=0; i<arrtmp.size(); i++) {
				if(byPositions)
					omitList[i] = UtilFunctions.toInt( arrtmp.get(i) );
				else {
					stmp = UtilFunctions.unquote( (String)arrtmp.get(i) );
					omitList[i] = colNames.get(stmp);
				}
			}
			Arrays.sort(omitList);
		}
		else
			omitList = null;
		// --------------------------------------------------------------------------
		// Missing value imputation
		if( inputSpec.containsKey(TfUtils.TXMETHOD_IMPUTE) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_IMPUTE);
			
			mvList = new int[arrtmp.size()];
			mvMethods = new byte[arrtmp.size()];
			mvConstants = new Object[arrtmp.size()];
			
			for(int i=0; i<arrtmp.size(); i++) {
				entry = (JSONObject)arrtmp.get(i);
				if (byPositions) {
					mvList[i] = UtilFunctions.toInt(entry.get(ID));
				}
				else {
					stmp = UtilFunctions.unquote((String) entry.get(NAME));
					mvList[i] = colNames.get(stmp);
				}
				
				stmp = UtilFunctions.unquote((String) entry.get(METHOD));
				if(stmp.equals(MV_METHOD_MEAN))
					btmp = (byte)1;
				else if ( stmp.equals(MV_METHOD_MODE))
					btmp = (byte)2;
				else if ( stmp.equals(MV_METHOD_CONSTANT))
					btmp = (byte)3;
				else
					throw new IOException("Unknown missing value imputation method (" + stmp + ") in transformation specification: " + specWithNames);
				mvMethods[i] = btmp;
				
				//txMethods.add( btmp );
				
				mvConstants[i] = null;
				if ( entry.containsKey(VALUE) )
					mvConstants[i] = entry.get(VALUE);
			}
			
			Integer[] idx = new Integer[mvList.length];
			for(int i=0; i < mvList.length; i++)
				idx[i] = i;
			Arrays.sort(idx, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					return (mvList[o1]-mvList[o2]);
				}
			});
			
			// rearrange mvList, mvMethods, and mvConstants according to permutation idx
			inplacePermute(mvList, mvMethods, mvConstants, idx);
		}
		else
			mvList = null;
		// --------------------------------------------------------------------------
		// Recoding
		if( inputSpec.containsKey(TfUtils.TXMETHOD_RECODE) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_RECODE);
			rcdList = new int[arrtmp.size()];
			for(int i=0; i<arrtmp.size(); i++) {
				if (byPositions)
					rcdList[i] = UtilFunctions.toInt(arrtmp.get(i));
				else {
					stmp = UtilFunctions.unquote( (String)arrtmp.get(i) );
					rcdList[i] = colNames.get(stmp);
				}
			}
			Arrays.sort(rcdList);
		}
		else
			rcdList = null;
		// --------------------------------------------------------------------------
		// Binning
		if( inputSpec.containsKey(TfUtils.TXMETHOD_BIN) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_BIN);
			
			binList = new int[arrtmp.size()];
			binMethods = new byte[arrtmp.size()];
			numBins = new Object[arrtmp.size()];
			
			for(int i=0; i<arrtmp.size(); i++) {
				entry = (JSONObject)arrtmp.get(i);
				
				if (byPositions) {
					binList[i] = UtilFunctions.toInt(entry.get(ID));
				}
				else {
					stmp = UtilFunctions.unquote((String) entry.get(NAME));
					binList[i] = colNames.get(stmp);
				}
				stmp = UtilFunctions.unquote((String) entry.get(METHOD));
				if(stmp.equals(BIN_METHOD_WIDTH))
					btmp = (byte)1;
				else if ( stmp.equals(BIN_METHOD_HEIGHT))
					throw new IOException("Equi-height binning method is not yet supported, in transformation specification: " + specWithNames);
				else
					throw new IOException("Unknown missing value imputation method (" + stmp + ") in transformation specification: " + specWithNames);
				binMethods[i] = btmp;
				
				numBins[i] = entry.get(TfUtils.JSON_NBINS);
				if ( ((Integer) numBins[i]).intValue() <= 1 ) 
					throw new IllegalArgumentException("Invalid transformation on column \"" + (String) entry.get(NAME) + "\". Number of bins must be greater than 1.");
			}
			
			Integer[] idx = new Integer[binList.length];
			for(int i=0; i < binList.length; i++)
				idx[i] = i;
			Arrays.sort(idx, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					return (binList[o1]-binList[o2]);
				}
			});
			
			// rearrange binList and binMethods according to permutation idx
			inplacePermute(binList, binMethods, numBins, idx);
		}
		else
			binList = null;
		// --------------------------------------------------------------------------
		// Dummycoding
		if( inputSpec.containsKey(TfUtils.TXMETHOD_DUMMYCODE) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_DUMMYCODE);
			dcdList = new int[arrtmp.size()];
			for(int i=0; i<arrtmp.size(); i++) {
				if (byPositions)
					dcdList[i] = UtilFunctions.toInt(arrtmp.get(i));
				else {
					stmp = UtilFunctions.unquote( (String)arrtmp.get(i) );
					dcdList[i] = colNames.get(stmp);
				}
			}
			Arrays.sort(dcdList);
		}
		else
			dcdList = null;
		// --------------------------------------------------------------------------
		// Scaling
		if(inputSpec.containsKey(TfUtils.TXMETHOD_SCALE) ) {
			JSONArray arrtmp = (JSONArray) inputSpec.get(TfUtils.TXMETHOD_SCALE);
			
			scaleList = new int[arrtmp.size()];
			scaleMethods = new byte[arrtmp.size()];
			
			for(int i=0; i<arrtmp.size(); i++) {
				entry = (JSONObject)arrtmp.get(i);
				
				if (byPositions) {
					scaleList[i] = UtilFunctions.toInt(entry.get(ID));
				}
				else {
					stmp = UtilFunctions.unquote((String) entry.get(NAME));
					scaleList[i] = colNames.get(stmp);
				}
				stmp = UtilFunctions.unquote((String) entry.get(METHOD));
				if(stmp.equals(SCALE_METHOD_M))
					btmp = (byte)1;
				else if ( stmp.equals(SCALE_METHOD_Z))
					btmp = (byte)2;
				else
					throw new IOException("Unknown missing value imputation method (" + stmp + ") in transformation specification: " + specWithNames);
				scaleMethods[i] = btmp;
			}
			
			Integer[] idx = new Integer[scaleList.length];
			for(int i=0; i < scaleList.length; i++)
				idx[i] = i;
			Arrays.sort(idx, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					return (scaleList[o1]-scaleList[o2]);
				}
			});
			
			// rearrange scaleList and scaleMethods according to permutation idx
			inplacePermute(scaleList, scaleMethods, null, idx);
		}
		else
			scaleList = null;
		// --------------------------------------------------------------------------
		
		// check for column IDs that are imputed with mode, but not recoded
		// These columns have be handled separately, because the computation of mode 
		// requires the computation of distinct values (i.e., recode maps)
		ArrayList<Integer> tmpList = new ArrayList<Integer>();
		if(mvList != null)
		for(int i=0; i < mvList.length; i++) {
			int colID = mvList[i];
			if(mvMethods[i] == 2 && (rcdList == null || Arrays.binarySearch(rcdList, colID) < 0) )
				tmpList.add(colID);
		}
		
		int[] mvrcdList = null;
		if ( tmpList.size() > 0 ) {
			mvrcdList = new int[tmpList.size()];
			for(int i=0; i < tmpList.size(); i++)
				mvrcdList[i] = tmpList.get(i);
		}
		// Perform Validity Checks
		
		/*
			      OMIT MVI RCD BIN DCD SCL
			OMIT     -  x   *   *   *   *
			MVI      x  -   *   *   *   *
			RCD      *  *   -   x   *   x
			BIN      *  *   x   -   *   x
			DCD      *  *   *   *   -   x
			SCL      *  *   x   x   x   -
		 */
		
		if(mvList != null)
			for(int i=0; i < mvList.length; i++) 
			{
				int colID = mvList[i];

				if ( omitList != null && Arrays.binarySearch(omitList, colID) >= 0 ) 
					throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A column can not be both omitted and imputed.");
				
				if(mvMethods[i] == 1) 
				{
					if ( rcdList != null && Arrays.binarySearch(rcdList, colID) >= 0 ) 
						throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A numeric column can not be recoded.");
					
					if ( dcdList != null && Arrays.binarySearch(dcdList, colID) >= 0 )
						// throw an error only if the column is not binned
						if ( binList == null || Arrays.binarySearch(binList, colID) < 0 )
							throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A numeric column can not be dummycoded.");
				}
			}
		
		if(scaleList != null)
		for(int i=0; i < scaleList.length; i++) 
		{
			int colID = scaleList[i];
			if ( rcdList != null && Arrays.binarySearch(rcdList, colID) >= 0 ) 
				throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A column can not be recoded and scaled.");
			if ( binList != null && Arrays.binarySearch(binList, colID) >= 0 ) 
				throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A column can not be binned and scaled.");
			if ( dcdList != null && Arrays.binarySearch(dcdList, colID) >= 0 ) 
				throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A column can not be dummycoded and scaled.");
		}
		
		if(rcdList != null)
		for(int i=0; i < rcdList.length; i++) 
		{
			int colID = rcdList[i];
			if ( binList != null && Arrays.binarySearch(binList, colID) >= 0 ) 
				throw new IllegalArgumentException("Invalid transformations on column ID " + colID + ". A column can not be recoded and binned.");
		}
		
		// Check if dummycoded columns are either recoded or binned.
		// If not, add them to recode list.
		ArrayList<Integer> addToRcd = new ArrayList<Integer>();
		if(dcdList != null)
		for(int i=0; i < dcdList.length; i++) 
		{
			int colID = dcdList[i];
			boolean isRecoded = (rcdList != null && Arrays.binarySearch(rcdList, colID) >= 0);
			boolean isBinned = (binList != null && Arrays.binarySearch(binList, colID) >= 0);
			// If colID is neither recoded nor binned, then, add it to rcdList.
			if ( !isRecoded && !isBinned )
				addToRcd.add(colID);
		}
		if ( addToRcd.size() > 0 ) 
		{
			int[] newRcdList = null;
			if ( rcdList != null)  
				newRcdList = Arrays.copyOf(rcdList, rcdList.length + addToRcd.size());
			else
				newRcdList = new int[addToRcd.size()];
			
			int i = (rcdList != null ? rcdList.length : 0);
			for(int idx=0; i < newRcdList.length; i++, idx++)
				newRcdList[i] = addToRcd.get(idx);
			Arrays.sort(newRcdList);
			rcdList = newRcdList;
		}
		// -----------------------------------------------------------------------------
		
		// Prepare output spec
		JSONObject outputSpec = new JSONObject();

		if (omitList != null)
		{
			JSONObject rcdSpec = new JSONObject();
			rcdSpec.put(TfUtils.JSON_ATTRS, toJSONArray(omitList));
			outputSpec.put(TfUtils.TXMETHOD_OMIT, rcdSpec);
		}
		
		if (mvList != null)
		{
			JSONObject mvSpec = new JSONObject();
			mvSpec.put(TfUtils.JSON_ATTRS, toJSONArray(mvList));
			mvSpec.put(TfUtils.JSON_MTHD, toJSONArray(mvMethods));
			mvSpec.put(TfUtils.JSON_CONSTS, toJSONArray(mvConstants));
			outputSpec.put(TfUtils.TXMETHOD_IMPUTE, mvSpec);
		}
		
		if (rcdList != null)
		{
			JSONObject rcdSpec = new JSONObject();
			rcdSpec.put(TfUtils.JSON_ATTRS, toJSONArray(rcdList));
			outputSpec.put(TfUtils.TXMETHOD_RECODE, rcdSpec);
		}
		
		if (binList != null)
		{
			JSONObject binSpec = new JSONObject();
			binSpec.put(TfUtils.JSON_ATTRS, toJSONArray(binList));
			binSpec.put(TfUtils.JSON_MTHD, toJSONArray(binMethods));
			binSpec.put(TfUtils.JSON_NBINS, toJSONArray(numBins));
			outputSpec.put(TfUtils.TXMETHOD_BIN, binSpec);
		}
		
		if (dcdList != null)
		{
			JSONObject dcdSpec = new JSONObject();
			dcdSpec.put(TfUtils.JSON_ATTRS, toJSONArray(dcdList));
			outputSpec.put(TfUtils.TXMETHOD_DUMMYCODE, dcdSpec);
		}
		
		if (scaleList != null)
		{
			JSONObject scaleSpec = new JSONObject();
			scaleSpec.put(TfUtils.JSON_ATTRS, toJSONArray(scaleList));
			scaleSpec.put(TfUtils.JSON_MTHD, toJSONArray(scaleMethods));
			outputSpec.put(TfUtils.TXMETHOD_SCALE, scaleSpec);
		}
		
		if (mvrcdList != null)
		{
			JSONObject mvrcd = new JSONObject();
			mvrcd.put(TfUtils.JSON_ATTRS, toJSONArray(mvrcdList));
			outputSpec.put(TfUtils.TXMETHOD_MVRCD, mvrcd);
		}
		
		// return output spec with IDs
		return outputSpec.toString();
	}
	
	private static JSONArray toJSONArray(int[] list) 
	{
		JSONArray ret = new JSONArray(list.length);
		for(int i=0; i < list.length; i++)
			ret.add(list[i]);
		return ret;
	}

	private static JSONArray toJSONArray(byte[] list) 
	{
		JSONArray ret = new JSONArray(list.length);
		for(int i=0; i < list.length; i++)
			ret.add(list[i]);
		return ret;
	}

	private static JSONArray toJSONArray(Object[] list) 
		throws JSONException 
	{
		return new JSONArray(list);
	}
	
	/**
	 * Helper function to move transformation metadata files from a temporary
	 * location to permanent location. These files (e.g., header before and
	 * after transformation) are generated by a single mapper, while applying
	 * data transformations. Note that, these files must be ultimately be placed
	 * under the existing metadata directory (txMtdPath), which is
	 * simultaneously read by other mappers. If they are not created at a
	 * temporary location, then MR tasks fail due to changing timestamps on
	 * txMtdPath.
	 * 
	 * @param fs
	 * @param tmpPath
	 * @param txMtdPath
	 * @throws IllegalArgumentException
	 * @throws IOException
	 */
	private static void moveFilesFromTmp(FileSystem fs, String tmpPath, String txMtdPath) throws IllegalArgumentException, IOException 
	{
		// move files from temporary location to txMtdPath
		MapReduceTool.renameFileOnHDFS(tmpPath + "/" + TfUtils.TXMTD_COLNAMES, txMtdPath + "/" + TfUtils.TXMTD_COLNAMES);
		MapReduceTool.renameFileOnHDFS(tmpPath + "/" + TfUtils.TXMTD_DC_COLNAMES, txMtdPath + "/" + TfUtils.TXMTD_DC_COLNAMES);
		MapReduceTool.renameFileOnHDFS(tmpPath + "/" + TfUtils.TXMTD_COLTYPES, txMtdPath + "/" + TfUtils.TXMTD_COLTYPES);
		
		if ( fs.exists(new Path(tmpPath +"/Dummycode/" + TfUtils.DCD_FILE_NAME)) ) 
		{
			if ( !fs.exists( new Path(txMtdPath + "/Dummycode/") )) 
				fs.mkdirs(new Path(txMtdPath + "/Dummycode/"));
			MapReduceTool.renameFileOnHDFS( tmpPath + "/Dummycode/" + TfUtils.DCD_FILE_NAME, txMtdPath + "/Dummycode/" + TfUtils.DCD_FILE_NAME);
		}
	}
	
	/**
	 * Helper function to determine the number of columns after applying
	 * transformations. Note that dummycoding changes the number of columns.
	 * 
	 * @param fs
	 * @param header
	 * @param delim
	 * @param tfMtdPath
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 * @throws JSONException 
	 */
	private static int getNumColumnsTf(FileSystem fs, String header, String delim, String tfMtdPath) throws IllegalArgumentException, IOException, DMLRuntimeException, JSONException {
		String[] columnNames = Pattern.compile(Pattern.quote(delim)).split(header, -1);
		int ret = columnNames.length;
		
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(tfMtdPath + "/spec.json"))));
		JSONObject spec = JSONHelper.parse(br);
		br.close();
		
		// fetch relevant attribute lists
		if ( !spec.containsKey(TfUtils.TXMETHOD_DUMMYCODE) )
			return ret;
		
		JSONArray dcdList = (JSONArray) ((JSONObject)spec.get(TfUtils.TXMETHOD_DUMMYCODE)).get(TfUtils.JSON_ATTRS);

		// look for numBins among binned columns
		for(Object o : dcdList) 
		{
			int id = UtilFunctions.toInt(o);
			
			Path binpath = new Path( tfMtdPath + "/Bin/" + UtilFunctions.unquote(columnNames[id-1]) + TfUtils.TXMTD_BIN_FILE_SUFFIX);
			Path rcdpath = new Path( tfMtdPath + "/Recode/" + UtilFunctions.unquote(columnNames[id-1]) + TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
			
			if ( TfUtils.checkValidInputFile(fs, binpath, false ) )
			{
				br = new BufferedReader(new InputStreamReader(fs.open(binpath)));
				int nbins = UtilFunctions.parseToInt(br.readLine().split(TfUtils.TXMTD_SEP)[4]);
				br.close();
				ret += (nbins-1);
			}
			else if ( TfUtils.checkValidInputFile(fs, rcdpath, false ) )
			{
				br = new BufferedReader(new InputStreamReader(fs.open(rcdpath)));
				int ndistinct = UtilFunctions.parseToInt(br.readLine());
				br.close();
				ret += (ndistinct-1);
			}
			else
				throw new DMLRuntimeException("Relevant transformation metadata for column (id=" + id + ", name=" + columnNames[id-1] + ") is not found.");
		}
		
		return ret;
	}
	
	/**
	 * Main method to create and/or apply transformation metdata using MapReduce.
	 * 
	 * @param jobinst
	 * @param inputMatrices
	 * @param shuffleInst
	 * @param otherInst
	 * @param resultIndices
	 * @param outputMatrices
	 * @param numReducers
	 * @param replication
	 * @return
	 * @throws Exception
	 */
	public static JobReturn mrDataTransform(MRJobInstruction jobinst, MatrixObject[] inputs, String shuffleInst, String otherInst, byte[] resultIndices, MatrixObject[] outputs, int numReducers, int replication) throws Exception {
		
		String[] insts = shuffleInst.split(Instruction.INSTRUCTION_DELIM);
		
		// Parse transform instruction (the first instruction) to obtain relevant fields
		TransformOperands oprnds = new TransformOperands(insts[0], inputs[0]);
		
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		FileSystem fs = FileSystem.get(job);
		
		// find the first file in alphabetical ordering of part files in directory inputPath 
		String smallestFile = CSVReblockMR.findSmallestFile(job, oprnds.inputPath);
		
		// find column names
		String headerLine = readHeaderLine(fs, oprnds.inputCSVProperties, smallestFile);
		HashMap<String, Integer> colNamesToIds = processColumnNames(fs, oprnds.inputCSVProperties, headerLine, smallestFile);
		String outHeader = getOutputHeader(fs, headerLine, oprnds);
		int numColumns = colNamesToIds.size();
		
		int numColumnsTf = 0;
		long numRowsTf = 0;
		
		ArrayList<Integer> csvoutputs= new ArrayList<Integer>();
		ArrayList<Integer> bboutputs = new ArrayList<Integer>();
		
		// divide output objects based on output format (CSV or BinaryBlock)
		for(int i=0; i < outputs.length; i++) 
		{
			if(outputs[i].getFileFormatProperties() != null 
					&& outputs[i].getFileFormatProperties().getFileFormat() == FileFormatProperties.FileFormat.CSV)
				csvoutputs.add(i);
			else
				bboutputs.add(i);
		}
		boolean isCSV = (csvoutputs.size() > 0);
		boolean isBB  = (bboutputs.size()  > 0);
		String tmpPath = MRJobConfiguration.constructTempOutputFilename();
		
		checkIfOutputOverlapsWithTxMtd(outputs, oprnds, isCSV, isBB, csvoutputs, bboutputs, fs);
		
		JobReturn retCSV = null, retBB = null;
		
		if (!oprnds.isApply) {
			// build specification file with column IDs insteadof column names
			String specWithIDs = processSpecFile(fs, oprnds.inputPath, 
							smallestFile, colNamesToIds, oprnds.inputCSVProperties, oprnds.spec);
			colNamesToIds = null; // enable GC on colNamesToIds

			// Build transformation metadata, including recode maps, bin definitions, etc.
			// Also, generate part offsets file (counters file), which is to be used in csv-reblock
			
			String partOffsetsFile =  MRJobConfiguration.constructTempOutputFilename();
			numRowsTf = GenTfMtdMR.runJob(oprnds.inputPath, oprnds.txMtdPath, specWithIDs, smallestFile, 
					partOffsetsFile, oprnds.inputCSVProperties, numColumns, replication, outHeader);
			
			if ( numRowsTf == 0 )
				throw new DMLRuntimeException(ERROR_MSG_ZERO_ROWS);
			
			// store the specFileWithIDs as transformation metadata
			MapReduceTool.writeStringToHDFS(specWithIDs, oprnds.txMtdPath + "/" + "spec.json");
			
			numColumnsTf = getNumColumnsTf(fs, outHeader, oprnds.inputCSVProperties.getDelim(), oprnds.txMtdPath);
			
			// Apply transformation metadata, and perform actual transformation 
			if(isCSV)
				retCSV = ApplyTfCSVMR.runJob(oprnds.inputPath, specWithIDs, oprnds.txMtdPath, tmpPath, 
					outputs[csvoutputs.get(0)].getFileName(), partOffsetsFile, 
					oprnds.inputCSVProperties, numColumns, replication, outHeader);
			
			if(isBB)
			{
				DMLConfig conf = ConfigurationManager.getDMLConfig();
				int blockSize = conf.getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
				CSVReblockInstruction rblk = prepDummyReblockInstruction(oprnds.inputCSVProperties, blockSize);
				
				AssignRowIDMRReturn ret1 = CSVReblockMR.runAssignRowIDMRJob(new String[]{oprnds.inputPath}, 
							new InputInfo[]{InputInfo.CSVInputInfo}, new int[]{blockSize}, new int[]{blockSize}, 
							rblk.toString(), replication, new String[]{smallestFile}, true, 
							oprnds.inputCSVProperties.getNAStrings(), specWithIDs);
				if ( ret1.rlens[0] == 0 )
					throw new DMLRuntimeException(ERROR_MSG_ZERO_ROWS);
					
				retBB = ApplyTfBBMR.runJob(oprnds.inputPath, insts[1], otherInst, 
							specWithIDs, oprnds.txMtdPath, tmpPath, outputs[bboutputs.get(0)].getFileName(), 
							ret1.counterFile.toString(), oprnds.inputCSVProperties, numRowsTf, numColumns, 
							numColumnsTf, replication, outHeader);
			}
			
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(partOffsetsFile), job);
				
		}
		else {
			colNamesToIds = null; // enable GC on colNamesToIds
			
			// copy given transform metadata (applyTxPath) to specified location (txMtdPath)
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(oprnds.txMtdPath), job);
			MapReduceTool.copyFileOnHDFS(oprnds.applyTxPath, oprnds.txMtdPath);
			
			// path to specification file
			String specWithIDs = (oprnds.spec != null) ? oprnds.spec : 
				MapReduceTool.readStringFromHDFSFile(oprnds.txMtdPath + "/" + "spec.json");
			numColumnsTf = getNumColumnsTf(fs, outHeader, 
												oprnds.inputCSVProperties.getDelim(), 
												oprnds.txMtdPath);
			
			if (isCSV) 
			{
				DMLConfig conf = ConfigurationManager.getDMLConfig();
				int blockSize = conf.getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
				CSVReblockInstruction rblk = prepDummyReblockInstruction(oprnds.inputCSVProperties, blockSize);
				
				AssignRowIDMRReturn ret1 = CSVReblockMR.runAssignRowIDMRJob(new String[]{oprnds.inputPath}, 
							new InputInfo[]{InputInfo.CSVInputInfo}, new int[]{blockSize}, new int[]{blockSize}, 
							rblk.toString(), replication, new String[]{smallestFile}, true, 
							oprnds.inputCSVProperties.getNAStrings(), specWithIDs);
				numRowsTf = ret1.rlens[0];
				
				if ( ret1.rlens[0] == 0 )
					throw new DMLRuntimeException(ERROR_MSG_ZERO_ROWS);
					
				// Apply transformation metadata, and perform actual transformation 
				retCSV = ApplyTfCSVMR.runJob(oprnds.inputPath, specWithIDs, oprnds.applyTxPath, tmpPath, 
							outputs[csvoutputs.get(0)].getFileName(), ret1.counterFile.toString(), 
							oprnds.inputCSVProperties, numColumns, replication, outHeader);
			}
			
			if(isBB) 
			{
				// compute part offsets file
				CSVReblockInstruction rblk = (CSVReblockInstruction) InstructionParser.parseSingleInstruction(insts[1]);
				CSVReblockInstruction newrblk = (CSVReblockInstruction) rblk.clone((byte)0);
				AssignRowIDMRReturn ret1 = CSVReblockMR.runAssignRowIDMRJob(new String[]{oprnds.inputPath}, 
							new InputInfo[]{InputInfo.CSVInputInfo}, new int[]{newrblk.brlen}, new int[]{newrblk.bclen}, 
							newrblk.toString(), replication, new String[]{smallestFile}, true, 
							oprnds.inputCSVProperties.getNAStrings(), specWithIDs);
				numRowsTf = ret1.rlens[0];
				
				if ( ret1.rlens[0] == 0 )
					throw new DMLRuntimeException(ERROR_MSG_ZERO_ROWS);
				
				// apply transformation metadata, as well as reblock the resulting data
				retBB = ApplyTfBBMR.runJob(oprnds.inputPath, insts[1], otherInst, specWithIDs, 
							oprnds.txMtdPath, tmpPath, outputs[bboutputs.get(0)].getFileName(), 
							ret1.counterFile.toString(), oprnds.inputCSVProperties, ret1.rlens[0], 
							ret1.clens[0], numColumnsTf, replication, outHeader);
			}
		}
		
		// copy auxiliary data (old and new header lines) from temporary location to txMtdPath
		moveFilesFromTmp(fs, tmpPath, oprnds.txMtdPath);

		// generate matrix metadata file for outputs
		if ( retCSV != null ) 
		{
			retCSV.getMatrixCharacteristics(0).setDimension(numRowsTf, numColumnsTf);
			
			CSVFileFormatProperties prop = new CSVFileFormatProperties(
												false, 
												oprnds.inputCSVProperties.getDelim(), // use the same header as the input
												false, Double.NaN, null);
			
			MapReduceTool.writeMetaDataFile (outputs[csvoutputs.get(0)].getFileName()+".mtd", 
												ValueType.DOUBLE, retCSV.getMatrixCharacteristics(0), 
												OutputInfo.CSVOutputInfo, prop);
			return retCSV;
		}

		if ( retBB != null )
		{
			retBB.getMatrixCharacteristics(0).setDimension(numRowsTf, numColumnsTf);
			
			MapReduceTool.writeMetaDataFile (outputs[bboutputs.get(0)].getFileName()+".mtd", 
					ValueType.DOUBLE, retBB.getMatrixCharacteristics(0), OutputInfo.BinaryBlockOutputInfo);
			return retBB;
		}
		
		return null;
			
	}
	
	private static CSVReblockInstruction prepDummyReblockInstruction(CSVFileFormatProperties prop, int blockSize) {
		StringBuilder sb = new StringBuilder();
		sb.append( ExecType.MR );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( CSVReBlock.OPCODE );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "0" );
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(DataType.MATRIX);
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(ValueType.DOUBLE);
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "1" );
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(DataType.MATRIX);
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(ValueType.DOUBLE);
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( blockSize );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( blockSize );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( prop.hasHeader() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( prop.getDelim() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( prop.isFill() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( prop.getFillValue() );

		return (CSVReblockInstruction) CSVReblockInstruction.parseInstruction(sb.toString());
	}

	private static String getOutputHeader(FileSystem fs, String headerLine, TransformOperands oprnds) throws IOException
	{
		String ret = null;
		
		if(oprnds.isApply)
		{
			BufferedReader br = new BufferedReader(new InputStreamReader( fs.open(new Path(oprnds.applyTxPath + "/" + TfUtils.TXMTD_COLNAMES)) ));
			ret = br.readLine();
			br.close();
		}
		else {
			if ( oprnds.outNamesFile == null )
				ret = headerLine;
			else {
				BufferedReader br = new BufferedReader(new InputStreamReader( fs.open(new Path(oprnds.outNamesFile)) ));
				ret = br.readLine();
				br.close();
			}
		}
		
		return ret;
	}
	
	/**
	 * Main method to create and/or apply transformation metdata in-memory, on a
	 * single node.
	 * 
	 * @param inst
	 * @param inputMatrices
	 * @param outputMatrices
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 * @throws JSONException 
	 * @throws IllegalArgumentException 
	 */
	public static JobReturn cpDataTransform(ParameterizedBuiltinCPInstruction inst, CacheableData<?>[] inputs, MatrixObject[] outputs) throws IOException, DMLRuntimeException, IllegalArgumentException, JSONException {
		TransformOperands oprnds = new TransformOperands(inst.getParameterMap(), inputs[0]);
		return cpDataTransform(oprnds, inputs, outputs);
	}

	public static JobReturn cpDataTransform(String inst, CacheableData<?>[] inputs, MatrixObject[] outputs) 
		throws IOException, DMLRuntimeException, IllegalArgumentException, JSONException 
	{
		String[] insts = inst.split(Instruction.INSTRUCTION_DELIM);
		// Parse transform instruction (the first instruction) to obtain relevant fields
		TransformOperands oprnds = new TransformOperands(insts[0], inputs[0]);
		
		return cpDataTransform(oprnds, inputs, outputs);
	}
		
	public static JobReturn cpDataTransform(TransformOperands oprnds, CacheableData<?>[] inputs, MatrixObject[] outputs) 
		throws IOException, DMLRuntimeException, IllegalArgumentException, JSONException 
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		FileSystem fs = FileSystem.get(job);
		// find the first file in alphabetical ordering of partfiles in directory inputPath 
		String smallestFile = CSVReblockMR.findSmallestFile(job, oprnds.inputPath);
		
		// find column names
		String headerLine = readHeaderLine(fs, oprnds.inputCSVProperties, smallestFile);
		HashMap<String, Integer> colNamesToIds = processColumnNames(fs, oprnds.inputCSVProperties, headerLine, smallestFile);
		String outHeader = getOutputHeader(fs, headerLine, oprnds);
		
		ArrayList<Integer> csvoutputs= new ArrayList<Integer>();
		ArrayList<Integer> bboutputs = new ArrayList<Integer>();
		
		// divide output objects based on output format (CSV or BinaryBlock)
		for(int i=0; i < outputs.length; i++) 
		{
			if(outputs[i].getFileFormatProperties() != null && outputs[i].getFileFormatProperties().getFileFormat() == FileFormatProperties.FileFormat.CSV)
				csvoutputs.add(i);
			else
				bboutputs.add(i);
		}
		boolean isCSV = (csvoutputs.size() > 0);
		boolean isBB  = (bboutputs.size()  > 0);
		
		checkIfOutputOverlapsWithTxMtd(outputs, oprnds, isCSV, isBB, csvoutputs, bboutputs, fs);
		
		JobReturn ret = null;
		
		if (!oprnds.isApply) {
			// build specification file with column IDs insteadof column names
			String specWithIDs = processSpecFile(fs, oprnds.inputPath, smallestFile, colNamesToIds, oprnds.inputCSVProperties, 
					oprnds.spec);
			MapReduceTool.writeStringToHDFS(specWithIDs, oprnds.txMtdPath + "/" + "spec.json");
	
			ret = performTransform(job, fs, oprnds.inputPath, colNamesToIds.size(), oprnds.inputCSVProperties, specWithIDs, 
					oprnds.txMtdPath, oprnds.isApply, outputs[0], outHeader, isBB, isCSV );
		}
		else {
			// copy given transform metadata (applyTxPath) to specified location (txMtdPath)
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(oprnds.txMtdPath), job);
			MapReduceTool.copyFileOnHDFS(oprnds.applyTxPath, oprnds.txMtdPath);
			
			// path to specification file (optionally specified)
			String specWithIDs = (oprnds.spec != null) ? 
				oprnds.spec : MapReduceTool.readStringFromHDFSFile(oprnds.txMtdPath + "/" + "spec.json");
			
			ret = performTransform(job, fs, oprnds.inputPath, colNamesToIds.size(), oprnds.inputCSVProperties, specWithIDs,  
					oprnds.txMtdPath, oprnds.isApply, outputs[0], outHeader, isBB, isCSV );
		}
		
		return ret;
	}
	
	/**
	 * Apply given transform metadata (incl recode maps) over an in-memory frame input in order to
	 * create a transformed numerical matrix. Note: The number of rows always remains unchanged, 
	 * whereas the number of column might increase or decrease. 
	 * 
	 * @param params
	 * @param input
	 * @param meta
	 * @param spec
	 * @return
	 * @throws DMLRuntimeException
	 * @throws  
	 */
	public static MatrixBlock cpDataTransform(HashMap<String,String> params, FrameBlock input, FrameBlock meta) 
		throws DMLRuntimeException
	{
		Encoder encoder = EncoderFactory.createEncoder(params.get("spec"), input.getNumColumns(), meta);
		return encoder.apply(input, new MatrixBlock(input.getNumRows(), input.getNumColumns(), false));
	}
	
	/**
	 * Helper function to fetch and sort the list of part files under the given
	 * input directory.
	 * 
	 * @param input
	 * @param fs
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	private static ArrayList<Path> collectInputFiles(String input, FileSystem fs) throws FileNotFoundException, IOException 
	{
		Path path = new Path(input);
		ArrayList<Path> files=new ArrayList<Path>();
		if(fs.isDirectory(path))
		{
			for(FileStatus stat: fs.listStatus(path, CSVReblockMR.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);

		return files;
	}
	
	private static int[] countNumRows(ArrayList<Path> files, CSVFileFormatProperties prop, FileSystem fs, TfUtils agents) throws IOException 
	{
		int[] rows = new int[2];
		int numRows=0, numRowsTf=0;
		
		OmitAgent oa = agents.getOmitAgent();
		
		if(!oa.isApplicable())
		{
			for(int fileNo=0; fileNo<files.size(); fileNo++)
			{
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
				if(fileNo==0 && prop.hasHeader() ) 
					br.readLine(); //ignore header
				
				while ( br.readLine() != null)
					numRows++;
				br.close();
			}
			numRowsTf = numRows;
		}
		else
		{
			String line = null;
			String[] words;
			
			Pattern delim = Pattern.compile(Pattern.quote(prop.getDelim()));
			
			for(int fileNo=0; fileNo<files.size(); fileNo++)
			{
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
				if(fileNo==0 && prop.hasHeader() ) 
					br.readLine(); //ignore header
				
				while ( (line=br.readLine()) != null)
				{
					numRows++;
					
					words = delim.split(line, -1);
					if(!oa.omit(words, agents))
						numRowsTf++;
				}
				br.close();
			}
		}
		
		rows[0] = numRows;
		rows[1] = numRowsTf;
		
		return rows;
	}
	
	/**
	 * Main method to create and/or apply transformation metdata in-memory, on a single node.
	 * 
	 * @param job
	 * @param fs
	 * @param inputPath
	 * @param ncols
	 * @param prop
	 * @param specFileWithIDs
	 * @param tfMtdPath
	 * @param applyTxPath
	 * @param isApply
	 * @param outputPath
	 * @param headerLine
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 * @throws JSONException 
	 * @throws IllegalArgumentException 
	 */
	private static JobReturn performTransform(JobConf job, FileSystem fs, String inputPath, int ncols, CSVFileFormatProperties prop, String specWithIDs, String tfMtdPath, boolean isApply, MatrixObject result, String headerLine, boolean isBB, boolean isCSV ) throws IOException, DMLRuntimeException, IllegalArgumentException, JSONException {
		
		String[] na = TfUtils.parseNAStrings(prop.getNAStrings());
		
		JSONObject spec = new JSONObject(specWithIDs);
		TfUtils agents = new TfUtils(headerLine, prop.hasHeader(), prop.getDelim(), na, spec, ncols, tfMtdPath, null, null );
		
		MVImputeAgent _mia = agents.getMVImputeAgent();
		RecodeAgent _ra = agents.getRecodeAgent();
		BinAgent _ba = agents.getBinAgent();
		DummycodeAgent _da = agents.getDummycodeAgent();

		// List of files to read
		ArrayList<Path> files = collectInputFiles(inputPath, fs);
				
		// ---------------------------------
		// Construct transformation metadata
		// ---------------------------------
		
		String line = null;
		String[] words  = null;
		
		int numColumnsTf=0;
		BufferedReader br = null;
		
		if (!isApply) {
			for(int fileNo=0; fileNo<files.size(); fileNo++)
			{
				br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
				if(fileNo==0 && prop.hasHeader() ) 
					br.readLine(); //ignore header
				
				line = null;
				while ( (line = br.readLine()) != null) {
					agents.prepareTfMtd(line);
				}
				br.close();
			}
			
			if(agents.getValid() == 0) 
				throw new DMLRuntimeException(ERROR_MSG_ZERO_ROWS);
			
			_mia.outputTransformationMetadata(tfMtdPath, fs, agents);
			_ba.outputTransformationMetadata(tfMtdPath, fs, agents);
			_ra.outputTransformationMetadata(tfMtdPath, fs, agents);
		
			// prepare agents for the subsequent phase of applying transformation metadata
			
			// NO need to loadTxMtd for _ra, since the maps are already present in the memory
			Path tmp = new Path(tfMtdPath);
			_mia.loadTxMtd(job, fs, tmp, agents);
			_ba.loadTxMtd(job, fs, tmp, agents);
			
			_da.setRecodeMapsCP( _ra.getCPRecodeMaps() );
			_da.setNumBins(_ba.getColList(), _ba.getNumBins());
			_da.loadTxMtd(job, fs, tmp, agents);
		}
		else {
			// Count the number of rows
			int rows[] = countNumRows(files, prop, fs, agents);
			agents.setTotal(rows[0]);
			agents.setValid(rows[1]);
			
			if(agents.getValid() == 0) 
				throw new DMLRuntimeException("Number of rows in the transformed output (potentially, after ommitting the ones with missing values) is zero. Cannot proceed.");
			
			// Load transformation metadata
			// prepare agents for the subsequent phase of applying transformation metadata
			Path tmp = new Path(tfMtdPath);
			_mia.loadTxMtd(job, fs, tmp, agents);
			_ra.loadTxMtd(job, fs, tmp, agents);
			_ba.loadTxMtd(job, fs, tmp, agents);
			
			_da.setRecodeMaps( _ra.getRecodeMaps() );
			_da.setNumBins(_ba.getColList(), _ba.getNumBins());
			_da.loadTxMtd(job, fs, tmp, agents);
		}
		
		// -----------------------------
		// Apply transformation metadata
		// -----------------------------
        
		numColumnsTf = getNumColumnsTf(fs, headerLine, prop.getDelim(), tfMtdPath);

		MapReduceTool.deleteFileIfExistOnHDFS(result.getFileName());
		BufferedWriter out=new BufferedWriter(new OutputStreamWriter(fs.create(new Path(result.getFileName()),true)));		
		StringBuilder sb = new StringBuilder();
		
		MatrixBlock mb = null; 
		if ( isBB ) 
		{
			int estNNZ = (int)agents.getValid() * ncols;
			mb = new MatrixBlock((int)agents.getValid(), numColumnsTf, estNNZ );
			
			if ( mb.isInSparseFormat() )
				mb.allocateSparseRowsBlock();
			else
				mb.allocateDenseBlock();
		}

		int rowID = 0; // rowid to be used in filling the matrix block
		
		for(int fileNo=0; fileNo<files.size(); fileNo++)
		{
			br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo))));
			if ( fileNo == 0 ) 
			{
				if ( prop.hasHeader() )
					br.readLine(); // ignore the header line from data file
				
				//TODO: fix hard-wired header propagation to meta data column names
				
				String dcdHeader = _da.constructDummycodedHeader(headerLine, agents.getDelim());
				numColumnsTf = _da.genDcdMapsAndColTypes(fs, tfMtdPath, ncols, agents);
				generateHeaderFiles(fs, tfMtdPath, headerLine, dcdHeader);
			}
			
			line = null;
			while ( (line = br.readLine()) != null) {
				words = agents.getWords(line);

				if(!agents.omit(words))
				{
					words = agents.apply(words);
	
					if (isCSV)
					{
						out.write( agents.checkAndPrepOutputString(words, sb) );
						out.write("\n");
					}
					
					if( isBB ) 
					{
						agents.check(words);
						for(int c=0; c<words.length; c++)
						{
							if(words[c] == null || words[c].isEmpty())
								;
							else 
								mb.appendValue(rowID, c, UtilFunctions.parseToDouble(words[c]));
						}
					}
					rowID++;
				}
			}
			br.close();
		}
		out.close();
		
		if(mb != null)
		{
			mb.recomputeNonZeros();
			mb.examSparsity();
			
			result.acquireModify(mb);
			result.release();
			result.exportData();
		}
		
		MatrixCharacteristics mc = new MatrixCharacteristics(agents.getValid(), numColumnsTf, (int) result.getNumRowsPerBlock(), (int) result.getNumColumnsPerBlock());
		JobReturn ret = new JobReturn(new MatrixCharacteristics[]{mc}, true);

		return ret;
	}
	
	public static void generateHeaderFiles(FileSystem fs, String txMtdDir, String origHeader, String newHeader) throws IOException {
		// write out given header line
		Path pt=new Path(txMtdDir+"/" + TfUtils.TXMTD_COLNAMES);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(origHeader+"\n");
		br.close();

		// write out the new header line (after all transformations)
		pt = new Path(txMtdDir + "/" + TfUtils.TXMTD_DC_COLNAMES);
		br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(newHeader+"\n");
		br.close();
	}
	
	private static void checkIfOutputOverlapsWithTxMtd(MatrixObject[] outputs, TransformOperands oprnds,
			boolean isCSV, boolean isBB, ArrayList<Integer> csvoutputs, ArrayList<Integer> bboutputs, FileSystem fs) throws DMLRuntimeException {
		if(isCSV) {
			checkIfOutputOverlapsWithTxMtd(oprnds.txMtdPath, outputs[csvoutputs.get(0)].getFileName(), fs);
		}
		else if(isBB) {
			checkIfOutputOverlapsWithTxMtd(oprnds.txMtdPath, outputs[bboutputs.get(0)].getFileName(), fs);
		}
	}
	
	@SuppressWarnings("deprecation")
	private static void checkIfOutputOverlapsWithTxMtd(String txMtdPath, String outputPath, FileSystem fs) 
		throws DMLRuntimeException 
	{
		Path path1 = new Path(txMtdPath).makeQualified(fs);
		Path path2 = new Path(outputPath).makeQualified(fs);
		
		String fullTxMtdPath = path1.toString();
		String fullOutputPath = path2.toString();
		
		if(path1.getParent().toString().equals(path2.getParent().toString())) {
			// Both txMtdPath and outputPath are in same folder, but outputPath can have suffix 
			if(fullTxMtdPath.equals(fullOutputPath)) {
				throw new DMLRuntimeException("The transform path \'" + txMtdPath 
						+ "\' cannot overlap with the output path \'" + outputPath + "\'");
			}
		}
		else if(fullTxMtdPath.startsWith(fullOutputPath) || fullOutputPath.startsWith(fullTxMtdPath)) {
			throw new DMLRuntimeException("The transform path \'" + txMtdPath 
					+ "\' cannot overlap with the output path \'" + outputPath + "\'");
		}
	}
	
	public static void spDataTransform(ParameterizedBuiltinSPInstruction inst, FrameObject[] inputs, MatrixObject[] outputs, ExecutionContext ec) throws Exception {
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		// Parse transform instruction (the first instruction) to obtain relevant fields
		TransformOperands oprnds = new TransformOperands(inst.getParams(), inputs[0]);
		
		JobConf job = new JobConf();
		FileSystem fs = FileSystem.get(job);
		
		checkIfOutputOverlapsWithTxMtd(oprnds.txMtdPath, outputs[0].getFileName(), fs);
		
		// find the first file in alphabetical ordering of partfiles in directory inputPath 
		String smallestFile = CSVReblockMR.findSmallestFile(job, oprnds.inputPath);
		
		// find column names and construct output header
		String headerLine = readHeaderLine(fs, oprnds.inputCSVProperties, smallestFile);
		HashMap<String, Integer> colNamesToIds = processColumnNames(fs, oprnds.inputCSVProperties, headerLine, smallestFile);
		int numColumns = colNamesToIds.size();
		String outHeader = getOutputHeader(fs, headerLine, oprnds);
		
		String tmpPath = MRJobConfiguration.constructTempOutputFilename();
		
		// Construct RDD for input data
		@SuppressWarnings("unchecked")
		JavaPairRDD<LongWritable, Text> inputData = (JavaPairRDD<LongWritable, Text>) sec.getRDDHandleForFrameObject(inputs[0], InputInfo.CSVInputInfo);
		JavaRDD<Tuple2<LongWritable,Text>> csvLines = JavaPairRDD.toRDD(inputData).toJavaRDD();
		
		long numRowsTf=0, numColumnsTf=0;
		JavaPairRDD<Long, String> tfPairRDD = null;
		
		if (!oprnds.isApply) {
			// build specification file with column IDs insteadof column names
			String specWithIDs = processSpecFile(fs, oprnds.inputPath, smallestFile,
						colNamesToIds, oprnds.inputCSVProperties, oprnds.spec);
			colNamesToIds = null; // enable GC on colNamesToIds

			// Build transformation metadata, including recode maps, bin definitions, etc.
			// Also, generate part offsets file (counters file), which is to be used in csv-reblock (if needed)
			String partOffsetsFile =  MRJobConfiguration.constructTempOutputFilename();
			numRowsTf = GenTfMtdSPARK.runSparkJob(sec, csvLines, oprnds.txMtdPath,  
													specWithIDs,partOffsetsFile, 
													oprnds.inputCSVProperties, numColumns, 
													outHeader);
			
			// store the specFileWithIDs as transformation metadata
			MapReduceTool.writeStringToHDFS(specWithIDs, oprnds.txMtdPath + "/" + "spec.json");
			
			numColumnsTf = getNumColumnsTf(fs, outHeader, oprnds.inputCSVProperties.getDelim(), oprnds.txMtdPath);
			
			tfPairRDD = ApplyTfCSVSPARK.runSparkJob(sec, csvLines, oprnds.txMtdPath, 
					specWithIDs, tmpPath, oprnds.inputCSVProperties, numColumns, outHeader);

			
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(partOffsetsFile), job);
		}
		else {
			colNamesToIds = null; // enable GC on colNamesToIds
			
			// copy given transform metadata (applyTxPath) to specified location (txMtdPath)
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(oprnds.txMtdPath), job);
			MapReduceTool.copyFileOnHDFS(oprnds.applyTxPath, oprnds.txMtdPath);
			
			// path to specification file
			String specWithIDs = (oprnds.spec != null) ? oprnds.spec :
					MapReduceTool.readStringFromHDFSFile(oprnds.txMtdPath + "/" + "spec.json");
			numColumnsTf = getNumColumnsTf(fs, outHeader, 
												oprnds.inputCSVProperties.getDelim(), 
												oprnds.txMtdPath);
			
			// Apply transformation metadata, and perform actual transformation 
			tfPairRDD = ApplyTfCSVSPARK.runSparkJob(sec, csvLines, oprnds.txMtdPath, 
					specWithIDs, tmpPath, oprnds.inputCSVProperties, numColumns, outHeader);
			
		}
		
		// copy auxiliary data (old and new header lines) from temporary location to txMtdPath
		moveFilesFromTmp(fs, tmpPath, oprnds.txMtdPath);

		// convert to csv output format (serialized longwritable/text)
		JavaPairRDD<LongWritable, Text> outtfPairRDD = 
				RDDConverterUtils.stringToSerializableText(tfPairRDD);
		
		if ( outtfPairRDD != null ) 
		{
			MatrixObject outMO = outputs[0];
			String outVar = outMO.getVarName();
			outMO.setRDDHandle(new RDDObject(outtfPairRDD, outVar));
			sec.addLineageRDD(outVar, inst.getParams().get("target"));
			
			//update output statistics (required for correctness)
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(outVar);
			mcOut.setDimension(numRowsTf, numColumnsTf);
			mcOut.setNonZeros(-1);
		}
	}
	

	/**
	 * Private class to hold the relevant input parameters to transform operation.
	 */
	private static class TransformOperands 
	{
		private String inputPath=null;
		private String txMtdPath=null;
		private String applyTxPath=null;
		private String spec=null;
		private String outNamesFile=null;
		private boolean isApply=false;
		private CSVFileFormatProperties inputCSVProperties = null;
		
		private TransformOperands(String inst, CacheableData<?> input) {
			inputPath = input.getFileName();
			inputCSVProperties = (CSVFileFormatProperties)input.getFileFormatProperties();
			String[] instParts = inst.split(Instruction.OPERAND_DELIM);
			txMtdPath = instParts[3];
			applyTxPath = instParts[4].startsWith("applymtd=") ? instParts[4].substring(9) : null;
			isApply = (applyTxPath != null);
			int pos = (applyTxPath != null) ? 5 : 4;
			if( pos<instParts.length )
				spec = instParts[pos].startsWith("spec=") ? instParts[pos++].substring(5) : null;
			if( pos<instParts.length )
				outNamesFile = instParts[pos].startsWith("outnames=") ? instParts[pos].substring(9) : null;
		}
		
		private TransformOperands(HashMap<String, String> params, CacheableData<?> input) {
			inputPath = input.getFileName();
			txMtdPath = params.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_MTD);
			spec = params.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_SPEC);
			applyTxPath = params.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_APPLYMTD);
			isApply = (applyTxPath != null);
			outNamesFile =  params.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_OUTNAMES); // can be null
			inputCSVProperties = (CSVFileFormatProperties)input.getFileFormatProperties();
		}
	}
}

