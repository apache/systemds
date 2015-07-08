/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

import com.google.common.base.Functions;
import com.google.common.collect.Ordering;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class DummycodeAgent extends TransformationAgent {	
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int[] _dcdList = null;
	private long numCols = 0;
	
	private HashMap<Integer, HashMap<String,String>> _finalMaps = null;
	private HashMap<Integer, HashMap<String,Long>> _finalMapsCP = null;
	private int[] _binList = null;
	private int[] _numBins = null;
	
	private int[] _domainSizes = null;			// length = #of dummycoded columns
	private long _dummycodedLength = 0;			// #of columns after dummycoded
	
	DummycodeAgent(int[] list) {
		_dcdList = list;
	}
	
	DummycodeAgent(JSONObject parsedSpec, long ncol) {
		numCols = ncol;
		
		Object obj = parsedSpec.get(TX_METHOD.DUMMYCODE.toString());
		if(obj == null) {
			return;
		}
		else {
			JSONArray attrs = (JSONArray) ((JSONObject)obj).get(JSON_ATTRS);
			
			_dcdList = new int[attrs.size()];
			for(int i=0; i < _dcdList.length; i++) 
				_dcdList[i] = ((Long) attrs.get(i)).intValue();
		}
	}
	
	public boolean isDummycoded() {
		return (_dcdList != null);
	}
	
	public int[] dcdList() {
		return _dcdList;
	}
	
	/**
	 * Method to output transformation metadata from the mappers. 
	 * This information is collected and merged by the reducers.
	 * 
	 * @param out
	 * @throws IOException
	 * 
	 */
	@Override
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TransformationAgent agent) throws IOException {
		// There is no metadata required for dummycode.
		// Required information is output from RecodeAgent.
		
		return;
	}
	
	@Override
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values,
			String outputDir, int colID, JobConf job) throws IOException {
		// TODO Auto-generated method stub
		
	}

	public void setRecodeMaps(HashMap<Integer, HashMap<String,String>> maps) {
		_finalMaps = maps;
	}
	
	public void setRecodeMapsCP(HashMap<Integer, HashMap<String,Long>> maps) {
		_finalMapsCP = maps;
	}
	
	public void setNumBins(int[] binList, int[] numbins) {
		_binList = binList;
		_numBins = numbins;
	}
	
	public int generateDummycodeMaps(FileSystem fs, String txMtdDir, int numCols, String header, String delim) throws IOException {
		if ( _dcdList == null ) 
			return numCols;
		
		Pattern _delim = Pattern.compile(Pattern.quote(delim));
		String[] names = _delim.split(header, -1);

		Path pt=new Path(txMtdDir+"/Dummycode/" + DCD_FILE_NAME);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		
		int sum=1;
		int idx = 0;
		for(int colID=1; colID <= numCols; colID++) 
		{
			if ( idx < _dcdList.length && _dcdList[idx] == colID )
			{
				br.write(colID + "," + UtilFunctions.quote(names[colID-1]) + "," + "1" + "," + sum + "," + (sum+_domainSizes[idx]-1) + "\n");
				sum += _domainSizes[idx];
				idx++;
			}
			else 
			{
				br.write(colID + "," + UtilFunctions.quote(names[colID-1]) + "," + "0" + "," + sum + "," + sum + "\n");
				sum += 1;
			}
				
		}
		br.close();
		
		return sum-1;
	}
	
	public String constructDummycodedHeader(String header, String delim) {
		
		if(_dcdList == null)
			// none of the columns are dummycoded, simply return the given header
			return header;
		
		Pattern _delim = Pattern.compile(Pattern.quote(delim));
		String[] names = _delim.split(header, -1);
		List<String> newNames = null;

		StringBuilder sb = new StringBuilder();
		
		// Dummycoding can be performed on either a recoded column or a binned column
		
		// process recoded columns
		if(_finalMapsCP != null) 
		{
			for(int i=0; i <_dcdList.length; i++) 
			{
				int colID = _dcdList[i];
				HashMap<String,Long> map = _finalMapsCP.get(colID);
				String colName = UtilFunctions.unquote(names[colID-1]);
				
				if ( map != null  ) 
				{
					// order map entries by their recodeID
					Ordering<String> valueComparator = Ordering.natural().onResultOf(Functions.forMap(map));
					newNames = valueComparator.sortedCopy(map.keySet());
					
					// construct concatenated string of map entries
					sb.setLength(0);
					for(int idx=0; idx < newNames.size(); idx++) 
					{
						if(idx==0) 
							sb.append( colName + DCD_NAME_SEP + newNames.get(idx));
						else
							sb.append( delim + colName + DCD_NAME_SEP + newNames.get(idx));
					}
					names[colID-1] = sb.toString();			// replace original column name with dcd name
					//newColumnLengths[colID-1] = newNames.size();
				}
			}
		}
		else if(_finalMaps != null) {
			for(int i=0; i <_dcdList.length; i++) {
				int colID = _dcdList[i];
				HashMap<String,String> map = _finalMaps.get(colID);
				String colName = UtilFunctions.unquote(names[colID-1]);
				
				if ( map != null ) 
				{
					// order map entries by their recodeID (represented as Strings .. "1", "2", etc.)
					Ordering<String> orderByID = new Ordering<String>() 
					{
			    		public int compare(String s1, String s2) {
			        		return Integer.compare(Integer.parseInt(s1), Integer.parseInt(s2));
			    		}
					};
					
					newNames = orderByID.onResultOf(Functions.forMap(map)).sortedCopy(map.keySet());
					// construct concatenated string of map entries
					sb.setLength(0);
					for(int idx=0; idx < newNames.size(); idx++) 
					{
						if(idx==0) 
							sb.append( colName + DCD_NAME_SEP + newNames.get(idx));
						else
							sb.append( delim + colName + DCD_NAME_SEP + newNames.get(idx));
					}
					names[colID-1] = sb.toString();			// replace original column name with dcd name
					//newColumnLengths[colID-1] = newNames.size();
				}
			}
		}
		
		// process binned columns
		if (_binList != null) 
			for(int i=0; i < _binList.length; i++) 
			{
				int colID = _binList[i];
				int numBins = _numBins[i];
				String colName = UtilFunctions.unquote(names[colID-1]);
				
				sb.setLength(0);
				for(int idx=0; idx < numBins; idx++) 
				{
					if(idx==0) 
						sb.append( colName + DCD_NAME_SEP + "Bin" + (idx+1) );
					else
						sb.append( delim + colName + DCD_NAME_SEP + "Bin" + (idx+1) );
				}
				names[colID-1] = sb.toString();			// replace original column name with dcd name
				//newColumnLengths[colID-1] = numBins;
			}
		
		sb.setLength(0);
		for(int colID=0; colID < names.length; colID++) 
		{
			if (colID == 0)
				sb.append(names[colID]);
			else
				sb.append(delim + names[colID]);
		}
		//System.out.println("DummycodedHeader: " + sb.toString());
		
		return sb.toString();
	}
	
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir) throws IOException {
		if ( _dcdList == null )
			return;
		
		// sort to-be dummycoded column IDs in ascending order. This is the order in which the new dummycoded record is constructed in apply() function.
		Arrays.sort(_dcdList);	
		_domainSizes = new int[_dcdList.length];

		_dummycodedLength = numCols;
		
		//HashMap<String, String> map = null;
		for(int i=0; i<_dcdList.length; i++) {
			int colID = _dcdList[i];
			
			// Find the domain size for colID using _finalMaps or _finalMapsCP
			int domainSize = 0;
			if(_finalMaps != null) {
				if(_finalMaps.get(colID) != null)
					domainSize = _finalMaps.get(colID).size();
			}
			else {
				if(_finalMapsCP.get(colID) != null)
					domainSize = _finalMapsCP.get(colID).size();
			}
			
			if ( domainSize != 0 ) {
				// dummycoded column
				_domainSizes[i] = domainSize;
			}
			else {
				// binned column
				if ( _binList != null )
				for(int j=0; j<_binList.length; j++) {
					if (colID == _binList[j]) {
						_domainSizes[i] = _numBins[j];
						break;
					}
				}
			}
			_dummycodedLength += _domainSizes[i]-1;
			//System.out.println("colID=" + colID + ", domainsize=" + _domainSizes[i] + ", dcdLength=" + _dummycodedLength);
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
		
		if ( _dcdList == null )
			return words;
		
		String[] nwords = new String[(int)_dummycodedLength];
		
		int rcdVal = 0;
		
		for(int colID=1, idx=0, ncolID=1; colID <= words.length; colID++) {
			if(idx < _dcdList.length && colID==_dcdList[idx]) {
				// dummycoded columns
				try {
				rcdVal = UtilFunctions.parseToInt(UtilFunctions.unquote(words[colID-1]));
				nwords[ ncolID-1+rcdVal-1 ] = "1";
				ncolID += _domainSizes[idx];
				idx++;
				} catch (Exception e) {
					System.out.println("Error in dummycoding: colID="+colID + ", rcdVal=" + rcdVal+", word="+words[colID-1] + ", domainSize=" + _domainSizes[idx] + ", dummyCodedLength=" + _dummycodedLength);
					throw new RuntimeException(e);
				}
			}
			else {
				nwords[ncolID-1] = words[colID-1];
				ncolID++;
			}
		}
		
		return nwords;
	}
	
	/**
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	@Override
	public int isTransformed(int colID)
	{
		if(_dcdList == null)
			return -1;
		
		for(int i=0; i < _dcdList.length; i++)
			if( _dcdList[i] == colID )
				return i;
		
		return -1;
	}
	
	@Override
	public void print() {
		System.out.print("Dummycoding List: \n    ");
		for(int i : _dcdList) {
			System.out.print(i + " ");
		}
		System.out.println();
	}

}
