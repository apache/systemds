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

import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class RecodeAgent extends TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public int[] _rcdList = null;	// List of attributes to recode
	
	// HashMap< columnID, HashMap<distinctValue, count> >
	private HashMap<Integer, HashMap<String, Long>> _rcdMaps  = new HashMap<Integer, HashMap<String, Long>>();
	
	RecodeAgent() { }
	
	RecodeAgent(int[] list) {
		_rcdList = list;
	}
	
	RecodeAgent(JSONObject parsedSpec) {
		Object obj = parsedSpec.get(TX_METHOD.RECODE.toString());
		if(obj == null) {
			
		}
		else {
			JSONArray attrs = (JSONArray) ((JSONObject)obj).get(JSON_ATTRS);
			
			_rcdList = new int[attrs.size()];
			for(int i=0; i < _rcdList.length; i++) 
				_rcdList[i] = ((Long) attrs.get(i)).intValue();
		}
	}
	
	void prepare(String[] words) {
		if ( _rcdList == null )
			return;
		String w = null;
		for (int colID : _rcdList) {
			w = UtilFunctions.unquote(words[colID-1].trim());
			if(_rcdMaps.get(colID) == null ) 
				_rcdMaps.put(colID, new HashMap<String, Long>());
			
			HashMap<String, Long> map = _rcdMaps.get(colID);
			Long count = map.get(w);
			if(count == null)
				map.put(w, new Long(1));
			else
				map.put(w, count+1);
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
		if ( _rcdList == null )
			return;
		
		try 
		{ 
			for(int i=0; i < _rcdList.length; i++) 
			{
				int colID = _rcdList[i];
				HashMap<String, Long> map = _rcdMaps.get(colID);
				
				if(map != null) 
				{
					IntWritable iw = new IntWritable(colID);
					for(String s : map.keySet()) 
						out.collect(iw, new DistinctValue(s, map.get(s)));
				}
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	private void writeMetadata(HashMap<String,Long> map, String outputDir, int colID, FileSystem fs) throws IOException {
		// output recode maps and mode
		
		String mode = null;
		Long count = null;
		int rcdIndex = 0, modeIndex = 0;
		long maxCount = Long.MIN_VALUE;
		
		Path pt=new Path(outputDir+"/Recode/"+ columnNames[colID-1] + RCD_MAP_FILE_SUFFIX);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		

		// remove NA strings
		if ( TransformationAgent.NAstrings != null ) 
		{
			for(String naword : TransformationAgent.NAstrings) 
				map.remove(naword);
		}
		
		if ( map.size() == 0 ) 
		{
			throw new RuntimeException("Can not proceed since the column id " + colID + " contains only the missing values, and not a single valid value.");
		}
		
		for(String w : map.keySet()) {
			//if ( !MVImputeAgent.isNA(w, TransformationAgent.NAstrings)) {
				count = map.get(w);
				++rcdIndex;
				
				// output (w, count, rcdIndex)
				br.write(UtilFunctions.quote(w) + TXMTD_SEP + rcdIndex + TXMTD_SEP + count  + "\n");
				
				if(maxCount < count) {
					maxCount = count;
					mode = w;
					modeIndex = rcdIndex;
				}
				
				// Replace count with recode index (useful when invoked from CP)
				map.put(w, (long)rcdIndex);
			//}
		}
		br.close();
		
		if ( mode == null ) {
			mode = "";
			maxCount = 0;
		}
		
		// output number of distinct values
		pt=new Path(outputDir+"/Recode/"+ columnNames[colID-1] + MODE_FILE_SUFFIX);
		br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(UtilFunctions.quote(mode) + "," + modeIndex + "," + maxCount );
		br.close();
		
		// output "mode"
		pt=new Path(outputDir+"/Recode/"+ columnNames[colID-1] + NDISTINCT_FILE_SUFFIX);
		br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		br.write(""+map.size());
		br.close();
	}
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs) throws IOException {
		if(_rcdList == null)
			return;
		for(int i=0; i<_rcdList.length; i++) {
			int colID = _rcdList[i];
			writeMetadata(_rcdMaps.get(colID), outputDir, colID, fs);
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
		HashMap<String, Long> map = new HashMap<String,Long>();
		
		DistinctValue d = new DistinctValue();
		String word = null;
		Long count = null, val = null;
		while(values.hasNext()) {
			d.reset();
			d = values.next();
			
			word = d.getWord();
			count = d.getCount();
			
			val = map.get(word);
			if(val == null) 
				map.put(word, count);
			else 
				map.put(word, val+count);
		}
		
		writeMetadata(map, outputDir, colID, FileSystem.get(job));
	}
	
	// ------------------------------------------------------------------------------------------------
	
	public HashMap<Integer, HashMap<String,Long>> getCPRecodeMaps() { return _rcdMaps; }
	
	HashMap<Integer, HashMap<String,String>> _finalMaps = null;
	public HashMap<Integer, HashMap<String,String>> getRecodeMaps() {
		return _finalMaps;
	}
	
	/**
	 * Method to load recode maps of all attributes, at once.
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir) throws IOException {
		if ( _rcdList == null )
			return;
		
		_finalMaps = new HashMap<Integer, HashMap<String, String>>();
		//Path txMtdDir = (DistributedCache.getLocalCacheFiles(job))[0];
		
		//FileSystem fs = FileSystem.getLocal(job);
		
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_rcdList.length;i++) {
				int colID = _rcdList[i];
				
				Path path = new Path( txMtdDir + "/Recode/" + columnNames[colID-1] + RCD_MAP_FILE_SUFFIX);
				TransformationAgent.checkValidInputFile(fs, path, true); 
				
				HashMap<String,String> map = new HashMap<String,String>();
				
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
				String line = null, word=null;
				String rcdIndex = null;
				
				// Example line to parse: "WN (1)67492",1,61975
				while((line=br.readLine())!=null) {
					
					// last occurrence of quotation mark
					int idxQuote = line.lastIndexOf('"');
					word = UtilFunctions.unquote(line.substring(0,idxQuote+1));
					
					int idx = idxQuote+2;
					while(line.charAt(idx) != TXMTD_SEP.charAt(0))
						idx++;
					rcdIndex = line.substring(idxQuote+2,idx); 
					
					map.put(word, rcdIndex);
				}
				br.close();
				_finalMaps.put(colID, map);
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
		if ( _rcdList == null )
			return words;
		
		for(int i=0; i < _rcdList.length; i++) {
			int colID = _rcdList[i];
			try {
				words[colID-1] = _finalMaps.get(colID).get(UtilFunctions.unquote(words[colID-1].trim()));
			} catch(NullPointerException e) {
				System.err.println("Maps for colID="+colID + " may be null (map = " + _finalMaps.get(colID) + ")");
				throw new RuntimeException(e);
			}
		}
			
		return words;
	}
	
	
	public String[] cp_apply(String[] words) {
		if ( _rcdList == null )
			return words;
		
		for(int i=0; i < _rcdList.length; i++) {
			int colID = _rcdList[i];
			try {
				words[colID-1] = Long.toString(_rcdMaps.get(colID).get(UtilFunctions.unquote(words[colID-1].trim())));
			} catch(NullPointerException e) {
				System.err.println("Maps for colID="+colID + " may be null (map = " + _rcdMaps.get(colID) + ")");
				throw new RuntimeException(e);
			}
		}
			
		return words;
	}
	
	
	public void printMaps() {
		for(Integer k : _rcdMaps.keySet()) {
			System.out.println("Column " + k);
			HashMap<String,Long> map = _rcdMaps.get(k);
			for(String w : map.keySet()) {
				System.out.println("    " + w + " : " + map.get(w));
			}
		}
	}
	
	public void print() {
		System.out.print("Recoding List: \n    ");
		for(int i : _rcdList) {
			System.out.print(i + " ");
		}
		System.out.println();
	}
	
	/*public static void main(String[] args) throws IllegalArgumentException, IOException {
		RecodeAgent ra = new RecodeAgent( new int[]{2} );
		
		FileSystem fs = FileSystem.get(new JobConf());
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path("data/recode/emp.csv"))));
		
		String line = br.readLine();
		while( (line=br.readLine()) != null) {
			String[] words = line.split(",");
			ra.prepare(words);
		}
		ra.printMaps();
	}*/
}
 