/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;

public class CSVAssignRowIDMapper extends MapReduceBase implements Mapper<LongWritable, Text, ByteWritable, OffsetCount>
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ByteWritable outKey=new ByteWritable();
	private long fileOffset=0;
	private long num=0;
	private boolean first=true;
	private OutputCollector<ByteWritable, OffsetCount> outCache=null;
	private String delim=" ";
	private boolean ignoreFirstLine=false;
	private boolean realFirstLine=false;
	private String filename="";
	private boolean headerFile=false;
	
	@Override
	public void map(LongWritable key, Text value,
			OutputCollector<ByteWritable, OffsetCount> out, Reporter report)
			throws IOException 
	{
		if(first) {
			first=false;
			fileOffset=key.get();
			outCache=out;
		}
		
		if(key.get()==0 && headerFile)//getting the number of colums
		{
			if(!ignoreFirstLine)
			{
				report.incrCounter(CSVReblockMR.NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim, -1).length);
				num++;
			}
			else
				realFirstLine=true;
		}
		else
		{
			if(realFirstLine)
			{
				report.incrCounter(CSVReblockMR.NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim, -1).length);
				realFirstLine=false;
			}
			num++;
		}
	}
	
	@Override
	public void configure(JobConf job)
	{	
		byte thisIndex;
		try {
			//it doesn't make sense to have repeated file names in the input, since this is for reblock
			thisIndex=MRJobConfiguration.getInputMatrixIndexesInMapper(job).get(0);
			outKey.set(thisIndex);
			FileSystem fs=FileSystem.get(job);
			Path thisPath=new Path(job.get("map.input.file")).makeQualified(fs);
			filename=thisPath.toString();
			String[] strs=job.getStrings(CSVReblockMR.SMALLEST_FILE_NAME_PER_INPUT);
			Path headerPath=new Path(strs[thisIndex]).makeQualified(fs);
			if(headerPath.toString().equals(filename))
				headerFile=true;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
		try {
			CSVReblockInstruction[] reblockInstructions = MRJobConfiguration.getCSVReblockInstructions(job);
			for(CSVReblockInstruction ins: reblockInstructions)
			{
				if(ins.input==thisIndex)
				{
					delim=Pattern.quote(ins.delim); 
					ignoreFirstLine=ins.hasHeader;
					break;
				}
			}
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void close() throws IOException
	{
		outCache.collect(outKey, new OffsetCount(filename, fileOffset, num));
	}
}
