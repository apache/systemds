/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.mr.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.BlockRow;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class CSVReblockMapper extends MapperBase implements Mapper<LongWritable, Text, TaggedFirstSecondIndexes, BlockRow>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long rowOffset=0;
	private boolean first=true;
	private long num=0;
	private HashMap<Long, Long> offsetMap=new HashMap<Long, Long>();
	private TaggedFirstSecondIndexes outIndexes=new TaggedFirstSecondIndexes();
	private BlockRow row;
	private String delim=" ";
	private boolean ignoreFirstLine=false;
	private boolean headerFile=false;
	private Pattern _compiledDelim = null;

	@Override
	public void map(LongWritable key, Text value,
			OutputCollector<TaggedFirstSecondIndexes, BlockRow> out, Reporter reporter)
			throws IOException 
	{
		if(first) {
			rowOffset=offsetMap.get(key.get());
			first=false;
		}
		
		if(key.get()==0 && headerFile && ignoreFirstLine)
			return;
		
		String[] cells = _compiledDelim.split( value.toString() );
		
		
		for(int i=0; i<representativeMatrixes.size(); i++)
			for(CSVReblockInstruction ins: csv_reblock_instructions.get(i))
			{
				int start=0;
				outIndexes.setTag(ins.output);
				long rowIndex=UtilFunctions.blockIndexCalculation(rowOffset+num+1, ins.brlen);
				row.indexInBlock=UtilFunctions.cellInBlockCalculation(rowOffset+num+1, ins.brlen);
				
				long col=0;
				for(; col<cells.length/ins.bclen; col++)
				{
					row.data.reset(1, ins.bclen);
					outIndexes.setIndexes(rowIndex, col+1);
					for(int k=0;k<ins.bclen; k++)
					{
						if(cells[k+start].isEmpty())
						{
							if(!ins.fill)
								throw new RuntimeException("Empty fields found in the input delimited file. Use \"fill\" option to read delimited files with empty fields.");
							row.data.appendValue(0, k, ins.fillValue);
						}
						else
							row.data.appendValue(0, k, Double.parseDouble(cells[k+start]));
					}
					out.collect(outIndexes, row);
					start+=ins.bclen;
				}
				outIndexes.setIndexes(rowIndex, col+1);
				int lastBclen=cells.length%ins.bclen;
				if(lastBclen!=0)
				{
					row.data.reset(1, lastBclen);
					for(int k=0;k<lastBclen; k++)
					{
						if(cells[k+start].isEmpty())
						{
							if(!ins.fill)
								throw new RuntimeException("Empty fields found in the input delimited file. Use \"fill\" option to read delimited files with empty fields.");
							row.data.appendValue(0, k, ins.fillValue);
						}
						else
							row.data.appendValue(0, k, Double.parseDouble(cells[k+start]));
					}
					out.collect(outIndexes, row);
				}
			}
		
		num++;
	}	
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		//get the number colums per block
		
		//load the offset mapping
		byte matrixIndex=representativeMatrixes.get(0);
		try {
			
			//Path[] paths=DistributedCache.getLocalCacheFiles(job);
			FileSystem fs = FileSystem.get(job);
			Path thisPath=new Path(job.get("map.input.file")).makeQualified(fs);
			String filename=thisPath.toString();
			Path headerPath=new Path(job.getStrings(CSVReblockMR.SMALLEST_FILE_NAME_PER_INPUT)[matrixIndex]).makeQualified(fs);
			if(headerPath.toString().equals(filename))
				headerFile=true;
			
			ByteWritable key=new ByteWritable();
			OffsetCount value=new OffsetCount();
			Path p=new Path(job.get(CSVReblockMR.ROWID_FILE_NAME));
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
			try {
				while (reader.next(key, value)) {
					if(key.get()==matrixIndex && filename.equals(value.filename))
						offsetMap.put(value.fileOffset, value.count);
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			} 
			
			reader.close();
			
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		CSVReblockInstruction ins=csv_reblock_instructions.get(0).get(0);
		delim = Pattern.quote(ins.delim);
		ignoreFirstLine=ins.hasHeader;
		row = new BlockRow();
		row.data = new MatrixBlock();
		int maxBclen=0;
	
		for(ArrayList<CSVReblockInstruction> insv: csv_reblock_instructions)
			for(CSVReblockInstruction in: insv)
			{	
				if(maxBclen<in.bclen)
					maxBclen=in.bclen;
			}
		
		
		//always dense since common csv usecase
		row.data.reset(1, maxBclen, false);		
	
		//precompile regex pattern for better efficiency
		_compiledDelim = Pattern.compile(delim);
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		//do nothing
	}	
}
