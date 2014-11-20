/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.nimble.hadoop.HDFSFileManager;

import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.Matrix.ValueType;

/**
 * 
 *
 */
@Deprecated
public class RemoveEmptyRows extends PackageFunction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";
	
	private Matrix _ret; 
	
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		return _ret;
	}

	@Override
	public void execute() 
	{
		Matrix mat = (Matrix) this.getFunctionInput(0);
		String fnameOld = mat.getFilePath(); 
		
		HashMap<Long,Long> keyMap = new HashMap<Long,Long>(); //old,new rowID
			
		try
		{		
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			FileInputFormat.addInputPath(job, path); 
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			
			//prepare output
			String fnameNew = createOutputFilePathAndName( OUTPUT_FILE );
			DataOutputStream ostream = HDFSFileManager.getOutputStreamStatic( fnameNew, true);
		
			//read and write if necessary
			InputSplit[] splits = informat.getSplits(job, 1);
		
			LongWritable key = new LongWritable();
			Text value = new Text();
			long ID = 1;
			
			try
			{
				//for obj reuse and preventing repeated buffer re-allocations
				StringBuilder sb = new StringBuilder();
				
				for(InputSplit split: splits)
				{
					RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);				
					try
					{
						while( reader.next(key, value) )
						{
							String cellStr = value.toString().trim();							
							StringTokenizer st = new StringTokenizer(cellStr, " ");
							long row = Integer.parseInt( st.nextToken() );
							long col = Integer.parseInt( st.nextToken() );
							double lvalue = Double.parseDouble( st.nextToken() );
							
							if( !keyMap.containsKey( row ) )
								keyMap.put(row, ID++);
							long rowNew = keyMap.get( row );
							
							sb.append(rowNew);
							sb.append(' ');
							sb.append(col);
							sb.append(' ');
							sb.append(lvalue);
							sb.append('\n');
							
							ostream.writeBytes( sb.toString() );	
							sb.setLength(0);
						}
					}
					finally
					{
						if( reader != null )
							reader.close();
					}
				}
				
				_ret = new Matrix(fnameNew, keyMap.size(), mat.getNumCols(), ValueType.Double);
			}
			finally
			{
				if( ostream != null )
					ostream.close();	
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException( "Unable to execute external function.", ex );
		}
	}
}
