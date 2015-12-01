/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.udf.lib;

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

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * 
 *
 */
@Deprecated
public class RemoveEmptyRows extends PackageFunction 
{	
	
	private static final long serialVersionUID = 1L;
	private static final String OUTPUT_FILE = "TMP";
	
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
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			FileInputFormat.addInputPath(job, path); 
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			
			//prepare output
			String fnameNew = createOutputFilePathAndName( OUTPUT_FILE );
			DataOutputStream ostream = MapReduceTool.getHDFSDataOutputStream( fnameNew, true );
		
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
