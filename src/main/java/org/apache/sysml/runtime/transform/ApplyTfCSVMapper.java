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

package org.apache.sysml.runtime.transform;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.wink.json4j.JSONException;

import org.apache.sysml.runtime.DMLRuntimeException;

public class ApplyTfCSVMapper implements Mapper<LongWritable, Text, NullWritable, Text> {
	
	boolean _firstRecordInSplit = true;
	boolean _partFileWithHeader = false;
	
	TfUtils tfmapper = null;
	Reporter _reporter = null;
	BufferedWriter br = null;
	JobConf _rJob = null;
	
	@Override
	public void configure(JobConf job) {
		try {
			_rJob = job;
			_partFileWithHeader = TfUtils.isPartFileWithHeader(job);
			tfmapper = new TfUtils(job);
			
			tfmapper.loadTfMetadata(job, true);
			
		} catch (IOException e) { throw new RuntimeException(e); }
		catch(JSONException e)  { throw new RuntimeException(e); }

	}
	
	@Override
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<NullWritable, Text> out, Reporter reporter) throws IOException  {
		
		if(_firstRecordInSplit)
		{
			_firstRecordInSplit = false;
			_reporter = reporter;
			
			// generate custom output paths so that order of rows in the 
			// output (across part files) matches w/ that from input data set
			String partFileSuffix = tfmapper.getPartFileID(_rJob, rawKey.get());
			Path mapOutputPath = new Path(tfmapper.getOutputPath() + "/transform-part-" + partFileSuffix);
			
			// setup the writer for mapper's output
			// the default part-..... files will be deleted later once the job finishes 
			br = new BufferedWriter(new OutputStreamWriter(FileSystem.get(_rJob).create( mapOutputPath, true)));
		}
		
		// output the header line
		if ( rawKey.get() == 0 && _partFileWithHeader ) 
		{
			_reporter = reporter;
			tfmapper.processHeaderLine();
			if ( tfmapper.hasHeader() )
				return;
		}
		
		// parse the input line and apply transformation
		String[] words = tfmapper.getWords(rawValue);
		
		if(!tfmapper.omit(words))
		{
			try {
				words = tfmapper.apply(words);
				String outStr = tfmapper.checkAndPrepOutputString(words);
				//out.collect(NullWritable.get(), new Text(outStr));
				br.write(outStr + "\n");
			} 
			catch(DMLRuntimeException e) {
				throw new RuntimeException(e.getMessage() + ": " + rawValue.toString());
			}
		}
	}

	@Override
	public void close() throws IOException {
		if ( br != null ) 
			br.close();
	}

}
