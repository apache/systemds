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
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

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
import org.apache.wink.json4j.JSONException;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;
import org.apache.sysml.runtime.matrix.mapred.CSVReblockMapper;
import org.apache.sysml.runtime.matrix.mapred.CSVReblockMapper.IndexedBlockRow;
import org.apache.sysml.runtime.matrix.mapred.MapperBase;

@SuppressWarnings("deprecation")
public class ApplyTfBBMapper extends MapperBase implements Mapper<LongWritable, Text, TaggedFirstSecondIndexes, CSVReblockMR.BlockRow>{
	
	boolean _partFileWithHeader = false;
	TfUtils tfmapper = null;
	Reporter _reporter = null;
	
	// variables relevant to CSV Reblock
	private IndexedBlockRow idxRow = null;
	private long rowOffset=0;
	private HashMap<Long, Long> offsetMap=new HashMap<Long, Long>();
	private boolean _first = true;
	private long num=0;
	
	@Override
	public void configure(JobConf job) {
		super.configure(job);
		try {
			_partFileWithHeader = TfUtils.isPartFileWithHeader(job);
			tfmapper = new TfUtils(job);
			tfmapper.loadTfMetadata(job, true);
			
			// Load relevant information for CSV Reblock
			ByteWritable key=new ByteWritable();
			OffsetCount value=new OffsetCount();
			Path p=new Path(job.get(CSVReblockMR.ROWID_FILE_NAME));
			
			FileSystem fs = FileSystem.get(job);
			Path thisPath=new Path(job.get("map.input.file")).makeQualified(fs);
			String thisfile=thisPath.toString();

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
			while (reader.next(key, value)) {
				// "key" needn't be checked since the offset file has information about a single CSV input (the raw data file)
				if(thisfile.equals(value.filename))
					offsetMap.put(value.fileOffset, value.count);
			}
			reader.close();

			idxRow = new CSVReblockMapper.IndexedBlockRow();
			int maxBclen=0;
		
			for(ArrayList<CSVReblockInstruction> insv: csv_reblock_instructions)
				for(CSVReblockInstruction in: insv)
				{	
					if(maxBclen<in.bclen)
						maxBclen=in.bclen;
				}
			
			//always dense since common csv usecase
			idxRow.getRow().data.reset(1, maxBclen, false);		

		} catch (IOException e) { throw new RuntimeException(e); }
 		 catch(JSONException e)  { throw new RuntimeException(e); }

	}
	
	@Override
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<TaggedFirstSecondIndexes,CSVReblockMR.BlockRow> out, Reporter reporter) throws IOException  {
		
		if(_first) {
			rowOffset=offsetMap.get(rawKey.get());
			_reporter = reporter;
			_first=false;
		}
		
		// output the header line
		if ( rawKey.get() == 0 && _partFileWithHeader ) 
		{
			tfmapper.processHeaderLine();
			if ( tfmapper.hasHeader() )
				return;
		}
		
		// parse the input line and apply transformation
		String[] words = tfmapper.getWords(rawValue);
		
		if(!tfmapper.omit(words))
		{
			words = tfmapper.apply(words);
			try {
				tfmapper.check(words);
				
				// Perform CSV Reblock
				CSVReblockInstruction ins = csv_reblock_instructions.get(0).get(0);
				idxRow = CSVReblockMapper.processRow(idxRow, words, rowOffset, num, ins.output, ins.brlen, ins.bclen, ins.fill, ins.fillValue, out);
			}
			catch(DMLRuntimeException e) {
				throw new RuntimeException(e.getMessage() + ":" + rawValue.toString());
			}
			num++;
		}
	}

	@Override
	public void close() throws IOException {
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
	}

}
