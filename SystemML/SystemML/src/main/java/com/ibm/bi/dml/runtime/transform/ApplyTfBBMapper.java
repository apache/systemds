/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;
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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.mr.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockMapper.IndexedBlockRow;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MapperBase;
import com.ibm.json.java.JSONObject;

@SuppressWarnings("deprecation")
public class ApplyTfBBMapper extends MapperBase implements Mapper<LongWritable, Text, TaggedFirstSecondIndexes, CSVReblockMR.BlockRow>{
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	ApplyTfHelper tfmapper = null;
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
			tfmapper = new ApplyTfHelper(job);
			JSONObject spec = tfmapper.parseSpec();
			tfmapper.setupTfAgents(spec);
			tfmapper.loadTfMetadata(spec);
			
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

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<TaggedFirstSecondIndexes,CSVReblockMR.BlockRow> out, Reporter reporter) throws IOException  {
		
		if(_first) {
			rowOffset=offsetMap.get(rawKey.get());
			_reporter = reporter;
			_first=false;
		}
		
		// output the header line
		if ( rawKey.get() == 0 && tfmapper._partFileWithHeader ) 
		{
			tfmapper.processHeaderLine(rawValue);
			if ( tfmapper._hasHeader )
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
		_reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_ROWS, tfmapper.getNumTransformedRows());
		_reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_COLS, tfmapper.getNumTransformedColumns());
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
	}

}
