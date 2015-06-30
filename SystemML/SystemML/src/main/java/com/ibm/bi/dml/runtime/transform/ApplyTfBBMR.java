package com.ibm.bi.dml.runtime.transform;
import java.util.HashSet;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.instructions.InstructionParser;
import com.ibm.bi.dml.runtime.instructions.mr.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.BlockRow;
import com.ibm.bi.dml.runtime.matrix.WriteCSVMR;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * MapReduce job that performs the actual data transformations, such as recoding
 * and binning. In contrast to ApplyTxCSVMR, this job generates the output in
 * BinaryBlock format. This job takes a data set as well as the transformation
 * metadata (which, for example, computed from GenTxMtdMR) as inputs.
 * 
 */

@SuppressWarnings("deprecation")
public class ApplyTfBBMR {
	
	public static JobReturn runJob(String inputPath, String rblkInst, String otherInst, String specPath, String mapsPath, String tmpPath, String outputPath, String partOffsetsFile, CSVFileFormatProperties inputDataProperties, long numRows, long numColsBefore, long numColsAfter, int replication, String headerLine) throws Exception {
		
		CSVReblockInstruction rblk = (CSVReblockInstruction) InstructionParser.parseSingleInstruction(rblkInst);
		
		long[] rlens = new long[]{numRows};
		long[] clens = new long[]{numColsAfter};
		int[] brlens = new int[]{rblk.brlen};
		int[] bclens = new int[]{rblk.bclen};
		byte[] realIndexes = new byte[]{rblk.input};
		byte[] resultIndexes = new byte[]{rblk.output};

		JobConf job = new JobConf(ApplyTfBBMR.class);
		job.setJobName("ApplyTfBB");

		/* Setup MapReduce Job */
		job.setJarByClass(ApplyTfBBMR.class);
		
		// set relevant classes
		job.setMapperClass(ApplyTxBBMapperOLD.class);
	
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, new String[]{inputPath}, new InputInfo[]{InputInfo.CSVInputInfo}, brlens, bclens, false, ConvertTarget.CELL);

		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);

		MRJobConfiguration.setCSVReblockInstructions(job, rblkInst);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInst);

		job.setInt("dfs.replication", replication);
		
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );

		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  null, 
				rblkInst, null, otherInst, resultIndexes);

		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				null, rblkInst, null, null, null, resultIndexes, mapoutputIndexes, false);

		//set up the number of reducers
		int numRed = WriteCSVMR.determineNumReducers(rlens, clens, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS), ret.numReducerGroups);
		job.setNumReduceTasks( numRed );

		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, new byte[]{rblk.output}, new byte[]{0}, new String[]{outputPath}, new OutputInfo[]{OutputInfo.BinaryBlockOutputInfo}, true, false);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(ApplyTfBBMapper.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(BlockRow.class);
		
		//configure reducer
		job.setReducerClass(CSVReblockReducer.class);
	
		//turn off adaptivemr
		job.setBoolean("adaptivemr.map.enable", false);

		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		
		// Add transformation metadata file as well as partOffsetsFile to Distributed cache
		DistributedCache.addCacheFile((new Path(mapsPath)).toUri(), job);
		DistributedCache.createSymlink(job);
		
		Path cachefile=new Path(new Path(partOffsetsFile), "part-00000");
		DistributedCache.addCacheFile(cachefile.toUri(), job);
		DistributedCache.createSymlink(job);
		
		job.set(MRJobConfiguration.TF_HAS_HEADER, 	Boolean.toString(inputDataProperties.hasHeader()));
		job.set(MRJobConfiguration.TF_DELIM, 		inputDataProperties.getDelim());
		if ( inputDataProperties.getNAStrings() != null)
			job.set(MRJobConfiguration.TF_NA_STRINGS, 	inputDataProperties.getNAStrings());
		job.set(MRJobConfiguration.TF_SPEC_FILE, 	specPath);
		job.set(MRJobConfiguration.TF_SMALLEST_FILE, CSVReblockMR.findSmallestFile(job, inputPath));
		job.set(MRJobConfiguration.OUTPUT_MATRICES_DIRS_CONFIG, outputPath);
		job.setLong(MRJobConfiguration.TF_NUM_COLS, numColsBefore);
		job.set(MRJobConfiguration.TF_TXMTD_PATH, mapsPath);
		job.set(MRJobConfiguration.TF_HEADER, headerLine);
		job.set(CSVReblockMR.ROWID_FILE_NAME, cachefile.toString());
		job.set(MRJobConfiguration.TF_TMP_LOC, tmpPath);

		RunningJob runjob=JobClient.runJob(job);
		
		MapReduceTool.deleteFileIfExistOnHDFS(cachefile, job);
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		for(int i=0; i<resultIndexes.length; i++) {
			ret.stats[i].setDimension(numRows, numColsAfter);
			ret.stats[i].setNonZeros(group.getCounter(Integer.toString(i)));
		}
		return new JobReturn(ret.stats, runjob.isSuccessful());
	}
	
}
