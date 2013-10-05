/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred.tentative;

import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedTripleIndexes;


//assume only two inputs to this job
public class ABMRMapper extends MapReduceBase 
implements Mapper<MatrixIndexes, MatrixBlock, TaggedTripleIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	protected Vector<Byte> representativeMatrixes=new Vector<Byte>();
	public static final String MATRIX_NUM_ROW_PREFIX_CONFIG="matrix.num.row.";
	public static final String MATRIX_NUM_COLUMN_PREFIX_CONFIG="matrix.num.column.";
	
	public static final String BLOCK_NUM_ROW_PREFIX_CONFIG="block.num.row.";
	public static final String BLOCK_NUM_COLUMN_PREFIX_CONFIG="block.num.column.";

	private TaggedTripleIndexes triplebuffer=new TaggedTripleIndexes();
	//private static final Log LOG = LogFactory.getLog(ABMRMapper.class);
	
	private byte representative;
	private long numRepeats=0;
	private int stepWidth=0;
	public static final String MATRIX_FILE_NAMES_CONFIG="input.matrices.dirs";
	static enum Counters {MAP_TIME };
	@Override
	public void map(MatrixIndexes indexes, MatrixBlock block,
			OutputCollector<TaggedTripleIndexes, MatrixBlock> out,
			Reporter report) throws IOException {
		long start=System.currentTimeMillis();
		triplebuffer.setTag(representative);
	//	System.out.println("inputput: "+indexes);
	/*	if(indexes.getRowIndex()>=100 || indexes.getColumnIndex()>=100)
			throw new IOException("indexes are wrong: "+indexes);*/
		if(representative==0)
		{
			for(long j=0; j<numRepeats; j++)
			{
				triplebuffer.setIndexes(indexes.getRowIndex(), j+1, indexes.getColumnIndex());
				out.collect(triplebuffer, block);
			//	System.out.println("######### output");
			//	System.out.println(triplebuffer.toString());
				/*if(triplebuffer.getFirstIndex()>=100 || triplebuffer.getSecondIndex()>=100 || triplebuffer.getThirdIndex()>=100)
					throw new IOException("indexes are wrong: "+triplebuffer);*/
			//	LOG.info(block.toString());*/
			}
		}else
		{
			for(long i=0; i<numRepeats; i++)
			{
				triplebuffer.setIndexes(i+1, indexes.getColumnIndex(), indexes.getRowIndex());
				out.collect(triplebuffer, block);
			//	System.out.println("######### output");
			//	System.out.println(triplebuffer.toString());
	/*			if(triplebuffer.getFirstIndex()>=100 || triplebuffer.getSecondIndex()>=100 || triplebuffer.getThirdIndex()>=100)
					throw new IOException("indexes are wrong: "+triplebuffer);*/
	//			LOG.info(block.toString());*/
			}
		}
		report.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
		
	}

	public void configure(JobConf job)
	{
		//get the indexes that this matrix file represents, 
		//since one matrix file can occur multiple times in a statement
		String[] matrices=job.getStrings(MATRIX_FILE_NAMES_CONFIG);
		
		String thisMatrixName=job.get("map.input.file");
	//	System.out.println("this matrix name is: "+thisMatrixName);
		
		for(int i=0; i<matrices.length; i++)
		{
			if(thisMatrixName.contains(matrices[i]))
			{
				representativeMatrixes.add((byte)i);
				//LOG.info("add to representative: "+i);
			}
		}
		
		representative=representativeMatrixes.get(0);
		if(representative==0)
		{
			byte other=1;
			long matrixNumColumn=job.getLong(MATRIX_NUM_COLUMN_PREFIX_CONFIG+other, -1);
			int blockNumColumn=job.getInt(BLOCK_NUM_COLUMN_PREFIX_CONFIG+other, -1);
			numRepeats=(long)Math.ceil((double)matrixNumColumn/(double)blockNumColumn);
			stepWidth=blockNumColumn;
		}else
		{
			byte other=0;
			long matrixNumRow=job.getLong(MATRIX_NUM_ROW_PREFIX_CONFIG+other, -1);
			int blockNumRow=job.getInt(BLOCK_NUM_ROW_PREFIX_CONFIG+other, -1);
			numRepeats=(long)Math.ceil((double)matrixNumRow/(double)blockNumRow);
			stepWidth=blockNumRow;
		}
	}
}
