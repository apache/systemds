/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.mr.CombineUnaryInstruction;
import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

@SuppressWarnings("rawtypes")
public class ValueSortMapper<KIN extends WritableComparable, VIN extends Writable, KOUT extends WritableComparable, VOUT extends Writable> extends MapReduceBase 
      implements Mapper<KIN, VIN, KOUT, VOUT>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int brlen;
	private int bclen;
	private CombineUnaryInstruction combineInstruction=null;
	private Converter<KIN, VIN, KOUT, VOUT> inputConverter;
	private IntWritable one=new IntWritable(1);
	private DoubleWritable combinedKey=new DoubleWritable();
	
	@SuppressWarnings("unchecked")
	public void map(KIN key, VIN value, OutputCollector<KOUT, VOUT> out,
			Reporter reporter) throws IOException {
		inputConverter.convert(key, value);
		while(inputConverter.hasNext())
		{
			Pair pair=inputConverter.next();
			if(combineInstruction==null)
			{
				//System.out.println("output: "+pair.getKey()+": "+pair.getValue());
				out.collect((KOUT) pair.getKey(), (VOUT)pair.getValue());
			}else
			{
				processCombineUnaryInstruction(pair, out);
			}
		}
	} 
	
	@SuppressWarnings("unchecked")
	private void processCombineUnaryInstruction(Pair pair, OutputCollector<KOUT, VOUT> out) 
		throws IOException
	{
		combinedKey.set(((MatrixCell)pair.getValue()).getValue());
		out.collect((KOUT)combinedKey, (VOUT)one);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public void configure(JobConf job)
	{
		try 
		{
			brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
			bclen = MRJobConfiguration.getNumColumnsPerBlock(job, (byte) 0);
			String str=job.get(SortMR.COMBINE_INSTRUCTION, null);
			if(str!=null && !str.isEmpty() && !"null".equals(str))
					combineInstruction=(CombineUnaryInstruction) CombineUnaryInstruction.parseInstruction(str);
			inputConverter = MRJobConfiguration.getInputConverter(job, (byte) 0);
			inputConverter.setBlockSize(brlen, bclen);
		} 
		catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}
}
