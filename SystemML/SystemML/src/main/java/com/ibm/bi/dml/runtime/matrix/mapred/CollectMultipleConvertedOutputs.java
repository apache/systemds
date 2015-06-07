/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


public class CollectMultipleConvertedOutputs 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	protected Converter[] outputConverters;
	protected MultipleOutputs multipleOutputs;
	protected MatrixCharacteristics[] matrixStats;
	
	public CollectMultipleConvertedOutputs(Converter[] convts, MatrixCharacteristics[] stats, 
			MultipleOutputs outputs)
	{
		outputConverters=convts;
		multipleOutputs=outputs;
		matrixStats=stats;
	}
	
	@SuppressWarnings("unchecked")
	public void collectOutput(Writable key, Writable value, int output, Reporter reporter) 
	throws IOException
	{
		Converter<Writable, Writable, Writable, Writable> conv=outputConverters[output];
		conv.setBlockSize(matrixStats[output].getRowsPerBlock(), matrixStats[output].getColsPerBlock());
		conv.convert(key, value);
	//	System.out.println("output before convert: "+key+" "+value);
		while(conv.hasNext())
		{
			Pair<Writable, Writable> outpair=conv.next();
			multipleOutputs.getCollector(Integer.toString(output), reporter).collect(outpair.getKey(), outpair.getValue());
		//	System.out.println("output in collectOutput "+outpair.getKey().toString()+":"+outpair.getValue());
		}
	}
	
	public void directOutput(Writable key, Writable value, int output, Reporter reporter) 
	throws IOException
	{
		//System.out.println("output before convert: "+key+" "+value +" --> output " + output);
		multipleOutputs.getCollector(Integer.toString(output), reporter).collect(key, value);
		//LOG.info("** output in collectOutput "+key+":"+value);
	}

	public void close() throws IOException {
		
		multipleOutputs.close();
	}

}
