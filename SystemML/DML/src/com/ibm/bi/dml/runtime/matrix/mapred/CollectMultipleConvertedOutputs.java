package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Vector;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.Pair;


public class CollectMultipleConvertedOutputs {
	
	/*static class BlockSize
	{
		public int brlen=1;
		public int bclen=1;
		public BlockSize(int br, int bc)
		{
			brlen=br;
			bclen=bc;
		}
	}*/
	
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
		conv.setBlockSize(matrixStats[output].numRowsPerBlock, matrixStats[output].numColumnsPerBlock);
		conv.convert(key, value);
	//	System.out.println("output before convert: "+key+" "+value);
		while(conv.hasNext())
		{
			Pair<Writable, Writable> outpair=conv.next();
			multipleOutputs.getCollector(Integer.toString(output), reporter).collect(outpair.getKey(), outpair.getValue());
		//	System.out.println("output in collectOutput "+outpair.getKey().toString()+":"+outpair.getValue());
		}
	}

	public void close() throws IOException {
		
		multipleOutputs.close();
	}

}
