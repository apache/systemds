/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import com.ibm.bi.dml.runtime.matrix.WriteCSVMR.RowBlock;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

public class TextCellToRowBlockConverter implements Converter<LongWritable, Text, MatrixIndexes, RowBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes indexes=new MatrixIndexes();
	private RowBlock rowBlock=new RowBlock();
	private Pair<MatrixIndexes, RowBlock> pair=new Pair<MatrixIndexes, RowBlock>(indexes, rowBlock);
	private FastStringTokenizer st = new FastStringTokenizer(' '); 
	private boolean hasValue=false;
	private boolean toIgnore=false;

	@Override
	public void convert(LongWritable k1, Text v1) {
		
		String str=v1.toString();
		
		//handle support for matrix market format
		if(str.startsWith("%")) {
			if(str.startsWith("%%"))
				toIgnore=true;
			hasValue=false;
			return;
		}
		else if(toIgnore) {
			toIgnore=false;
			hasValue=false;
			return;
		}
		
		//reset the tokenizer
		st.reset( str );
				
		//convert text to row block
		indexes.setIndexes( st.nextLong(), st.nextLong() );
		if(rowBlock.container==null || rowBlock.container.length<1)
			rowBlock.container=new double[1];
		rowBlock.numCols = 1;
		rowBlock.container[0] = st.nextDouble();
		hasValue = true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, RowBlock> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}
	
	public static void main(String[] args) throws Exception {
		TextToBinaryCellConverter conv=new TextToBinaryCellConverter();
		LongWritable lw=new LongWritable(1);
		Text[] texts=new Text[]{new Text("%%ksdjl"), new Text("%ksalfk"), new Text("1 1 1"), new Text("1 2 10.0"), new Text("%ksalfk")};
		
		for(Text text: texts)
		{
			conv.convert(lw, text);
			while(conv.hasNext())
			{
				Pair pair=conv.next();
				System.out.println(pair.getKey()+": "+pair.getValue());
			}
		}
	}

	@Override
	public void setBlockSize(int rl, int cl) {
		
	}
}

