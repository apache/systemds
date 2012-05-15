package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;


public class TextToBinaryCellConverter 
implements Converter<LongWritable, Text, MatrixIndexes, MatrixCell>
{
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixCell value=new MatrixCell();
	private Pair<MatrixIndexes, MatrixCell> pair=
		new Pair<MatrixIndexes, MatrixCell>(indexes, value);
	private boolean hasValue=false;

	@Override
	public void convert(LongWritable k1, Text v1) {
		String[] strs=v1.toString().split(" ");
		if(strs.length==1)
			strs=v1.toString().split(",");
		indexes.setIndexes(Long.parseLong(strs[0]), Long.parseLong(strs[1]));
		value.setValue(Double.parseDouble(strs[2]));
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, MatrixCell> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}
	
	public static void main(String[] args) throws Exception {
		TextToBinaryCellConverter conv=new TextToBinaryCellConverter();
		conv.convert(new LongWritable(1), new Text("1 2 10.0"));
		while(conv.hasNext())
		{
			Pair pair=conv.next();
			System.out.println(pair.getKey()+": "+pair.getValue());
		}
	}

	@Override
	public void setBlockSize(int rl, int cl) {
		
	}
}
