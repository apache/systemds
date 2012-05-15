package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;


public class BinaryCellToTextConverter 
implements Converter<MatrixIndexes, MatrixCell, NullWritable, Text>{

	private Text value=new Text();
	private Pair<NullWritable, Text> pair=new Pair<NullWritable, Text>(NullWritable.get(), value);
	private boolean hasValue=false;

	@Override
	public void convert(MatrixIndexes k1, MatrixCell v1) {
		double v=((MatrixCell)v1).getValue();
		value.set(k1.getRowIndex()+" "+k1.getColumnIndex()+" "+v);
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<NullWritable, Text> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	public static void main(String[] args) throws Exception {
		BinaryCellToTextConverter conv=new BinaryCellToTextConverter();
		conv.convert(new MatrixIndexes(1, 2), new MatrixCell(10));
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
