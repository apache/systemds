package com.ibm.bi.dml.runtime.matrix.io;

import java.util.StringTokenizer;

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
	private boolean toIgnore=false;

	@Override
	public void convert(LongWritable k1, Text v1) {
		
		String str=v1.toString();
		//added to support matrix market format
		if(str.startsWith("%%"))
		{
			toIgnore=true;
			hasValue=false;
			return;
		}
		else if(str.startsWith("%"))
		{
			hasValue=false;
			return;
		}else if(toIgnore)
		{
			toIgnore=false;
			hasValue=false;
			return;
		}
		
		String cellStr = str.toString().trim();							
		StringTokenizer st = new StringTokenizer(cellStr, " ");
		indexes.setIndexes( Long.parseLong(st.nextToken()), 
				            Long.parseLong(st.nextToken()) );
		value.setValue( Double.parseDouble(st.nextToken()) );
		hasValue = true;
		
		/* FIXME,
		String[] strs=str.split(" ");
		if(strs.length==1)
			strs=str.split(",");
		indexes.setIndexes(Long.parseLong(strs[0]), Long.parseLong(strs[1]));
		value.setValue(Double.parseDouble(strs[2]));
		hasValue=true;
		*/
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
