/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public class AddDummyWeightConverter implements Converter<Writable, Writable, MatrixIndexes, WeightedPair>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Converter toCellConverter=null;
	private WeightedPair outValue=new WeightedPair();
	private Pair<MatrixIndexes, WeightedPair> pair=new Pair<MatrixIndexes, WeightedPair>();
	private int rlen;
	private int clen;
	public AddDummyWeightConverter()
	{
		outValue.setWeight(1.0);
		outValue.setOtherValue(0);
		pair.setValue(outValue);
	}
	
	@Override
	public void convert(Writable k1, Writable v1) {
		if(toCellConverter==null)
		{
			if(v1 instanceof Text)
				toCellConverter=new TextToBinaryCellConverter();
			else if(v1 instanceof MatrixBlock)
				toCellConverter=new BinaryBlockToBinaryCellConverter();
			else
				toCellConverter=new IdenticalConverter();
			toCellConverter.setBlockSize(rlen, clen);
		}
		toCellConverter.convert(k1, v1);
	}

	@Override
	public boolean hasNext() {
		return toCellConverter.hasNext();
	}

	@Override
	public Pair<MatrixIndexes, WeightedPair> next() {
		Pair<MatrixIndexes, MatrixCell> temp=toCellConverter.next();
		pair.setKey(temp.getKey());
		outValue.setValue(temp.getValue().getValue());
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
		
		if(toCellConverter==null)
		{
			rlen=rl;
			clen=cl;
		}else
			toCellConverter.setBlockSize(rl, cl);
	}

}
