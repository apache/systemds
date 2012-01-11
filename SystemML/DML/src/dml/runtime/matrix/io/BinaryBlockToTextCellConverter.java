package dml.runtime.matrix.io;

import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.runtime.util.UtilFunctions;


public class BinaryBlockToTextCellConverter implements 
Converter<MatrixIndexes, MatrixBlock, NullWritable, Text>{

	private Iterator sparseInterator=null;
	private double[] denseArray=null;
	private int denseArraySize=0;
	private int nextInDenseArray=-1;
	private boolean sparse=true;
	private int thisBlockWidth=0;
	private MatrixIndexes startIndexes=new MatrixIndexes();
	private boolean hasValue=false;
	private int brow;
	private int bcolumn;
	
	private Text value=new Text();
	private Pair<NullWritable, Text> pair=new Pair<NullWritable, Text>(NullWritable.get(), value);
	
	private void reset()
	{
		sparseInterator=null;
		denseArray=null;
		denseArraySize=0;
		nextInDenseArray=-1;
		sparse=true;
		thisBlockWidth=0;
	}
	
	@Override
	public void convert(MatrixIndexes k1, MatrixBlock v1) {
		reset();
		startIndexes.setIndexes(UtilFunctions.cellIndexCalculation(k1.getRowIndex(), brow,0), 
				UtilFunctions.cellIndexCalculation(k1.getColumnIndex(),bcolumn,0));
		sparse=v1.isInSparseFormat();
		thisBlockWidth=v1.getNumColumns();
		if(sparse)
		{
			if(v1.getSparseMap()==null)
				return;
			sparseInterator=v1.getSparseMap().entrySet().iterator();
		}
		else
		{
			if(v1.getDenseArray()==null)
				return;
			denseArray=v1.getDenseArray();
			nextInDenseArray=0;
			denseArraySize=v1.getNumRows()*v1.getNumColumns();
		}
		hasValue=(v1.getNonZeros()>0);
	}

	@Override
	public boolean hasNext() {
		if(sparse)
		{
			if(sparseInterator==null)
				hasValue=false;
			else
				hasValue=sparseInterator.hasNext();
		}else
		{
			if(denseArray==null)
				hasValue=false;
			else
			{
				while(nextInDenseArray<denseArraySize && denseArray[nextInDenseArray]==0)
					nextInDenseArray++;
				hasValue=(nextInDenseArray<denseArraySize);
			}
		}
		return hasValue;
	}

	@Override
	public Pair<NullWritable, Text> next() {
		if(!hasValue)
			return null;
		long i, j;
		double v;
		if(sparse)
		{
			if(sparseInterator==null)
				return null;
			else
			{
				Entry<CellIndex, Double> e=(Entry<CellIndex, Double>) sparseInterator.next();
				i=e.getKey().row + startIndexes.getRowIndex();
				j=e.getKey().column + startIndexes.getColumnIndex();
				v=e.getValue();
			}
				
		}else
		{
			if(denseArray==null)
				return null;
			else
			{
				i=startIndexes.getRowIndex() + nextInDenseArray/thisBlockWidth;
				j=startIndexes.getColumnIndex() + nextInDenseArray%thisBlockWidth;
				v=denseArray[nextInDenseArray];
				nextInDenseArray++;
			}
		}
		value.set(i+" "+j+" "+v);
		return pair;
	}

	public void setBlockSize(int nr, int nc) {
		brow=nr;
		bcolumn=nc;
	}
	
	public static void main(String[] args) throws Exception {
		
		MatrixBlock m1=new MatrixBlock(3, 2, false);
		m1.setValue(0, 0, 1);
		//m1.setValue(0, 1, 2);
		m1.setValue(1, 0, 3);
		//m1.setValue(1, 1, 4);
		m1.setValue(2, 0, 5);
		//m1.setValue(2, 1, 6);
		System.out.println("matrix m1: ");
		m1.print();
		
		MatrixIndexes ind=new MatrixIndexes(10, 10);
		
		BinaryBlockToTextCellConverter conv=new BinaryBlockToTextCellConverter();
		conv.setBlockSize(3, 2);
		conv.convert(ind, m1);
		while(conv.hasNext())
		{
			Pair pair=conv.next();
			System.out.println(pair.getKey()+": "+pair.getValue());
		}
	}
	
}
