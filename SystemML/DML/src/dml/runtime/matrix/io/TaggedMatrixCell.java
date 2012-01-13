package dml.runtime.matrix.io;

public class TaggedMatrixCell extends TaggedMatrixValue{
	
	public TaggedMatrixCell(MatrixCell b, byte t) {
		super(b, t);
	}

	public TaggedMatrixCell()
	{    
		super();
        tag=-1;
     	base=new MatrixCell();
	}

	public TaggedMatrixCell(TaggedMatrixCell value) {
		tag=value.getTag();
		base=new MatrixCell(value.base);
	}
}
