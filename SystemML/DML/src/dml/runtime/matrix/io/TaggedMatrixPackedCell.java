package dml.runtime.matrix.io;

public class TaggedMatrixPackedCell extends TaggedMatrixValue{

	public TaggedMatrixPackedCell(MatrixPackedCell b, byte t)
	{
		super(b, t);
	}
	public TaggedMatrixPackedCell()
	{
		super();
        tag=-1;
     	base=new MatrixPackedCell();
	}
	public TaggedMatrixPackedCell(TaggedMatrixPackedCell that)
	{
		this.tag=that.tag;
		base=new MatrixPackedCell((MatrixPackedCell) that.base);
	}
}