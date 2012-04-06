package dml.runtime.matrix;

import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class MatrixFormatMetaData extends MatrixDimensionsMetaData {

	protected InputInfo iinfo;
	protected OutputInfo oinfo;
	
	public MatrixFormatMetaData(MatrixCharacteristics mc, OutputInfo oinfo_, InputInfo iinfo_ ) {
		super(mc);
		oinfo = oinfo_;
		iinfo = iinfo_;
	}
	
	public InputInfo getInputInfo() {
		return iinfo;
	}
	
	public OutputInfo getOutputInfo() {
		return oinfo;
	}
}
