package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * Represents a matrix in controlprogram. This class contains method to read
 * matrices from HDFS and convert them to a specific format/representation. It
 * is also able to write several formats/representation of matrices to HDFS.
 * 
 * @author Felix Hamborg
 * 
 */
public class MatrixObject extends Data {
	/*
	 * Container object that holds the actual data.
	 */
	protected MatrixBlock _data;

	// The name of HDFS file by which the data is backed up.
	private String _hdfsFileName; // file name

	/*
	 * Object that holds the metadata associated with the matrix, which
	 * includes: 1) Matrix dimensions, if available 2) Number of non-zeros, if
	 * available 3) Block dimensions, if applicable 4) InputInfo -- subsequent
	 * operations that use this Matrix expect it to be in this format.
	 * 
	 * When the matrix is written to HDFS (local file system, as well?), one
	 * must get the OutputInfo that matches with InputInfo stored inside _mtd.
	 */
	protected MetaData _metaData;

	/**
	 * Default constructor
	 */
	public MatrixObject() {
		super(DataType.MATRIX, ValueType.DOUBLE); // DOUBLE is the default value type
		_data = null;
		_metaData = null;
		_hdfsFileName = null;
	}

	/**
	 * Constructor that takes both HDFS filename and associated metadata.
	 */
	public MatrixObject(ValueType vt, String file, MetaData mtd) {
		super(DataType.MATRIX, vt);
		_hdfsFileName = file; // HDFS file path
		_metaData = mtd; // Metadata
		_data = null;
	}

	/**
	 * Constructor that takes only the HDFS filename.
	 */
	public MatrixObject(ValueType vt, String file) {
		super(DataType.MATRIX, vt);
		_hdfsFileName = file; // HDFS file path
		_metaData = null;
		_data = null;
	}

	public MetaData getMetaData() {
		return _metaData;
	}

	@Override
	public void setMetaData(MetaData mtd) {
		_metaData = mtd;
	}
	
	@Override
	public void updateMatrixCharacteristics(MatrixCharacteristics mc) {
		((MatrixDimensionsMetaData)_metaData).setMatrixCharacteristics(mc);
	}

	public void removeMetaData() {
		_metaData = null;
	}

	public MatrixBlock getData() {
		return _data;
	}

	public void setData(MatrixBlock data) {
		_data = data;
	}

	public double getValue(int row, int col) {
		return _data.getValue(row, col);
	}

	public String getFileName() {
		return _hdfsFileName;
	}

	public void setFileName(String file) {
		this._hdfsFileName = file;
	}

	public int getNumRows() {
		return _data.getNumRows();
	}

	public int getNumColumns() {
		return _data.getNumColumns();
	}

	/**
	 * All operations on MatrixObjects are responsible for updating the result
	 * MatrixObject with the following information: 1) resulting data; 2)
	 * dimensions, nonzeros; and 3) InputInfo -- the format in which subsequent
	 * operations (from MR side) should read this matrix.
	 */

	public MatrixObject binaryOperations(BinaryOperator bin_op, MatrixObject that, MatrixObject result) throws DMLRuntimeException,
			DMLUnsupportedOperationException {
		
		MatrixBlock result_data = (MatrixBlock) (_data.binaryOperations(bin_op, that._data, new MatrixBlock() ));
		
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		
		return result;
	}

	private void updateMatrixMetaData() throws DMLRuntimeException {
		//MatrixCharacteristics mc = new MatrixCharacteristics(_data.getNumRows(), _data.getNumColumns(), -1, -1, _data.getNonZeros());
		if ( _metaData == null ) {
			throw new DMLRuntimeException("Metadata can not be null at this point!");
		}
		MatrixCharacteristics mc = ((MatrixDimensionsMetaData)_metaData).getMatrixCharacteristics();
		mc.setDimension(_data.getNumRows(), _data.getNumColumns());
		mc.setNonZeros(_data.getNonZeros());
	}
	
	public MatrixObject aggregateBinaryOperations(
			AggregateBinaryOperator ab_op, MatrixObject that,
			MatrixObject result) throws DMLRuntimeException,
			DMLUnsupportedOperationException {

		// perform computation and update result data
		MatrixBlock result_data = (MatrixBlock) (_data
				.aggregateBinaryOperations(_data, that._data, new MatrixBlock(), ab_op));
		//int x = result_data.getNonZeros();
		//result_data.examSparsity();
		result.setData(result_data);
		
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
			
		return result;
	}

	public MatrixObject aggregateUnaryOperations(AggregateUnaryOperator au_op, 
			MatrixObject result) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		MatrixBlock result_data = (MatrixBlock) (_data.aggregateUnaryOperations(
				au_op, new MatrixBlock(), _data.getNumRows(), _data.getNumColumns(),
				new MatrixIndexes(1, 1), true));
		result.setData(result_data);
		result.updateMatrixMetaData();

		return result;
	}
	
	public CM_COV_Object cmOperations(CMOperator cm_op) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return this._data.cmOperations(cm_op);
	}

	public CM_COV_Object cmOperations(CMOperator cm_op, MatrixObject weights) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return this._data.cmOperations(cm_op, weights._data);
	}

	public CM_COV_Object covOperations(COVOperator cov_op, MatrixObject that) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return this._data.covOperations(cov_op, that._data);
	}

	public CM_COV_Object covOperations(COVOperator cov_op, MatrixObject that, MatrixObject weights) throws DMLRuntimeException, DMLUnsupportedOperationException {
		return this._data.covOperations(cov_op, that._data, weights._data);
	}

	public MatrixObject scalarOperations(ScalarOperator sc_op, MatrixObject result) throws DMLUnsupportedOperationException,
			DMLRuntimeException {
		
		MatrixBlock result_data = (MatrixBlock) (_data.scalarOperations(sc_op, new MatrixBlock()));
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();

		return result;
	}

	public MatrixObject reorgOperations(ReorgOperator r_op, MatrixObject result)
			throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		MatrixBlock result_data = (MatrixBlock) (_data.reorgOperations(r_op, new MatrixBlock(),
				0, 0, 0));
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		
		return result;
	}

	public MatrixObject appendOperations(ReorgOperator r_op, MatrixObject result)
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		MatrixBlock result_data = result.getData();
		if(result_data == null)
			result_data = (MatrixBlock) (_data.appendOperations(r_op, new MatrixBlock(), 0, 0, 0));
		else 
			result_data = (MatrixBlock) (_data.appendOperations(r_op, result_data, 0, 0, 0));
		
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		
		return result;
	}
	
	public MatrixObject unaryOperations(UnaryOperator u_op, MatrixObject result)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixBlock result_data = (MatrixBlock) (_data.unaryOperations(u_op, new MatrixBlock()));

		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		
		return result;
	}
	
	public MatrixObject indexOperations(long rl, long ru, long cl, long cu, MatrixObject result) throws DMLRuntimeException {
		MatrixBlock result_data = (MatrixBlock) _data.slideOperations(rl, ru, cl, cu, new MatrixBlock());
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		return result;
	}

	public MatrixObject sortOperations(MatrixObject weights, MatrixObject result) throws DMLRuntimeException, DMLUnsupportedOperationException {
		MatrixBlock result_data = null;
		if ( weights != null )
			result_data = (MatrixBlock) (_data.sortOperations(weights._data, new MatrixBlock()));
		else
			result_data = (MatrixBlock) (_data.sortOperations(null, new MatrixBlock()));
		result.setData(result_data);
		// update the metadata for resulting matrix
		result.updateMatrixMetaData();
		return result;
	}
	
	public MatrixObject valuePick(MatrixObject quantiles, MatrixObject result) throws DMLRuntimeException, DMLUnsupportedOperationException {
		MatrixBlock result_data = null;
		result_data = (MatrixBlock) (_data.pickValues(quantiles._data, new MatrixBlock()));
		result.setData(result_data);
		result.updateMatrixMetaData();
		return result;
	}
	
	public double valuePick(double quantile) throws DMLRuntimeException {
		if (quantile < 0 || quantile > 1) {
			throw new DMLRuntimeException("Requested quantile (" + quantile + ") is invalid -- it must be between 0 and 1.");
		}
		return _data.pickValue(quantile);
	}
	
	public double interQuartileMean() throws DMLRuntimeException {
		return _data.interQuartileMean();
	}

	private MatrixObject finalizeCtableOperations(HashMap<CellIndex,Double> ctable, MatrixObject result) throws DMLRuntimeException {
		MatrixBlock result_data = new MatrixBlock(ctable);
		result.setData(result_data);
		result.updateMatrixMetaData();
		return result;
	}
	
	// F=ctable(A,B,W)
	public MatrixObject tertiaryOperations(SimpleOperator op, MatrixObject input2, MatrixObject input3, MatrixObject result)
	throws DMLUnsupportedOperationException, DMLRuntimeException {
		HashMap<CellIndex,Double> ctable = new HashMap<CellIndex,Double>();
		_data.tertiaryOperations(op, input2.getData(), input3.getData(), ctable);
		return finalizeCtableOperations(ctable, result);
	}

	// F=ctable(A,B) or F=ctable(A,B,1);
	public MatrixObject tertiaryOperations(SimpleOperator op, MatrixObject input2, ScalarObject input3, MatrixObject result)
	throws DMLUnsupportedOperationException, DMLRuntimeException {
		HashMap<CellIndex,Double> ctable = new HashMap<CellIndex,Double>();
		_data.tertiaryOperations(op, input2.getData(), input3.getDoubleValue(), ctable);
		return finalizeCtableOperations(ctable, result);
	}
	
	// F=ctable(A,1) or F = ctable(A,1,1)
	public MatrixObject tertiaryOperations(SimpleOperator op, ScalarObject input2, ScalarObject input3, MatrixObject result)
	throws DMLUnsupportedOperationException, DMLRuntimeException {
		HashMap<CellIndex,Double> ctable = new HashMap<CellIndex,Double>();
		_data.tertiaryOperations(op, input2.getDoubleValue(), input3.getDoubleValue(), ctable);
		return finalizeCtableOperations(ctable, result);
	}

	// F=ctable(A,1,W)
	public MatrixObject tertiaryOperations(SimpleOperator op, ScalarObject input2, MatrixObject input3, MatrixObject result)
	throws DMLUnsupportedOperationException, DMLRuntimeException {
		HashMap<CellIndex,Double> ctable = new HashMap<CellIndex,Double>();
		_data.tertiaryOperations(op, input2.getDoubleValue(), input3.getData(), ctable);
		return finalizeCtableOperations(ctable, result);
	}

	public MatrixObject randOperations(long rows, long cols, double minValue, double maxValue, long seed, double sparsity) throws DMLRuntimeException{
		Random random=new Random();
		_data = new MatrixBlock();
		
		if(sparsity > MatrixBlock.SPARCITY_TURN_POINT)
			_data.reset((int)rows, (int)cols, false);
		else
			_data.reset((int)rows, (int)cols, true);
		
		double currentValue;
		random.setSeed(seed);
		for(int r = 0; r < _data.getNumRows(); r++)
		{
			for(int c = 0; c < _data.getNumColumns(); c++)
			{
				if(random.nextDouble() > sparsity)
					continue;
				currentValue = random.nextDouble();//((double) random.nextInt(0, maxRandom) / (double) maxRandom);
				currentValue = (currentValue * (maxValue - minValue) + minValue);
				_data.setValue(r, c, currentValue);
			}
		}
		updateMatrixMetaData();
		
		return this;
	}
	
	public void readMatrix() throws DMLRuntimeException {
		//DataConverter.readDouble1DArrayMatrixFromHDFSText(_hdfsFileName, _data);
		MatrixFormatMetaData iimd = (MatrixFormatMetaData)_metaData;
		MatrixCharacteristics mc = iimd.getMatrixCharacteristics();
		try {
			_data = DataConverter.readMatrixFromHDFS(_hdfsFileName, iimd.getInputInfo(), mc.get_rows(), mc.get_cols(), mc.numRowsPerBlock, mc.get_cols_per_block());
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		} 
	}

	public boolean isBuffered() {
		if (_data != null)
			return true; // _data.isBuffered();
		return false;
	}

	/**
	 * Deletes internal data object.
	 */
	public void cleanData() {
		_data.reset();
	}

	public void writeInMemoryMatrixToHDFS(String path, OutputInfo oinfo) throws DMLRuntimeException {
		// Get the dimension information from the metadata stored within MatrixObject
		MatrixCharacteristics mc = ((MatrixFormatMetaData)_metaData).getMatrixCharacteristics();
		
		// Write the matrix to HDFS in requested format
		try {
			DataConverter.writeMatrixToHDFS(_data, path, oinfo, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block());
			MapReduceTool.writeMetaDataFile(path+".mtd", mc, oinfo);
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	/**
	 * Writes in-memory matrix to disk.
	 * 
	 * @throws DMLRuntimeException
	 */
	public void writeData() throws DMLRuntimeException {
		
		if ( isBuffered() ) {
			writeInMemoryMatrixToHDFS(_hdfsFileName, InputInfo.getMatchingOutputInfo(((MatrixFormatMetaData)_metaData).getInputInfo()));
		}
		else {
			throw new DMLRuntimeException("Can not write a matrix that is not in the buffer.");
		}
		//cleanData();
	}
	
	public String toString() { 
		StringBuilder str = new StringBuilder();
		str.append("MatrixObject: ");
		str.append(_hdfsFileName + ", ");
		if ( _metaData instanceof NumItemsByEachReducerMetaData ) {
			str.append("NumItemsByEachReducerMetaData");
		} 
		else {
			MatrixFormatMetaData md = (MatrixFormatMetaData)_metaData;
			if ( md != null ) {
				MatrixCharacteristics mc = ((MatrixDimensionsMetaData)_metaData).getMatrixCharacteristics();
				str.append(mc.toString());
				
				InputInfo ii = md.getInputInfo();
				if ( ii == null )
					str.append("null");
				else {
					if ( InputInfo.inputInfoToString(ii) == null ) {
						try {
							throw new DMLRuntimeException("Unexpected input format");
						} catch (DMLRuntimeException e) {
							e.printStackTrace();
						}
					}
					str.append(InputInfo.inputInfoToString(ii));
				}
			}
			else {
				str.append("null, null");
			}
		}
		return str.toString();
	}
}
