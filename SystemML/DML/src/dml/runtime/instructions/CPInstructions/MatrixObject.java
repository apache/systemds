package dml.runtime.instructions.CPInstructions;

import java.io.IOException;

import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.MatrixFormatMetaData;
import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.BinaryOperator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.runtime.matrix.operators.ScalarOperator;
import dml.runtime.matrix.operators.UnaryOperator;
import dml.runtime.util.DataConverter;
import dml.runtime.util.MapReduceTool;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

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
				new MatrixIndexes(1, 1)));
		result.setData(result_data);
		result.updateMatrixMetaData();

		return result;
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
							// TODO Auto-generated catch block
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
