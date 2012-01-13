package dml.runtime.instructions.CPInstructions;

import dml.lops.OutputParameters.Format;
import dml.runtime.util.DataConverter;
import dml.utils.DMLRuntimeException;

/**
 * Represents a matrix in controlprogram. This class contains method to read
 * matrices from HDFS and convert them to a specific format/representation. It
 * is also able to write several formats/representation of matrices to HDFS.
 * 
 * @author Felix Hamborg
 * 
 */
public class MatrixObject extends Data {

	enum DataType {
		EMPTY, ARRAY1D, ARRAY2D
	};

	private String _name;
	private int _rows;
	private int _cols;
	private int _rowsPerBlock;
	private int _colsPerBlock;
	private Format _format;

	private DataType _outputDataType = DataType.EMPTY;
	private double[] data1D = null;
	private double[][] data2D = null;

	public MatrixObject(String name, int rows, int cols, int rowsPerBlock, int colsPerBlock, Format format) {
		_name = name;
		_rows = rows;
		_cols = cols;
		_rowsPerBlock = rowsPerBlock;
		_colsPerBlock = colsPerBlock;
		_format = format;
	}

	public String getName() {
		return _name;
	}

	public int get_rows() {
		return _rows;
	}

	public void set_rows(int rows) {
		_rows = rows;
	}

	public int get_cols() {
		return _cols;
	}

	public void set_cols(int cols) {
		_cols = cols;
	}

	public int get_rowsPerBlock() {
		return _rowsPerBlock;
	}

	public void set_rowsPerBlock(int rowsPerBlock) {
		_rowsPerBlock = rowsPerBlock;
	}

	public int get_colsPerBlock() {
		return _colsPerBlock;
	}

	public void set_colsPerBlock(int colsPerBlock) {
		_colsPerBlock = colsPerBlock;
	}

	public double[] get1DDoubleArray() throws DMLRuntimeException {
		switch (_format) {
		case BINARY:
			return DataConverter
					.readDouble1DArrayMatrixFromHDFSBlock(_name, _rows, _cols, _rowsPerBlock, _colsPerBlock);
		case TEXT:
			return DataConverter.readDouble1DArrayMatrixFromHDFSText(_name, _rows, _cols);

		default:
			throw new DMLRuntimeException(_name + ": wrong format: " + _format.toString());
		}
	}

	public void setData(double[] matrix) {
		_outputDataType = DataType.ARRAY1D;
		data1D = matrix;
	}

	public void setData(double[][] matrix) {
		_outputDataType = DataType.ARRAY2D;
		data2D = matrix;
	}

	public void setName(String name) {
		_name = name;
	}

	/**
	 * Deletes internal data object.
	 */
	public void cleanData() {
		_outputDataType = DataType.EMPTY;
		data1D = null;
		data2D = null;
	}

	/**
	 * Writes matrix to disk. After successful write, internal data will be
	 * deleted, to gain memory.
	 * 
	 * @throws DMLRuntimeException
	 */
	public void writeDataAndClean() throws DMLRuntimeException {
		switch (_outputDataType) {
		case ARRAY1D:
			DataConverter
					.writeDouble1DArrayMatrixToHDFSBlock(_name, _rows, _cols, _rowsPerBlock, _colsPerBlock, data1D);
			break;

		case ARRAY2D:
			DataConverter.writeDouble2DArrayMatrixToHDFSBlock(_name, _rowsPerBlock, _colsPerBlock, data2D);
			break;
		}

		cleanData();
	}
}
