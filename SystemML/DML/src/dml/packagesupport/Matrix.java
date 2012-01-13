package dml.packagesupport;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FSDataInputStream;
import org.nimble.hadoop.HDFSFileManager;

/**
 * Class to represent the matrix input type
 * 
 * @author aghoting
 * 
 */
public class Matrix extends FIO {

	private static final long serialVersionUID = -1058329938431848909L;
	String filePath;
	long rows, cols;
	ValueType vType;

	public enum ValueType {
		Double, Integer
	};

	/**
	 * Constructor that takes matrix file path, num rows, num cols, and matrix
	 * value type as parameters.
	 * 
	 * @param path
	 * @param rows
	 * @param cols
	 * @param vType
	 */

	public Matrix(String path, long rows, long cols, ValueType vType) {
		super(Type.Matrix);
		filePath = path;
		this.rows = rows;
		this.cols = cols;
		this.vType = vType;
	}

	/**
	 * Method to get file path for matrix.
	 * 
	 * @return
	 */
	public String getFilePath() {
		return filePath;
	}

	/**
	 * Method to get the number of rows in the matrix.
	 * 
	 * @return
	 */
	public long getNumRows() {
		return rows;
	}

	/**
	 * Method to get the number of cols in the matrix.
	 * 
	 * @return
	 */

	public long getNumCols() {
		return cols;
	}

	/**
	 * Method to get value type for this matrix.
	 * 
	 * @return
	 */
	public ValueType getValueType() {
		return vType;
	}

	/**
	 * Method to get matrix as double array. This should only be used if the
	 * user knows the matrix fits in memory. We are using the dense
	 * representation.
	 * 
	 * @return
	 */
	public double[][] getMatrixAsDoubleArray() {

		double[][] arr = new double[(int) rows][(int) cols];

		try {

			String[] files;

			// get file names for this matrix.

			files = HDFSFileManager.getFileNamesWithPrefixStatic(this
					.getFilePath() + "/");
			String line = null;

			// read each file into memory.
			for (String file : files) {

				FSDataInputStream inStrm;
				inStrm = HDFSFileManager.getInputStreamStatic(file);
				BufferedReader br = new BufferedReader(new InputStreamReader(
						inStrm));

				while ((line = br.readLine()) != null) {

					StringTokenizer tk = new StringTokenizer(line);
					int i, j;
					double val;

					i = Integer.parseInt(tk.nextToken());
					j = Integer.parseInt(tk.nextToken());
					val = Double.parseDouble(tk.nextToken());

					arr[i - 1][j - 1] = val;
				}
			}
		} catch (Exception e) {
			throw new PackageRuntimeException(e.toString());
		}

		return arr;

	}
}