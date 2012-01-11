package dml.runtime.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import dml.runtime.matrix.io.BinaryBlockToBinaryCellConverter;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.TextToBinaryCellConverter;
import dml.utils.DMLRuntimeException;

/**
 * This class provides methods to convert data from SystemML's internal and
 * external data representations to representations needed external functions,
 * such as those in JLapack, by and vice versa.
 * 
 * @author Felix Hamborg
 * 
 */
public class DataConverter {
	private static Configuration conf = new Configuration();

	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @param blockSizeRows
	 *            Number of rows in normal blocks
	 * @param blockSizeCols
	 *            Number of cols in normal blocks
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static double[] readDouble1DArrayMatrixFromHDFSBlock(String dir, int numRows, int numCols,
			int blockSizeRows, int blockSizeCols) throws DMLRuntimeException {
		double[] ret = new double[numRows * numCols];
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
				BinaryBlockToBinaryCellConverter conv = new BinaryBlockToBinaryCellConverter();
				conv.setBlockSize(blockSizeRows, blockSizeCols);

				while (reader.next(indexes, value)) {
					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						int pos = (int) (pair.getKey().getColumnIndex() - 1 + (pair.getKey().getRowIndex() - 1)
								* numCols);
						ret[pos] = pair.getValue().getValue();
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}

		return ret;
	}

	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static double[] readDouble1DArrayMatrixFromHDFSText(String dir, int numRows, int numCols)
			throws DMLRuntimeException {
		double[] ret = new double[numRows * numCols];
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			if (!fs.isDirectory(new Path(dir))) {
				subpaths = new Path[] { new Path(dir) };
			}

			LongWritable indexes = new LongWritable();
			Text value = new Text();

			for (Path path : subpaths) {
				// SequenceFile.Reader reader = new SequenceFile.Reader(fs,
				// path, conf);
				TextToBinaryCellConverter conv = new TextToBinaryCellConverter();
				FSDataInputStream fi = fs.open(path);
				BufferedReader br = new BufferedReader(new InputStreamReader(fi));
				String line = null;
				while ((line = br.readLine()) != null) {
					value = new Text(line);

					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						int pos = (int) (pair.getKey().getColumnIndex() - 1 + (pair.getKey().getRowIndex() - 1)
								* numCols);
						ret[pos] = pair.getValue().getValue();
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}

		return ret;
	}

	/**
	 * Reads a matrix from HDFS (in cell format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix is expected to have
	 * @param numCols
	 *            Number of columns which the matrix is expected to have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static double[] readDouble1DArrayMatrixFromHDFSCell(String dir, int numRows, int numCols)
			throws DMLRuntimeException {
		double[] ret = new double[numRows * numCols];

		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell value = new MatrixCell();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

				while (reader.next(indexes, value)) {
					long i = indexes.getRowIndex() - 1;
					long j = indexes.getColumnIndex() - 1;
					long p = i * numCols + j;
					if (p > (int) p)
						throw new DMLRuntimeException("Matrix is too large");

					ret[(int) p] = value.getValue();
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}

		return ret;
	}

	public static void writeDouble2DArrayMatrixToHDFSBlock(String dir, int blockSizeRows, int blockSizeCols,
			double[][] matrix) throws DMLRuntimeException {
		int numRows = matrix.length;
		int numCols = matrix[0].length;
		try {
			SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, new Path(dir),
					MatrixIndexes.class, MatrixBlock.class);

			MatrixIndexes index = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();
			for (int i = 0; i < matrix.length; i += numRows) {
				int rows = Math.min(numRows, (matrix.length - i));
				for (int j = 0; j < matrix[i].length; j += numCols) {
					int cols = Math.min(numCols, (matrix[i].length - j));
					index.setIndexes(((i / numRows) + 1), ((j / numCols) + 1));
					value = new MatrixBlock(rows, cols, true);
					for (int k = 0; k < rows; k++) {
						for (int l = 0; l < cols; l++) {
							value.setValue(k, l, matrix[i + k][j + l]);
						}
					}
					writer.append(index, value);
				}
			}

			writer.close();
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static void writeDouble1DArrayMatrixToHDFSBlock(String dir, int numRows, int numCols, int blockSizeRows,
			int blockSizeCols, double[] array) throws DMLRuntimeException {

		double[][] matrix = new double[numRows][numCols];
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				matrix[i][j] = array[i + j * numRows];
			}
		}
		writeDouble2DArrayMatrixToHDFSBlock(dir, blockSizeRows, blockSizeCols, matrix);
	}

	private static Path[] getSubDirs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new Configuration());
		ArrayList<Path> paths = new ArrayList<Path>();
		for (FileStatus cur : fs.listStatus(new Path(dir))) {
			paths.add(cur.getPath());
		}
		return paths.toArray(new Path[paths.size()]);
	}
}
