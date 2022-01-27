package org.apache.sysds.runtime.iogen.exp;

import org.apache.sysds.common.Types;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Util {

	// Load Random 2D data from file
	private double[][] load2DData(String fileName, int nrows, int ncols) throws Exception {

		Path path = Paths.get(fileName);
		FileChannel inStreamRegularFile = FileChannel.open(path);
		int bufferSize = ncols * 8;

		double[][] result = new double[nrows][ncols];
		try {
			for(int r = 0; r < nrows; r++) {
				inStreamRegularFile.position((long) r * ncols * 8);
				ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize);
				inStreamRegularFile.read(buffer);
				buffer.flip();

				for(int c = 0; c < ncols; c++) {
					result[r][c] = buffer.getDouble();
				}
			}
			inStreamRegularFile.close();
		}
		catch(IOException e) {
			throw new Exception("Can't read matrix from ByteArray", e);
		}
		return result;
	}

	public String readEntireTextFile(String fileName) throws IOException {
		String text = new String(Files.readAllBytes(Paths.get(fileName)), StandardCharsets.UTF_8);
		return text;
	}

	public void createLog(String fileName, String text) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		writer.write(text);
		writer.write("\n");
		writer.close();
	}

	public void addLog(String fileName, String log) {
		try(Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName, true), "utf-8"))) {
			writer.write(log);
			writer.write("\n");
		}
		catch(Exception ex) {
		}
	}

	public Types.ValueType[] getSchema(String fileName) throws IOException {
		String[] sschema = readEntireTextFile(fileName).trim().split(",");
		Types.ValueType[] result = new Types.ValueType[sschema.length];
		for(int i = 0; i < sschema.length; i++)
			result[i] = Types.ValueType.valueOf(sschema[i]);
		return result;
	}

	public String[][] loadFrameData(String fileName, int nrows, int ncols, String delimiter)
		throws IOException {
		String[][] result = new String[nrows][ncols];

		try(BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			String line;
			int row = 0;
			while((line = br.readLine()) != null) {
				String[] data = line.split(delimiter);
				for(int i = 0; i < data.length; i++) {
					String[] value = data[i].split("::");
					if(value.length ==2) {
						int col = Integer.parseInt(value[0]);
						result[row][col] = value[1];
					}
				}
				row++;
			}
		}
		return result;
	}
}
