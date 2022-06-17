package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Util {

	public String readEntireTextFile(String fileName) throws IOException {
		String text = Files.readString(Paths.get(fileName));
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

	public Map<String, Integer> getSchemaMap(String fileName) throws IOException {
		Map<String, Integer> schemaMap = new HashMap<>();
		try(BufferedReader br = new BufferedReader(new FileReader(fileName,StandardCharsets.UTF_8))) {
			String line;
			while((line = br.readLine()) != null) {
				String[] colSchema = line.split(",");
				schemaMap.put(colSchema[0], Integer.parseInt(colSchema[1]));
			}
		}
		return schemaMap;
	}

	public String[][] loadFrameData(String fileName,String delimiter, int ncols)
		throws IOException {
		ArrayList<String[]> sampleRawLines = new ArrayList<>();
		try(BufferedReader br = new BufferedReader(new FileReader(fileName,StandardCharsets.UTF_8))) {
			String line;
			while((line = br.readLine()) != null) {
				String[] data = line.split(delimiter);
				String[] colsData = new String[ncols];
				for(int i = 0; i < data.length; i++) {
					String[] value = data[i].split("::");
					if(value.length ==2) {
						int col = Integer.parseInt(value[0]);
						colsData[col] = value[1];
					}
				}
				sampleRawLines.add(colsData);
			}
		}

		int nrows = sampleRawLines.size();
		String[][] result = new String[nrows][ncols];
		for(int i=0; i< nrows; i++)
			result[i] = sampleRawLines.get(i);

		return result;
	}

	public MatrixBlock loadMatrixData(String fileName,  String delimiter) throws IOException {
		int ncols = 0;
		try(BufferedReader br = new BufferedReader(new FileReader(fileName,StandardCharsets.UTF_8))) {
			String line;
			while((line = br.readLine()) != null) {
				String[] data = line.split(delimiter);
				ncols = Math.max(ncols, Integer.parseInt( data[data.length-1].split("::")[0]));
			}
		}
		String[][] dataString = loadFrameData(fileName,delimiter, ncols+1);
		double[][] data = new double[dataString.length][dataString[0].length];
		for(int i=0;i<dataString.length;i++)
			for(int j=0;j<dataString[0].length;j++)
				if(dataString[i][j]!=null)
					data[i][j] = Double.parseDouble(dataString[i][j]);
				else
					data[i][j] =0;
		MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
		return mb;
	}
}
