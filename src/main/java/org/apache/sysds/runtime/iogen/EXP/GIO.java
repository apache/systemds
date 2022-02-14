package org.apache.sysds.runtime.iogen.EXP;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.ArrayList;
import java.util.HashSet;

public class GIO {

	public static void main(String[] args) throws Exception {
		String sampleRawFileName;
		String sampleFrameFileName;
		String sampleRawDelimiter;
		String schemaFileName;
		String dataFileName;
		long nrows;

		sampleRawFileName = System.getProperty("sampleRawFileName");
		sampleFrameFileName = System.getProperty("sampleFrameFileName");
		sampleRawDelimiter = System.getProperty("delimiter");
		if(sampleRawDelimiter.equals("\\t"))
			sampleRawDelimiter = "\t";
		schemaFileName = System.getProperty("schemaFileName");
		dataFileName = System.getProperty("dataFileName");
		nrows = Long.parseLong(System.getProperty("nrows"));


		Util util = new Util();
		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
		int ncols = sampleSchema.length;

		String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, ncols, sampleRawDelimiter);

		ArrayList<Types.ValueType> newSampleSchema = new ArrayList<>();
		ArrayList<ArrayList<String>> newSampleFrame = new ArrayList<>();

		for(int c = 0; c < sampleFrameStrings[0].length; c++) {
			HashSet<String> valueSet = new HashSet<>();
			for(int r = 0; r < sampleFrameStrings.length; r++)
				valueSet.add(sampleFrameStrings[r][c]);
			if(valueSet.size() > 1) {
				ArrayList<String> tempList = new ArrayList<>();
				for(int r = 0; r < sampleFrameStrings.length; r++) {
					tempList.add(sampleFrameStrings[r][c]);
				}
				newSampleFrame.add(tempList);
				newSampleSchema.add(sampleSchema[c]);
			}
		}

		sampleFrameStrings = new String[newSampleFrame.get(0).size()][newSampleFrame.size()];

		for(int row = 0; row < sampleFrameStrings.length; row++) {
			for(int col = 0; col < sampleFrameStrings[0].length; col++) {
				sampleFrameStrings[row][col] = newSampleFrame.get(col).get(row);
			}
		}

		sampleSchema = new Types.ValueType[newSampleSchema.size()];
		for(int i = 0; i < newSampleSchema.size(); i++)
			sampleSchema[i] = newSampleSchema.get(i);


		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);

		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
		GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
		FrameReader fr = gr.getReader();
		FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, nrows, sampleSchema.length);

	}
}
