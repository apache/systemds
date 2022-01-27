package org.apache.sysds.runtime.iogen.exp;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.ArrayList;
import java.util.HashSet;

public class GIONestedExperimentHDFS {

	public static void main(String[] args) throws Exception {

		String sampleRawFileName = args[0];
		String sampleFrameFileName = args[1];
		Integer sampleNRows = Integer.parseInt(args[2]);
		String delimiter = args[3];
		String schemaFileName = args[4];
		String dataFileName = args[5];

		Float percent = Float.parseFloat(args[6]);
		String datasetName = args[7];
		String LOG_HOME =args[8];

		if(delimiter.equals("\\t"))
			delimiter = "\t";

		Util util = new Util();
		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
		int ncols = sampleSchema.length;

		ArrayList<Types.ValueType> newSampleSchema = new ArrayList<>();
		ArrayList<ArrayList<String>> newSampleFrame = new ArrayList<>();

		String[][] sampleFrameStrings =  util.loadFrameData(sampleFrameFileName, sampleNRows, ncols, delimiter);

		for(int c = 0; c < sampleFrameStrings[0].length; c++) {
			HashSet<String> valueSet = new HashSet<>();
			for(int r=0; r<sampleFrameStrings.length;r++)
				valueSet.add(sampleFrameStrings[r][c]);
			if(valueSet.size()>3){
				ArrayList<String> tempList = new ArrayList<>();
				for(int r=0; r<sampleFrameStrings.length;r++) {
					tempList.add(sampleFrameStrings[r][c]);
				}
				newSampleFrame.add(tempList);
				newSampleSchema.add(sampleSchema[c]);
			}
		}

		sampleFrameStrings = new String[newSampleFrame.get(0).size()][newSampleFrame.size()];

		for(int row=0; row<sampleFrameStrings.length; row++){
			for(int col=0; col<sampleFrameStrings[0].length; col++){
				sampleFrameStrings[row][col] = newSampleFrame.get(col).get(row);
			}
		}

		sampleSchema = new Types.ValueType[newSampleSchema.size()];
		for(int i=0; i< newSampleSchema.size();i++)
			sampleSchema[i] = newSampleSchema.get(i);

		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);

		double tmpTime = System.nanoTime();
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
		GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
		FrameReader fr = gr.getReader();
		double generateTime = (System.nanoTime() - tmpTime) / 1000000000.0;

		tmpTime = System.nanoTime();
		FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, sampleSchema.length);
		double readTime = (System.nanoTime() - tmpTime) / 1000000000.0;

		String log= datasetName+","+ frameBlock.getNumRows()+","+ ncols+","+percent+","+ sampleNRows+","+ generateTime+","+readTime;
		util.addLog(LOG_HOME, log);
	}
}
