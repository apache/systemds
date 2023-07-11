package org.apache.sysds.test.functions.iogen.baseline;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.template.TemplateUtil;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.apache.sysds.runtime.io.FrameReader.checkValidInputFile;
import static org.apache.sysds.runtime.io.FrameReader.createOutputFrameBlock;

public class FrameReaderXMLJackson {
	protected TemplateUtil.SplitOffsetInfos _offsets;

	public FrameBlock readFrameFromHDFS(String fname, Types.ValueType[] schema, Map<String, Integer> schemaMap,
		String beginToken, String endToken, long rlen, long clen) throws IOException, DMLRuntimeException {
		//prepare file access
		JobConf jobConf = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fileSystem = IOUtilFunctions.getFileSystem(path, jobConf);
		FileInputFormat.addInputPath(jobConf, path);

		//check existence and non-empty file
		checkValidInputFile(fileSystem, path);

		// allocate output frame block
		String[] lnames = createOutputNamesFromSchemaMap(schemaMap);

		TextInputFormat informat = new TextInputFormat();
		informat.configure(jobConf);
		InputSplit[] splits = informat.getSplits(jobConf, 1);
		splits = IOUtilFunctions.sortInputSplits(splits);

		FrameBlock ret = computeSizeAndCreateOutputFrameBlock(informat, jobConf, schema, lnames, splits, beginToken, endToken);
		readXMLLFrameFromHDFS(splits, informat, jobConf, schema, schemaMap, ret);

		return ret;
	}

	protected void readXMLLFrameFromHDFS(InputSplit[] splits, TextInputFormat informat, JobConf jobConf,
		Types.ValueType[] schema, Map<String, Integer> schemaMap, FrameBlock dest) throws IOException {
		int rpos = 0;
		for(int i = 0; i < splits.length; i++) {
			RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[i], jobConf, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			TemplateUtil.SplitInfo splitInfo = _offsets.getSeqOffsetPerSplit(i);
			rpos = _offsets.getOffsetPerSplit(i);
			readXMLFrameFromInputSplit(reader, splitInfo, key, value, rpos, schema, schemaMap, dest);
		}
	}

	protected FrameBlock computeSizeAndCreateOutputFrameBlock(TextInputFormat informat, JobConf job,
		Types.ValueType[] schema, String[] names, InputSplit[] splits, String beginToken, String endToken)
		throws IOException, DMLRuntimeException {

		int row = 0;
		// count rows in parallel per split
		try {
			_offsets = new TemplateUtil.SplitOffsetInfos(splits.length);
			for(int i = 0; i < splits.length; i++) {
				TemplateUtil.SplitInfo splitInfo = new TemplateUtil.SplitInfo();
				_offsets.setSeqOffsetPerSplit(i, splitInfo);
				_offsets.setOffsetPerSplit(i, row);
			}

			int splitIndex = 0;
			for(InputSplit inputSplit : splits) {
				int nrows = 0;
				TemplateUtil.SplitInfo splitInfo = _offsets.getSeqOffsetPerSplit(splitIndex);
				ArrayList<Pair<Long, Integer>> beginIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(inputSplit,
					informat, job, beginToken).getKey();
				ArrayList<Pair<Long, Integer>> endIndexes = TemplateUtil.getTokenIndexOnMultiLineRecords(inputSplit,
					informat, job, endToken).getKey();
				int tokenLength = endToken.length();

				int i = 0;
				int j = 0;

				if(beginIndexes.get(0).getKey() > endIndexes.get(0).getKey()) {
					nrows++;
					for(; j < endIndexes.size() && beginIndexes.get(0).getKey() > endIndexes.get(j).getKey(); j++)
						;
				}

				while(i < beginIndexes.size() && j < endIndexes.size()) {
					Pair<Long, Integer> p1 = beginIndexes.get(i);
					Pair<Long, Integer> p2 = endIndexes.get(j);
					int n = 0;
					while(p1.getKey() < p2.getKey() || (p1.getKey() == p2.getKey() && p1.getValue() < p2.getValue())) {
						n++;
						i++;
						if(i == beginIndexes.size()) {
							break;
						}
						p1 = beginIndexes.get(i);
					}
					j += n - 1;
					splitInfo.addIndexAndPosition(beginIndexes.get(i - n).getKey(), endIndexes.get(j).getKey(),
						beginIndexes.get(i - n).getValue(), endIndexes.get(j).getValue() + tokenLength);
					j++;
					nrows++;
				}
				if(splitIndex < splits.length - 1) {
					RecordReader<LongWritable, Text> reader = informat.getRecordReader(inputSplit, job, Reporter.NULL);
					LongWritable key = new LongWritable();
					Text value = new Text();

					StringBuilder sb = new StringBuilder();

					for(long ri = 0; ri < beginIndexes.get(beginIndexes.size() - 1).getKey(); ri++) {
						reader.next(key, value);
					}
					if(reader.next(key, value)) {
						String strVar = value.toString();
						sb.append(strVar.substring(beginIndexes.get(beginIndexes.size() - 1).getValue()));
						while(reader.next(key, value)) {
							sb.append(value.toString());
						}
						_offsets.getSeqOffsetPerSplit(splitIndex + 1).setRemainString(sb.toString());
					}
				}
				splitInfo.setNrows(nrows);
				_offsets.setOffsetPerSplit(splitIndex, row);
				row += nrows;
				splitIndex++;
			}
		}

		catch(Exception e) {
			throw new IOException("Sequence Scatter Row Counting Error: " + e.getMessage(), e);
		}
		FrameBlock ret = createOutputFrameBlock(schema, names, row);
		return ret;
	}

	protected static int readXMLFrameFromInputSplit(RecordReader<LongWritable, Text> reader,
		TemplateUtil.SplitInfo splitInfo, LongWritable key, Text value, int rpos, Types.ValueType[] schema,
		Map<String, Integer> schemaMap, FrameBlock dest) throws IOException {
		int rlen = splitInfo.getNrows();
		int ri;
		int row = 0;
		int beginPosStr, endPosStr;
		String remainStr = "";
		String str = "";
		StringBuilder sb = new StringBuilder(splitInfo.getRemainString());
		long beginIndex = splitInfo.getRecordIndexBegin(0);
		long endIndex = splitInfo.getRecordIndexEnd(0);
		boolean flag;
		XmlMapper mapper = new XmlMapper();
		if(sb.length() > 0) {
			ri = 0;
			while(ri < beginIndex) {
				reader.next(key, value);
				sb.append(value.toString());
				ri++;
			}
			reader.next(key, value);
			String valStr = value.toString();
			sb.append(valStr.substring(0, splitInfo.getRecordPositionBegin(0)));
			addRow(sb.toString(), mapper, dest, schema, schemaMap, row + rpos);
			row++;
			sb = new StringBuilder(valStr.substring(splitInfo.getRecordPositionBegin(0)));
		}
		else {
			ri = -1;
		}

		int rowCounter = 0;
		while(row < rlen) {
			flag = reader.next(key, value);
			if(flag) {
				ri++;
				String valStr = value.toString();
				if(ri >= beginIndex && ri <= endIndex) {
					beginPosStr = ri == beginIndex ? splitInfo.getRecordPositionBegin(rowCounter) : 0;
					endPosStr = ri == endIndex ? splitInfo.getRecordPositionEnd(rowCounter) : valStr.length();
					sb.append(valStr.substring(beginPosStr, endPosStr));
					remainStr = valStr.substring(endPosStr);
					continue;
				}
				else {
					str = sb.toString();
					sb = new StringBuilder();
					sb.append(remainStr).append(valStr);
					if(rowCounter + 1 < splitInfo.getListSize()) {
						beginIndex = splitInfo.getRecordIndexBegin(rowCounter + 1);
						endIndex = splitInfo.getRecordIndexEnd(rowCounter + 1);
					}
					rowCounter++;
				}
			}
			else {
				str = sb.toString();
				sb = new StringBuilder();
			}
			addRow(str, mapper, dest, schema, schemaMap, row+rpos);
			row++;
		}
		return row + rpos;
	}

	private static void addRow(String str, XmlMapper mapper, FrameBlock dest, Types.ValueType[] schema,
		Map<String, Integer> schemaMap, int row) throws JsonProcessingException {
		JsonNode root = mapper.readTree(str);
		Map<String, String> map = new HashMap<>();
		addKeys("", root, map, new ArrayList<>());
		for(Map.Entry<String, Integer> entry : schemaMap.entrySet()) {
			String strCellValue = map.get(entry.getKey());
			if(strCellValue != null) {
				try {
					dest.set(row, entry.getValue(),
						UtilFunctions.stringToObject(schema[entry.getValue()], strCellValue));
				}
				catch(Exception e) {
				}
			}
		}
	}

	private static void addKeys(String currentPath, JsonNode jsonNode, Map<String, String> map, List<Integer> suffix) {
		if(jsonNode.isObject()) {
			ObjectNode objectNode = (ObjectNode) jsonNode;
			Iterator<Map.Entry<String, JsonNode>> iter = objectNode.fields();
			String pathPrefix = currentPath.isEmpty() ? "" : currentPath + "/";

			while(iter.hasNext()) {
				Map.Entry<String, JsonNode> entry = iter.next();
				addKeys(pathPrefix + entry.getKey(), entry.getValue(), map, suffix);
			}
		}
		else if(jsonNode.isArray()) {
			ArrayNode arrayNode = (ArrayNode) jsonNode;
			for(int i = 0; i < arrayNode.size(); i++) {
				suffix.add(i + 1);
				addKeys(currentPath + "-" + i, arrayNode.get(i), map, suffix);
				if(i + 1 < arrayNode.size()) {
					suffix.remove(suffix.size() - 1);
				}
			}

		}
		else if(jsonNode.isValueNode()) {
			if(currentPath.contains("/") && !currentPath.contains("-")) {
				suffix = new ArrayList<>();
			}
			ValueNode valueNode = (ValueNode) jsonNode;
			if(currentPath.endsWith("/"))
				currentPath = currentPath.substring(0, currentPath.length() - 1);
			map.put("/" + currentPath, valueNode.asText());
		}
	}

	protected String[] createOutputNamesFromSchemaMap(Map<String, Integer> schemaMap) {
		String[] names = new String[schemaMap.size()];
		schemaMap.forEach((key, value) -> names[value] = key);
		return names;
	}
}
