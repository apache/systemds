package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.commons.lang.StringUtils;
import org.apache.spark.api.java.function.Function2;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.instructions.spark.data.CountLinesInfo;

public class CountLines implements Function2<Integer, Iterator<String>, Iterator<Tuple2<Integer, CountLinesInfo>>> {
	private static final long serialVersionUID = -2611946238807543849L;
	private String delim = null;
	
	public CountLines(String delim) {
		this.delim = delim; 
	}
	
	private long getNumberOfColumns(String line) {
		return StringUtils.countMatches(line, delim) + 1;
	}

	@Override
	public Iterator<Tuple2<Integer, CountLinesInfo>> call(Integer partNo, Iterator<String> lines) throws Exception {
		long nline = 0;
		long estimateOfNumLines = -1;
		while (lines.hasNext()) {
			String line = lines.next();
			estimateOfNumLines = Math.max(estimateOfNumLines, getNumberOfColumns(line));
			nline = nline + 1;
		}

		// Package up the result in a format that Spark understands
		ArrayList<Tuple2<Integer, CountLinesInfo>> retVal = new ArrayList<Tuple2<Integer, CountLinesInfo>>();
		CountLinesInfo info = new CountLinesInfo();
		info.setNumLines(nline);
		info.setExpectedNumColumns(estimateOfNumLines);
		retVal.add(new Tuple2<Integer, CountLinesInfo>(partNo, info));
		return retVal.iterator();
	}
	
}

