package com.ibm.bi.dml.runtime.instructions.spark.utils;

import java.io.Serializable;

import org.apache.hadoop.io.Text;
import org.apache.spark.Accumulator;
import org.apache.spark.sql.Row;
import org.apache.spark.mllib.linalg.Vector;

import com.ibm.bi.dml.runtime.io.IOUtilFunctions;

public class RowAnalysisFunctionHelper implements Serializable 
{
	private static final long serialVersionUID = 2310303223289674477L;

	private Accumulator<Double> _aNnz = null;
	private String _delim = null;
	
	public RowAnalysisFunctionHelper( Accumulator<Double> aNnz ) {
		_aNnz = aNnz;
	}
	
	public RowAnalysisFunctionHelper( Accumulator<Double> aNnz, String delim ) {
		_aNnz = aNnz;
		_delim = delim;
	}
	
	public String analyzeText(Text v1) throws Exception {
		//parse input line
		String line = v1.toString();
		String[] cols = IOUtilFunctions.split(line, _delim);
		
		//determine number of non-zeros of row (w/o string parsing)
		long lnnz = 0;
		for( String col : cols ) {
			if( !col.isEmpty() && !col.equals("0") && !col.equals("0.0") ) {
				lnnz++;
			}
		}
		
		//update counters
		_aNnz.add( (double)lnnz );
		
		return line;
	}
	
	public Row analyzeRow(Row arg0) throws Exception {
		//determine number of non-zeros of row
		long lnnz = 0;
		if(arg0 != null) {
			for(int i = 0; i < arg0.length(); i++) {
				if(RowToBinaryBlockFunctionHelper.getDoubleValue(arg0, i) != 0) {
					lnnz++;
				}
			}
		}
		else {
			throw new Exception("Error while analyzing row");
		}
		
		//update counters
		_aNnz.add( (double)lnnz );
		
		return arg0;
	}
	
	public Row analyzeVector(Row row)  {
		Vector vec = (Vector) row.get(0); // assumption: 1 column DF
		long lnnz = 0;
		for(int i = 0; i < vec.size(); i++) {
			if(vec.apply(i) == 0) {
				lnnz++;
			}
		}
		
		//update counters
		_aNnz.add( (double)lnnz );
		return row;
	}
}