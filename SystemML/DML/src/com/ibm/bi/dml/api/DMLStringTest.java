package com.ibm.bi.dml.api;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.utils.DMLException;

public class DMLStringTest {
	public static void main(String[] args) throws IOException, ParseException, DMLException {
		
		DMLScript d = new DMLScript();
		String s = "B = read(\"data/fileB\", rows=3, cols=1, format=\"text\") \n" +
				"i=0 \n" +
				"x=0 \n" +
				"while (i<2) {\n" +
				"x = x+sum(B) \n" +
				"print(\"Sum = \" +  x)\n" +
				"i = i+1 \n" +
				"} \n";
		
		InputStream is = new ByteArrayInputStream(s.getBytes());
		String[] executionOptions = null;
		String[] scriptArgs = {"1", "-d"};
		d.executeScript(is, executionOptions, "1", "-d");
		
	}
}