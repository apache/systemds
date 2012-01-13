package dml.runtime.test.numericalStability;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class NISTParser {

	public static class ReferenceResults
	{
		public double mean=0.0;
		public double std=0.0;
		public double autocorrelation=0.0;
		
		public String toString()
		{
			return "mean: "+mean+", std: "+std+", auto correlation: "+autocorrelation;
		}
		
	}
	static ReferenceResults parseNISTFile(String input, String output) throws IOException
	{
		ReferenceResults ret=new ReferenceResults();
		BufferedReader reader=new BufferedReader(new FileReader(new File(input)));
		BufferedWriter writer=new BufferedWriter(new FileWriter(new File(output)));
		String line=null;
		int n=0;
		int certifiedValuesStart=-100;
		int valueStart=-100;
		int index=0;
		while((line=reader.readLine())!=null)
		{
			n++;
			if(line.startsWith("File Format:"))
			{
				/*
				 * Header          : lines  1 to   60     (=   60)
               Certified Values: lines 41 to   43     (=    3)
               Data            : lines 61 to 1061     (= 1001)
				 */
				
				line=reader.readLine();//get header
				line=reader.readLine();//get certified values
				int start=line.indexOf("lines");
				int end=line.indexOf("to", start+5);
				certifiedValuesStart=Integer.parseInt(line.substring(start+5, end).trim());
				line=reader.readLine();//get certified values
				start=line.indexOf("lines");
				end=line.indexOf("to", start+5);
				valueStart=Integer.parseInt(line.substring(start+5, end).trim());
				n+=3;
			}
			
			if(certifiedValuesStart>0 && n==certifiedValuesStart)
			{
				int start=line.indexOf(":");
				int end=line.indexOf("(exact)", start+1);
				if(end<0) end=line.length();
				ret.mean=Double.parseDouble(line.substring(start+1, end).trim());
			}else if(certifiedValuesStart>0 && n==certifiedValuesStart+1)
			{
				int start=line.indexOf(":");
				int end=line.indexOf("(exact)", start+1);
				if(end<0) end=line.length();
				ret.std=Double.parseDouble(line.substring(start+1, end).trim());
			}else if(certifiedValuesStart>0 && n==certifiedValuesStart+2)
			{
				int start=line.indexOf(":");
				int end=line.indexOf("(exact)", start+1);
				if(end<0) end=line.length();
				ret.autocorrelation=Double.parseDouble(line.substring(start+1, end).trim());
			}
			
			if(valueStart>0 && n>=valueStart)
			{
				index++;
				writer.write(index+" 1 "+line.trim());
				writer.newLine();
			}
		}
		reader.close();
		writer.close();
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println(NISTParser.parseNISTFile("ReferenceData/PiDigits.dat", "ReferenceData/PiDigits.dat.mtx"));
	}
}
