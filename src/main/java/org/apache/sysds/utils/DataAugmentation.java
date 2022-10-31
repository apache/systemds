/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

public class DataAugmentation 
{
	/**
	 * This function returns a new frame block with error introduced in the data:
	 * Typos in string values, null values, outliers in numeric data and swapped elements.
	 * 
	 * @param input Original frame block
	 * @param pTypo Probability of introducing a typo in a row
	 * @param pMiss Probability of introducing missing values in a row
	 * @param pDrop Probability of dropping a value inside a row
	 * @param pOut Probability of introducing outliers in a row
	 * @param pSwap Probability swapping two elements in a row
	 * @return A new frameblock with corrupted elements
	 * 
	 */
	public static FrameBlock dataCorruption(FrameBlock input, double pTypo, double pMiss, double pDrop, double pOut, double pSwap) {
		List<Integer> numerics = new ArrayList<>();
		List<Integer> strings = new ArrayList<>();
		List<Integer> swappable = new ArrayList<>();

		FrameBlock res = preprocessing(input, numerics, strings, swappable);
		res = typos(res, strings, pTypo);
		res = miss(res, pMiss, pDrop);
		res = outlier(res, numerics, pOut, 0.5, 3);
		
		return res;
	}
	
	/**
	 * This function returns a new frame block with a labels column added, and build the lists
	 * with column index of the different types of data.
	 * 
	 * @param frame Original frame block
	 * @param numerics Empty list to return the numeric positions
	 * @param strings Empty list to return the string positions
	 * @param swappable Empty list to return the swappable positions
	 * @return A new frameblock with a labels column
	 * 
	 */
	public static FrameBlock preprocessing(FrameBlock frame, List<Integer> numerics, List<Integer> strings, List<Integer> swappable) {
		FrameBlock res = new FrameBlock(frame);
		for(int i=0;i<res.getNumColumns();i++) {
			if(res.getSchema()[i].isNumeric())
				numerics.add(i);
			else if(res.getSchema()[i].equals(ValueType.STRING))
				strings.add(i);
			if(i!=res.getNumColumns()-1 && res.getSchema()[i].equals(res.getSchema()[i+1]))
				swappable.add(i);
		}
		
		String[] labels = new String[res.getNumRows()];
		Arrays.fill(labels, "");
		res.appendColumn(labels);
		res.getColumnNames()[res.getNumColumns()-1] = "errorLabels";
		
		return res;
	}
	
	/**
	 * This function modifies the given, preprocessed frame block to add typos to the string values,
	 * marking them with the label typos.
	 * 
	 * @param frame Original frame block
	 * @param strings List with the columns of string type that can be changed, generated during preprocessing or manually selected
	 * @param pTypo Probability of adding a typo to a row
	 * @return A new frameblock with typos
	 * 
	 */
	public static FrameBlock typos(FrameBlock frame, List<Integer> strings, double pTypo) 
	{
		if(!frame.getColumnName(frame.getNumColumns()-1).equals("errorLabels")) {
			throw new IllegalArgumentException("The FrameBlock passed has not been preprocessed.");
		}
		if(strings.isEmpty()) 
			return frame;
		
		Random rand = new Random();
		for(int r=0;r<frame.getNumRows();r++) {
			int c = strings.get(rand.nextInt(strings.size()));
			String s = (String) frame.get(r, c);
			if(s.length()!=1 && rand.nextDouble()<=pTypo) {
				int i = rand.nextInt(s.length());
				if(i==s.length()-1)             s = swapchr(s, i-1, i);
				else if(i==0)                   s = swapchr(s, i, i+1);
				else if(rand.nextDouble()<=0.5) s = swapchr(s, i, i+1);
				else                            s = swapchr(s, i-1, i);
				frame.set(r, c, s);
				String label = (String) frame.get(r, frame.getNumColumns()-1);
				frame.set(r, frame.getNumColumns()-1,
					label.equals("") ? "typo" : (label + ",typo"));
			}
		}
		return frame;
	}
	
	/**
	 * This function modifies the given, preprocessed frame block to add missing values to some of the rows,
	 * marking them with the label missing.
	 * 
	 * @param frame Original frame block
	 * @param pMiss Probability of adding missing values to a row
	 * @param pDrop Probability of dropping a value
	 * @return A new frameblock with missing values
	 * 
	 */
	public static FrameBlock miss(FrameBlock frame, double pMiss, double pDrop) {
		if(!frame.getColumnName(frame.getNumColumns()-1).equals("errorLabels")) {
			throw new IllegalArgumentException("The FrameBlock passed has not been preprocessed.");
		}
		Random rand = new Random();
		for(int r=0;r<frame.getNumRows();r++) {
			if(rand.nextDouble()<=pMiss) {
				int dropped = 0;
				for(int c=0;c<frame.getNumColumns()-1;c++) {
					Object xi = frame.get(r, c);
					if(xi!=null && !xi.equals(0) && rand.nextDouble()<=pDrop) {
						frame.set(r, c, null);
						dropped++;
					}
				}
				if(dropped>0) {
					String label = (String) frame.get(r, frame.getNumColumns()-1);
					frame.set(r, frame.getNumColumns()-1,
						label.equals("") ? "missing" : (label + ",missing"));
				}
			}
		}
		return frame;
	}
	
	/**
	 * This function modifies the given, preprocessed frame block to add outliers to some
	 * of the numeric data of the frame, adding or  several times the standard deviation,
	 * and marking them with the label outlier.
	 * 
	 * @param frame Original frame block
	 * @param numerics List with the columns of numeric type that can be changed, generated during preprocessing or manually selected
	 * @param pOut Probability of introducing an outlier in a row
	 * @param pPos Probability of using positive deviation
	 * @param times Times the standard deviation is added
	 * @return A new frameblock with outliers
	 * 
	 */
	public static FrameBlock outlier(FrameBlock frame, List<Integer> numerics, double pOut, double pPos, int times) {
		
		if(!frame.getColumnName(frame.getNumColumns()-1).equals("errorLabels")) {
			throw new IllegalArgumentException("The FrameBlock passed has not been preprocessed.");
		}
		if(numerics.isEmpty())
			return frame;
		
		Map<Integer, Double> stds = new HashMap<>();
		Random rand = new Random();
		for(int r=0;r<frame.getNumRows();r++) {
			if(rand.nextDouble()>pOut) continue;
			int c = numerics.get(rand.nextInt(numerics.size()));
			if(!stds.containsKey(c)) {
				FrameBlock ftmp = frame.slice(0, 
					frame.getNumColumns()-1, c, c, new FrameBlock());
				MatrixBlock mtmp = DataConverter.convertToMatrixBlock(ftmp);
				double sum = mtmp.sum();
				double mean = sum/mtmp.getNumRows();
				MatrixBlock diff = mtmp.scalarOperations(InstructionUtils
					.parseScalarBinaryOperator("-", false, mean), new MatrixBlock());
				double sumsq = diff.sumSq();
				stds.put(c, Math.sqrt(sumsq/mtmp.getNumRows()));
			}
			Double std = stds.get(c);
			boolean pos = rand.nextDouble()<=pPos;
			switch(frame.getSchema()[c]) {
				case INT32: {
					Integer val = (Integer) frame.get(r, c);
					frame.set(r, c, val + (pos?1:-1)*(int)Math.round(times*std));
					break;
				}
				case INT64: {
					Long val = (Long) frame.get(r, c);
					frame.set(r, c, val + (pos?1:-1)*Math.round(times*std));
					break;
				}
				case FP32: {
					Float val = (Float) frame.get(r, c);
					frame.set(r, c, val + (pos?1:-1)*(float)(times*std));
					break;
				}
				case FP64: {
					Double val = (Double) frame.get(r, c);
					frame.set(r, c, val + (pos?1:-1)*times*std);
					break;
				}
				default: //do nothing
			}
			String label = (String) frame.get(r, frame.getNumColumns()-1);
			frame.set(r, frame.getNumColumns()-1,
				label.equals("") ? "outlier": (label + ",outlier"));
		}
		
		return frame;
	}
	
	/**
	 * This function modifies the given, preprocessed frame block to add swapped fields of the same ValueType
	 * that are consecutive, marking them with the label swap.
	 * 
	 * @param frame Original frame block
	 * @param swappable List with the columns that are swappable, generated during preprocessing
	 * @param pSwap Probability of swapping two fields in a row
	 * @return A new frameblock with swapped elements
	 * 
	 */
	public static FrameBlock swap(FrameBlock frame, List<Integer> swappable, double pSwap) {
		if(!frame.getColumnName(frame.getNumColumns()-1).equals("errorLabels")) {
			throw new IllegalArgumentException("The FrameBlock passed has not been preprocessed.");
		}
		Random rand = new Random();
		for(int r=0;r<frame.getNumRows();r++) {
			if(rand.nextDouble()<=pSwap) {
				int i = swappable.get(rand.nextInt(swappable.size()));
				Object tmp = frame.get(r, i);
				frame.set(r, i, frame.get(r, i+1));
				frame.set(r, i+1, tmp);
				String label = (String) frame.get(r, frame.getNumColumns()-1);
				frame.set(r, frame.getNumColumns()-1,
					label.equals("") ? "swap" : (label + ",swap"));
			}
		}
		return frame;
	}
	
	private static String swapchr(String str, int i, int j) {
		if (j == str.length() - 1) 
			return str.substring(0, i) + str.charAt(j)
				+ str.substring(i + 1, j) + str.charAt(i);
		return str.substring(0, i) + str.charAt(j)
			+ str.substring(i + 1, j) + str.charAt(i)
			+ str.substring(j + 1, str.length());
	}
}
