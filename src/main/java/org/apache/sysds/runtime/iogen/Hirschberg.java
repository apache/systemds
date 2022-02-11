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

package org.apache.sysds.runtime.iogen;

import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.Arrays;

public class Hirschberg {

	public Pair<ArrayList<String>, String> getLCS(String x, String y, int pxy, int pgap) {
		int i, j; // initialising variables
		int m = x.length(); // length of gene1
		int n = y.length(); // length of gene2

		// table for storing optimal substructure answers
		int dp[][] = new int[n + m + 1][n + m + 1];

		for(int[] x1 : dp)
			Arrays.fill(x1, 0);

		// initialising the table
		for(i = 0; i <= (n + m); i++) {
			dp[i][0] = i * pgap;
			dp[0][i] = i * pgap;
		}

		// calculating the minimum penalty
		for(i = 1; i <= m; i++) {
			for(j = 1; j <= n; j++) {
				if(x.charAt(i - 1) == y.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1];
				}
				else {
					dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1] + pxy, dp[i - 1][j] + pgap), dp[i][j - 1] + pgap);
				}
			}
		}

		// Reconstructing the solution
		int l = n + m; // maximum possible length
		i = m;
		j = n;
		int xpos = l;
		int ypos = l;

		// Final answers for the respective strings
		int xans[] = new int[l + 1];
		int yans[] = new int[l + 1];

		while(!(i == 0 || j == 0)) {
			if(x.charAt(i - 1) == y.charAt(j - 1)) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) y.charAt(j - 1);
				i--;
				j--;
			}
			else if(dp[i - 1][j - 1] + pxy == dp[i][j]) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) y.charAt(j - 1);
				i--;
				j--;
			}
			else if(dp[i - 1][j] + pgap == dp[i][j]) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) '_';
				i--;
			}
			else if(dp[i][j - 1] + pgap == dp[i][j]) {
				xans[xpos--] = (int) '_';
				yans[ypos--] = (int) y.charAt(j - 1);
				j--;
			}
		}
		while(xpos > 0) {
			if(i > 0)
				xans[xpos--] = (int) x.charAt(--i);
			else
				xans[xpos--] = (int) '_';
		}
		while(ypos > 0) {
			if(j > 0)
				yans[ypos--] = (int) y.charAt(--j);
			else
				yans[ypos--] = (int) '_';
		}
		// Since we have assumed the answer to be n+m long, we need to remove the extra
		// gaps in the starting id represents the index from which the arrays xans, yans are useful
		int id = 1;
		for(i = l; i >= 1; i--) {
			if((char) yans[i] == '_' && (char) xans[i] == '_') {
				id = i + 1;
				break;
			}
		}

		StringBuilder sb = new StringBuilder();
		ArrayList<String> pattern = new ArrayList<>();
		for(i = id; i <= l; i++) {
			if(xans[i] == yans[i])
				sb.append((char) xans[i]);
			else {
				if(sb.length() > 0)
					pattern.add(sb.toString());
				sb = new StringBuilder();
			}
		}

		if(sb.length() > 0)
			pattern.add(sb.toString());

		//		System.out.println("");
		//		for(i = id; i <= l; i++)
		//			System.out.print((char) yans[i]);
		//
		sb = new StringBuilder();
		for(int bi = id; bi <= l; bi++) {
			if(xans[bi] == yans[bi]) {
				sb.append((char) xans[bi]);
				System.out.print((char) xans[bi]);
			}
			//else
			//System.out.print("*");
		}
		System.out.println();
		if(sb.length() > 0) {
			//            StringBuilder stringBuilder = new StringBuilder();
			//            for (String s: pattern){
			//                stringBuilder.append(s).append("_");
			//            }
			//            if (stringBuilder.length()>0)
			//                stringBuilder.deleteCharAt(stringBuilder.length()-1);
			return new Pair<>(pattern, sb.toString());
		}
		else
			return null;
	}

	public Pair<ArrayList<String>, String> getLCS(String x, String y) {
		int i, j; // initialising variables
		int m = x.length(); // length of gene1
		int n = y.length(); // length of gene2

		// table for storing optimal substructure answers
		int dp[][] = new int[n + m + 1][n + m + 1];

		for(int[] x1 : dp)
			Arrays.fill(x1, 0);

		// initialising the table
		for(i = 0; i <= (n + m); i++) {
			dp[i][0] = i;
			dp[0][i] = i;
		}

		// calculating the minimum penalty
		for(i = 1; i <= m; i++) {
			for(j = 1; j <= n; j++) {
				if(x.charAt(i - 1) == y.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1];
				}
				else {
					dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]);
				}
			}
		}

		// Reconstructing the solution
		int l = n + m; // maximum possible length
		i = m;
		j = n;
		int xpos = l;
		int ypos = l;

		// Final answers for the respective strings
		int xans[] = new int[l + 1];
		int yans[] = new int[l + 1];

		while(!(i == 0 || j == 0)) {
			if(x.charAt(i - 1) == y.charAt(j - 1)) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) y.charAt(j - 1);
				i--;
				j--;
			}
			else if(dp[i - 1][j - 1] == dp[i][j]) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) y.charAt(j - 1);
				i--;
				j--;
			}
			else if(dp[i - 1][j] == dp[i][j]) {
				xans[xpos--] = (int) x.charAt(i - 1);
				yans[ypos--] = (int) '_';
				i--;
			}
			else if(dp[i][j - 1] == dp[i][j]) {
				xans[xpos--] = (int) '_';
				yans[ypos--] = (int) y.charAt(j - 1);
				j--;
			}
		}
		while(xpos > 0) {
			if(i > 0)
				xans[xpos--] = (int) x.charAt(--i);
			else
				xans[xpos--] = (int) '_';
		}
		while(ypos > 0) {
			if(j > 0)
				yans[ypos--] = (int) y.charAt(--j);
			else
				yans[ypos--] = (int) '_';
		}
		// Since we have assumed the answer to be n+m long, we need to remove the extra
		// gaps in the starting id represents the index from which the arrays xans, yans are useful
		int id = 1;
		for(i = l; i >= 1; i--) {
			if((char) yans[i] == '_' && (char) xans[i] == '_') {
				id = i + 1;
				break;
			}
		}

		StringBuilder sb = new StringBuilder();
		ArrayList<String> pattern = new ArrayList<>();
		for(i = id; i <= l; i++) {
			if(xans[i] == yans[i])
				sb.append((char) xans[i]);
			else {
				if(sb.length() > 0)
					pattern.add(sb.toString());
				sb = new StringBuilder();
			}
		}

		if(sb.length() > 0)
			pattern.add(sb.toString());

		//		System.out.println("");
		//		for(i = id; i <= l; i++)
		//			System.out.print((char) yans[i]);
		//
		sb = new StringBuilder();
		for(int bi = id; bi <= l; bi++) {
			if(xans[bi] == yans[bi]) {
				sb.append((char) xans[bi]);
				System.out.print((char) xans[bi]);
			}
			//else
			//System.out.print("*");
		}
		System.out.println();
		if(sb.length() > 0) {
			//            StringBuilder stringBuilder = new StringBuilder();
			//            for (String s: pattern){
			//                stringBuilder.append(s).append("_");
			//            }
			//            if (stringBuilder.length()>0)
			//                stringBuilder.deleteCharAt(stringBuilder.length()-1);
			return new Pair<>(pattern, sb.toString());
		}
		else
			return null;
	}

	public static void main(String[] args) {
		int misMatchPenalty = 3;
		int gapPenalty = 2;
		Hirschberg hirschberg = new Hirschberg();
		String s1 = "123";
		String s2 = "12369666";
		System.out.println(hirschberg.getLCS(s1, s2, misMatchPenalty, gapPenalty).getValue());
	}

	public ArrayList<String> getLCS(ArrayList<String> list, int pxy, int pgap) {

		if(list.size() < 2)
			return null;

		String str1 = list.get(0);
		String str2 = list.get(1);

		Pair<ArrayList<String>, String> pattern = getLCS(str1, str2, pxy, pgap);
		if(pattern != null) {
			String intersect = pattern.getValue();
			ArrayList<String> intersectPattern = pattern.getKey();
			for(int i = 2; i < list.size(); i++) {
				pattern = getLCS(intersect, list.get(i));
				if(pattern != null) {
					intersect = pattern.getValue();
					intersectPattern = pattern.getKey();
				}
				else
					intersect = null;
			}
			if(intersect != null)
				return intersectPattern;
		}
		return null;
	}
}
//	public ArrayList<String> getLCS(ArrayList<String> list, int pxy, int pgap) {
//		if(list.size() < 2)
//			return null;
//
//		if(pattern != null) {
//			String intersect = pattern.getValue();
//			ArrayList<String> intersectPattern = pattern.getKey();
//			for(int i = 2; i < list.size(); i++) {
//				if(i == 199)
//					System.out.print(i + " >>  " + list.get(i) + "\n");
//				pattern = getLCS(intersect, list.get(i));
//				if(pattern != null) {
//					intersect = pattern.getValue();
//					intersectPattern = pattern.getKey();
//				}
//				else
//					intersect = null;
//			}
//			if(intersect != null)
//				return intersectPattern;
//
//		}
//
//		Hirschberg2 hirschberg2 = new Hirschberg2();
//		String str1 = list.get(0);
//		String str2 = list.get(1);
//		Pair<ArrayList<String>, String pattern = hirschberg2.algC(str1.length(), str2.length(), str1, str2);
//		if(pattern != null) {
//			String intersect = pattern.getValue();
//			ArrayList<String> intersectPattern = pattern.getKey();
//			for(int i = 2; i < list.size(); i++) {
//				if(i == 199)
//					System.out.print(i + " >>  " + list.get(i) + "\n");
//				pattern = getLCS(intersect, list.get(i));
//				if(pattern != null) {
//					intersect = pattern.getValue();
//					intersectPattern = pattern.getKey();
//				}
//				else
//					intersect = null;
//			}
//			if(intersect != null)
//				return intersectPattern;
//
//		}
//
//	}


