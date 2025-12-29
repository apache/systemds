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

package org.apache.sysds.runtime.einsum;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class EinsumContext {
    public Integer outRows;
    public Integer outCols;
    public Character outChar1;
    public Character outChar2;
    public HashMap<Character, Integer> charToDimensionSize;
    public String equationString;
    public ArrayList<String> newEquationStringInputsSplit;
	public HashMap<Character, Integer> characterAppearanceCount;

    private EinsumContext(){};
    public static EinsumContext getEinsumContext(String eqStr, ArrayList<MatrixBlock> inputs){
        EinsumContext res = new EinsumContext();

        res.equationString = eqStr;
		HashMap<Character, Integer> charToDimensionSize = new HashMap<>();
        HashMap<Character, Integer> characterAppearanceCount = new HashMap<>();
		ArrayList<String> newEquationStringSplit = new ArrayList<>();
		Character outChar1 = null;
		Character outChar2 = null;

        Iterator<MatrixBlock> it = inputs.iterator();
        MatrixBlock curArr = it.next();
        int i = 0;

		char c = eqStr.charAt(i);
		for(i = 0; i < eqStr.length(); i++) {
			StringBuilder sb = new StringBuilder(2);
			for(;i < eqStr.length(); i++){
				c = eqStr.charAt(i);
				if  (c == ' ') continue;
				if  (c == ',' || c == '-' ) break;
				if (!Character.isAlphabetic(c)) {
					throw new RuntimeException("Einsum: only alphabetic characters are supported for dimensions: "+c);
				}
				sb.append(c);
				if (characterAppearanceCount.containsKey(c)) characterAppearanceCount.put(c, characterAppearanceCount.get(c) + 1) ;
				else characterAppearanceCount.put(c, 1);
			}
			String s = sb.toString();
			newEquationStringSplit.add(s);

			if(s.length() > 0){
				if (charToDimensionSize.containsKey(s.charAt(0)))
					if (charToDimensionSize.get(s.charAt(0)) != curArr.getNumRows())
						throw new RuntimeException("Einsum: character "+c+" has multiple conflicting sizes");
				charToDimensionSize.put(s.charAt(0), curArr.getNumRows());
			}
			if(s.length() > 1){
				if (charToDimensionSize.containsKey(s.charAt(1)))
					if (charToDimensionSize.get(s.charAt(1)) != curArr.getNumColumns())
						throw new RuntimeException("Einsum: character "+c+" has multiple conflicting sizes");
				charToDimensionSize.put(s.charAt(1), curArr.getNumColumns());
			}
			if(s.length() > 2) throw new RuntimeException("Einsum: only up-to 2D inputs strings allowed ");

			if( c==','){
				curArr = it.next();
			}
			else if (c=='-') break;

			if (i == eqStr.length() - 1) {throw new RuntimeException("Einsum: missing '->' substring "+c);}
		}

		if (i == eqStr.length() - 1 || eqStr.charAt(i+1) != '>') throw new RuntimeException("Einsum: missing '->' substring "+c);
		i+=2;

		StringBuilder sb = new StringBuilder(2);

		for(;i < eqStr.length(); i++){
			c = eqStr.charAt(i);
			if  (c == ' ') continue;
			if (!Character.isAlphabetic(c)) {
				throw new RuntimeException("Einsum: only alphabetic characters are supported for dimensions: "+c);
			}
			sb.append(c);
		}
		String s = sb.toString();
		if(s.length() > 0) outChar1 = s.charAt(0);
		if(s.length() > 1) outChar2 = s.charAt(1);
		if(s.length() > 2) throw new RuntimeException("Einsum: only up-to 2D output allowed ");

		res.outRows=(outChar1 == null ? 1 : charToDimensionSize.get(outChar1));
		res.outCols=(outChar2 == null ? 1 : charToDimensionSize.get(outChar2));

        res.outChar1 = outChar1;
        res.outChar2 = outChar2;
        res.newEquationStringInputsSplit = newEquationStringSplit;
		res.characterAppearanceCount = characterAppearanceCount;
		res.charToDimensionSize = charToDimensionSize;
        return res;
    }
}
