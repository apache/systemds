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
package org.apache.sysds.runtime.util;

import org.apache.commons.lang3.StringUtils;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

/**
 * Stemmer, implementing the Porter Stemming Algorithm
 *
 * The Stemmer class transforms a word into its root form.  The input
 * word can be provided a character at time (by calling add()), or at once
 * by calling one of the various stem(something) methods.
 */

public class PorterStemmer
{
   /* m() measures the number of consonant sequences between 0 and j. if c is
      a consonant sequence and v a vowel sequence, and <..> indicates arbitrary
      presence,

         <c><v>       gives 0
         <c>vc<v>     gives 1
         <c>vcvc<v>   gives 2
         <c>vcvcvc<v> gives 3
         ....
   */

	private static int calcM(String word)
	{
		int l = word.length() ;
		int count = 0;
		boolean currentConst = false;
		for(int c = 0; c < l; c++) {
			if(cons(word, c))
			{
				if(!currentConst && c != 0) {
					count += 1;
				}
				currentConst = true;
			}
			else
				currentConst = false;

		}
		return  count;
	}

	/* doublec(j) is true <=> j,(j-1) contain a double consonant. */

	private static boolean doublec(String word)
	{  int len = word.length() - 1;
		if (len < 1) return false;
		if (word.charAt(len) != word.charAt(len - 1)) return false;
		return cons(word, len);
	}

   /* cvc(i) is true <=> i-2,i-1,i has the form consonant - vowel - consonant
      and also if the second c is not w,x or y. this is used when trying to
      restore an e at the end of a short word. e.g.

         cav(e), lov(e), hop(e), crim(e), but
         snow, box, tray.
*	*/
	private static boolean cvc(String word)
	{
		int len = word.length();
		int l = len - 1;
		if (len < 3)
			return false;
		if(!cons(word, l) | cons(word, l-1) | !cons(word, (l-2)))
			return false;
		String ch = String.valueOf(word.charAt(l));
		String exceptions = "wxy";
		return !exceptions.contains(ch);
	}

	/* vowelinstem() is true <=> 0,...j contains a vowel */
	private static boolean  vowelinStem(String word, String suffix) {
		int length = word.length() - suffix.length();
		for(int i=0; i<length; i++)
			if(!cons(word, i))
				return true;

		return false;
	}

	/* cons(i) is true <=> b[i] is a consonant. */

	private static boolean cons(String stem, int i)
	{
		String vowels = "aeiou";
		char ch = stem.charAt(i);
		if(vowels.contains(String.valueOf(stem.charAt(i))))
			return false;
		if(ch == 'y')
		{
			if(i == 0)
				return true;
			else
				return (!cons(stem, i - 1));
		}
		return true;
	}
	// process the collection of tuples to find which prefix matches the case.
	private static String processMatched(String word, HashMap<String,String> suffixAndfix, int mCount)
	{
		String stemmed = null;
		Iterator<Entry<String,String>> it = suffixAndfix.entrySet().iterator();
		while (it.hasNext() && (stemmed == null)) {
			Entry<String,String> pair = it.next();
			stemmed = replacer(word, pair.getKey().toString(), pair.getValue(), mCount);
			it.remove();
		}
		return stemmed;
	}

	//	replace the suffix with suggeston
	private static String replacer(String word, String orig, String replace, int mCount)
	{
		int l = word.length();
		int suffixLength = orig.length();

		if (word.endsWith(orig))
		{
			String stem = word.substring( 0, l - suffixLength);
			int  m = calcM( stem );
			if (m > mCount)
				return stem.concat(replace);
			else
				return word;

		}

		return null;
	}

	/* step1() gets rid of plurals and -ed or -ing. e.g.
	i.e., condition & suffix -> replacement
		SSES -> SS
		IES  -> I
		SS -> SS
		S -> ""
		(m > 0) EED -> EE
		vowelSequence(ED) -> ""
		vowelsequence(ING) -> ""
		any("at, bl, iz")  -> add(e)
		doubleconsonant and not("l", "s", "z") -> remove single letter from end
		(m == 1 and cvc) -> add(e)
		turns terminal y to i when there is another vowel in the stem.
   */

	private static String step1(String word)
	{
		boolean flag = false;
		if (word.endsWith("s"))
		{
			if (word.endsWith("sses"))
				word = StringUtils.removeEnd(word, "es");
			else if (word.endsWith("ies")) {
				word = StringUtils.removeEnd(word, "ies").concat("i");
			}
			else if (!word.endsWith("ss") && word.endsWith("s"))
				word = StringUtils.removeEnd(word, "s");
		}
		if (word.endsWith("eed"))
		{
			if (calcM(word) > 1)
				word = StringUtils.removeEnd(word, "d");
		}
		else if(word.endsWith("ed") && vowelinStem(word, "ed")) {
			word = StringUtils.removeEnd(word, "ed");
			flag = true;
		}
		else if(word.endsWith("ing") && vowelinStem(word, "ing"))
		{
			word = StringUtils.removeEnd(word, "ing");

			flag = true;
		}

		if (flag)
		{
			if(word.endsWith("at") || word.endsWith("bl") || word.endsWith("iz"))
				word = word.concat("e");
			int m = calcM(word);
			String last = String.valueOf(word.charAt(word.length() - 1));
			if (doublec(word) && !"lsz".contains(last))
				word = word.substring(0, word.length() - 1);
			else if (m == 1 && cvc(word))
				word = word.concat("e");
		}
		if (word.endsWith("y") && vowelinStem(word, "y"))
			word = StringUtils.removeEnd(word, "y").concat("i");

		return word;
	}

	// step2() maps double suffices to single ones

	private static String step2(String word) {
		int len = word .length();
		if (len == 0) return word;
		HashMap<String, String> suffixAndfix = new HashMap<>();
		suffixAndfix.put("ational", "ate");
		suffixAndfix.put("tional","tion");
		suffixAndfix.put("enci","ence");
		suffixAndfix.put("anci","ance");
		suffixAndfix.put("izer","ize");
		suffixAndfix.put("bli","ble");
		suffixAndfix.put("alli", "al");
		suffixAndfix.put("entli","ent");
		suffixAndfix.put("eli","e");
		suffixAndfix.put("ousli","ous");
		suffixAndfix.put("ization","ize");
		suffixAndfix.put("ation","ate");
		suffixAndfix.put("ator","ate");
		suffixAndfix.put("alism","al");
		suffixAndfix.put("iveness", "ive");
		suffixAndfix.put("fulness","ful");
		suffixAndfix.put("ousness", "ous");
		suffixAndfix.put("aliti", "al");
		suffixAndfix.put("iviti","ive");
		suffixAndfix.put("biliti", "ble");
		suffixAndfix.put("log", "logi");
		suffixAndfix.put("icate", "ic");
		suffixAndfix.put("ative","");
		suffixAndfix.put("alize","al");
		suffixAndfix.put("iciti","ic");
		suffixAndfix.put("ical","ic");

		String stemmed = processMatched(word, suffixAndfix, 0);
		return (stemmed != null)? stemmed: word;
	}
	// handles -ic-, -full, -ness etc.
	private static String step3(String word) {
		int len = word .length();
		if (len == 0) return word;
		HashMap<String, String> suffixAndfix = new HashMap<>();
		suffixAndfix.put("icate", "ic");
		suffixAndfix.put("ative","");
		suffixAndfix.put("alize","al");
		suffixAndfix.put("iciti","ic");
		suffixAndfix.put("ical","ic");
		suffixAndfix.put("ful","");
		suffixAndfix.put("ness","");

		String stemmed = processMatched(word, suffixAndfix, 0);
		return (stemmed != null)? stemmed: word;

	}

	// takes off -ant, -ence etc., in context <c>vcvc<v>
	private static String step4(String word)
	{
		// first part.
		String[] suffix = new String[] {"al", "ance", "ence", "er", "ic", "able", "ible", "ant",
			"ement", "ment", "ent"};
		String stemmed = null;
		int i = 0;
		while(stemmed == null && i < suffix.length)
		{
			stemmed = replacer(word, suffix[i], "", 1);
			i++;
		}
		// exceptions
		if(stemmed == null)
		{
			if(word.length() > 4)
			{
				char ch = word.charAt(word.length() - 4);
				if(ch == 's' || ch == 't')
				{
					stemmed = replacer(word, "ion", "", 1);
				}
			}
		}
		// exceptions
		if (stemmed == null)
		{
			suffix = new String[] {"ou",  "ism", "ate", "iti", "ous", "ive", "ize"};
			i = 0;
			while(stemmed == null && i < suffix.length)
			{
				stemmed = replacer(word, suffix[i], "", 1);
				i++;
			}
		}

		return (stemmed != null)? stemmed: word;
	}
	//	handle the last e and l
	private static String step5(String word)
	{
		String stem = StringUtils.removeEnd(word, "e");
		if(word.endsWith("e") && calcM(word) > 1)
			word =  stem;
		if(word.endsWith("e") && calcM(word)  == 1 && !cvc(stem))
			word = stem;
		if(word.endsWith("l") && doublec(word) && calcM(word) > 1)
			word = word.substring(0, word.length() - 1);

		return word;
	}
	public static String stem (String word)
	{
		if(word.length() >= 3) {
			word = step1(word);
			word = step2(word);
			word = step3(word);
			word = step4(word);
			if(word.length() > 0)
				word = step5(word);
		}
		return word;
	}
}