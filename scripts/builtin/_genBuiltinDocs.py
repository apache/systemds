#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

#-------------------------------------------------------------
# This python script does take all files within the current directory and creates a corresponding markdown file out of the existing headers
#   If there are specific parts of the header missing a warning within the terminal will be displayed
#   The finished markdown file is then placed in the same directory
#-------------------------------------------------------------
# OUTPUTS
#-------------------------------------------------------------
#   Name                            Type    Default     Meaning
#-------------------------------------------------------------
#   SystemDs_Builtin_Markdown       .md    -------     A .md file of all files with a parsable header
#-------------------------------------------------------------

import re
import os
from mdutils.mdutils import MdUtils

file_data_array = [] #Contains all valid header information

incorrect_file_data = [] #Contains all file names with invalid header

ALLOW_DIFFERENT_SIZED_HEADERS = False #Global FLAG to identify if Headers with only 3 arguments are valid or not

THROW_EXCEPTION = False #Global FLAG to throw Exception on existing warnings - If set True no md file will be created unless all files have valid headers!


#This class contains all necessary variables for the complete representation of the markdown file
class File_data:
    def __init__(self, inputParameterCount, inputParams, I_HEADER_MISSING, I_DESCRIPTION_MISSING, outputParameterCount, outputParams, O_HEADER_MISSING, O_DESCRIPTION_MISSING, description, additional, fileName):
        self.param_count = inputParameterCount
        self.param = inputParams
        self.input_header_missing = I_HEADER_MISSING
        self.input_description_missing = I_DESCRIPTION_MISSING
        self.output_param_count = outputParameterCount
        self.output = outputParams
        self.output_header_missing = O_HEADER_MISSING
        self.output_description_missing = O_DESCRIPTION_MISSING
        self.description = description
        self.additional = additional
        self.fileName = fileName


#class to make colored output possible
class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'


#This functions simply iterates over all files, given at the specific location and creates the corresponding markdown files
def callWithAbsoluteParameters():
    directory = os.path.dirname(os.path.realpath(__file__))

    for entry in os.scandir(directory):
        if entry.path.endswith(".dml") and entry.is_file():
            with open(entry.path) as f:
                parseFile(f.readlines(), f.name)

    cleaned_array = cleanFileDataArray()
    createMarkDownFile(cleaned_array)


#Function to start parsing the file, it skips the lincence agreement and then forwards the rest to the parse Information Function
def parseFile(lines, fileName):
    fileName = os.path.basename(fileName)
    fileName = fileName.split('.')[0]

    EMPTY_LINES = True
    markDownArray = []
    license_break_count = 0

    lineBreakRegex = re.compile('#.[\-]+')  # Match all lines which look like #---- or # ---
    emptyLineRegex = re.compile('^\s*$') # Matches all empty lines within the header
    emptyCommentLineRegex = re.compile('^#\s{1,}$')  # Match lines whick look like this # followed by abritrary number of spaces
    commentLineRegex = re.compile('^#.*$') #Matches any line with a # in front
    functionStartRegex = re.compile('^[A-Za-z_]*\s*=+') #Because we know that the start of the function name must be within the range of a-Z we can simply search for it to find the start of the program

    for line in lines:
        if lineBreakRegex.match(line) and license_break_count < 2:  #Skip Apache License info
            license_break_count += 1
            continue
        elif(license_break_count < 2):   #skip all lines within the apache license header
            continue
        elif(license_break_count >= 2) and EMPTY_LINES and (emptyLineRegex.match(line) or emptyCommentLineRegex.match(line)):
            continue
        elif functionStartRegex.match(line):
            break
        elif bool(commentLineRegex.match(line)) is False:
            continue
        else:
            EMPTY_LINES = False
            markDownArray.append(line)

    if len(markDownArray) > 0: #If our markDownArray is empty we append the name to the incorrect file_data_array
        parseInformation(markDownArray, fileName)
    else:
        incorrect_file_data.append(fileName)


def parseInformation(markDownArray, fileName):
    markDownArray = [e[1:] for e in markDownArray]  # First I cut the first character of all lines because it is simply a # character

    INPUT = False #Then I start to set some global flags to identify the status of my program
    INPUT_FINISHED = False
    OUTPUT = False
    OUTPUT_FINISHED = False
    DESCRIPTION = True  #When I start parsing the File, the first line(s) should ALWAYS contain the description therefore I start with True
    commentLineCount = 0 #To track how many comment_lines where found inside the headerRegex
    additionalInfos = [] #If there are information which do not match the syntax or are not identifyable it is put inside this array
    description = [] #contains the description of the corresponding file
    I_DESCRIPTION_MISSING = False
    I_HEADER_MISSING = False
    inputParameterCount = 0
    inputParameters = []
    O_DESCRIPTION_MISSING = False
    O_HEADER_MISSING = False
    outputParameterCount = 0
    outputParameters = []

    commentLineRegex = re.compile('^#\s{0,}$')
    emptyLineRegex = re.compile('^\s*$')
    headerRegex = re.compile('\s{1,}[A-Za-z]{2,}')
    inputStringRegex = re.compile('^#*\s{1,}input', flags=re.I)
    outputStringRegex = re.compile('^#*\s{0,}(output|return)', flags=re.I)

    for line in markDownArray:
        if INPUT:
            inputParameterCount, INPUT, INPUT_FINISHED, I_HEADER_MISSING, commentLineCount = parseParam(inputParameters, inputParameterCount, INPUT, INPUT_FINISHED, I_HEADER_MISSING, commentLineCount, additionalInfos, line)

        if OUTPUT:
            outputParameterCount, OUTPUT, OUTPUT_FINISHED, O_HEADER_MISSING, commentLineCount = parseParam(outputParameters, outputParameterCount, OUTPUT, OUTPUT_FINISHED, O_HEADER_MISSING, commentLineCount, additionalInfos, line)

        if headerRegex.match(line) and DESCRIPTION:
            if inputStringRegex.match(line):
                DESCRIPTION = False
                INPUT = True
                continue

            description.append(line)
        else:
            DESCRIPTION = False

        if emptyLineRegex.match(line) or commentLineRegex.match(line):
            continue

        if inputStringRegex.match(line) and INPUT_FINISHED == False:
            INPUT = True

        if outputStringRegex.match(line) and OUTPUT_FINISHED is False:
            OUTPUT = True

    file_data_array.append(File_data(inputParameterCount, inputParameters, I_HEADER_MISSING, I_DESCRIPTION_MISSING, outputParameterCount, outputParameters, O_HEADER_MISSING, O_DESCRIPTION_MISSING, description, additionalInfos, fileName))


def createMarkDownFile(file_array):
    mdFile = MdUtils(file_name='_genBuiltinDocs_Out', title='Markdown Files for scripts')
    mdFile.new_header(level=1, title='Overview')  # style is set 'atx' format by default.

    for entry in file_array:
        inputParameters = entry.param
        outputParameters = entry.output
        inputParameterCount = entry.param_count
        outputParameterCount = entry.output_param_count
        description = entry.description
        additionalInfos = entry.additional
        fileName = entry.fileName

        #First I strip all the newline characters from the end of each string and replace the newlines within strings wit spaces
        inputParameters = replaceNewlineCharacter(inputParameters)
        outputParameters = replaceNewlineCharacter(outputParameters)
        description = replaceNewlineCharacter(description)

        titleString = fileName + "-Function"
        mdFile.new_header(level=2, title=titleString)  # style is set 'atx' format by default.

        for line in description:
            mdFile.new_paragraph(line)

        mdFile.new_header(level=3, title="Usage")
        usage = fileName + "("

        for argument in inputParameters[4::4]:
            usage += (argument + ", ")

        if len(inputParameters) > 0:
            usage = usage[:-2] + ')'
        else:
            usage = usage + ')'

        mdFile.insert_code(usage, language='python')
        mdFile.new_header(level=3, title="Arguments")

        if inputParameterCount > 0:
            rows = int(len(inputParameters)/inputParameterCount)
            mdFile.new_table(columns=inputParameterCount, rows=rows, text=inputParameters, text_align='center')

        mdFile.new_header(level=3, title="Returns")

        if outputParameterCount > 0:
            rows = int(len(outputParameters) / outputParameterCount)
            mdFile.new_table(columns=outputParameterCount, rows=rows, text=outputParameters, text_align='center')
        else:
            for line in additionalInfos:
                mdFile.new_paragraph(line)

    mdFile.create_md_file()


def cleanFileDataArray():
    #Here I go through all elements of the file_data_array check if everything is correct and if not create a file  or meaningfull output
    new_file_data_array = []
    missingValues = False
    for entry in file_data_array:
        missingParts = False
        missingParameterString = ''
        inputParameterCount = entry.param_count
        I_HEADER_MISSING = entry.input_header_missing
        I_DESCRIPTION_MISSING = entry.input_description_missing
        outputParameterCount = entry.output_param_count
        O_HEADER_MISSING = entry.output_header_missing
        O_DESCRIPTION_MISSING = entry.output_description_missing
        description = entry.description
        fileName = entry.fileName

        if len(description) == 0:
            missingParts = True
            missingParameterString += "There is no description of the function available!\n"

        if I_HEADER_MISSING is True:
            missingParts = True
            missingParameterString += "There is either no input parameter header given or the syntax is incorrect!\n"

        if I_DESCRIPTION_MISSING is True:
            missingParts = True
            missingParameterString += "There is either no dedicated Input section header or syntax is incorrect!\n"

        if inputParameterCount == 0:
            missingParts = True
            missingParameterString += "There are either no parameters given within the file or syntax is incorrect!\n"

        if O_DESCRIPTION_MISSING is True:
            missingParts = True
            missingParameterString += "There is either no dedicated Output section header or syntax is incorrect!\n"

        if O_HEADER_MISSING is True:
            missingParts = True
            missingParameterString += "There is either no output parameter header given or the syntax is incorrect!\n"

        if outputParameterCount == 0:
            os.system('color')
            missingParts = True
            missingParameterString += "There are either no output parameters given within the file or syntax is incorrect!\n"

        if missingParts:
            missingValues = True
            os.system('color')
            print(f'{bcolors.WARNING}For the File: {bcolors.UNDERLINE}{bcolors.OKGREEN}' +  fileName + f'{bcolors.ENDC}{bcolors.WARNING} following errors occured: {bcolors.ENDC}')
            print(missingParameterString)
        else:
            new_file_data_array.append(entry)

    for entry in incorrect_file_data:
        os.system('color')
        print(f'{bcolors.WARNING}For the File: {bcolors.UNDERLINE}{bcolors.OKGREEN}' + entry + f'{bcolors.ENDC}{bcolors.WARNING} no header was found at all!{bcolors.ENDC}')

    if len(incorrect_file_data) > 0:
        missingValues = True

    if missingValues and THROW_EXCEPTION is True:
        raise Exception(f'{bcolors.FAIL}At least one file does not fit the required Syntax!{bcolors.ENDC}')

    return new_file_data_array


def parseParam(param_array, param_count, FLAG, FLAG_FIN, FLAG_HEADER_MISSING, comment_line_count, additional_infos, line):
    lineBreakRegex = re.compile('.[\-]+')  # Match all lines which look like ---- or  ---
    commentLineRegex = re.compile('^#\s{0,}$')
    emptyLineRegex = re.compile('^\s*$')
    outputStringRegex = re.compile('^#*\s{0,}(output|return)', flags=re.I)  # #\s{1,}.*input  <-- other possibility to find all input fields, problem here that it matches all lines with input in the test
    simpleWordRegex = re.compile('[A-Za-z]+')
    continuedParamDescribtionRegex = re.compile('^#+\s{10,}')  # if my parameter input matches this i obviously have a continuation of the param describtion
    arbritraryNumberOfSpacesRegex = re.compile('\s{1,}')

    if lineBreakRegex.match(line):
        comment_line_count += 1
        if comment_line_count > 2:
            FLAG = False
            comment_line_count = 0
        return param_count, FLAG, FLAG_FIN, FLAG_HEADER_MISSING, comment_line_count

    if comment_line_count == 1:
        # current line should only contain parameter names
        # In case there are no parameter names given but only the parameters, we set the parameter header missing Flag
        if len(arbritraryNumberOfSpacesRegex.split(line)) >= 7:
            FLAG_HEADER_MISSING = True
            comment_line_count += 1
            param_count = 4
            return param_count, FLAG, FLAG_FIN, FLAG_HEADER_MISSING, comment_line_count

        for parameter in line.split(' '):
            if simpleWordRegex.match(parameter):
                param_array.append(parameter)
                param_count += 1

        #Here I check if there are headers with a different amount of parameters if yes and the global flag is set to false I return a Warning
        if param_count < 4 and ALLOW_DIFFERENT_SIZED_HEADERS is False:
            FLAG_HEADER_MISSING = True
            return param_count, FLAG, FLAG_FIN, FLAG_HEADER_MISSING, comment_line_count

    if comment_line_count == 2:
        if continuedParamDescribtionRegex.match(line):
            splitted_line = continuedParamDescribtionRegex.split(line)
            if (len(splitted_line)) == 2:
                param_array[-1] += splitted_line[1]
        elif commentLineRegex.match(line) or emptyLineRegex.match(line):
            FLAG_FIN = True
            FLAG = False
            comment_line_count = 0
        else:
            splitParameterString(line, param_array, param_count)
    else:
        additional_infos.append(line)

    if (outputStringRegex.match(line) or commentLineRegex.match(line)) and len(param_array) > 1:
        FLAG = False
        FLAG_FIN = True
        comment_line_count = 0

    return param_count, FLAG, FLAG_FIN, FLAG_HEADER_MISSING, comment_line_count


def splitParameterString(line, array, size):
    arbritraryNumberOfSpacesRegex = re.compile('\s{1,}')
    #I know that the first occurence of [A-Za-z]\s is the first Parameter so I can split here
    new_line = arbritraryNumberOfSpacesRegex.split(line, size)
    if len(new_line) > size:
        for i in range(size):
            array.append(new_line[i+1])


def replaceNewlineCharacter(array):
    array = list(map(str.strip, array))
    array = [s.replace("\n", " ") for s in array]
    return array


if __name__ == '__main__':
    callWithAbsoluteParameters()