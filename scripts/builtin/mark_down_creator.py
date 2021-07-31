import re
import os
import sys
from mdutils.mdutils import MdUtils

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# This python script does take all files within the current directory and creates a corresponding markdown file out of the existing headers
#   If there are specific parts of the header missing a warning within the terminal will be displayed
#   The finished markdown file is then placed in the same directory
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# OUTPUTS
# -----------------------------------------------------------------------------------------------------------------------------------------------------
#   Name                            Type    Default     Meaning
# -----------------------------------------------------------------------------------------------------------------------------------------------------
#   SystemDs_Builtin_Markdown       .dm     -------     A .dm file of all files with a parsable header
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Additional Information
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# If the output within the terminal does not look correct because the os isnt supporting colored output,
#   please swap the comment prefix in line 300 and 301




#This Array contains all valid information
file_data_array = []

#This Array contains all invalid file names with corresponding missing values
incorrect_file_data = []

#Global FLAG to identify if Headers with only 3 arguments are valid or not!
ALLOW_DIFFERENT_SIZED_HEADERS = False



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

#class to make a pretty output possible
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
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

    lineBreakRegex = re.compile('#.[\-]+')  # Match all lines which look like #---- or # ---
    emptyLineRegex = re.compile('^\s*$') # Matches all empty lines within the header
    emptyCommentLineRegex = re.compile('^#\s{1,}$')  # Match lines whick look like this # followed by abritrary number of spaces
    commentLineRegex = re.compile('^#.*$') #Matches any line with a # in front
    functionStartRegex = re.compile('^[A-Za-z_]*\s*=+') #Because we know that the start of the function name must be within the range of a-Z we can simply search for it to find the start of the program

    license_break_count = 0

    for line in lines:
        if lineBreakRegex.match(line) and license_break_count < 2:  #First we skip the apache license info
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

    #If we don´t have anything in our markDownArray we don´t need to continue
    if len(markDownArray) > 0:
        parseInformation(markDownArray, fileName)
    else:
        incorrect_file_data.append(fileName)

def parseInformation(markDownArray, fileName):
    markDownArray = [e[1:] for e in markDownArray]  # First I cut the first character of all lines because it is simply a # character

    #Then I start to set some global flags to identify the status of my program
    #As soon as INPUT_FINISHED is True I know that I already parsed some inputStringRegex parameters
    INPUT = False
    INPUT_FINISHED = False

    OUTPUT = False
    OUTPUT_FINISHED = False

    #When I start parsing the File, the first line(s) should ALWAYS contain the description therefore I start with True
    DESCRIPTION = True

    #To track how many comment_lines where found inside the headerRegex
    commentLineCount = 0

    #If there are information which do not match the syntax or are not identifyable I put it inside this array
    additionalInfos = []

    #contains the description of the corresponding file
    description = []

    #Number of parameters and array of parameters
    I_DESCRIPTION_MISSING = False
    I_HEADER_MISSING = False
    inputParameterCount = 0
    inputParameters = []

    # Number of outputs and array of parameters
    O_DESCRIPTION_MISSING = False
    O_HEADER_MISSING = False
    outputParameterCount = 0
    outputParameters = []


    commentLineRegex = re.compile('^#\s{0,}$')
    emptyLineRegex = re.compile('^\s*$')
    headerRegex = re.compile('\s{1,}[A-Za-z]{2,}')
    inputStringRegex = re.compile('^#*\s{1,}input', flags=re.I)  #\s{1,}.*input  <-- other possibility to find all input fields, problem here that it matches all lines with input in the test
    outputStringRegex = re.compile('^#*\s{0,}(output|return)', flags=re.I)

    for line in markDownArray:
        if INPUT:
            inputParameterCount, INPUT, INPUT_FINISHED, I_HEADER_MISSING, commentLineCount = parseParam(inputParameters, inputParameterCount, INPUT, INPUT_FINISHED, I_HEADER_MISSING, commentLineCount, additionalInfos, line)


        if OUTPUT:
            outputParameterCount, OUTPUT, OUTPUT_FINISHED, O_HEADER_MISSING, commentLineCount = parseParam(outputParameters, outputParameterCount, OUTPUT, OUTPUT_FINISHED, O_HEADER_MISSING, commentLineCount, additionalInfos, line)

        #First we care about the headerRegex, if everything worked out only the headerRegex information should be left
        if headerRegex.match(line) and DESCRIPTION:
            if inputStringRegex.match(line):
                DESCRIPTION = False
                INPUT = True
                continue
            description.append(line)
        else:
            DESCRIPTION = False

        #Within the parsing process I want to skip all empty lines or simple comment_lines
        if emptyLineRegex.match(line) or commentLineRegex.match(line):
            continue

        if inputStringRegex.match(line) and INPUT_FINISHED == False:
            INPUT = True

        if outputStringRegex.match(line) and OUTPUT_FINISHED is False:
            OUTPUT = True

    file_data = File_data(inputParameterCount, inputParameters, I_HEADER_MISSING, I_DESCRIPTION_MISSING, outputParameterCount, outputParameters, O_HEADER_MISSING, O_DESCRIPTION_MISSING, description, additionalInfos, fileName)
    file_data_array.append(file_data)


def createMarkDownFile(file_array):
    mdFile = MdUtils(file_name='SystemDs_Builtin_Markdown', title='Markdown Files for scripts')
    mdFile.new_header(level=1, title='Overview')  # style is set 'atx' format by default.

    for entry in file_array:
        #First I gather all Parameters from the Entry
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
        #usage would come next
        mdFile.new_header(level=3, title="Usage")

        #Creating the UsageString
        usage = fileName + "("

        for argument in inputParameters[4::4]:
            usage += (argument + ", ")

        if len(inputParameters) > 0:
            usage = usage[:-2] + ')'
        else:
            usage = usage + ')'

        mdFile.insert_code(usage, language='python')

        #arguments
        mdFile.new_header(level=3, title="Arguments")

        if inputParameterCount > 0:
            rows = int(len(inputParameters)/inputParameterCount)
            mdFile.new_table(columns=inputParameterCount, rows=rows, text=inputParameters, text_align='center')


        #Returns
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
    for entry in file_data_array:
        missingValues = False
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
            missingValues = True
            missingParameterString += "There is no description of the function available!\n"

        if I_HEADER_MISSING is True:
            missingValues = True
            missingParameterString += "There is either no input parameter header given or the syntax is incorrect!\n"

        if I_DESCRIPTION_MISSING is True:
            missingValues = True
            missingParameterString += "There is either no dedicated Input section header or syntax is incorrect!\n"

        if inputParameterCount == 0:
            missingValues = True
            missingParameterString += "There are either no parameters given within the file or syntax is incorrect!\n"

        if O_DESCRIPTION_MISSING is True:
            missingValues = True
            missingParameterString += "There is either no dedicated Output section header or syntax is incorrect!\n"

        if O_HEADER_MISSING is True:
            missingValues = True
            missingParameterString += "There is either no output parameter header given or the syntax is incorrect!\n"

        if outputParameterCount == 0:
            missingValues = True
            missingParameterString += "There are either no output parameters given within the file or syntax is incorrect!\n"

        if missingValues:
            print('\033[93m' + "For the File: " + '\033[4m\033[92m' + fileName + '\033[0m' + '\033[93m' + " following errors occured:" + '\033[0m')
            print(missingParameterString)
        else:
            new_file_data_array.append(entry)

    for entry in incorrect_file_data:
        print(f'{bcolors.WARNING}For the File: {bcolors.UNDERLINE}{bcolors.OKGREEN}' + entry + f'{bcolors.ENDC}{bcolors.WARNING} NO HEADER WAS FOUND AT ALL:{bcolors.ENDC}')
        #print('For the File: ' + entry + ' NO HEADER WAS FOUND AT ALL') # Print without color

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
        # case parameter name if there are some
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
        # now I have to parse my parameters
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

