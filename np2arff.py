import numpy as np

def write_to_weka(filename,relationname,attributenames,attributes,classes):
    """ writes NumPy arrays with data to WEKA format .arff files

        input: relationname (string with a description), attributenames (list
        of the names of different attributes), attributes (array of attributes,
        one row for each attribute, WEKA treats last row as classlabels by
        default), comment (short description of the content)."""

    nbrattributes = len(attributenames)
    if attributes.shape[1] != nbrattributes:
        raise Exception('Number of attribute names is not equal to length of attributes')

    f = open(filename, 'w')
    f.write('@RELATION '+relationname+'\n')

    for a in attributenames:
        if a == 'Type':
            f.write('@ATTRIBUTE '+str(a)+' {') #assume values are numeric
            for b in range(len(classes)):
                f.write(classes[b]+',')
            f.write('}\n')
        else:
            f.write('@ATTRIBUTE '+str(a)+' NUMERIC\n') #assume values are numeric

    f.write('\n')
    f.write('@DATA\n') #write the data, one attribute vector per line
    for i in range(attributes.shape[0]):
        for j in range(nbrattributes):
            f.write(str(attributes[i,j]))
            if j < nbrattributes-1:
                f.write(', ')
        f.write('\n')
    f.close()
