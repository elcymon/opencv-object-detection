import os

def createTrainvalTxt(baseDirDataSet):
    buffer = ''
    baseDir = baseDirDataSet+'/images'
    for filename in os.listdir(baseDir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        s = 'images/'+filenameOnly+'.jpg'+' '+'labels/'+filenameOnly+'.xml\n'
        print (repr(s))
        img_file, anno = s.strip("\n").split(" ")
        print(repr(img_file), repr(anno))
        buffer+=s
    with open(baseDirDataSet+'/structure/trainval.txt', 'w') as file:
        file.write(buffer)  
    print('Done')   
#Usage
createTrainvalTxt('litter')