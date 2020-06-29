import random, codecs

def split_file(file,out1,out2,percentage=0.75,isShuffle=True,seed=42):

    random.seed(seed)
    with codecs.open(file, 'r', "utf-8") as fin, \
         codecs.open(out1, 'w', "utf-8") as foutBig, \
         codecs.open(out2, 'w', "utf-8") as foutSmall:
        nLines = sum(1 for line in fin)
        fin.seek(0)

        nTrain = int(nLines*percentage) 
        nValid = nLines - nTrain

        i = 0
        for line in fin:
            r = random.random() if isShuffle else 0 # so that always evaluated to true when not isShuffle
            if (i < nTrain and r < percentage) or (nLines - i > nValid):
                foutBig.write(str(line))
                i += 1
            else:
                foutSmall.write(str(line))
                
split_file("./yeast/yeast.txt", 
           "./yeast/yeast_train.txt",
           "./yeast/yeast_val.txt",
           percentage=0.9)