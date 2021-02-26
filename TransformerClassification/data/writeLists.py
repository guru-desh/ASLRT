import glob

def writeList(arkLoc: str, listFile: str, listLabel: str):
    files = glob.glob(arkLoc)
    listF = open(listFile, "w")
    allFiles = "\n".join(files)
    listF.write(allFiles)
    fileNames = [" ".join(i.split("/")[-1].split(".")[1].split("_"))+"\n" for i in files]
    listL = open(listLabel, "w")
    listL.writelines(fileNames)


writeList("alphapose/ark/*", "alphapose/lists/train.data", "alphapose/lists/train.en")
writeList("alphapose/ark/*", "alphapose/lists/dev.data", "alphapose/lists/dev.en")
writeList("alphapose/ark/*", "alphapose/lists/test.data", "alphapose/lists/test.en")
