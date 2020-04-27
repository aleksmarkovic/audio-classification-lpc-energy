#import matplotlib as mtl
import matplotlib.pyplot as plt
#import seaborn
#from IPython.display import Audio
import numpy as np
#import librosa
#import librosa.display
import pysptk
import sox
import os
import scipy
from scipy.io import wavfile
import soundfile as sf
import wave
from pathlib import Path
from scipy.spatial import distance
import math


#PRIMJER
#def primjer():
#    seaborn.set(style="whitegrid")
#    mtl.rcParams['figure.figsize'] = (16, 5)
#    
#    #sr, x = wavfile.read(pysptk.util.example_audio_file())
#    sr, x = wavfile.read("sm04010103201.wav")
#    assert sr == 16000
#    x = x.astype(np.float64)
#    print(x.shape)
#    librosa.display.waveplot(x, sr=sr)
#    #title("Raw waveform of example audio flle")
#    Audio(x, rate=sr)




def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def CutVoices(fname):
    nVoices = 1
    M = open(fname + ".lab","r").readlines()
    content=[x.strip('\n') for x in M]
    numOfLines=file_len(fname + ".lab")
    
    for i in range(0, numOfLines):
        content[i]=content[i].split(" ")
    print(content)

    for i in range(0, numOfLines):
        tfm=sox.Transformer()
        tfm.trim(float(content[i][0])*10**-7, float(content[i][1])*10**-7)
        Path("RawVoices/" + content[i][2]).mkdir(parents = True, exist_ok = True)
        tfm.build(fname + ".wav", "RawVoices/" + content[i][2] + "/" + content[i][2] + "-" + str(nVoices) + ".raw")
        nVoices = nVoices + 1
    return


def ToRaw(fl):
    dirs = []
    f = []
    os.system("sox " + fl + " " + fl + ".raw")

    for (dirpath, dirnames, filenames) in os.walk("Voices/"):
        dirs.extend(dirnames)
        break
    for folder in dirs:
        for (dirpath, dirnames, filenames) in os.walk("Voices/" + folder + "/"):
            Path("RawVoices/" + folder).mkdir(parents = True, exist_ok = True)
#            data, samplerate = sf.read('sm04010103201.wav')
#            f.extend(filenames)
            for filename in filenames:
                print(filename)
                os.system("sox Voices/" + folder + "/" + filename + " RawVoices/" + folder + "/" + filename[:4] + ".raw")
                #data, samplerate = sf.read("Voices/" + folder + "/" + filename)
                #sf.write("RawVoices/" + folder + "/" + filename + ".raw", data, samplerate=44100, subtype='FLOAT')

    results = {}
#    print(len(f))
#    for sound in f:
#        print(sound)
    #data, samplerate = sf.read('sm04010103201.wav')
    #sf.write('sm04010103201.ogg', data, samplerate)
   # sf.write('sm04010103201.raw', data, samplerate=44100, subtype='FLOAT')
  # sox.transform() BILO BI DOBRO SA SOX-om
  

def lpcCoeff():
    for (dirpath, dirnames, filenames) in os.walk("Voices/"):
        for dirname in dirnames:
            for filename in os.listdir("Voices/" + dirname + "/"):
                #Prvo gre za sve wav-ove
                Path("LPCCoeff/Wav/" + dirname).mkdir(parents = True, exist_ok = True)         
                filenamePath = "Voices/" + dirname + "/" + filename
                print(filenamePath)
                os.system("frame -l 320 -p 160 " + filenamePath + " | window -l 320 -L 512 |" + " lpc -l 512 -m 12 -f 0.00000001" + " | lpc2c -m 12 -M 12" + " | vstat -l 13 -o 1" + " | x2x +fa12 > LPCCoeff/Wav/" + dirname + "/" + filename[:4] + ".lpc.txt")
    
    for (dirpath, dirnames, filenames) in os.walk("RawVoices/"):
        for dirname in dirnames:
            for filename in os.listdir("RawVoices/" + dirname + "/"): 
               #svaki segment raw
                Path("LPCCoeff/RawSegments/" + dirname).mkdir(parents = True, exist_ok = True)    
                filenamePath = "RawVoices/" + dirname + "/" + filename
#                print(filenamePath)
                os.system("x2x +sf " + filenamePath + " | frame -l 320 -p 160 " + filenamePath + " | window -l 320 -L 512 |" + " lpc -l 512 -m 12 -f 0.00000001" + " | lpc2c -m 12 -M 12" + " | vstat -l 13 -o 1" + " | x2x +fa13 > LPCCoeff/RawSegments/" + dirname + "/" + filename[:4] + ".lpc.txt")
#                os.system("x2x +sf " + filenamePath + " | frame -l 320 -p 160 "  + " | window -l 320 -L 512 |" + " lpc -l 512 -m 12 -f 0.00000001" + " | lpc2c -m 12 -M 12" + " | x2x +fa13 > LPCCoeff/RawSegments/" + dirname + "/" + filename[:4] + ".lpc.txt")

def lpcCoeffSingle(filename):
    Path("LPCCoeff/RawSingle/").mkdir(parents = True, exist_ok = True)     
    os.system("x2x +sf " + filename + " | frame -l 320 -p 160 "  + " | window -l 320 -L 512 |" + " lpc -l 512 -m 12 -f 0.00000001" + " | lpc2c -m 12 -M 12" + " | x2x +fa13 > LPCCoeff/RawSingle/"  + filename + ".lpc.txt")

#    PO POLONIJU TO JE ENERGIJA
def CalcWindow():
    for (dirpath, dirnames, filenames) in os.walk("RawVoices/"):
        for dirname in dirnames:
            for filename in os.listdir("RawVoices/" + dirname + "/"): 
               #svaki segment raw
                Path("Window/" + dirname).mkdir(parents = True, exist_ok = True)    
                filenamePath = "RawVoices/" + dirname + "/" + filename
#                print(filenamePath)
                os.system("frame -l 320 -p 320 " + filenamePath + " | window -l 320" + " | x2x +fa > Window/" + dirname + "/" + filename[:4] + ".window")
#                os.system("x2x +sf " + filenamePath + " | frame < " + filenamePath + " | window | " + "lpc -m 20" + " | x2x +fa13 > LPCCoeff/RawSegments/" + dirname + "/" + filename[:4] + ".lpc.txt")
    ShowPositiveWindow()

def ShowPositiveWindow():
    bezz = ""
    zv = ""
    nisko = ""
    for (dirpath, dirnames, filenames) in os.walk("Window/"):
        for dirname in dirnames:
            for filename in os.listdir("Window/" + dirname + "/"): 
                filePath = "Window/" + dirname + "/" + filename
                fileData = []

                print(filePath)
                with open(filePath, "r") as f:
                    for line in f:
                        tmpData = line.split("\n")
                        for newLine in tmpData:                            
                            fileData.append(newLine.split(" "))
                    fileData = [x for x in fileData if x != ['']]
                    fileData = np.array(fileData, dtype=np.float32)
                    suma = 0
                    for x in fileData:
                        if np.isnan(x):
                            continue
                        else:
#                            print(x**2)
                            x = abs(x) / 10**37
#                        print(x)
                        suma += x**2
                        
#                    suma += [x for x in fileData]
                    print(suma, "\n")
                    if suma > float(50):
                        zv += filename + ", "
                    else:
                        bezz += filename + ", "

#                    if suma > float(10**-10):
#                        nisko += filename + ", "
#                    elif suma < float(10**-10) and suma > float(10**-5):
#                        bezz += filename + ", "
#                    else:
                        zv += filename + ", "
    print("Zvucni: " + zv)
    print("Bezzvucni: " + bezz)
    print("Nisko: " + nisko)
                

def ReadValues(singleRaw):
    fileData1 = {}
    fileData2 = {}
    
#    for (dirpath, dirnames, filenames) in os.walk("LPCCoeff/test/"):
#        for dirname in dirnames:
#            for filename in os.listdir("LPCCoeff/test/" + dirname + "/"): 
#                filePath = "LPCCoeff/test/" + dirname + "/" + filename
#                refName = filename.split(".")[0]
#                fileData = { refName: [] }
#                with open(filePath, "r") as f:
#                    for line in f:
#                        fileData[refName].append(line.split("\t"))
##                    print(fileData)
#                
#                fileData[refName] = [x for x in fileData[refName][0] if x != '-nan']
#                fileData1[refName] = np.array(fileData[refName], dtype=np.float32)    
#    for (dirpath, dirnames, filenames) in os.walk("LPCCoeff/test/"):
#        for filename in os.listdir("LPCCoeff/test/"): 
#            filePath = "LPCCoeff/test/" + filename
#            refName = filename.split(".")[0]
#            fileData = { refName: [] }
#            with open(filePath, "r") as f:
#                for line in f:
#                    fileData[refName].append(line.split(" "))
#                    print(fileData)
#            
#            fileData[refName] = [x for x in fileData[refName][0] if x != '-nan']
#            fileData1[refName] = np.array(fileData[refName], dtype=np.float32)   
#                print("\n")
#                
#                Multi RAW reader
    for (dirpath, dirnames, filenames) in os.walk("LPCCoeff/test/"):
        for filename in os.listdir("LPCCoeff/test/"): 
            filePath = "LPCCoeff/test/" + filename
            refName = filename.split(".")[0]
            fileData = { refName: [] }
            with open(filePath, "r") as f:
                for line in f:
                    tmpData = line.split("\n")
#                        fileData.append(line.split("\n"))
#                        for data in tmpData:
                    for newLine in tmpData:                            
                        fileData[refName].append(newLine.split(" "))
            fileData[refName] = [x for x in fileData[refName] if x != ['']]
            fileData1[refName] = np.array(fileData[refName], dtype=np.float32)

#                print(len(fileData))

#            print(fileData1)
#                print("\n")
                
#                Single RAW reader
    refName = singleRaw.split(".")[0]
    fileData = { refName: [] }
    filePath = "LPCCoeff/RawSingle/" + singleRaw
    with open(filePath, "r") as f:
        for line in f:
            tmpData = line.split("\n")
#                        fileData.append(line.split("\n"))
#                        for data in tmpData:
            for newLine in tmpData:                            
                fileData[refName].append(newLine.split("\t"))
    fileData[refName] = [x for x in fileData[refName] if x != ['']]
    fileData2[refName] = np.array(fileData[refName], dtype=np.float32)

#    print(len(fileData))
#
#    print(fileData)
#    print("\n")
                
    return fileData1, fileData2

def Euclidian(wav, rawSegments): 
#    print(rawSegments["a-11"])
#    print(rawSegments["a-11"][0])
#    print(wav["a-11"])

    euclidResult = {}
    cnt = 0
    resultString = ""
    for a in rawSegments["sm04010105160"]:
        minEuclidian = np.nan
        minEuclidianName = ""
#        print(a)
#        print(wav)
        for b in wav:
            wavB = wav[b]
#            print(wavB)
#            break
            for zz in wav[b]:
#                print(zz)
#                if np.isnan(wav[b][0]):
#                    continue
#                    wavB = np.zeros(13)
                dist = distance.euclidean(a[1:], zz[1:])
    #            dist = math.sqrt(sum([(a1 - b1) ** 2 for a1, b1 in zip(a, wav[b])]))
    #            dist = 0
    #            for one in range(0, 13):
    #                dist += (a - wav[b])**2
    #            dist = math.sqrt(one)
    #            print(a)
    #            print(wav[b])
    #            break
    #            maxV = np.max(wav[b])
    #            dist = np.linalg.norm((a - wav[b])/maxV) * maxV
    #            dist = np.sqrt(((a - wav[b]) ** 2).sum())
    #            print(dist)
                if np.isnan(minEuclidian):
                    minEuclidian = dist
                    minEuclidianName = b
                elif minEuclidian > dist:
                    minEuclidian = dist
                    minEuclidianName = b
            
#            print(wav[b].all())
#            print(dist)

#            print(minEuclidianName, minEuclidian)
        euclidResult.update( { minEuclidianName: minEuclidian })
        resultString = resultString + minEuclidianName[:1]
    print(resultString)
    return resultString
#    for euclid in euclidResult:
#        if euclidResult[euclid]
#        print(euc)
#            print(a, "\n")
#            print(wav[b])

def FilterOne(firstString):
    newString = ""
    one = ""
    two = ""
    cnt = 0
    print("\n \n")
    for i in range(0, len(firstString)):
        cnt += 1
        if cnt == 3:
            cnt = 0;
            if firstString[i] == firstString[i - 1] and firstString[i] == firstString[i - 2]:  
                newString += firstString[i]
            else:
                newString += firstString[i]
            #        if i >= 1 and i <= len(firstString) - 2:
#            if firstString[i] == firstString[i - 1] and firstString[i] == firstString[i + 1]:  
#                continue
#            else:
#                newString += firstString[i]
#    for i in range(0, len(firstString)):
#        if i >= 1 and i <= len(firstString) - 2:
#            if firstString[i] != firstString[i - 1] and firstString[i] != firstString[i + 1]:
#                continue
#            else:
#                newString += firstString[i]
#                
#    print(newString + "\n")
#
#    firstString = newString
#    newString = ""
#    for i in range(0, len(firstString)):
#        if i >= 1 and i <= len(firstString) - 2:    
#            if firstString[i] == firstString[i - 1] and firstString[i] == firstString[i + 1]:    
#                continue;
#            else:
#                newString += firstString[i]
                
    print(newString + "\n")
#        if one == "":
#            one = char
#        elif two == "":
#            two = char
#            
#        if one != two and one != "":
#            one = ""
#            two = ""
#        else:
#            newString += char
#            one = ""
#            two = ""


def Energies(wav, rawSegments):

    for raw in rawSegments:
        energy = 0
        cnt = 0
        for segment in rawSegments[raw]:
#            print(segment)
            energy += segment[0]
            cnt += 1
        energy = energy / cnt
        print(raw, "energy - ", energy)
        
    for v in wav:
        energy = wav[v][0]
#        print(energy)
        
        
def printWave(path, signal):
    #spf = wave.open(path, "r")
    
    # Extract Raw Audio from Wav File
    #signal = spf.readframes(-1)
    #signal = np.frombuffer(signal, "Int16")
    
    
    # If Stereo
   # if spf.getnchannels() == 2:
     #   print("Just mono files")
        #sys.exit(0)
    
    plt.figure(2)
    plt.title("Signal Wave...")
    plt.plot(signal)
    plt.show()
    
    return signal
    
def calculateEnergy(signal):
    #arr = np.array([2, 3, 2])
    energy = np.sum(signal**2)
    #energy = pysptk.sptk.
#    energy = 0
#    for en in signal:
#        energy = energy + en**2
    return energy

def eachSoundEnergy():
    f = []
    for (dirpath, dirnames, filenames) in os.walk("Voices/"):
        f.extend(filenames)
        break
    results = {}
    for sound in f:
        spf = wave.open("Voices/" + sound, "r")        
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, "Int16")
        fftSignal = np.fft.fft(signal)
        results.update({ sound : calculateEnergy(fftSignal) })
        #print(sound + " " + str(calculateEnergy(fftSignal)))
        printWave("Voices/" + sound, signal)

    return results



#CutVoices("sm04010103301")
#ToRaw("sm04010105160.wav")
lpcCoeff()
#lpcCoeffSingle("sm04010105160.wav")
wav, rawSegments = ReadValues("sm04010105160.raw.lpc.txt")
resultString = Euclidian(wav, rawSegments)
FilterOne(resultString)
#Energies(wav, rawSegments)
#eachSoundEnergy()
#CalcWindow()

#MOJE
#sjeci_glasove("sm04010105160.wav")
#lpcCoeff();
#toRaw()
#signal = printWave()
#print(signal[:29000])
#calculateEnergy(signal)
#energies = eachSoundEnergy()
#print(energies.values())
#
#occurencies = sum(1 for i in energies.values() if i > 0)
#energyMean = 0
#for energy in energies.values():
#    energyMean = energyMean + energy
#
#for energy in energies:
#    if energies[energy] > 0:
#        print(energy)
#    
#energyMean = energyMean / 138
#print("Mean: ", energyMean)
#print (occurencies) #TRIBAN MAKNAT IMAGINARNI KUS
##somewhat = energies.values()
##print("MIN: ", min(somewhat))
##print("MAX: ", max(somewhat))
#
#primjer()
##print("MEAN: ", np.array([somewhat[k] for k in somewhat]).mean())
