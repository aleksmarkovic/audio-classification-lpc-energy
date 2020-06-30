import matplotlib.pyplot as plt
import numpy as np
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
import librosa
import json


classification = { "voiced": ["b", "d", "g", "z", "Z", "dZ", "D", "a", "e", "i", "o", "u"],
                  "unvoiced": ["p", "t", "k", "s", "S", "C", "cc", "h", "f"],
                  "silence": "sil" }
energyResult = {}
lpcResult = {}
alpha = 1.2
increment = 0.05
correctRatio = 0

correctcorrect = 0


def FileLength(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def CutVoices(fname):
    nVoices = 1
    global numberOfSilence
    numberOfSilence = 0
    M = open("Materials/lab_sm04/" + fname + ".lab","r").readlines()
    content = [x.strip("\n") for x in M]
    numOfLines = FileLength("Materials/lab_sm04/" + fname + ".lab")
    
    for i in range(0, numOfLines):
        content[i]=content[i].split(" ")
#    print(content)

    for i in range(0, numOfLines):
        tfm=sox.Transformer()
        tfm.trim(float(content[i][0])*10**-7, float(content[i][1])*10**-7)
        Path("Voices/" + fname + "/" + content[i][2]).mkdir(parents = True, exist_ok = True)
        tfm.build("Materials/wav_sm04/" + fname + ".wav", "Voices/" + fname + "/" + content[i][2] + "/" + content[i][2] + "-" + str(nVoices) + ".wav")
        nVoices = nVoices + 1
        if content[i][2] == "sil":
            numberOfSilence += 1
    return

def lpcCoeff(fname):
    frameLength = 320
    hopLength = 160
    results = {}
    for (dirpath, dirnames, filenames) in os.walk("Voices/" + fname + "/"):
        for dirname in dirnames:
            for filename in os.listdir("Voices/"  + fname + "/" + dirname + "/"):
                filenamePath = "Voices/" + fname + "/" + dirname + "/" + filename
                Path("LPC/" + fname + "/").mkdir(parents = True, exist_ok = True)      
                f = open("LPC/"  + fname + "/" + filename + ".lpc.txt", "w")
#                Path("LPC/" + fname + "/" + dirname).mkdir(parents = True, exist_ok = True)      
#                print(filenamePath)
                result = np.empty((0, 13))
                spf = wave.open(filenamePath, "r")        
                # Extract Raw Audio from Wav File
                samples = spf.readframes(-1)
                samples = np.frombuffer(samples, "Int16")

                if len(samples) >= 320:
                        frames = librosa.util.frame(samples, frameLength, hopLength).astype(np.float64).T
                        frames *= pysptk.blackman(frameLength)                        
                        lpcCoeff = pysptk.sptk.lpc(frames[0], 12)
        #                lpcCoeff = pysptk.sptk.lpc2c(lpcCoeff, 12)
                        result = np.append(result, np.array([lpcCoeff]), axis=0)
                        results.update( { filename[:-4]: result[0] } )                            
#                        print(result)
                np.savetxt(f, result, fmt='%.6f')
                f.close()
    return results

def sqdist(vector):
    return sum(x*x for x in vector)

def lpcClassification(lpcData):
    mean = np.zeros(12)
    tmpDict = lpcData.copy()
    voiced = []
    unvoiced = []
    silence = []
    for lpc in lpcData: #ZA SVE MEMBERS VECTORA
        if lpc.split("-")[0] in classification["voiced"]:
            voiced.append(lpc)
#            print(lpc + " is VOICED")
        elif lpc.split("-")[0] in classification["unvoiced"]:    
            unvoiced.append(lpc)
#            print(lpc + " is UNVOICED")
        elif lpc.split("-")[0] == classification["silence"]:    
            silence.append(lpc)
#            print(lpc + " is SILENCE")
        else:
            del(tmpDict[lpc])
            continue
#        print(lpcData[lpc])
        if lpc[:3] != "sil":            
            mean += lpcData[lpc][1:]
    lpcData = tmpDict
    mean = mean / len(lpcData)

    voicedResult = []
    unvoicedResult = []
    silenceResult = []
    

    meanEuclidDistance = 0
    for lpc in lpcData:
        meanEuclidDistance += distance.euclidean(lpcData[lpc][1:], mean)
    meanEuclidDistance = meanEuclidDistance / len(lpcData)
    for lpc in lpcData:
        if  distance.euclidean(lpcData[lpc][1:], mean) >= meanEuclidDistance * alpha:
            voicedResult.append(lpc)
            lpcResult.update({ lpc: "VOICED" })
            print(lpc + "VOICED")
        else:
            unvoicedResult.append(lpc)
            lpcResult.update({ lpc: "UNVOICED" })
            print(lpc + "UNVOICED")

def CalculateEnergy(signal):
    energy = 0
    N = len(signal)
    e = 10**-8
    for S in signal:    
        energy += (1/N) * np.sum(S**2)
    energy += 10*math.log(math.e + energy)
    return energy/10**6

def EachSoundEnergy(fname):
    results = {}
    for (dirpath, dirnames, filenames) in os.walk("Voices/"  + fname + "/"):
        for dirname in dirnames:
            for filename in os.listdir("Voices/" + fname + "/" + dirname + "/"): 
                filePath = "Voices/" + fname + "/" + dirname + "/" + filename
                fileData = []
                spf = wave.open("Voices/" + fname + "/" + dirname +"/" + filename, "r")        
                # Extract Raw Audio from Wav File
                signal = spf.readframes(-1)
                signal = np.frombuffer(signal, "Int16")
                results.update({ filename[:-4]: CalculateEnergy(signal) })
    return results

def ClassificateEnergy(energyResults):
    mean = 0
    tmpDict = energyResults.copy()
    voiced = []
    unvoiced = []
    silence = []
    for energy in energyResults:
        if energy.split("-")[0] in classification["voiced"]:
            voiced.append(energy)
#            print(energy + " is VOICED")
        elif energy.split("-")[0] in classification["unvoiced"]:    
            unvoiced.append(energy)
#            print(energy + " is UNVOICED")
        elif energy.split("-")[0] == classification["silence"]:    
            silence.append(energy)
#            print(energy + " is SILENCE")
        else:
            del(tmpDict[energy])
            continue
        mean += energyResults[energy]   

    energyResults = tmpDict
    mean = mean / len(energyResults)

    voicedResult = []
    unvoicedResult = []
    silenceResult = []

    minValues = sorted(list(energyResults.values()))
    minValues = np.array(minValues)
    minValues = minValues[:numberOfSilence]
    
    for energy in energyResults:
        if energyResults[energy] >= mean:
            energyResult.update({ energy: "VOICED" })
            voicedResult.append(energy)
            print(energy + " is VOICED")
        elif energyResults[energy] in minValues:
            energyResult.update({ energy: "SILENCE" })
            silenceResult.append(energy)
#            print(energy + " is UNVOICED", energyResults[energy])
            print(energy + "is SILENCE")
        else:
            energyResult.update({ energy: "UNVOICED" })
            unvoicedResult.append(energy)
            print(energy + " is UNVOICED", energyResults[energy])
    
    correctNumbers = 0

    for v in voicedResult:
        if v in voiced:
            correctNumbers += 1
#    print("VOICED CORRECT: ", correctNumbers/len(voiced) * 100, "%")
            
    correctNumbers = 0
    for v in unvoicedResult:
        if v in unvoiced:
            correctNumbers += 1
#    print("UNVOICED CORRECT: ", correctNumbers/len(unvoiced) * 100, "%")
    
    correctNumbers = 0
    for v in silenceResult:
        if v in silence:
            correctNumbers += 1
#    print("SILENCE CORRECT: ", correctNumbers/len(silence) * 100, "%")


def FinalClassification(fname):
    perc = 0
    finalResult = {}
    for l in lpcResult:
#        if l not in energyResult:
#            continue
        if lpcResult[l] == energyResult[l]:
#            print(l + " is 100% " + lpcResult[l])
            finalResult.update({ l: lpcResult[l] })
        elif energyResult[l] == "SILENCE":
#            print(l + " is 100% " + energyResult[l])
            finalResult.update({ l: energyResult[l] })
        else:
#            print(l + " is 50% ")
            finalResult.update({ l: lpcResult[l] })
        
        print (l[0])
        if finalResult[l] == "VOICED" and l[0] in classification["voiced"]:
            perc += 1
        elif finalResult[l] == "UNVOICED" and l[0] in classification["unvoiced"]:
            perc += 1
        elif finalResult[l] == "SILENCE":
            perc +=1
            
            
    perc = perc/len(lpcResult) * 100
    correctRatio = perc
    perc = "{:.2f}".format(perc)
    print("CORRECT IS: ", perc, "%")
    finalResult.update({ "FINAL RESULT":  str(perc) + "%" })
#    f = open("Results/" + fname,"w")
#    f.write(str(finalResult))
#    f.close()

    jsonFile = json.dumps(finalResult)
    f = open("Results/" + fname + ".json", "w")
    f.write(jsonFile)
    f.close()
    
    return correctRatio
    
materials = []
for (dirpath, dirnames, filenames) in os.walk("Materials/wav_sm04"):
    materials = filenames

cnt = 0
for file in materials:    
    if cnt == 10:
        break
    fname = file[:-4]
    print(fname)
    CutVoices(fname)
    lpcCoeffs = lpcCoeff(fname)
    energyValues = EachSoundEnergy(fname)   
    lpcClassification(lpcCoeffs)
    ClassificateEnergy(energyValues)
    correctRatio = FinalClassification(fname)
    
    lastCorrectRatio = 0
    incrementTry = 0
    nextStop = 0
    while (1):
        print(correctRatio, " ", alpha)
        if correctRatio > lastCorrectRatio:
            alpha += increment
        elif correctRatio < lastCorrectRatio and nextStop == 0:
            alpha -= increment
            nextStop = 1
        elif incrementTry == 3:
            break
        else:
            alpha += increment
            incrementTry += 1
        print(correctRatio)
        lastCorrectRatio = correctRatio    

        lpcClassification(lpcCoeffs)
        correctRatio = FinalClassification(fname)

    cnt += 1
    energyResult.clear
    lpcResult.clear
    currentRatio = 0    
    alpha = 1.2