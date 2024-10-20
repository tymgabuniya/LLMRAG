# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:44:47 2024

@author: tymur
"""

import pandas as pd
import math
import numpy as np
import gensim.downloader
from mistralai import Mistral

from fastapi import FastAPI

embeddings = gensim.downloader.load('word2vec-google-news-300')
client = Mistral(api_key="zTZl98Dv7VDRUbPdbEGn9umIVaf78vw5")

#### Utility functions

lemur_stop_words = ['','.', ',', '?', '!', "'", '"', "''", '`', '``', '*', '-', '/', '+','an',
       'a', 'about', 'above', 'according', 'across', 'after',
       'afterwards', 'again', 'against', 'albeit', 'all', 'almost',
       'alone', 'along', 'already', 'also', 'although', 'always', 'am',
       'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody',
       'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'apart',
       'are', 'around', 'as', 'at', 'av', 'be', 'became', 'because',
       'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
       'behind', 'being', 'below', 'beside', 'besides', 'between',
       'beyond', 'both', 'but', 'by', 'can', 'cannot', 'canst', 'certain',
       'cf', 'choose', 'contrariwise', 'cos', 'could', 'cu', 'day', 'do',
       'does', "doesn't", 'doing', 'dost', 'doth', 'double', 'down',
       'dual', 'during', 'each', 'either', 'else', 'elsewhere', 'enough',
       'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone',
       'everything', 'everywhere', 'except', 'excepted', 'excepting',
       'exception', 'exclude', 'excluding', 'exclusive', 'far', 'farther',
       'farthest', 'few', 'ff', 'first', 'for', 'formerly', 'forth',
       'forward', 'from', 'front', 'further', 'furthermore', 'furthest',
       'get', 'go', 'had', 'halves', 'hardly', 'has', 'hast', 'hath',
       'have', 'he', 'hence', 'henceforth', 'her', 'here', 'hereabouts',
       'hereafter', 'hereby', 'herein', 'hereto', 'hereupon', 'hers',
       'herself', 'him', 'himself', 'hindmost', 'his', 'hither',
       'hitherto', 'how', 'however', 'howsoever', 'i', 'ie', 'if', 'in',
       'inasmuch', 'inc', 'include', 'included', 'including', 'indeed',
       'indoors', 'inside', 'insomuch', 'instead', 'into', 'inward',
       'inwards', 'is', 'it', 'its', 'itself', 'just', 'kind', 'kg', 'km',
       'last', 'latter', 'latterly', 'less', 'lest', 'let', 'like',
       'little', 'ltd', 'many', 'may', 'maybe', 'me', 'meantime',
       'meanwhile', 'might', 'moreover', 'most', 'mostly', 'more', 'mr',
       'mrs', 'ms', 'much', 'must', 'my', 'myself', 'namely', 'need',
       'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
       'nonetheless', 'noone', 'nope', 'nor', 'not', 'nothing',
       'notwithstanding', 'now', 'nowadays', 'nowhere', 'of', 'off',
       'often', 'ok', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
       'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out',
       'outside', 'over', 'own', 'per', 'perhaps', 'plenty', 'provide',
       'quite', 'rather', 'really', 'round', 'said', 'sake', 'same',
       'sang', 'save', 'saw', 'see', 'seeing', 'seem', 'seemed',
       'seeming', 'seems', 'seen', 'seldom', 'selves', 'sent', 'several',
       'shalt', 'she', 'should', 'shown', 'sideways', 'since', 'slept',
       'slew', 'slung', 'slunk', 'smote', 'so', 'some', 'somebody',
       'somehow', 'someone', 'something', 'sometime', 'sometimes',
       'somewhat', 'somewhere', 'spake', 'spat', 'spoke', 'spoken',
       'sprang', 'sprung', 'stave', 'staves', 'still', 'such',
       'supposing', 'than', 'that', 'the', 'thee', 'their', 'them',
       'themselves', 'then', 'thence', 'thenceforth', 'there',
       'thereabout', 'thereabouts', 'thereafter', 'thereby', 'therefore',
       'therein', 'thereof', 'thereon', 'thereto', 'thereupon', 'these',
       'they', 'this', 'those', 'thou', 'though', 'thrice', 'through',
       'throughout', 'thru', 'thus', 'thy', 'thyself', 'till', 'to',
       'together', 'too', 'toward', 'towards', 'ugh', 'unable', 'under',
       'underneath', 'unless', 'unlike', 'until', 'up', 'upon', 'upward',
       'upwards', 'us', 'use', 'used', 'using', 'very', 'via', 'vs',
       'want', 'was', 'we', 'week', 'well', 'were', 'what', 'whatever',
       'whatsoever', 'when', 'whence', 'whenever', 'whensoever', 'where',
       'whereabouts', 'whereafter', 'whereas', 'whereat', 'whereby',
       'wherefore', 'wherefrom', 'wherein', 'whereinto', 'whereof',
       'whereon', 'wheresoever', 'whereto', 'whereunto', 'whereupon',
       'wherever', 'wherewith', 'whether', 'whew', 'which', 'whichever',
       'whichsoever', 'while', 'whilst', 'whither', 'who', 'whoa',
       'whoever', 'whole', 'whom', 'whomever', 'whomsoever', 'whose',
       'whosoever', 'why', 'will', 'wilt', 'with', 'within', 'without',
       'worse', 'worst', 'would', 'wow', 'ye', 'yet', 'year', 'yippee',
       'you', 'your', 'yours', 'yourself', 'yourselves'] 

def remove_stops(list_a,list_b=lemur_stop_words):
    
    list_aa = []
    for d in list_a:
        if d in list_b:
            continue
        else:
            list_aa.append(d)
            
    return(list_aa)

def removeQuestion(list):
    
    try:
        
        return(list[list.index("Question ")+len("Question "):])
    
    except ValueError:
        
        return list

def is_nan(x):
    
    try:
        math.isnan(x)
        return True
    except TypeError:
        return False
    
def handle_abbrev(docm):
    # Remove the dots in the abbreviations, handle hyphens and apostrophes 
    
    lengthDocm=len(docm)
    inx0=1
    while inx0 <= lengthDocm-3:
    
        if docm[inx0]=='.' and (docm[inx0+1].isupper()) and (docm[inx0-1].isupper()):# Abbreviations of the type U.S.A. to USA
            if docm[inx0+2]==' ' or (docm[inx0+2].isupper()) or docm[inx0+2]=='.':
                docm=docm[:inx0]+docm[inx0+1:]
                lengthDocm-=1
            else:
                inx0+=1
        if docm[inx0]=='-':# Delete the hyphens
            docm=docm[:inx0]+docm[inx0+1:]
            lengthDocm-=1
        if docm[inx0]=='’' or docm[inx0]=="'":# Similarly to hyphens
            docm=docm[:inx0]+docm[inx0+1:]
            lengthDocm-=1
        else:
            inx0+=1
        
    return(docm)

def preprocess_first(docm):
    # Handle the tabs, new lines and the carriage returns. Unify the use of hyphens and the use of and
    
    docm=docm.replace('\t',' ').replace('\n',' ').replace('\r',' ')
    docm=docm.replace('--','-').replace(' — ','-').replace(' & ', ' and ')
        
    return (docm)

def preprocess_third(docm):
    # Handle punctuation and some other symbols by substituting them with an empty space

    docm=docm.replace('?',' ').replace('!',' ').replace(',',' ').replace(';',' ').replace('“',' ').replace('”',' ')
    docm=docm.replace('. ',' ').replace(' .',' ').replace('•',' ').replace('…',' ').replace('-',' ').replace('.',' ')
    docm=docm.replace(' > ',' ').replace('<',' ').replace(' - ',' ').replace('’',' ').replace('_',' ').replace('/',' ')
    docm=docm.replace('[',' ').replace(']',' ').replace('|',' ').replace('{',' ').replace('}',' ').replace('@',' ')
    docm=docm.replace(':',' ').replace('(',' ').replace(')',' ').replace('=',' ').replace('£',' ').replace('%',' ')
    docm=docm.replace('#',' ').replace('~',' ').replace('$',' ').replace('^',' ').replace('€',' ').replace('■',' ').replace('‘',' ')
    
    return(docm)

def preprocess(docm):
    # Pre-process the document: STEP 1: Prepare the document for abbreviation, hyphens and apostrophe handeling; 
    # STEP 2: Join the abbreviation letters, parts separated by hyphens (in some cases) and by apostrophe. STEP 3: Remove
    # an uppercasing. STEP 4: Remove punctuation and other symbols.
    
    docm=preprocess_first(docm)
    docm=handle_abbrev(docm)
    docm=docm.lower()
    docm=preprocess_third(docm)
    
    return(docm) 

def calcCosineSim(x,y):
    
    xModulus = np.sqrt(np.sum(sq_uFun(x)))
    yModulus = np.sqrt(np.sum(sq_uFun(y))) 
    
    return(np.dot(x,y)/(xModulus*yModulus))

def sqElement(x):
    
    return x**2

sq_uFun = np.frompyfunc(sqElement,1,1)

def getQuestionsData(rows, dumData = 1):
    
    namesLayer1 = []
    namesLayer2 = []
    indexQuestions = []
    
    if dumData == 1:
    
        for i in range(len(rows)):
            
            rr0 = rows[i][0]
            rr1 = rows[i][1]
            
            if is_nan(rr0) == False:
                
                if "Question " in rr0:
                    
                    indexQuestions.append(i)
                
            namesLayer1.append(rr0)
            namesLayer2.append(rr1)
                        
    else:
    
        for i in range(len(rows)):
            
            rr0 = rows[i]
            
            if is_nan(rr0) == False:
                
                if "Question " in rr0:
                    
                    indexQuestions.append(i)
                    namesLayer1.append(rr0)
                else:
                    namesLayer1.append(' ')
                
            namesLayer2.append(rr0)
                    
    return([namesLayer1, namesLayer2, indexQuestions])

def loadDat(dumDat = 1):

    if dumDat == 1:
    
        data = pd.read_excel("Dataset 1 (Sustainability Research Results).xlsx", header = [0,1], index_col = 0)
        inxess = data.index
        inx_drop = [i for i in range(len(inxess)) if is_nan(inxess[i])]
        data.drop(labels = inxess[inx_drop[0]], axis = 0, inplace = True)
        data.dropna(axis = 1, how = 'all', inplace = True)
        rows = data.index
        
        inxToSeparate = np.where(rows == "Questions")[0][0]
        dataA = data.iloc[:inxToSeparate,:]
        dataA = dataA.transpose()
        dataB = data.iloc[inxToSeparate+1:,:]
    
    else: 

        data = pd.read_excel("Dataset 2 (Christmas Research Results).xlsx", header = [0,1], index_col = 0)
        inxess = data.index
        inx_drop = [i for i in range(len(inxess)) if is_nan(inxess[i])]
        data.drop(labels = inxess[inx_drop[0]], axis = 0, inplace = True)
        data.dropna(axis = 1, how = 'all', inplace = True)
        rows = data.index
        
        inxToSeparate = np.where(rows == "Questions")[0][0]
        dataA = data.iloc[:inxToSeparate,:]
        dataA = dataA.transpose()
        dataB = data.iloc[inxToSeparate+1:,:]
        
    [rowsPartALayer1, rowsPartALayer2, indexQuestionsPartA] = getQuestionsData(dataA.index) 
    [rowsPartBLayer1, rowsPartBLayer2, indexQuestionsPartB] = getQuestionsData(dataB.index, dumData = 3)
        
    return ([dataA, dataB, rowsPartALayer1, rowsPartALayer2, indexQuestionsPartA, rowsPartBLayer1, rowsPartBLayer2, indexQuestionsPartB])

def loadData(dumyData = 1):
    
    if dumyData == 1:
        
        return(loadDat())
        
    elif dumyData == 2:
        
        return(loadDat(dumDat = 2))
        
    else:
        
        [data1A, data1B, data1rowsPartALayer1, data1rowsPartALayer2, data1indexQuestionsPartA, data1rowsPartBLayer1, data1rowsPartBLayer2, data1indexQuestionsPartB] = loadDat()
        [data2A, data2B, data2rowsPartALayer1, data2rowsPartALayer2, data2indexQuestionsPartA, data2rowsPartBLayer1, data2rowsPartBLayer2, data2indexQuestionsPartB] = loadDat(dumDat = 2)
        return ([loadDat(), loadDat(dumDat = 2)])
    
def processData(X):
    
    xTransformed = []
    xEmbeds = []
    
    for i,x in enumerate(X):
        
        x = removeQuestion(x)
        x = preprocess(x)
        x = x.split(' ')
        x = remove_stops(x)
        tokens = []
        embeds = []    
        if len(x) == 0:
            xEmbeds.append([])
            xTransformed.append([])
            continue  
        for token in x:
            #print(token)
            try:
                embedding = embeddings[token]
            except KeyError:
                #print(qq)
                continue
            tokens.append(token)
            embeds.append(embedding)
        xEmbeds.append(embeds)
        xTransformed.append(tokens)
        
    return([xTransformed,xEmbeds])

def getMostSimilarDataParts(embedsQuery, embedsData, k = 3, dumMaximum = 1):
    
    nQuery = len(embedsQuery)
    
    most_similar = []
    inxs = []

    for i in range(k):
        
        simty = 0
        passage = embedsData[i]
        nPassage = len(passage)
        
        if nPassage == 0:
            
            most_similar.append(-1000)
            continue
        
        if dumMaximum == 1:
            
            for vectrP in passage:
                
                simlrty = []
                for vectrQ in embedsQuery:
                    simlrty.append(calcCosineSim(vectrP,vectrQ))
                simty += np.max(simlrty)
                
        else:
                
            for vectrP in passage:
                
                simlrty = 0
                for vectrQ in embedsQuery:
                    simlrty += calcCosineSim(vectrP,vectrQ)
                simty += simlrty/nQuery

                
        most_similar.append(simty/nPassage)
    
    inxsSorted = list(np.argsort(most_similar))
    most_similarSorted = list(np.sort(most_similar))

    ##print("STEP A:")
    ##print(most_similar)
    ##print(most_similarSorted)
    ##print(inxsSorted)
    
    for i in range(k,len(embedsData),1):
        
        ##print(i)
        simty = 0
        passage=embedsData[i]
        nPassage = len(passage)
    
        if nPassage == 0:
            continue
            
        if dumMaximum == 1:
            
            for vectrP in passage:
                
                simlrty = []
                for vectrQ in embedsQuery:
                    simlrty.append(calcCosineSim(vectrP,vectrQ))
                simty += np.max(simlrty)
                
        else:
                
            for vectrP in passage:
                
                simlrty = 0
                for vectrQ in embedsQuery:
                    simlrty += calcCosineSim(vectrP,vectrQ)
                simty += simlrty/nQuery    
    
        simty = simty/nPassage
        
        ##print("STEP B:")
        ##print(simty)
        ##print(most_similarSorted)
        ##print(inxsSorted)
    
        if most_similarSorted[0] > simty:
            ##print("STEP C:")
            continue
        else:
        
            ##print("STEP D:")
            ##print(most_similarSorted)
            ##print(inxsSorted)
        
            if most_similarSorted[-1] <= simty:
                most_similarSorted.append(simty)
                inxsSorted.append(i)
                most_similarSorted = most_similarSorted[1:]
                inxsSorted = inxsSorted[1:]
            
                ##print("STEP E:")
                ##print(most_similarSorted)
                ##print(inxsSorted)
            else:            
                        
                ##print("STEP F:")
                ##print(most_similarSorted)
                ##print(inxsSorted)
            
                most_similarSorted.insert(0,most_similarSorted[0])
                inxsSorted.insert(0,inxsSorted[0])
            
                inxWhereLess = np.max(np.where(most_similarSorted <= simty)[0])
                most_similarSorted[inxWhereLess] = simty
                inxsSorted[inxWhereLess] = i
            
                most_similarSorted = most_similarSorted[1:]
                inxsSorted = inxsSorted[1:]
            
            
                ##print("STEP G:")
                ##print(most_similarSorted)
                ##print(inxsSorted)
            
    ##print("END:")
    ##print(most_similarSorted)
    ##print(inxsSorted)
    
    return ([most_similarSorted, inxsSorted])

def getSubset(listJoinedQuestions, data, inxsSorteda, inxsSortedb, indexes, inxess):
    
    ##print("PART A")

    questionsSelecteda = []
    inxsQuestionsSelecteda = []
    for ii in inxsSorteda:
        ##print(ii)
        inxQuest = np.where(np.array(indexes) <= ii)[0]    
        
        ##if len(inxQuest) != 0:
            
            ##print("-----------------------------------------------------------------------------------")
            ##print(inxQuest)
            ##print(inxess[indexes[inxQuest[-1]]])
            ##print("-----------------------------------------------------------------------------------")
            
        ##else:
            
            ##print("ZERO")
                    
        
        if len(inxQuest) == 0:
            questionsSelecteda.append('Beginning')
            ##print("Beginning")
            inxsQuestionsSelecteda.append(0)
            
            continue
        
        elif len(listJoinedQuestions) != 0 and inxQuest[-1] in listJoinedQuestions:
            ##print("JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ")
            ##print(inxQuest[-1])
            for iiJoined in listJoinedQuestions:
                    
                inxSelect = indexes[iiJoined]
                questionsSelecteda.append(inxess[inxSelect])
                ##print(inxess[inxSelect])
                inxsQuestionsSelecteda.append(inxSelect)
                
            continue
            
        else:
            inxSelect = indexes[inxQuest[-1]]
            questionsSelecteda.append(inxess[inxSelect])
            ##print(inxess[inxSelect])
            inxsQuestionsSelecteda.append(inxSelect)
        ##print("KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        
    ##print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

    questionsSelectedb = []
    inxsQuestionsSelectedb = []
    for ii in inxsSortedb:
        ##print(ii)
        inxQuest = np.where(np.array(indexes) <= ii)[0]
        
        ##if len(inxQuest) != 0:
            
            ##print("-----------------------------------------------------------------------------------")
            ##print(inxQuest)
            ##print(inxess[indexes[inxQuest[-1]]])
            ##print("-----------------------------------------------------------------------------------")
            
        ##else:
            
            ##print("ZERO")
        
        if len(inxQuest) == 0:
            questionsSelectedb.append('Beginning')
            ##print("Beginning")
            inxsQuestionsSelectedb.append(0)
            
            continue
            
        elif len(listJoinedQuestions) != 0 and inxQuest[-1] in listJoinedQuestions:
            ##print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            for iiJoined in listJoinedQuestions:
                    
                inxSelect = indexes[iiJoined]
                questionsSelectedb.append(inxess[inxSelect])
                ##print(inxess[inxSelect])
                inxsQuestionsSelectedb.append(inxSelect)
                    
            continue
            
        else:
            inxSelect = indexes[inxQuest[-1]]
            questionsSelectedb.append(inxess[inxSelect])
            ##print(inxess[inxSelect])
            inxsQuestionsSelectedb.append(inxSelect)
        ##print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
            
    setIndexesPassages=set(inxsQuestionsSelecteda).union(set(inxsQuestionsSelectedb))
    
    ##print("PART B")
    
    selectedParts=[]
    for ii in setIndexesPassages:
        inx = np.where(np.array(indexes)>ii+1)
        if len(inx[0]) == 0:
            selectedParts.append([ii,len(inxess)])
        else:
            selectedParts.append([ii,indexes[np.min(inx)]])
        ##print(inx)
    
    ##print(selectedParts)
    
    ##print("PART C")
    
    subTables=[]
    for i,inx0 in enumerate(selectedParts):
        ##print("-----------------------------------------")
        ##print(i)
        ##print("Section")
        ##print("-----------------------------------------")
        i0 = inx0[0]
        i1 = inx0[1]
        print(data.iloc[i0:i1])
        subTables.append(data.iloc[i0:i1])
        
    return(subTables)
    
        

app = FastAPI()

contextualInfo = []

@app.post("/response")
def makeQueries(query: str, contextWindow: int, dumyData: int):
    
    [tokenizedQuery, embedingsQuery] = processData([query])
    
    if dumyData <= 2:
        
        [dataA, dataB, rowsALayer1, rowsALayer2, indexQA, rowsBLayer1, rowsBLayer2, indexQB] = loadData(dumyData)
        
        [tokenizedALayer1, embedingsALayer1] = processData(rowsALayer1)
        [tokenizedALayer2, embedingsALayer2] = processData(rowsALayer2)
        [tokenizedBLayer1, embedingsBLayer1] = processData(rowsBLayer1)
        [tokenizedBLayer2, embedingsBLayer2] = processData(rowsBLayer2)
        
        [mostSimilarSortedALayer1, indexesALayer1] = getMostSimilarDataParts(embedingsQuery[0], embedingsALayer1, contextWindow)
        [mostSimilarSortedALayer2, indexesALayer2] = getMostSimilarDataParts(embedingsQuery[0], embedingsALayer2, contextWindow)
        [mostSimilarSortedBLayer1, indexesBLayer1] = getMostSimilarDataParts(embedingsQuery[0], embedingsBLayer1, contextWindow)
        [mostSimilarSortedBLayer2, indexesBLayer2] = getMostSimilarDataParts(embedingsQuery[0], embedingsBLayer2, contextWindow)

        
    else:
        
        [data1, data2] = loadData(dumyData)
        [data1A, data1B, data1rowsPartALayer1, data1rowsPartALayer2, data1indexQuestionsPartA, data1rowsPartBLayer1, data1rowsPartBLayer2, data1indexQuestionsPartB] = data1
        [data2A, data2B, data2rowsPartALayer1, data2rowsPartALayer2, data2indexQuestionsPartA, data2rowsPartBLayer1, data2rowsPartBLayer2, data2indexQuestionsPartB] = data2
        
        [data1tokenizedALayer1, data1embedingsALayer1] = processData(data1rowsPartALayer1)
        [data1tokenizedALayer2, data1embedingsALayer2] = processData(data1rowsPartALayer2)
        [data1tokenizedBLayer1, data1embedingsBLayer1] = processData(data1rowsPartBLayer1)
        [data1tokenizedBLayer2, data1embedingsBLayer2] = processData(data1rowsPartBLayer2)
        [data2tokenizedALayer1, data2embedingsALayer1] = processData(data2rowsPartALayer1)
        [data2tokenizedALayer2, data2embedingsALayer2] = processData(data2rowsPartALayer2)
        [data2tokenizedBLayer1, data2embedingsBLayer1] = processData(data2rowsPartBLayer1)
        [data2tokenizedBLayer2, data2embedingsBLayer2] = processData(data2rowsPartBLayer2)
        
        [data1mostSimilarSortedALayer1, data1indexesALayer1] = getMostSimilarDataParts(embedingsQuery[0], data1embedingsALayer1, contextWindow)
        [data1mostSimilarSortedALayer2, data1indexesALayer2] = getMostSimilarDataParts(embedingsQuery[0], data1embedingsALayer2, contextWindow)
        [data1mostSimilarSortedBLayer1, data1indexesBLayer1] = getMostSimilarDataParts(embedingsQuery[0], data1embedingsBLayer1, contextWindow)
        [data1mostSimilarSortedBLayer2, data1indexesBLayer2] = getMostSimilarDataParts(embedingsQuery[0], data1embedingsBLayer2, contextWindow)
        [data2mostSimilarSortedALayer1, data2indexesALayer1] = getMostSimilarDataParts(embedingsQuery[0], data2embedingsALayer1, contextWindow)
        [data2mostSimilarSortedALayer2, data2indexesALayer2] = getMostSimilarDataParts(embedingsQuery[0], data2embedingsALayer2, contextWindow)
        [data2mostSimilarSortedBLayer1, data2indexesBLayer1] = getMostSimilarDataParts(embedingsQuery[0], data2embedingsBLayer1, contextWindow)
        [data2mostSimilarSortedBLayer2, data2indexesBLayer2] = getMostSimilarDataParts(embedingsQuery[0], data2embedingsBLayer2, contextWindow)
        
    if dumyData == 1:
    
        subTablesA = getSubset(listJoinedQuestions = [9, 10, 11], data = dataA, inxsSorteda = indexesALayer1, inxsSortedb = indexesALayer2, indexes = indexQA, inxess = rowsALayer1)
        subTablesB = getSubset(listJoinedQuestions = [14, 15, 16], data = dataB, inxsSorteda = indexesBLayer1, inxsSortedb = indexesBLayer2, indexes = indexQB, inxess = rowsBLayer1)
        
        subTables = subTablesA + subTablesB

    elif dumyData == 2:

        subTablesA = getSubset(listJoinedQuestions = [], data = dataA, inxsSorteda = indexesALayer1, inxsSortedb = indexesALayer2, indexes = indexQA, inxess = rowsALayer1)
        subTablesB = getSubset(listJoinedQuestions = [], data = dataB, inxsSorteda = indexesBLayer1, inxsSortedb = indexesBLayer2, indexes = indexQB, inxess = rowsBLayer1)
        
        subTables = subTablesA + subTablesB

    else:

        data1subTablesA = getSubset(listJoinedQuestions = [9, 10, 11], data = data1A, inxsSorteda = data1indexesALayer1, inxsSortedb = data1indexesALayer2, indexes = data1indexQuestionsPartA, inxess = data1rowsPartALayer1)
        data1subTablesB = getSubset(listJoinedQuestions = [14, 15, 16], data = data1B, inxsSorteda = data1indexesBLayer1, inxsSortedb = data1indexesBLayer2, indexes = data1indexQuestionsPartB, inxess = data1rowsPartBLayer1)
        data2subTablesA = getSubset(listJoinedQuestions = [], data = data2A, inxsSorteda = data2indexesALayer1, inxsSortedb = data2indexesALayer2, indexes = data2indexQuestionsPartA, inxess = data2rowsPartALayer1)
        data2subTablesB = getSubset(listJoinedQuestions = [], data = data2B, inxsSorteda = data2indexesBLayer1, inxsSortedb = data2indexesBLayer2, indexes = data2indexQuestionsPartB, inxess = data2rowsPartBLayer1)
        
        subTables = data1subTablesA + data1subTablesB + data2subTablesA + data2subTablesB
        
    prompt = f"""
    Context information is below.
    ---------------------
    {subTables}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages
        )
    
    contextualInfo.append(subTables)
            
    
    return(chat_response.choices[0].message.content)
    
@app.get("/context")

def returnContext():
    
    
    return(contextualInfo[0])

#query = ["Who likes Amazon?"]
#[tokenizedQuery, embedingsQuery] = processData(query)