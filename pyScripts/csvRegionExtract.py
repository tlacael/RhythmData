import csv
import os
import glob
import eyed3 as d3
from geopy import geocoders
from pyechonest import song
from Levenshtein import ratio
import numpy as np
'''
Example

Get dataa from id3 tags for mp3's in a folder:
songDict = getSongInfoFromFiles('/Users/Tlacael/NYU/Latin_Music_Data/LMD/')

Add region data to songDict:
songDict = addRegionDataToSong(songDict,'/Users/Tlacael/lib/gTrends/csvsubfolder')

'''
def plotTempos(tempoList):
    tempoByGenre=[]
    for i in range(10):
        temp=[cur[0] for cur in tempoList if cur[0] != 'not in db' and cur[1]==i]
        tempoByGenre.append(np.array(temp))
    tempoByGenre = np.array(tempoByGenre)
    


def getTempos(allTags):
    tempos=[]
    for tag in allTags:
        try: 
            temp = song.search(title=tag.title.encode('ASCII', 'ignore'), 
                    artist=tag.artist.encode('ASCII', 'ignore'))[0].get_audio_summary()['tempo']
            tempos.append(temp)
            print temp
        except:
            print 'not in db'
            tempos.append('NA')
    return tempos


def getLatLong(regionList, keysToTry={}, firstTime=True, numTries=0):

    '''
    feed it a list of regions and it adds lat long data
    as a dict, with keys as the names of the region.
    regionDict['regionNameSearchInput']=['GoogleFormatedRegion', '[lat,long]]
    Recursive on missed list
    '''
    if firstTime:
        keysToTry=regionList.keys()
        print 'first time'

    g = geocoders.GoogleV3()
    maxTries = 20
    missedList=[]
    
    if not keysToTry or numTries > maxTries:
        return keysToTry

    for key in keysToTry:
        if key=='NA':
            regionList[key] = ['NA', 'NA']
            print 'none'
            continue
        try:
            place, (lat,lng) = g.geocode(key.encode('utf-8'))
            regionList[key] = [place, [lat,lng]]
            print 'found ', place 
        except:
            try:
                place, (lat,lng) = g.geocode(key, exactly_one=False)[0]
                regionList[key] = [place, [lat,lng]]
                print 'found multiple', place 
            except:
                regionList[key] = []
                missedList.append(key)
                print 'missed',regionList[key]
    
    print 'recursing. try: ', numTries+1
    getLatLong(regionList, missedList, False, numTries+1)

def findID3tagLocation(songDict):
    LMDdir='/Users/Tlacael/NYU/Latin_Music_Data/LMD/'
    for item in songDict:
        maxRatio=0
        path=[]
        for r,d,f in os.walk(LMDdir+item[1]):
            for files in f:
                if files.endswith('.mp3'):
                    temp = ratio(item[0], files)
                    if temp > maxRatio:
                        maxRatio=temp
                        path = os.path.join(r,files)
        item.append(os.path.join(r,paths)
    return
            

def getNamesAndGenreFromFolder(folder, ext = '.mp3'):
    
    names = []

    for r,d,f in os.walk(folder):
        predGenre= r.split('/')[-1]
        actGenre= r.split('/')[-2]
        #print actGenre, predGenre
        for files in f:
            if files.endswith(ext):
                pass
                names.append([files, actGenre, predGenre])
    return names



def getNamesFromFolder(folder, ext = '.csv'):
    names = []
    for r,d,f in os.walk(folder):
        for files in f:
            if files.endswith(ext):
                names.append(os.path.join(r,files))
    return names

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def extractRegion(filename):
    '''
    Scans a csv file that has artist, genre, region columns
    adds info to an artist dict 
        - artistdict['artistName']=['genreName', 'regionName']
    and makes a set of all regions as a dict with empty values
    to add lat long info later.
    return both.
    '''

    ifile = open(filename, "rb")
    r = csv.reader(ifile, delimiter=',')    
    
    artistDict  = {}
    regionDict = {}
    for row in r:
        #remove space at beginning

        artistDict[row[0].strip().decode('utf-8')]=\
            [row[1].strip().decode('utf-8'), 
             row[2].strip().decode('utf-8')]
        
        regionDict[row[2].strip().encode('utf-8')]=[]
    


    return artistDict, regionDict



def extractRegion_old(filename):
    ifile  = open(filename, "rb")
    #reader = csv.reader(utf_8_encoder(ifile))
    reader = csv.reader(ifile)
    artist=unicode(filename[50:-4], 'ASCII', 'ignore')
    result=dict([])
    rows = [row for row in reader]
    #remove rows with size less than 2 for dict conversion
    notDone=True;
    while notDone:
        try:
            rows.remove([])
        except:
            notDone=False

    #csvDict = dict(rows)

    idx=[this[0] for this in rows]
    try:
        pos=idx.index('City')
        return artist, rows[pos+1][0]
    except:
        return artist, 'none'

def getSongInfoFromFiles(folder):
    names=getNamesFromFolder(folder, ext=".mp3")
    allTags=[d3.load(name).tag for name in names]
    songDict=dict([])
    dup=2;
    for tag in allTags:
        if not tag.artist:
            continue
        print tag.artist.encode('utf-8')
        if not songDict.has_key(tag.title):
            try:
                songDict.update({tag.title:{'artist':tag.artist,'album':tag.album, 'genre':tag.genre.name}})
            except:
                print 'missing info - skipped: ', tag.title
        else:   
            done=False
            while not done:
                if not songDict.has_key(tag.title+str(dup)):
                    songDict.update({tag.title+str(dup):{'artist':tag.artist,'album':tag.album, 'genre':tag.genre.name}})
                    done=True
                else:
                    dup+=1

    return songDict
    #x=[tag.artist for tag in allTags]
    
def addRegionDataToSong(songDict,csvFolder):
    regions=getAllRegionsAsDict(csvFolder)
    i=0
    for key in songDict:
        print songDict[key]['artist']
        try:
            songDict[key]['region']=regions[songDict[key]['artist']]
        except:
            print "artist not on file"
            songDict[key]['region']='NA'
    return songDict


def getAllRegionsAsDict(folder):
    
    names = getNamesFromFolder(folder)
    result=dict([])
    i=0;
    prev=0
    prevReg=0
    for name in names:
        artist,region = extractRegion(name)
        if prev == artist:
            print "duplicate ",artist, region, "prev: ",prev, prevReg
        result[artist]=region
        prev=artist
        prevReg=region
    return result



