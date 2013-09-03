import csv
import os
import eyed3 as d3
from geopy import geocoders
from pyechonest import song
'''
Example

Get dataa from id3 tags for mp3's in a folder:
songDict = getSongInfoFromFiles('/Users/Tlacael/NYU/Latin_Music_Data/LMD/')

Add region data to songDict:
songDict = addRegionDataToSong(songDict,'/Users/Tlacael/lib/gTrends/csvsubfolder')

'''
def getTempos(allTags)
    tempos=[]
    for tag in allTags:
        try: 
            temp = song.search(title=tag.title.encode('ASCII', 'ignore'), artist=tag.artist.encode('ASCII', 'ignore'))[0].get_audio_summary()['tempo']
            tempos.append(temp)
            print temp
        except:
            print 'not in db'
            tempos.append('NA')

def getLatLong(regionList, option=False):
    g = geocoders.GoogleV3()

    missedList=[]

    for key in regionList:
        if regionList[key]=='none':
            print 'none'
            continue
        try:
            place, (lat,lng) = g.geocode(regionList[key])
            regionList[key]={'region':regionList[key], 'latlong':[lat,lng]}
            print 'found ', place 
        except:
            missedList.append(key)
            print 'missed'
    return regionList, missedList

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
    ifile  = open(filename, "rb")
    reader = csv.reader(utf_8_encoder(ifile))
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



