from pyechonest import song, track
from subprocess import Popen, PIPE
import os
import eyed3 as d3
import cPickle as pickle
import sys
import json
import time as t

def uploadToEcho(filenames):
    retry=[]
    successTracks = []
    i=0
    if not os.path.isdir('echoNestUploads'):
        os.mkdir('echoNestUploads')
     
    for f in filenames:
        time = t.localtime()
        time = str(time.tm_hour) +'_' + str(time.tm_min) + '_' + str(time.tm_sec)
        try:
            print 'uploading...', f
            rTrack = track.track_from_file(open(f, 'r'), 'mp3')
            if rTrack:
                print rTrack.tempo
                print rTrack.time_signature
                print 'gather track info'
                trackInfo = [f, rTrack.tempo, rTrack.time_signature, rTrack.id]
                successTracks.append(trackInfo)
                pickle.dump(trackInfo, open('echoNestUploads/'+str(i)+'-'+time+'.pkl', 'w'))
                try:
                    print 'saving id3'
                    tag = d3.load(f).tag
                    tag.bpm = rTrack.tempo
                    tag.copyright_url = 'echo'
                except:
                    print 'id3 failed'
        except:
            print "couldn't upload"
            retry.append(f)
        i+=1

    return successTracks, retry
               
def getENMFPcode(filename):
    (stdout, stderr) = Popen(["/Users/Tlacael/lib/ENMFP_codegen/codegen.Darwin",filename, "5", "20"], stdout=PIPE).communicate()
    code = json.loads(stdout)[0]

    return code

def lookupEcho(code):
    track = song.identify(query_obj=code)
    return track


def scanFolder(folder):

    if isinstance(folder, list):
        files = folder
    elif os.path.isdir(folder):
        files = getNamesAndGenreFromFolder(folder)
    else:
        return -1
    notAtEcho=[]
    notScanned=[]
    foundTracks = []
    retry=[]
    i=0
    if not os.path.isdir('echoNestResults'):
        os.mkdir('echoNestResults')
        
    for f in files:
        time = t.localtime()
        time = str(time.tm_hour) +'_' + str(time.tm_min) + '_' + str(time.tm_sec)
        try:
            print 'decodeing'
            print f
            code = getENMFPcode(f)
            try:
                print 'looking up on echonest'
                track = song.identify(query_obj=code)
            except:
                print "couldn't connect to server"
                retry.append(f)
            if not track:
                print 'song not at echonest'
                notAtEcho.append(f)
                continue
                
            print 'track:',track
            track=track[0]
            trackInfo = [f, track.audio_summary['tempo'], track.audio_summary['time_signature'], track.id]
            foundTracks.append(trackInfo)
            pickle.dump(foundTracks, open('echoNestResults/'+str(i)+'-'+time+'.pkl', 'w'))
            try:
                tag = d3.load(f).tag
                tag.copyright_url = 'echo'
                tag.bpm = track.audio_summary['tempo']
                tag.save()
            except:
                print 'id3 fail'
        except :
            print  "couldn't scan", sys.exc_info()[0]
            notScanned.append(f)
                
        i+=1
    return foundTracks, notAtEcho, notScanned, retry


def getNamesAndGenreFromFolder(folder, ext = '.mp3'):

    names = []

    for r,d,f in os.walk(folder):
        #print actGenre, predGenre
        for files in f:
            if files.endswith(ext):
                names.append(os.path.join(r, files))
    return names

