from urllib.parse import unquote
import xml.etree.ElementTree as ET, json

# Normalizes rekordbox file URLs/paths into local decoded paths.
# In:
# - path: raw path from rekordbox XML.
# Out:
# - normalized and URL-decoded file path.
def normalize_filepath(path: str):
    if path.startswith("file://localhost/"):
        path = path.removeprefix("file://localhost/")
    # URL decode the path to handle accented characters
    return unquote(path)

# find list of playlists that contain the given song
# In:
# - trackId: track ID stored in rekordbox XML
# - allPlaylists: playlists imported from rekordbox XML (and converted to JSON)
def find_matching_playlists(trackId: str, allPlaylists: list):
    matchingPlaylists = []
    for playlist in allPlaylists:
        if trackId in playlist['tracks']:
            matchingPlaylists.append(playlist['name'])
    return matchingPlaylists

# load rekordbox XML file and extract desired info into a JSON file
# In:
# - xmlPath: path to rekordbox XML file
# - outJsonPath: path to output JSON file
# - allowedFolders: rekordbox folders that will be imported
def parse_rekordbox(xmlPath, outJsonPath, allowedFolders):
    tree = ET.parse(xmlPath)
    rootXML = tree.getroot()

    # map to easily identify which playlists a track is in.
    # also lets us keep track of tracks that are actually used.
    trackIdToPlaylistMap = {}

    # Extract playlists
    # playlists and folders are saved as nodes
    #   folders have child nodes
    #   playlists have child tracks
    playlists = []
    playlistsXML = rootXML.find('PLAYLISTS').find('NODE')
    for nodeXML in playlistsXML.findall('NODE'):
        name = nodeXML.get('Name')
        if name:
            if name in allowedFolders:
                for subNodeXML in nodeXML.findall('NODE'):
                    playlistName = subNodeXML.get('Name')
                    trackIds = []
                    for trackXML in subNodeXML.findall('TRACK'):
                        id = trackXML.get('Key')
                        trackIds.append(id)

                        # append trackId to map
                        if id in trackIdToPlaylistMap.keys():
                            trackIdToPlaylistMap[id].append(playlistName)
                        else:
                            trackIdToPlaylistMap[id] = [playlistName]
                    playlists.append({'name': playlistName,
                                      'tracks': trackIds})

    # Extract tracks (only do tracks we need)
    tracks = []
    collectionXML = rootXML.find('COLLECTION')
    for trackXML in collectionXML.findall('TRACK'):
        id = trackXML.get('TrackID')
        if id in trackIdToPlaylistMap.keys():

            trackPath = trackXML.get('Location')
            # skip .m4a files (unsupported)
            if trackPath.endswith('.m4a'):
                continue

            tracks.append({
                'id': id,
                'name': trackXML.get('Name'),
                'artist': trackXML.get('Artist'),
                'path': normalize_filepath(trackPath),
                'bpm': trackXML.get('AverageBpm'),
                'key': trackXML.get('Tonality'),
                'length': trackXML.get('TotalTime'),
                'sampleRate': trackXML.get('SampleRate'),
                'playlists': find_matching_playlists(id, playlists)
            })
    
    with open(outJsonPath,'w', encoding='utf-8') as f:
        outJson = {'tracks': tracks, 'playlists': playlists}
        json.dump(outJson, f, indent=2)


if __name__ == "__main__":
    rekordboxXMLPath = 'C:\\Users\\carlo\\Music\\rekordbox\\rekordbox.xml'
    outJSONPath = 'D:\\projects\\music-ml\\out\\tracks.json'
    allowedFolders = {'Dubstep', 'Riddim'}
    parse_rekordbox(rekordboxXMLPath, outJSONPath, allowedFolders)
