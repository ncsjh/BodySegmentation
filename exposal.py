import pandas as pd

data=pd.read_excel('exposal.xlsx', index_col=0)
def getViews():
    return list(set(data.index.values))
def getPositions(view):
    frame = data[data.index == view]
    return list(set(frame.position.values))

def getExposal(view, position):
    frame = data[data.index == view]
    kvp=frame[frame['position']==position].values[0,1]
    ma=frame[frame['position']==position].values[0,2]
    msec=frame[frame['position']==position].values[0,3]
    mas=frame[frame['position']==position].values[0,4]
    return {'kvp':kvp, 'ma':ma, 'msec':msec, 'mas':mas}