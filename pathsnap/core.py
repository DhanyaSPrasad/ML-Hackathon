"""PathSnap data loading, candidate generation, features and route decoding."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from math import radians, sin, cos, atan2, sqrt

FEATURES=['distance_m','bearing_diff','road_length_m','road_bearing','gps_speed_mps','time_gap_s','heading_change','candidate_rank']

def read_trip(folder):
    folder=Path(folder); tid=folder.name
    def read(name,cols):
        p=folder/f'{tid}.{name}'
        if not p.exists(): raise FileNotFoundError(p)
        return pd.read_csv(p,sep=r'\s+',header=None,names=cols,engine='python')
    track=read('track',['lon','lat','timestamp'])
    nodes=read('nodes',['lon','lat'])
    arcs=read('arcs',['from_node','to_node'])
    route=read('route',['arc_id'])['arc_id'].astype(int).to_numpy()
    return track,nodes,arcs,route

def haversine(lon1,lat1,lon2,lat2):
    R=6371000.; p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def bearing(lon1,lat1,lon2,lat2):
    p1,p2=np.radians(lat1),np.radians(lat2); dl=np.radians(lon2-lon1)
    y=np.sin(dl)*np.cos(p2); x=np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.degrees(np.arctan2(y,x))+360)%360

def make_network(nodes,arcs):
    ij=arcs[['from_node','to_node']].to_numpy(dtype=int)
    valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<len(nodes))&(ij[:,1]<len(nodes))
    ij=ij[valid]; arc_ids=np.flatnonzero(valid)
    a=nodes.iloc[ij[:,0]][['lon','lat']].to_numpy(); b=nodes.iloc[ij[:,1]][['lon','lat']].to_numpy()
    mid=(a+b)/2
    lat0=float(nodes.lat.mean()); sx=111320*np.cos(np.radians(lat0)); sy=110540
    xy=np.column_stack((mid[:,0]*sx,mid[:,1]*sy))
    tree=cKDTree(xy)
    lengths=haversine(a[:,0],a[:,1],b[:,0],b[:,1])
    bearings=bearing(a[:,0],a[:,1],b[:,0],b[:,1])
    return {'ij':ij,'ids':arc_ids,'a':a,'b':b,'mid':mid,'xy':xy,'tree':tree,'lengths':lengths,'bearings':bearings,'sx':sx,'sy':sy}

def candidate_rows(track,net,route=None,k=8,radius_m=180):
    n=len(track); rows=[]; candidate_ids=[]; candidate_dists=[]
    coords=track[['lon','lat']].to_numpy(); lat0=np.radians(float(np.mean(coords[:,1])))
    xy=np.column_stack((coords[:,0]*net['sx'],coords[:,1]*net['sy']))
    kk=min(k,len(net['ids']))
    ds,ix=net['tree'].query(xy,k=kk)
    if kk==1: ds,ix=ds[:,None],ix[:,None]
    speed=np.zeros(n); gaps=np.ones(n)
    if n>1:
        gaps[1:]=np.maximum(np.diff(track.timestamp.to_numpy()),0.01)
        speed[1:]=haversine(coords[:-1,0],coords[:-1,1],coords[1:,0],coords[1:,1])/gaps[1:]
    gpshead=np.zeros(n)
    if n>1: gpshead[1:]=bearing(coords[:-1,0],coords[:-1,1],coords[1:,0],coords[1:,1])
    target=np.full(n,-1,dtype=int)
    if route is not None and len(route):
        # For each GPS point, choose the geometrically nearest directed arc from the supplied reference route.
        # This uses route labels only while constructing offline training/development labels.
        route_ids=np.unique(route[route>=0])
        id_to_pos={int(a):i for i,a in enumerate(net['ids'])}
        route_pos=np.asarray([id_to_pos[int(a)] for a in route_ids if int(a) in id_to_pos],dtype=int)
        if len(route_pos):
            route_tree=cKDTree(net['xy'][route_pos])
            _,near=route_tree.query(xy,k=1)
            target=net['ids'][route_pos[np.asarray(near,dtype=int)]]
    for i in range(n):
        order=np.argsort(ds[i]);
        for rank,j in enumerate(ix[i][order]):
            d=float(ds[i][rank]); aid=int(net['ids'][j]); bd=abs((float(net['bearings'][j])-gpshead[i]+180)%360-180)
            rows.append([d,bd,float(net['lengths'][j]),float(net['bearings'][j]),float(speed[i]),float(gaps[i]),0.0,rank])
            candidate_ids.append(aid); candidate_dists.append(d)
    X=pd.DataFrame(rows,columns=FEATURES)
    y=np.array([int(a==target[i]) for i in range(n) for a in net['ids'][ix[i][np.argsort(ds[i])]]],dtype=int) if route is not None else None
    return X,y,np.asarray(candidate_ids),np.asarray(candidate_dists),target

def viterbi(candidate_ids, distances, scores, arcs, k=8, transition_penalty=3.0):
    """Decode candidate arcs; adjacent directed arcs connect when previous.to == current.from."""
    n=len(scores)//k
    if n==0:return np.array([],dtype=int)
    cand=np.asarray(candidate_ids).reshape(n,k); dist=np.asarray(distances).reshape(n,k); prob=np.asarray(scores).reshape(n,k)
    # arcs rows map arc id to directed endpoint pair
    edge=arcs.to_numpy(dtype=int); valid=(edge[:,0]>=0)&(edge[:,1]>=0)
    edge= edge
    dp=np.full((n,k),-1e12); back=np.zeros((n,k),dtype=int)
    dp[0]=np.log(np.maximum(prob[0],1e-9))-dist[0]/80.
    for t in range(1,n):
        for j,aid in enumerate(cand[t]):
            if aid>=len(edge): continue
            frm,to=edge[aid]
            for q,prev in enumerate(cand[t-1]):
                if prev>=len(edge): continue
                connected=(edge[prev,1]==frm)
                trans=0.0 if connected else -transition_penalty
                val=dp[t-1,q]+trans
                if val>dp[t,j]: dp[t,j]=val; back[t,j]=q
            dp[t,j]+=np.log(max(prob[t,j],1e-9))-dist[t,j]/80.
    state=int(np.argmax(dp[-1])); out=np.zeros(n,dtype=int)
    for t in range(n-1,-1,-1):
        out[t]=cand[t,state]
        if t: state=back[t,state]
    return out

def trip_features(folder,k=8,max_observations=None):
    track,nodes,arcs,route=read_trip(folder)
    if max_observations and len(track)>max_observations:
        keep=np.linspace(0,len(track)-1,max_observations).astype(int)
        track=track.iloc[keep].reset_index(drop=True)
    net=make_network(nodes,arcs)
    X,y,cids,dists,target=candidate_rows(track,net,route,k)
    return track,nodes,arcs,route,net,X,y,cids,dists,target
