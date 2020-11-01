#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:26:43 2017

@author: Yuting1
"""
#import os
#os.environ["PYSPARK_SUBMIT_ARGS"] = (
#    "--packages graphframes:graphframes:0.5.0-spark1.6-s_2.10 pyspark-shell"
#)


import pyspark
import sys

#from pyspark.graphframes import graphframes


sc=pyspark.SparkContext()
input_path=sys.argv[1]
communities_path=sys.argv[2]
betweenness_path=sys.argv[3]
data=sc.textFile(input_path)\
    .map(lambda x: x.split(","))
header=data.first()
ratings=data.filter(lambda x:x!=header)\
    .map(lambda x:(int(x[1]),int(x[0])))
    
edge=ratings.join(ratings)\
    .filter(lambda x:x[1][0]!=x[1][1])\
    .map(lambda x:(x[1],1))\
    .reduceByKey(lambda x,y:x+y)\
    .filter(lambda x:x[1]>=3)\
    .map(lambda x:(x[0][0],x[0][1]))


adj_list=edge.map(lambda x:(x[0],[x[1]]))\
    .reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],x[1]))
adj_list=adj_list.collectAsMap()
vertex_list=adj_list.keys()
vertex=edge.map(lambda x:x[0]).distinct()


#edgeweight=[]

def betweenness(node):
    #
    
    edgeweight=[]
    distance={k:0 for k in vertex_list}
    path={k:[] for k in vertex_list}
    count={k:0 for k in vertex_list}
    to_explore=[]
    S=[]

    count[node]=1
    distance[node]=0
    to_explore.append(node)
    while len(to_explore)>0:
        v=to_explore.pop(0)
        S.append(v)
        for item in adj_list[v]:
            if distance[item]==0 and item !=node:
                distance[item]=distance[v]+1
                to_explore.append(item)
            if distance[item]==distance[v]+1:
                path[item].append(v)
                count[item]+=count[v]
                

    credit={k:1.0 for k in vertex_list}
    while len(S)>0:
        v=S.pop()
        for item in path[v]:
            credit[item]+=credit[v]/count[v]
            edgeweight.append(((v,item),credit[v]/count[v]*count[item]))
    return edgeweight


def truncate(f):
    a=str(f)
    pos=a.index('.')
    return a[0:pos+1]+a[pos+1]

a=vertex.map(lambda x:betweenness(x))
b=a\
    .flatMap(lambda x: [(item[0],item[1]) for item in x])\
    .map(lambda x:(tuple(sorted(x[0])),x[1]))\
    .reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],x[1]/2))\
    .map(lambda x:(x[0],float(truncate(x[1]))))
between=b.sortByKey()\
    .map(lambda x:(x[0][0],x[0][1],x[1]))

between.coalesce(1).saveAsTextFile(betweenness_path)

