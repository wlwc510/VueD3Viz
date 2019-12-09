import networkx as nx

import matplotlib.pyplot as plt

import pandas as pd

# import numpy as np

ords=pd.read_csv("order-sentence.csv",index_col=[0],encoding="utf-8")

orders=ords.values.flatten().tolist()

orderdict=dict(zip(orders[0::2],orders[1::2]))

print(orderdict)

pairs=pd.read_csv("pairs.csv",index_col=[0],encoding="utf-8")

sensors=pd.read_csv("sensors.csv",index_col=[0],encoding="utf-8")

applications=pd.read_csv("applications.csv",index_col=[0],encoding="utf-8")

merge_result_tuples = [(getattr(xi,"head"),getattr(xi,"tail")) for xi in pairs.itertuples(index=True)]

# 替换成句子

# relilist = []
#
# for reli in merge_result_tuples:
#
#     h=reli[0]
#
#     t=reli[1]
#
#     relilist.append((orderdict.get(h),orderdict.get(t)))
#
# merge_result_tuples = relilist

sensors=sensors.values.flatten().tolist()

#替换成句子

# senilist=[]
#
# for seni in sensors:
#
#     senilist.append(orderdict.get(seni))
#
# sensors=senilist

applications=applications.values.flatten().tolist()

cm=merge_result_tuples

for a in applications:

    for r in cm:

        if r[0] == a:

            merge_result_tuples.remove(r)

for s in sensors:

    for r in cm:

        if r[1] == s:

            merge_result_tuples.remove(r)

print("输出sensors全部节点：{}".format(len(sensors)))

print("输出applications全部节点：{}".format(len(applications)))

ca=applications

cs=sensors

for a in ca[:]:

    for s in cs[:]:

        if a == s:

            sensors.remove(s)

            applications.remove(a)

print(applications[0])

print("输出sensors后全部节点：{}".format(len(sensors)))

print("输出applications后全部节点：{}".format(len(applications)))

# sensors = sensors[0:12]
#
# applications = applications[0:12]

# 替换成句子

# appilist = []
#
# for appi in applications:
#
#     appilist.append(orderdict.get(appi))
#
# applications = appilist

sl=len(sensors)

poss=[]

count=0

for left in range(sl):

    posi = (0.1, (100 + 100 * count) % 1000 / 1000)

    poss.append(posi)

    count+=1

print("poss",poss)

al=len(applications)

posa=[]

count=0

for right in range(al):

    posi=(0.9,(120+100*count)%1000/1000)

    posa.append(posi)

    count+=1

print("posa",posa)

posdict1=dict(zip(sensors,poss))

posdict2=dict(zip(applications,posa))

posdict1.update(posdict2)

color1=['g' for i in range(12)]

color2=['b' for i in range(12)]

colors=color1+color2

print("color:", colors)

# merge_result_tuples=merge_result_tuples[0:30]

G=nx.Graph()

# G=nx.DiGraph()

G.add_nodes_from(sensors)

G.add_nodes_from(applications)

print("输出sensors全部节点：{}".format(len(sensors)))

print(sensors)

print("输出applications全部节点：{}".format(len(applications)))

print(applications)

print("输出G全部节点：{}".format(G.number_of_nodes()))

cm=merge_result_tuples

for r in cm[:]:

    tag=False

    for s in sensors:

        if r[1]== s:

            merge_result_tuples.remove(r)

        if r[0] == s:

            tag=True

    if not tag and r in merge_result_tuples:

        merge_result_tuples.remove(r)

for r in cm[:]:

    tag = False

    for a in applications:

        if r[0] == a:

            merge_result_tuples.remove(r)

        if r[1] == a:

            tag=True

    if not tag and r in merge_result_tuples:

        merge_result_tuples.remove(r)

print("输出merge_result_tuples全部边的数量：{}".format(len(merge_result_tuples)))

G.add_edges_from(merge_result_tuples)

print("输出全部节点：{}".format(G.nodes()))

print("输出全部节点：{}".format(G.number_of_nodes()))

print("输出全部边：{}".format(G.edges()))

print("输出全部边的数量：{}".format(G.number_of_edges()))

# posc=nx.get_node_attributes(G,'pos')

nx.draw(G,with_labels=True,node_color=colors)

# nx.draw_node_network,pos=posdict1

# nx.draw(G,with_labels=True,node_color=colors)

# ,with_labels=True,arrowsize=0.5,arrowstyle='->'

# nx.draw_networkx_nodes(G,posdict2,node_color='blue',with_label=True)
#
# nx.draw_networkx_edges(G,posdict2,arrowstyle='->',arrowsize=10)

plt.show()



