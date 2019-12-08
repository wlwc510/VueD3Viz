import os
import pandas as pd

rootdirSentence=["D:/publications/preparing/00biblio-clas/Rs project data/Token_sentence"]

rootdirSensor=["D:/publications/preparing/00biblio-clas/Rs project data/Rs field label"]

rootdirApplication=["D:/publications/preparing/00biblio-clas/Rs project data/Rs application label"]

sentencedir=[]

sentencelist=[]

order=[]

orderNo=[]

sensordir=[]

applicationdir=[]

sensors=[]

applications=[]

for i in rootdirSentence:  # 定义函数，查找所有文件

    for parent, dirnames, filenames in os.walk(i):

        for filename in filenames:

            if filename.endswith(".txt") :

                sentencedir.append(os.path.join(parent, filename))

                order.append(filename.replace(".txt",""))

for i in rootdirSensor:  # 定义函数，查找所有文件

    for parent, dirnames, filenames in os.walk(i):

        for filename in filenames:

            if filename.endswith(".txt") :

                sensordir.append(os.path.join(parent, filename))

for i in rootdirApplication:  # 定义函数，查找所有文件

    for parent, dirnames, filenames in os.walk(i):

        for filename in filenames:

            if filename.endswith(".txt") :

                applicationdir.append(os.path.join(parent, filename))

count=0

for txt in sentencedir:

    orderF = order[count]

    with open(txt, "r") as f:

        cu = f.read()

        culist = cu.split("\n")

        subcount=1

        for cuone in culist:

            # if cuone.isspace():

            if cuone.strip()=="":

                print("为空")

            else:

                sentencelist.append(cuone)

                # print(str(cuone))

                orderNo.append(str(orderF)+"-"+str(subcount))

                subcount=subcount+1

                # print(orderF,"-",subcount)

        f.close()

        count=count+1

print(sentencelist)

print(orderNo)

print("sentence:",len(sentencelist), "orderno", len(orderNo))

count=0

for txt in sensordir:

    orderF = order[count]

    with open(txt, "r") as f:

        cu = f.read()

        culist = cu.split("\n")

        subcount=1

        for cuone in culist:

            # if cuone.isspace():

            if cuone.strip()=="":

                test=1

            elif str(cuone)=="1":

                sensors.append(str(orderF)+"-"+str(subcount))

                # orderNo.append(str(orderF)+"-"+str(subcount))

                subcount=subcount+1

                # print(orderF, "-", subcount)

            else:

                subcount = subcount + 1




        f.close()

        count=count+1

print("sensors", sensors)

count = 0

for txt in applicationdir:

    orderF = order[count]

    with open(txt, "r") as f:

        cu = f.read()

        culist = cu.split("\n")

        subcount = 1

        for cuone in culist:

            # if cuone.isspace():

            if cuone.strip() == "":

                test=1

            elif str(cuone) == "1":

                applications.append(str(orderF) + "-" + str(subcount))

                # orderNo.append(str(orderF)+"-"+str(subcount))

                subcount = subcount + 1

                # print(orderF, "-", subcount)

            else:

                subcount = subcount + 1

        f.close()

        count = count + 1

print("applications", applications)

pairs=[]

pairh=[]

pairt=[]

for st in sensors:

    stl=st.split("-")

    a=stl[0]

    # p.append(stl[1])
    for at in applications:

        atl=at.split("-")

        if a==atl[0]:

            pair=st+":"+at

            # tpair=[]
            #
            # tpair.append(())

            pairs.append(pair)

            pairh.append(st)

            pairt.append(at)

print(pairs)

df = pd.DataFrame({'orderno':orderNo,'sentence':sentencelist})

# df = pd.DataFrame(orderNo,columns="orderno")

df.to_csv("order-sentence.csv",encoding="utf-8")

df = pd.DataFrame({'sensors':sensors})

df.to_csv("sensors.csv",encoding="utf-8")

df = pd.DataFrame({'applications':applications})

df.to_csv("applications.csv",encoding="utf-8")

df = pd.DataFrame({'pairs':pairs})

df.to_csv("pairs_str.csv",encoding="utf-8")

df = pd.DataFrame({'head':pairh,'tail':pairt})

df.to_csv("pairs.csv",encoding="utf-8")






