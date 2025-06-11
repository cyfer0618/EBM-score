import requests
from bs4 import BeautifulSoup as bs
import ssl
import pandas as pd
import numpy as np
import codecs
import time
import re
import os
from urllib.request import urlopen
#from . import models
#from .models import Sentences, WordOptions, Wordsinsentence
import json
wfile = open("edges.txt", 'w+', encoding='utf-8')
xfile = open("nodes.txt", 'w+', encoding='utf-8')
yfile = open("noSHR.txt", 'w+', encoding='utf-8')
afile = open("morpho.txt", 'w+', encoding='utf-8')
bfile = open("noSHR_.txt", 'w+', encoding='utf-8')

number_of_sentence = 0
dict_verb = {}
dict_noun = {}
dict_person = {}
edge_dict = {}
node_dict = []

def getdatafromsite(inputsent):  # Scrapping data from site
    #inputline = inputsent.line
    #inputtype = inputsent.linetype
    problem = []
    pbwords = []
    s_type = 'WX'

    s_d = inputsent

    s_c = s_d.replace(" ", "+")
    # for utilising the sanskrit heritage app, the url has been specified
    # urlname = ("http://sanskrit.inria.fr/cgi-bin/SKT/sktgraph?lex=SH&st=t&us=f&cp=t&text=" +
    #            s_c + "&t=" + s_type + "&topic=&mode=g&corpmode=&corpdir=&sentno=")

    #urlname = "http://sanskrit.inria.fr/cgi-bin/SKT/sktgraph?lex=SH;st=t;us=f;cp=t;text=surenxrapramuKEH+inxrAxiBiH samagrEH+samaswEH+vibuXEH+xevEH+upewya+samIpam+Agawya;t=WX;topic=;mode=g;corpmode=;corpdir=;sentno="
    #urlname = "https://sanskrit.inria.fr/cgi-bin/SKT/sktgraph.cgi?t=WX;lex=SH;font=roma;cache=f;st=t;us=f;text="+s_c+"surenxrapramuKEH+inxrAxiBiH+samagrEH+samaswEH+vibuXEH+xevEH+upewya+samIpam+Agawya;topic=;abs=f;allSol=;corpmode=;corpdir=;sentno=;cpts="

    urlname = "http://sanskrit.inria.fr/cgi-bin/SKT/sktgraph.cgi?t=SL;lex=SH;font=roma;cache=f;st=t;us=t;text="+s_c+";topic=;abs=f;allSol=;corpmode=;corpdir=;sentno=;cpts="

    print(urlname)
    #page = requests.get(urlname)
    context = ssl._create_unverified_context()

    #soup = bs(page.text, 'html.parser')
    res = urlopen(urlname,context=context)
    soup = bs(res, 'html.parser')
    

    table = soup.table
    tablebody = table.find('table', {'class': 'center'})
    t = pd.DataFrame(
        columns=['id', 'level', 'color_class', 'position', 'chunk_no', 'word', 'lemma', 'pre_verb', 'morph', 'colspan',
                 'wordlenth', 'aux_inf', 'subminp', 'maxp', 'endposition'])
    # columns=['wordid', 'level', 'color_class', 'position', 'chunk_no', 'lemma', 'pre_verb', 'morph', 'colspan',
    #              'wordlength', 'aux_info', 'word', 'subminp', 'maxp', 'endposition'])

    # t = pd.DataFrame(
    #     columns=['id', 'level', 'color_class', 'position', 'word', 'lemma', 'morph', 'colspan'])

    i = 0
    id_ = 0
    if not (tablebody):  #### wronginputs
        print(inputsent)
        bfile.write("URL : "+urlname+"\n")
        bfile.write("Sentence : "+inputsent+"\n")
        print('no table body of given inputline')
        return None

    # for valid entries corresponding to Wordsinsentence

    for child in tablebody.children:
        if (child.name == 'tr'):
            if i < 1:
                linechar = []
                c = 0
                for char in child.children:
                    linechar.append(char.string)
                    c += 1
                i += 1
                line_header = "".join(linechar)
                linechunks = line_header.split("\xa0")
                continue
            position_ = 0
            j = 0
            for wordtable in child.children:
                c = 0
                #print("1 ",wordtable.contents)
                for ch in linechar[0:position_]:
                    if (re.match('\xa0', ch)):  # or (re.match('_',ch))
                        c += 1
                    # if the contents exist in wordtable
                    # following assignings are carried out.
                if (wordtable.contents):
                    color_ = wordtable.table.get('class')[0]
                    colspan_ = wordtable.get('colspan')
                    word_ = wordtable.table.tr.td.string
                    onclickdatas_ = wordtable.table.tr.td.get('onclick')
                    #print("onclickdatas_2 ",onclickdatas_)
                    #print("word_ ",word_)

                    for onclickdata_ in onclickdatas_.split("<br>"):  # required splits carried out at positions stated
                        #morphslist_ = re.findall(r'{ (.*?) }', onclickdata_)  # .split(' | ')
                        #morphslist_1 = re.findall(r'{ (.*?) }', onclickdata_)  # .split(' | ')
                        morphslist_ = re.findall(r'{(.*?)}', onclickdata_)  # .split(' | ')
                        #print("morphslist_1",morphslist_1)
                        #print("morphslist_",morphslist_)
                        new_morphslist_ = []
                        for x in morphslist_:
                            if x[0] == " ":
                                continue
                            else:
                                new_morphslist_.append(x)
                        #print("new_morphslist_",new_morphslist_)
                        morphslist_ = new_morphslist_
                        #exist
                        #morphslist_2 = list(set(morphslist_) - set(morphslist_1))
                        #morphslist_ = morphslist_2
                        #print("onclickdata_4 ",onclickdata_)
                        #print("morphslist_",morphslist_)
                        #ldata = str(re.search(r'{.*?}\[(.*)\]', onclickdata_).group(1))
                        ldata = str(re.search(r'(.*?)\[(.*)\]', onclickdata_).group(2))
                        #print("ldata_1",ldata)

                        ldata = str(re.sub(r'</?a.*?>|</?i>', "", ldata))
                        #print("ldata_2",ldata)
                        if (ldata.find(']') != -1):
                            temp = str(re.search(r'(.*?)\]', ldata).group(1))
                            ldata = temp

                        lemmadata = ldata.split(" ")
                        #print(lemmadata)
                        if len(lemmadata) > 1:
                            auxi_ = " ".join(lemmadata[1:])
                        else:
                            auxi_ = ""
                        lemmas_ = "".join(lemmadata[0])
                        #print("lemmas_",lemmas_)
                        #lemmalists_ = lemmas_.split("-")
                        lemmalists_ = lemmas_.split("|")
                        #print("lemmalists_1",lemmalists_)
                        if (len(lemmalists_) > 1):
                            preverb_ = ",".join(lemmalists_[0:(len(lemmalists_) - 1)])
                            lemmalist_ = "".join(lemmalists_[-1:]).split("_")
                        else:
                            preverb_ = ""
                            lemmalist_ = "".join(lemmalists_[0]).split("_")
                        #print("preverb_",preverb_)
                        #print("lemmalists_2",lemmalists_)
                        if (len(lemmalist_) > 1):
                            auxi_ = auxi_ + " sence of lemma = " + "".join(lemmalist_[1:(len(lemmalist_))])
                            lemma_ = "".join(lemmalist_[0])
                        else:
                            lemma_ = "".join(lemmalist_[0])
                        #morphs_ = str(morphslist_[0])
                        morphs_ = str(morphslist_[0])
                        for morph_ in morphs_.split(" | "):
                            #minp = min(position_)
                            #print("Line 144 : ",id_, i, color_, position_, c + 1, word_, lemma_, preverb_, morph_, int(colspan_), len(word_), auxi_, 0, 0, int(position_) + int(colspan_))
                            t.loc[id_] = [id_, i, color_, position_, c + 1, word_, lemma_, preverb_, morph_, int(colspan_), len(word_), auxi_, 0, 0, int(position_) + int(colspan_)]
                            #print(colspan_)
                            # t.loc[id_] = [id_, i, color_, position_, word_, lemma_, morph_,
                            #               int(colspan_)]
                            if (re.match(r'grey_back', color_)):
                                if not (word_ == 'pop'):
                                    yfile.write("URL : "+urlname+"\n")

                                    yfile.write("Unrecognised Word : "+word_+" "+lemma_+"\n")

                                    yfile.write("Sentence : "+inputsent+"\n")
                                    problem.append(id_)  # filling entries to problem list
                                else:
                                    id_ = id_ - 1
                            id_ += 1

                    position_ += int(colspan_)
                else:
                    position_ += 1
            i = i + 1
            dict_ = {'t':t,'line_header':line_header}

    #print(dict_)
    nt = t
    #exit(1)
            #wfile.write(t)

    wordfromchunk = {}
    colspanofchunk = {}
    for c in t.chunk_no.unique():
        df1 = t.loc[t['chunk_no'] == c]
        minp = min(df1.position)
        for i in df1.index:
            t.loc[i, 'subminp'] = t.loc[i, 'position'] - minp

            t.loc[i, 'maxp'] = t.loc[i, 'subminp'] + t.loc[i, 'colspan']

        wordfromchunk[c] = []
        df1 = t.loc[t['chunk_no'] == c]
        for w in df1.word.unique():
            wordfromchunk[c].append(w)
        colspanofchunk[c] = max(df1['maxp'])

    words = t.word.unique()
    levelofword = {}
    posofword = {}
    idsofword = {}
    colspanofword = {}
    for w in words:
        levelofword[w] = min(t.loc[t['word'] == w].level)
        posofword[w] = min(t.loc[t['word'] == w].subminp)
        colspanofword[w] = max(t.loc[t['word'] == w].colspan)
        idsofword[w] = t.loc[t['word'] == w].id.unique()
    sentwords = inputsent.split(' ')
    chunknum = {}
    c = 0
    for sw in sentwords:
        c = c + 1
        chunknum[sw] = c

    maxlevel = max(t.level)
    levelrange = range(1, maxlevel + 1)
    chunkrange = range(1, max(t.chunk_no) + 1)
    positionrange = range(max(t['position']) + 1)
    maxpos = max(t['position'] + t['colspan'])
    levelpos = {}
    levelwordpos = {}

    df = t
    for l in levelrange:
        levelpos[l] = []
        levelwordpos[l] = []
        df1 = t.loc[t['level'] == l]
        for p in df1.position.unique():
            levelwordpos[l].append(p)
        for p in positionrange:
            check = True
            for i in df1.index:
                if (p == df1.loc[i, 'position']) or ((p > df1.loc[i, 'position']) and (p < df1.loc[i, 'endposition'])):
                    check = False
                    break
            if check:
                levelpos[l].append(p)

    dragdata = {}
    links = {}
    ic = 0

            # # for dw in wordsinsentence :
            # for dw in wordsdata:
            #     if dw.isSelected:
            #         lemma = dw.lemma
            #         if not str(dw.pre_verb) == '':
            #             lemma = dw.pre_verb + '-' + lemma
            #         if not str(dw.aux_info) == '':
            #             if dw.aux_info[-18:-1] == 'sence of lemma = ':
            #                 lemma = lemma + '-' + dw.aux_info[-1:]
            #                 if not str(dw.aux_info[:-18]) == ' ':
            #                     lemma = lemma + ' (' + dw.aux_info[:-18] + ')'
            #             else:
            #                 lemma = lemma + ' (' + dw.aux_info + ')'
            #         data1 = {

            #             "properties": {
            #                 "title": str(dw.id) + ' : ' + dw.word + '<br>[' + lemma + ']',
            #                 "inputs": {
            #                     "in-" + str(dw.id): {
            #                         "label": dw.morph
            #                     }
            #                 },
            #                 "outputs": {
            #                     "out-" + str(dw.id): {
            #                         "label": ' '
            #                     }
            #                 }
            #             }
            #         }
            #         # print('here')
            #         dragdata['word_' + str(dw.id)] = data1

            #         if not dw.parent == -1:
            #             link1 = {

            #                 "fromOperator": 'word_' + str(dw.parent),
            #                 "fromConnector": "out-" + str(dw.parent),
            #                 "fromSubConnector": '0',
            #                 "toOperator": 'word_' + str(dw.id),
            #                 "toConnector": "in-" + str(dw.id),
            #                 "toSubConnector": "0",
            #                 "relationame": dw.relation

            #             }
            #             links[ic] = link1
            #             ic = ic + 1

    conflictslp = {};
    conflictslp1 = {};
    conflictslp1color = {}
    for i in df.index:
        conflictslp[
            str(df.loc[i].level) + '-' + str(df.loc[i].position) + '-' + str(df.loc[i].endposition) + '-' + df.loc[i].color_class] = []
    for key in conflictslp.keys():
        l = int(key.split('-')[0])
        p = int(key.split('-')[1])
        e = int(key.split('-')[2])

        for i in df.index:
            if (l == df.loc[i].level) and ((df.loc[i].position > p) and df.loc[i].position < e):
                if not str(df.loc[i].level) + '-' + str(df.loc[i].position) in conflictslp[key]:
                    conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))
            if not (l == df.loc[i].level):
                if ((df.loc[i].position > p - 1) and df.loc[i].position < e - 1):
                    if not str(df.loc[i].level) + '-' + str(df.loc[i].position) in conflictslp[key]:
                        conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))
            if ((df.loc[i].position < p) and df.loc[i].endposition > p + 1):
                conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))

    for key in conflictslp:
        conflictslp1[key.split('-')[0] + '-' + key.split('-')[1]] = conflictslp[key]
        conflictslp1color[key.split('-')[0] + '-' + key.split('-')[1]] = key.split('-')[3]

            
    #context dictionary containing every detailed bit of the word in the sentence
    # context = {'line': inputsent,'line_header':line_header, 'wordsdata': wordsdata, 'words': sentwords, 'chunknum': chunknum,
    #            'sentid': sent_id, 'dragdata': json.dumps(dragdata), 'links': json.dumps(links),
    #            'conflictslp': json.dumps(conflictslp1), 'colorlp': json.dumps(conflictslp1color),
    #            'levelofword': levelofword, 'levelrange': levelrange, 'posofword': posofword, 'idsofword': idsofword,
    #            'wordfromchunk': wordfromchunk, 'chunkrange': chunkrange, 'colspanofchunk': colspanofchunk,
    #            'colspanofword': colspanofword,'allwords': words, 'positionrange': positionrange, 'levelpos': levelpos, 'levelwordpos': levelwordpos, 'wordsinsentence': wordsinsentence, 'chunkwordids': chunkword
    context = {'line': inputsent,'line_header':line_header, 'chunknum': chunknum, 'conflictslp': json.dumps(conflictslp1), 'colorlp': json.dumps(conflictslp1color),
                       'levelofword': levelofword, 'levelrange': levelrange, 'posofword': posofword, 'idsofword': idsofword,
                       'wordfromchunk': wordfromchunk, 'chunkrange': chunkrange, 'colspanofchunk': colspanofchunk,
                       'colspanofword': colspanofword,'allwords': words, 'positionrange': positionrange, 'levelpos': levelpos, 'levelwordpos': levelwordpos}

    #handling the corner cases in sandhi check from the file named all_sandhi.txt
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'all_sandhi.txt')
    s = pd.read_csv(path, encoding='utf-8', sep=',')
    df_2 = pd.DataFrame(data=s)
    keys = conflictslp1.keys()
    for key in keys:
        value = conflictslp1[key]
        l = int(key.split('-')[0])
        p = int(key.split('-')[1])

        #extracting word from dataframe of words corresponding to key level and positon
        word_df1 = df[(df['level'] == l) & (df['position'] == p)]
        word_df1 = word_df1['word'].values[0]

        if len(value) == 0:
            a = 1
            #print("no conflicts_1")
            #print("no conflict_1",word_df1)
        elif len(value) != 0:
            for v in value:
                lv = int(v.split('-')[0])
                pv = int(v.split('-')[1])

                # extracting word from dataframe of words corresponding to value of the key level and positon

                word_df2 = df[(df['level'] == lv) & (df['position'] == pv)]
                word_df2 = word_df2['word'].values[0]
                if p > pv:
                #c1 = word_df2
                #c2 = word_df1
                    if len(word_df2) < len(word_df1):
                        n = len(word_df2)
                    else:
                        n = len(word_df1)
                    t = word_df2[-n:]
                    d = 0

                    for l1, l2 in zip(t, word_df1):
                        if l1 == l2:
                            d = d + 1

                    if n > 2 :
                        a = 1
                        #print("full conflict_1 : characters greater than 2")
                        #print("full conflict ",word_df1,word_df2)
                    elif d == 2:
                        C2 = word_df1[:2]
                        C1 = word_df2[-2:]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            a = 1
                            #print("conflict_1 ",word_df1,word_df2)
                            #print("conflict")
                        else:
                            value.remove(str(lv) + '-' + str(pv))
                    else:
                        C2 = word_df1[:1]
                        C1 = word_df2[-1:]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            a = 1
                            #print("conflict_2 ",word_df1,word_df2)
                            #print("conflict")
                        else:
                            #removing the cases of overlapping and sandhi
                            value.remove(str(lv) + '-' + str(pv))

                elif pv > p:
                    #c1 = word_df1
                    #c2 = word_df2
                    if len(word_df2) < len(word_df1):
                        n = len(word_df2)
                    else:
                        n = len(word_df1)
                    t = word_df1[-n:]
                    d = 0

                    for l1, l2 in zip(t, word_df2):
                        if l1 == l2:
                            d = d + 1

                    if n > 2:
                        a = 1
                        #print("full conflict : characters greater than 2")
                        #print("full conflict_2 ",word_df1,word_df2)
                    elif d == 2:
                        C1 = word_df1[-2:]
                        C2 = word_df2[:2]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            a = 1
                            #print("conflict")
                            #print("conflict_3 ",word_df1,word_df2)
                        else:
                            value.remove(str(lv) + '-' + str(pv))
                    else:
                        C1 = word_df1[-1:]
                        C2 = word_df2[:1]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            a = 1
                            #print("conflict")
                            #print("conflict_4 ",word_df1,word_df2)
                        else:
                            value.remove(str(lv) + '-' + str(pv))


                else:
                    l1 = len(word_df1)
                    l2 = len(word_df2)
                    if l1 <= l2:
                        d = 0
                        #c1 = word_df1
                        #c2 = word_df2
                        for letter1, letter2 in zip(word_df1, word_df2):
                            if letter1 == letter2:
                                d = d + 1
                            else:
                                break

                        if l1 > 2:
                            a = 1
                            #print("full conflict_3 ",word_df1,word_df2)
                            #print("full conflict : characters greater than 2")
                        elif d == 2:
                            C1 = word_df1[-2:]
                            C2 = word_df2[:2]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                a = 1
                                #print("3 conflict")
                                #print("conflict_5 ",word_df1,word_df2)
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                        else:
                            C1 = word_df1[-1:]
                            C2 = word_df2[:1]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                a = 1
                                #print("conflict")
                                #print("conflict_6 ",word_df1,word_df2)
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                    else:
                        #c1 = word_df2
                        #c2 = word_df1
                
                        d = 0
                        for letter1, letter2 in zip(word_df1, word_df2):
                            if letter1 == letter2:
                                d = d + 1
                            else:
                                break

                        if l2 > 2:
                            a = 1
                            #print("full conflict_4 ",word_df1,word_df2)
                            #print("full conflict : characters greater than 2")
                        elif d == 2:
                            C2 = word_df1[:2]
                            C1 = word_df2[-2:]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                a = 1
                                #print("conflict")
                                #print("conflict_7 ",word_df1,word_df2)
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                        else:
                            C2 = word_df1[:1]
                            C1 = word_df2[-1:]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                a = 1
                                #print("conflict")
                                #print("conflict_8 ",word_df1,word_df2)
                            else:
                                value.remove(str(lv) + '-' + str(pv))

    context['allvar'] = context
            #return context
    #print(context)
    #exit(1)
    dict_levelpos2words = {}
    dict_id2lem = {}
    dict_id2cng = {}
    dict_word = {}

    #First level and position to id
    for x in range(len(nt)):
        #print(nt['morph'][x])
        #print(nt['id'][x],nt['level'][x],nt['position'][x])
        #exit(1)
        l = nt['level'][x]
        p = nt['position'][x]
        l_p = str(l) + "-" + str(p)
        if l_p in dict_levelpos2words :
            dict_levelpos2words[l_p].append(nt['id'][x])
        else:
            dict_levelpos2words[l_p] = []
            dict_levelpos2words[l_p].append(nt['id'][x])
        dict_id2lem[nt['id'][x]] = nt['lemma'][x]
        dict_id2cng[nt['id'][x]] = nt['morph'][x]
        dict_word[nt['id'][x]] = {}
        dict_word[nt['id'][x]]["lemma"] = nt['lemma'][x]
        temp = nt['morph'][x].split(" ")
        if temp[0] == "*":
            nt['morph'][x] = "m. " + temp[1] + " " + temp[2]
        if len(temp) == 3:
            nmorph = temp[2] + " " + temp[1] + " " + temp[0]
            if nmorph in dict_noun:
                dict_word[nt['id'][x]]["cng"] = int(dict_noun[nmorph])
        if nt['morph'][x] in dict_noun:
            #print(1,nt['morph'][x])
            dict_word[nt['id'][x]]["cng"] = int(dict_noun[nt['morph'][x]])
        elif nt['morph'][x] in dict_verb:
            #print(2)
            dict_word[nt['id'][x]]["cng"] = int((-1) * (int(dict_verb[nt['morph'][x]]) * 10))
        elif len(nt['morph'][x].split(" ")) == 5:
            verbs = nt['morph'][x].split(" ")
            pre_verb = " ".join(verbs[:3])
            post_verb = " ".join(verbs[3:])
            # print(pre_verb)
            # print(post_verb)
            # print(pre_verb in dict_verb)
            # print(post_verb in dict_person)
            if pre_verb in dict_verb and post_verb in dict_person:
                pre_numb = int((-1) * ((int(dict_verb[pre_verb]) * 10) + int(dict_person[post_verb])))
                dict_word[nt['id'][x]]["cng"] = int(pre_numb)
            else :
                #print(pre_verb,post_verb)
                #print()
                afile.write("5 : "+str(nt['morph'][x])+"\n")
                try :
                    dict_word[nt['id'][x]]["cng"] = int(nt['morph'][x])
                except:
                    if nt['morph'][x] == "?":
                        dict_word[nt['id'][x]]["cng"] = 0
                    else:
                        dict_word[nt['id'][x]]["cng"] = int(-1)
        elif len(nt['morph'][x].split(" ")) == 4:
            verbs = nt['morph'][x].split(" ")
            pre_verb = " ".join(verbs[:2])
            post_verb = " ".join(verbs[2:])
            # print(pre_verb)
            # print(post_verb)
            # print(pre_verb in dict_verb)
            # print(post_verb in dict_person)
            if pre_verb in dict_verb and post_verb in dict_person:
                pre_numb = int((-1) * ((int(dict_verb[pre_verb]) * 10) + int(dict_person[post_verb])))
                dict_word[nt['id'][x]]["cng"] = int(pre_numb)
            else :
                #print(pre_verb,post_verb)
                #print()
                afile.write("4 : "+str(nt['morph'][x])+"\n")
                try :
                    dict_word[nt['id'][x]]["cng"] = int(nt['morph'][x])
                except:
                    if nt['morph'][x] == "?":
                        dict_word[nt['id'][x]]["cng"] = 0
                    else:
                        dict_word[nt['id'][x]]["cng"] = int(-1)
            #print(3,nt['morph'][x])
        else:
            afile.write("Else : "+nt['morph'][x]+"\n")
            try :
                dict_word[nt['id'][x]]["cng"] = int(nt['morph'][x])
            except:
                if nt['morph'][x] == "?":
                    dict_word[nt['id'][x]]["cng"] = 0
                else:
                    dict_word[nt['id'][x]]["cng"] = int(-1)
        dict_word[nt['id'][x]]["tuple"] = (dict_word[nt['id'][x]]["lemma"],int(dict_word[nt['id'][x]]["cng"]))

    # print(dict_word)
    # exit(1)
    dict_levelpos2words = {}
    #print(context['conflictslp'])
    p = context['conflictslp']
    r = json.loads(p)
    #print(nt)

    #nt.to_csv('file_name.csv', encoding='utf-8')
    #exit(1)
    source = {}
    target = {}
    
    edge_dict[int(number_of_sentence)] = []
    #edgeID:xx source: "1" target: "2"
    i = 0
    #j = 0
    for x in range(len(nt)):
        l = nt['level'][x]
        p = nt['position'][x]
        l_p = str(l) + "-" + str(p)
        for y in range(x+1,len(nt)):
            nl = nt['level'][y]
            np = nt['position'][y]
            nl_np = str(nl) + "-" + str(np)
            
            #print(nl_np)
            #exit(1)
            
            if nl_np in r[l_p]:
                #print(1)
                #print(nt['lemma'][x],nt['lemma'][y],nl_np,l_p)
                continue
            elif nl_np == l_p:
                #print(2)
                #print(nt['lemma'][x],nt['lemma'][y],nl_np,l_p)
                continue
            else:
                #print(nt['lemma'][x],nt['lemma'][y],nl_np,l_p)
                temp = {}
                temp["edgeID"] = i
                temp["source"] = int(x)
                temp["target"] = int(y)
                source[i] = {}
                source[i]["id"] = x
                source[i]["lemma"] = dict_word[nt['id'][x]]["lemma"]
                source[i]["cng"] = int(dict_word[nt['id'][x]]["cng"])
                source[i]["tuple"] = (source[i]["lemma"],source[i]["cng"])
                target[i] = {}
                target[i]["id"] = y
                target[i]["lemma"] = dict_word[nt['id'][y]]["lemma"]
                target[i]["cng"] = int(dict_word[nt['id'][y]]["cng"])
                target[i]["tuple"] = (target[i]["lemma"],target[i]["cng"])
                i += 1
                edge_dict[int(number_of_sentence)].append(temp)

                temp = {}
                temp["edgeID"] = i
                temp["source"] = int(y)
                temp["target"] = int(x)
                source[i] = {}
                source[i]["id"] = y
                source[i]["lemma"] = dict_word[nt['id'][y]]["lemma"]
                source[i]["cng"] = int(dict_word[nt['id'][y]]["cng"])
                source[i]["tuple"] = (source[i]["lemma"],source[i]["cng"])
                target[i] = {}
                target[i]["id"] = x
                target[i]["lemma"] = dict_word[nt['id'][x]]["lemma"]
                target[i]["cng"] = int(dict_word[nt['id'][x]]["cng"])
                target[i]["tuple"] = (target[i]["lemma"],target[i]["cng"])
                i += 1
                edge_dict[int(number_of_sentence)].append(temp)
        #print(edge_dict)
    #exit(1)


    # for keys,values in r.items():
    #     l = int(keys.split('-')[0])
    #     p = int(keys.split('-')[0])
    #     for x in range(len(nt)):

    #print("Source : ",source)
    #print("Target : ",target)



        #dict_levelpos2words[keys] = dict_levelpos2words
    #print(edge_dict)
    #exit(1)

    return edge_dict,source,target,dict_word,dict_,context,nt,soup

# function to add to JSON 
def write_json_edge(data, filename='edges.json'): 
    with open(filename,'w+') as f: 
        json.dump(data, f, indent=4) 

# function to add to JSON 
def write_json_node(data, filename='nodes.json'): 
    with open(filename,'w+') as f: 
        json.dump(data, f, indent=4) 

if __name__ == "__main__":
    #inputsent = "surenxrapramuKEH inxrAxiBiH samagrEH samaswEH vibuXEH xevEH upewya samIpam Agawya"
    #inputsent = "surenxrapramuKEH"
    #inputsent = "pATayawaH"
    #inputsent = "Buvam"
    file_name = "noun.csv"
    j = 0
    #dict_edge = {}
    #dict_node = {}
    k =0
    # Nom. sg. masc.
    with open(file_name, 'r', encoding='utf-8',  errors='ignore') as f:
        for line in f:
            inputsent = line.split("\n")[0].split(",")
            if k < 5:
                k += 1
                continue
            if inputsent[1] == "xt?":
                continue
            #print(str(inputsent))
            #print(str(inputsent[1].split(" ")[2]))
            gen = str(inputsent[1].split(" ")[2])
            gender = gen[0] + "."
            number = inputsent[1].split(" ")[1]
            vachan = inputsent[1].split(" ")[0].lower()
            morph = str(gender) + " " + str(number) + " " + str(vachan)
            dict_noun[morph] = inputsent[0]
            nvachan = vachan[0] + "."
            nmorph = str(gender) + " " + str(number) + " " + str(nvachan)
            dict_noun[nmorph] = inputsent[0]
            # nom. sg. m.

    # print(dict_noun)
    # exit(1)
    file_name = "verb.csv"
    
    #dict_edge = {}
    #dict_node = {}

    with open(file_name, 'r', encoding='utf-8',  errors='ignore') as f:
        for line in f:
            inputsent = line.split("\n")[0].split(",")
            verb = inputsent[2].split(" ")
            if len(verb) == 3 and verb[1] == "[*]":
                for j in range(11):
                    verb[1] = "["+str(j)+"]"
                    verb[2] = "ac."
                    nverb = " ".join(verb)
                    dict_verb[nverb] = inputsent[1]
                    verb[2] = "md."
                    nnverb = " ".join(verb)
                    dict_verb[nnverb] = inputsent[1]
            elif len(verb) == 3 and verb[2] == "ac./ps./md.":
                verb[2] = "ac."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
                verb[2] = "ps."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
                verb[2] = "md."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
            elif len(verb) == 2 and verb[1] == "ac./ps./md.":
                verb[1] = "ac."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
                verb[1] = "ps."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
                verb[1] = "md."
                nverb = " ".join(verb)
                dict_verb[nverb] = inputsent[1]
            else:
                dict_verb[inputsent[2]] = inputsent[1]
    dict_noun["iic."] = str(3)
    dict_noun["prep."] = str(2)
    dict_noun["adv."] = str(2)
    dict_noun["conj."] = str(2)
    dict_noun["part."] = str(2)
    dict_noun["tasil"] = str(2)
    dict_noun["ind."] = str(1)
    dict_noun["iiv."] = str(3)
    dict_noun["ca. abs."] = str(-230)
    dict_noun["ca. inf."] = str(-220)
    dict_noun["pft. md. sg. 1"] = str(-151)
    dict_noun["pft. md. sg. 3"] = str(-153)
    dict_noun["ca. per. pft."] = str(-160)
    dict_noun["impft. [vn.] ac. sg. 2"] = str(-42)
    dict_noun["impft. [vn.] ac. sg. 1"] = str(-41)
    dict_noun["ca. impft. ac. sg. 2"] = str(-42)
    dict_noun["ca. impft. ac. sg. 3"] = str(-43)
    dict_noun["ca. impft. ac. pl. 1"] = str(-47)
    dict_noun["ca. impft. ac. pl. 3"] = str(-49)
    dict_noun["ca. impft. ps. sg. 1"] = str(-271)
    dict_noun["ca. imp. ac. sg. 2"] = str(-32)
    dict_noun["ca. imp. ac. sg. 3"] = str(-33)
    dict_noun["pr. [vn.] md. sg. 1"] = str(-11)
    dict_noun["imp. [vn.] ac. sg. 2"] = str(-32)
    #dict_noun["des. imp. ac. sg. 2"] = str(-32)
    #dict_noun["imp. [vn.] ac. sg. 2"] = str(-32)
    #dict_noun["imp. [vn.] ac. sg. 2"] = str(-32)
    dict_noun["ca. imp. ac. pl. 1"] = str(-47)
    dict_noun["ca. pr. ps. sg. 1"] = str(-241)
    dict_noun["ca. pr. ps. pl. 3"] = str(-249)
    dict_noun["ca. fut. ac. sg. 3"] = str(-53)
    dict_noun["ca. fut. ac. sg. 1"] = str(-51)
    dict_noun["ca. fut. ac. sg. 2"] = str(-52)
    dict_noun["ca. opt. ac. sg. 3"] = str(-23)
    dict_noun["aor. [6] ac. sg. 3"] = str(-123)
    #print(dict_noun)
    #exit(1)

    dict_person = {"sg. 1" : 1 , "sg. 2" : 2 , "sg. 3" : 3 , "du. 1" : 4 , "du. 2" : 5, "du. 3" : 6, "pl. 1" : 7,"pl. 2" : 8, "pl. 3" : 9 }


    file_name = "followaa"
    j = 0
    #dict_node = {}
    with open(file_name, 'r', encoding='utf-8',  errors='ignore') as f:
        for line in f:
            inputsent = line.split("\n")[0]
            #print(line)
            #inputsent = "surenxrapramuKE"
            result = getdatafromsite(inputsent)

            if result == None:
                yfile.write(inputsent+"\n")
                continue
            edge_dict,source,target,word2id,dict_,context,nt,soup = result
            # dict_edge = {}
            # dict_edge["sentid"] = j
            # dict_edge["source"] = source
            # dict_edge["target"] = target
            #write_json_edge(edge_dict)
            #json_edge[j] = edge_dict
            
            #print(dict_,context)
            #exit(1)
            
            number_of_sentence += 1
            dict_node = {}
            dict_node["sentid"] = j
            dict_node["nodes"] = word2id
            # wfile.write(str(dict_edge))
            # wfile.write("\n")
            xfile.write(str(dict_node))
            xfile.write("\n")
            node_dict.append(dict_node)
            file_ = str(j)+".csv"
            nt.to_csv("details/"+file_, encoding='utf-8')
            file_ = str(j)+".html"
            with open("details/"+file_, "w", encoding='utf-8') as file:
                file.write(str(soup))
            #json.dump(edge_dict, zfile)
            # print(str(dict_edge))
            # print(str(dict_node))
            #if j == 1:
            #      break
            #time.sleep(1)
            #     print("dict_edge ",dict_edge)
            #     print("dict_node ",dict_node)
            j += 1
print(j)
write_json_edge(edge_dict)
write_json_node(node_dict)
wfile.write("\n")
