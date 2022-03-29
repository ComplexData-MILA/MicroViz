try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import zipfile
import docx
from collections import defaultdict
import sys
import pandas as pd
import os
from tqdm import tqdm
import difflib

"""
Module that extract text from MS XML Word document (.docx).
(Inspired by python-docx <https://github.com/mikemaccana/python-docx>)
"""

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA = WORD_NAMESPACE + 'p'
TEXT = WORD_NAMESPACE + 't'

def get_highlighted_docx_text(path):
    
    document = docx.Document(path)

    highlight_map = {'RED (6)':"Slot", \
                    'YELLOW (7)': "Matched", \
                    'BRIGHT_GREEN (4)': "Substitution", \
                    'GRAY_25 (16)': "Deletion", \
                    'TEAL (10)': "Insertion"
                    }

    highlight_summ = {'RED (6)':"Slot", \
                    'YELLOW (7)': "Match", \
                    'BRIGHT_GREEN (4)': "Sub", \
                    'GRAY_25 (16)': "Del", \
                    'TEAL (10)': "Ins"
                    }


    highlights = []
    full_templ = ""
    for paragraph in document.paragraphs:
        highlight = ""
        templ = ""
        for run in paragraph.runs:
            highlight_color = str(run.font.highlight_color)
            # print(highlight_color)
            if highlight_color != 'None':
                if highlight_map[highlight_color] != 'Matched' and run.text not in highlight_map.values() and run.text.strip()!='':
                    # print(highlight_map[highlight_color], run.text)
                    templ += ("<"+highlight_summ[highlight_color]+">"+\
                                run.text+\
                                "</"+highlight_summ[highlight_color]+"/> ")
                    # highlight += (run.text+" ")
                else:
                    templ += (run.text)

            else:
                templ += (run.text)

        if "<" in templ:
            templ += "\n------------------\n"
            full_templ += templ
    
    return full_templ
        # if highlight:
            # highlights.append(highlight)
    # for h in highlights:
        # print(h)
    # print(template)
    # print(word_map.values())
    # print("\n\n")


def get_docx_text(path):
    """
    Take the path of a docx file as argument, return the text in unicode.
    """
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = XML(xml_content)

    paragraphs = []
    for paragraph in tree.getiterator(PARA):
        texts = [node.text
                 for node in paragraph.getiterator(TEXT)
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))

    return '\n\n'.join(paragraphs).split("----")[0].split("\n")[-1]

def get_differences(templates):

    temp1 = []
    temp2 = []
    changed = []
    changed_txt = []
    same = []
    encountered = set()

    d = difflib.Differ()
    for i in tqdm(range(len(templates))):
        for j in range(len(templates)):
            if i == j or (i,j) in encountered or (j,i) in encountered:
                continue
            encountered.add((i,j))
            operation_counter = {'+':0,'-':0,'?':0}
            common_words = 0
            str1 = templates[i]
            str2 = templates[j]
            diff = d.compare(str1.split(), str2.split())
            changed_words = ""
            for dd in diff:
                words = dd.split()
                if len(words) == 2:
                    operation_counter[words[0]] += 1
                    changed_words += (words[0]+words[1]+"\n")
                else:
                    common_words += 1

            words_changed = abs(operation_counter['+']-operation_counter['-'])
            changed.append(words_changed)
            same.append(common_words)
            temp1.append('T'+str(i+1))
            temp2.append('T'+str(j+1))
            changed_txt.append(changed_words)

    diff_df = pd.DataFrame()
    diff_df['T1'] = temp1
    diff_df['T2'] = temp2
    diff_df['Extra words'] = changed
    diff_df['Same words'] = same
    diff_df['Changed text'] = changed_words

    return diff_df
    


def get_full_text(filename):
    full_txt = []
    cnt = 0
    extras = []
    folder_name = ["".join(s) for s in filename.split('/')[:-1]][0]+"/"
    print(folder_name)
    list_of_micro_clusters = pd.read_csv(filename)['LSH label'].unique()
    cluster_ids = []
    for i in tqdm(list_of_micro_clusters):
        path = folder_name+str(i)+"/template_1/text.docx"
        cluster_ids.append(i)
        if not os.path.isfile(path):
            full_txt.append("")
            extras.append("")
            cnt += 1
            continue
        template1 = get_docx_text(path)
        template2 = get_highlighted_docx_text(path)
        # # print(template1)
        # if "<" in template2:
        #     print(template2)
        #     print("\n------------------\n")
        full_txt.append(template1)
        extras.append(template2)
        # full_txt += 
    diff_df = get_differences(full_txt)
    dd = pd.DataFrame(columns=['LSH label','Template', 'Extra'])
    dd['LSH label'] = cluster_ids
    dd['Template'] = full_txt
    dd['Extra'] = extras
    # print(extras)
    dd.to_csv(folder_name+"template_texts.csv",index='LSH label')
    diff_df.to_csv(folder_name+"template_differences.csv", index=False)
    # print("Count = " + str(cnt))
    print(diff_df)
    return dd

filename = sys.argv[1]
df = get_full_text(filename)