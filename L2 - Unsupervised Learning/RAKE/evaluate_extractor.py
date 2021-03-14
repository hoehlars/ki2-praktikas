# -*- coding: utf-8 -*-
from rake import RakeKeywordExtractor
import os
import argparse

def main():
    # SET N AS THE NUMBER OF KEYWORDS TO EVALUATE FROM THE KEYWORDS LISTS
    N = 50
    
    rake = RakeKeywordExtractor('stopwords_simple.txt')
    
    tp_total = 0
    fp_total = 0
    fn_total = 0

    parser = argparse.ArgumentParser(description="Extract Keywords for a Document")
    parser.add_argument('-f', '--file_name', help='Name of the input file: e.g. tech-20916454.txt',type=str, required=True)
    args = parser.parse_args()

    base_path = 'selected_corpus'
    file_name = args.file_name

    content = open(os.path.join(base_path, file_name), 'rt', encoding='utf-8').read()
    keywordsExpected = [x.strip().lower() for x in open(os.path.join(base_path, file_name.replace('.txt', '.key')), 'rt', encoding='utf-8')]

    keywordsExtracted = set(rake.extract(content, incl_scores=False)[0:N])
    #keywordsExpected = [x.lower() for x in set(listfromfilelines(keyfile)[0:N])]

    tp, fp, fn = confusionMatrix(keywordsExtracted, keywordsExpected)
    p, r, f1 = getF1(tp, fp, fn)

    tp_total += tp
    fp_total += fp
    fn_total += fn

    print("F1 for top " + str(N) + " keywords in " + file_name + ":\t" + str(f1))

    printFinalScores(tp_total, fp_total, fn_total)

    print('EXTRACTED KEYWORDS:\n')
    for keyword in keywordsExtracted:
        print('{}\n'.format(keyword))

    

def printFinalScores(tp_total, fp_total, fn_total):
    precision, recall, f1 = getF1(tp_total, fp_total, fn_total);
    
    print("Precision overall for top N keywords:\t" + str(precision))
    print("Recall overall for top N keywords:\t" + str(recall))
    print("F1 overall for top N keywords:\t" + str(f1))


def getF1(tp, fp, fn):
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    f1 = 0
    if (precision+recall)>0:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1


def confusionMatrix(keywordsExtracted, keywordsExpected):
    true_positives = len(keywordsExtracted.intersection(keywordsExpected))
    false_positives = len(keywordsExtracted)-true_positives
    false_negatives = len(keywordsExpected)-true_positives
    return true_positives, false_positives, false_negatives


def listfromfilelines(file):
    """ Returns a list from the files lines"""
    with open(file, 'r') as f:
        list = [line.strip() for line in f]
    return list
    

if __name__ == '__main__':
    main()