import json
from utils.progress import WorkSplitter

'''
Preliminary exploration of the data set to find some exceptions and prepare for subsequent processing
'''
progress = WorkSplitter()
max_print = 5

progress.section("Duplication of sample")
print_num = 0
Rtrain = open('datax/small/data_train.json', 'r', encoding='utf-8')
start = True
duplication_num = 0
dic_set = set()
for line in Rtrain.readlines():
    dic = json.loads(line)
    if start:
        start = False
        dic_set.add(dic_set.add(json.dumps(dic)))
    else:
        if json.dumps(dic) not in dic_set:
            dic_set.add(json.dumps(dic))
        else:
            duplication_num += 1
            if print_num < max_print:
                print(dic)
                print_num += 1
Rtrain.close()
print('In the training set, the number of sample duplications is {0}'.format(duplication_num))

progress.section("Exception in fact field")
print_num = 0
Rtrain = open('datax/small/data_train.json', 'r', encoding='utf-8')
for line in Rtrain.readlines():
    dic = json.loads(line)
    if len(dic['fact']) < 25:
        if print_num < max_print:
            print(dic)
            print_num += 1
Rtrain.close()


progress.section("Duplications of accusation and law")
print_num = 0
Rtrain = open('datax/small/data_train.json', 'r', encoding='utf-8')
accu_duplication_num, law_duplication_num = 0, 0
for line in Rtrain.readlines():
    dic = json.loads(line)
    unique_accusation = list(set(dic["meta"]["accusation"]))
    unique_law = list(set(dic["meta"]["relevant_articles"]))
    if len(unique_accusation) != len(dic["meta"]["accusation"]):
        accu_duplication_num += 1
        if print_num < max_print:
            print(dic)
            print_num += 1
    if len(unique_law) != len(dic["meta"]["relevant_articles"]):
        law_duplication_num += 1
        if print_num < max_print:
            print(dic)
            print_num += 1
Rtrain.close()
print('In the training set, the number of accusation duplications is {0}'.format(accu_duplication_num))
print('In the training set, the number of law duplications is {0}'.format(law_duplication_num))

progress.section("Exceptions in the accusation field")
print_num = 0
Rtrain = open('datax/small/data_train.json', 'r', encoding='utf-8')
accu = open('datax/accu.txt', 'r', encoding='utf-8')
accu_set = set()
for line in accu.readlines():
    if line.strip() not in accu_set:
        accu_set.add(line.strip())

for line in Rtrain.readlines():
    dic = json.loads(line)
    accu_len = len(dic["meta"]["accusation"])
    for i in range(accu_len):
        if dic["meta"]["accusation"][i] not in accu_set:
            if print_num < max_print:
                print(dic)
                print_num += 1
