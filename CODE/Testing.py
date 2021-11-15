import pandas as pd

f1 = pd.read_csv('C:\\Users\\Swars\\Desktop\\FYP\\TESTINGnew\\After_mapping_75.csv')
f2=pd.read_csv('C:\\Users\\Swars\\Desktop\\FYP\\TESTINGnew\\testing.csv',delimiter=':')
f = pd.merge(left=f1, right=f2, how='left', on='request_id')
f.to_csv('C:\\Users\\Swars\\Desktop\\FYP\\TESTINGnew\\After_mapping_final.csv')

c=0
t=0
inc=0
f1 = pd.read_csv('C:\\Users\\Swars\\Desktop\\FYP\\TESTINGnew\\After_mapping_final.csv')
for i,data in f1.iterrows():
    test_set = set(data['offer_id_x'])
    if(type(data['offer_id_y'])!=str):
        data['offer_id_y']=str(data['offer_id_y'])
    validate_set = set(list(data['offer_id_y']))
    '''out = test_set.intersection(validate_set)
    if(len(out)>0): inc=inc+1'''
    if(test_set & validate_set):
        c=c+1
    t=t+1
print("\n")
print("-"*100)
print("Correct Predictions\t|\t"+str(c))
print("Total Predictions\t|\t"+str(t))
print("Accuracy\t\t\t|\t"+str(c/t*100))
print("-"*100)
