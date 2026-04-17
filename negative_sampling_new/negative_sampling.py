import json
file1 = open("../src/data/only_path.txt",'r',encoding="utf-8")
file2 = open("./positive_sample.txt","w",encoding="utf-8")
lines = file1.readlines()
for line in lines:
    dict = json.loads(line)
    dict["type"] = 1
    dict_str = json.dumps(dict) + "\n"
    file2.write(dict_str)
