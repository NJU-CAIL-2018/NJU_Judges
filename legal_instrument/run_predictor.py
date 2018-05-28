import legal_instrument.system_path as constant
import json

from legal_instrument.predictor import Predictor

p = Predictor()
with open(constant.DATA_TEST, "r",
          encoding="UTF-8") as f:
    line = f.readline()
    out_file = open('./judge/output.txt', 'w', )
    while line:
        obj = json.loads(line)
        s = str(p.predict([obj['fact']])[0]) + "\n"
        print(s.replace('\'', '\"'))
        out_file.write(s.replace('\'', '\"'))
        line = f.readline()
print("out")
