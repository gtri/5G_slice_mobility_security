from pathlib import Path
import json

filenames = [
Path('final-config/Extended-Background-sst-1.ue.json'),
Path('final-config/Extended-Background-sst-1---2.ue.json'),
Path('final-config/Extended-Background-sst-2.ue.json'),
Path('final-config/Extended-Background-sst-2---2.ue.json'),
Path('final-config/Extended-Background-sst-3.ue.json'),
Path('final-config/Extended-Background-sst-3---2.ue.json'),
]

ue_list = []
for filename in filenames:
    with filename.open('r+', encoding="UTF-8") as f:
        json_data = json.load(f)

        for ue in json_data:
            ue_id = int(str(ue['imsi']).split('000')[-1])
            ue['ue_id'] = ue_id

    for ue in json_data:
        ue_list.append(ue)

with open('final-config/ue-list.json', "wt") as f:
    json.dump(ue_list, f)

    # print(json_data)
    # json.dump(json_data, f)
