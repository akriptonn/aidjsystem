import json

def __parse_args_json__(r, extracted_features):
    data = {}
    for kunci in extracted_features:
        looper = 0
        data[kunci] = []
        if (isinstance(extracted_features[kunci]["feature"], list)):
            looper = extracted_features[kunci]["feature"]
        else:
            looper = [extracted_features[kunci]["feature"]]
        for obj in r[kunci]:
            t_dic = {}
            idx = 0
            if (len(looper)==1):
                for feature in looper:
                    t_dic = obj[feature]
            else:
                for feature in looper:
                    t_dic[feature] = obj[feature] 
            if (extracted_features[kunci]['iterator'] is None):
                data[kunci].append("")
                idx = len(data[kunci])-1 
            else:
                while (len(data[kunci])-1<int(obj[extracted_features[kunci]['iterator']])):
                    data[kunci].append("")
                idx = int(obj[extracted_features[kunci]['iterator']])
            data[kunci][idx] = t_dic  
    return data

def file_read_json_direct(json_path, extracted_features):
    if (json_path.split(".")[-1]=='json'): 
        with open(json_path) as f:
            r = json.load(f)
            data = parse_args_json(r, extracted_features)  
    else:
        raise FileNotFoundError("File must JSON, instead "+json_path+" passed")
    return data

def parse_args_json(json_obj, extracted_features):
    return __parse_args_json__(json_obj, extracted_features)

def single_parse_models_settings_json(json_obj, retrieved='dir'): 
    t_dic = parse_args_json(json_obj, {'model_settings': {"iterator":None, "feature":['name', retrieved]}})
    return (t_dic['model_settings'][t_dic['model_settings'].index([isi for isi in t_dic['model_settings'] if isi['name']=='genre'][-1])][retrieved], t_dic['model_settings'][t_dic['model_settings'].index([isi for isi in t_dic['model_settings'] if isi['name']=='key'][-1])][retrieved], t_dic['model_settings'][t_dic['model_settings'].index([isi for isi in t_dic['model_settings'] if isi['name']=='energy'][-1])][retrieved])
