import json

def tsv_to_json(tsv_filename, json_filename, instruction=None):
    data_list = []
    with open(tsv_filename) as f_tsv, open(json_filename, 'w') as f_json:
        for line in f_tsv:
            source, target, gen_type = line.strip().split('\t')
            data = {'instruction': instruction, 'input': source, 'output': target,
                    'gen_type': gen_type}
            data_list.append(data)
        json.dump(data_list, f_json, indent=4, separators=(',', ':'))
    return data_list

def main():

    instruction = 'Parse the input sentence into COGS meaning representation.'

    filenames = ["train", "dev", "test", "gen"]

    for lf in ["cogs_LF", "varfree_LF"]:
        for fname in filenames:
            tsv_filename = f"../../data/{lf}/{fname}.tsv"
            json_filename = f"data/{lf}/{fname}.json"
            _ = tsv_to_json(tsv_filename, json_filename, instruction=instruction)

if __name__ == '__main__':
    main()
