# coding=utf-8

import json

V_trans_not_omissible = [
  'liked', 'helped', 'found', 'loved', 'poked',
  'admired', 'adored', 'appreciated', 'missed', 'respected',
  'threw', 'tolerated', 'valued', 'worshipped', 'discovered',
  'held', 'stabbed', 'touched', 'pierced', 'tossed'
]
V_trans_omissible = [
  'ate', 'painted', 'drew', 'cleaned', 'cooked',
  'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
  'called', 'heard', 'packed', 'saw', 'noticed',
  'studied', 'examined', 'observed', 'knew', 'investigated'
]

V_unacc = [
  'rolled', 'froze', 'burned', 'shortened', 'floated',
  'grew', 'slid', 'broke', 'crumpled', 'split',
  'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
  'doubled', 'improved', 'inflated', 'enlarged', 'reddened',
]

V_unerg = [
  'slept', 'smiled', 'laughed', 'sneezed', 'cried',
  'talked', 'danced', 'jogged', 'walked', 'ran',
  'napped', 'snoozed', 'screamed', 'stuttered', 'frowned',
  'giggled', 'scoffed', 'snored', 'smirked', 'gasped'
]


verbs_lemmas = {
  'ate':'eat', 'painted':'paint', 'drew':'draw', 'cleaned':'clean',
  'cooked':'cook', 'dusted':'dust', 'hunted':'hunt', 'nursed':'nurse',
  'sketched':'sketch', 'washed':'wash', 'juggled':'juggle', 'called':'call',
  'eaten':'eat', 'drawn':'draw', 'baked':'bake', 'liked':'like', 'knew':'know',
  'helped':'help', 'saw':'see', 'found':'find', 'heard':'hear', 'noticed':'notice',
   'admired':'admire', 'adored':'adore', 'appreciated':'appreciate',
  'missed':'miss', 'respected':'respect', 'tolerated':'tolerate', 'valued':'value',
  'worshipped':'worship', 'observed':'observe', 'discovered':'discover', 'held':'hold',
  'stabbed':'stab', 'touched':'touch', 'pierced':'pierce', 'poked':'poke',
  'known':'know', 'seen':'see', 'hit':'hit', 'hoped':'hope', 'said':'say',
  'believed':'believe', 'confessed':'confess', 'declared':'declare', 'proved':'prove',
  'thought':'think', 'supported':'support', 'wished':'wish', 'dreamed':'dream',
  'expected':'expect', 'imagined':'imagine', 'envied':'envy', 'wanted':'want',
  'preferred':'prefer', 'needed':'need', 'intended':'intend', 'tried':'try',
  'attempted':'attempt', 'planned':'plan','craved':'crave','hated':'hate','loved':'love',
  'enjoyed':'enjoy', 'rolled':'roll', 'froze':'freeze', 'burned':'burn', 'shortened':'shorten',
  'floated':'float', 'grew':'grow', 'slid':'slide', 'broke':'break', 'crumpled':'crumple',
  'split':'split', 'changed':'change', 'snapped':'snap', 'tore':'tear', 'collapsed':'collapse',
  'decomposed':'decompose', 'doubled':'double', 'improved':'improve', 'inflated':'inflate',
  'enlarged':'enlarge', 'reddened':'redden', 'popped':'pop', 'disintegrated':'disintegrate',
  'expanded':'expand', 'cooled':'cool', 'soaked':'soak', 'frozen':'freeze', 'grown':'grow',
  'broken':'break', 'torn':'tear', 'slept':'sleep', 'smiled':'smile', 'laughed':'laugh',
  'sneezed':'sneeze', 'cried':'cry', 'talked':'talk', 'danced':'dance', 'jogged':'jog',
  'walked':'walk', 'ran':'run', 'napped':'nap', 'snoozed':'snooze', 'screamed':'scream',
  'stuttered':'stutter', 'frowned':'frown', 'giggled':'giggle', 'scoffed':'scoff',
  'snored':'snore', 'snorted':'snort', 'smirked':'smirk', 'gasped':'gasp',
  'gave':'give', 'lent':'lend', 'sold':'sell', 'offered':'offer', 'fed':'feed',
  'passed':'pass', 'rented':'rent', 'served':'serve','awarded':'award', 'promised':'promise',
  'brought':'bring', 'sent':'send', 'handed':'hand', 'forwarded':'forward', 'mailed':'mail',
  'posted':'post','given':'give', 'shipped':'ship', 'packed':'pack', 'studied':'study',
  'examined':'examine', 'investigated':'investigate', 'thrown':'throw', 'threw':'throw',
  'tossed':'toss', 'meant':'mean', 'longed':'long', 'yearned':'yearn', 'itched':'itch',
  'loaned':'loan', 'returned':'return', 'slipped':'slip', 'wired':'wire', 'crawled':'crawl',
  'shattered':'shatter', 'bought':'buy', 'squeezed':'squeeze', 'teleported':'teleport',
  'melted':'melt', 'blessed':'bless'
}

animate_nouns = [
    'girl', 'boy', 'cat', 'dog', 'baby', 'child', 'teacher', 'frog', 'chicken', 'mouse',
    'lion', 'monkey', 'bear', 'giraffe', 'horse', 'bird', 'duck', 'bunny', 'butterfly', 'penguin',
    'student', 'professor', 'monster', 'hero', 'sailor', 'lawyer', 'customer', 'scientist', 'princess', 'president',
    'cow', 'crocodile', 'goose', 'hen', 'deer', 'donkey', 'bee', 'fly', 'kitty', 'tiger',
    'wolf', 'zebra', 'mother', 'father', 'patient', 'manager', 'director', 'king', 'queen', 'kid',
    'fish', 'moose',  'pig', 'pony', 'puppy', 'sheep', 'squirrel', 'lamb', 'turkey', 'turtle',
    'doctor', 'pupil', 'prince', 'driver', 'consumer', 'writer', 'farmer', 'friend', 'judge', 'visitor',
    'guest', 'servant', 'chief', 'citizen', 'champion', 'prisoner', 'captain', 'soldier', 'passenger', 'tenant',
    'politician', 'resident', 'buyer', 'spokesman', 'governor', 'guard', 'creature', 'coach', 'producer', 'researcher',
    'guy', 'dealer', 'duke', 'tourist', 'landlord', 'human', 'host', 'priest', 'journalist', 'poet'
]

inanimate_nouns = [
    'cake', 'donut', 'cookie', 'box', 'rose', 'drink', 'raisin', 'melon', 'sandwich', 'strawberry',
    'ball', 'balloon', 'bat', 'block', 'book', 'crayon', 'chalk', 'doll', 'game', 'glue',
    'lollipop', 'hamburger', 'banana', 'biscuit', 'muffin', 'pancake', 'pizza', 'potato', 'pretzel', 'pumpkin',
    'sweetcorn', 'yogurt', 'pickle', 'jigsaw', 'pen', 'pencil', 'present', 'toy', 'cracker', 'brush',
    'radio', 'cloud', 'mandarin', 'hat', 'basket', 'plant', 'flower', 'chair', 'spoon', 'pillow',
    'gumball', 'scarf', 'shoe', 'jacket', 'hammer', 'bucket', 'knife', 'cup', 'plate', 'towel',
    'bottle', 'bowl', 'can', 'clock', 'jar', 'penny', 'purse', 'soap', 'toothbrush', 'watch',
    'newspaper', 'fig', 'bag', 'wine', 'key', 'weapon', 'brain', 'tool', 'crown', 'ring',
    'leaf', 'fruit', 'mirror', 'beer', 'shirt', 'guitar', 'chemical', 'seed', 'shell', 'brick',
    'bell', 'coin', 'button', 'needle', 'molecule', 'crystal', 'flag', 'nail', 'bean', 'liver'
]

on_nouns = [
    'table', 'stage', 'bed', 'chair', 'stool', 'road', 'tree', 'box', 'surface', 'seat',
    'speaker', 'computer', 'rock', 'boat', 'cabinet', 'TV', 'plate', 'desk', 'bowl', 'bench',
    'shelf', 'cloth', 'piano', 'bible', 'leaflet', 'sheet', 'cupboard', 'truck', 'tray', 'notebook',
    'blanket', 'deck', 'coffin', 'log', 'ladder', 'barrel', 'rug', 'canvas', 'tiger', 'towel',
    'throne', 'booklet', 'sock', 'corpse', 'sofa', 'keyboard', 'book', 'pillow', 'pad', 'train',
    'couch', 'bike', 'pedestal', 'platter', 'paper', 'rack', 'board', 'panel', 'tripod', 'branch',
    'machine', 'floor', 'napkin', 'cookie', 'block', 'cot', 'device', 'yacht', 'dog', 'mattress',
    'ball', 'stand', 'stack', 'windowsill', 'counter', 'cushion', 'hanger', 'trampoline', 'gravel', 'cake',
    'carpet', 'plaque', 'boulder', 'leaf', 'mound', 'bun', 'dish', 'cat', 'podium', 'tabletop',
    'beach', 'bag', 'glacier', 'brick', 'crack', 'vessel', 'futon', 'turntable', 'rag', 'chessboard'
]

in_nouns = [
    'house', 'room', 'car', 'garden', 'box', 'cup', 'glass', 'bag', 'vehicle', 'hole',
    'cabinet', 'bottle', 'shoe', 'storage', 'cot', 'vessel', 'pot', 'pit', 'tin', 'can',
    'cupboard', 'envelope', 'nest', 'bush', 'coffin', 'drawer', 'container', 'basin', 'tent', 'soup',
    'well', 'barrel', 'bucket', 'cage', 'sink', 'cylinder', 'parcel', 'cart', 'sack', 'trunk',
    'wardrobe', 'basket', 'bin', 'fridge', 'mug', 'jar', 'corner', 'pool', 'blender', 'closet',
    'pile', 'van', 'trailer', 'saucepan', 'truck', 'taxi', 'haystack', 'dumpster', 'puddle', 'bathtub',
    'pod', 'tub', 'trap', 'bun', 'microwave', 'bookstore', 'package', 'cafe', 'train', 'castle',
    'bunker', 'vase', 'backpack', 'tube', 'hammock', 'stadium', 'backyard', 'swamp', 'monastery', 'refrigerator',
    'palace', 'cubicle', 'crib', 'condo', 'tower', 'crate', 'dungeon', 'teapot', 'tomb', 'casket',
    'jeep', 'shoebox', 'wagon', 'bakery', 'fishbowl', 'kennel', 'china', 'spaceship', 'penthouse', 'pyramid'
]

beside_nouns = [
    'table', 'stage', 'bed', 'chair', 'book', 'road', 'tree', 'machine', 'house', 'seat',
    'speaker', 'computer', 'rock', 'car', 'box', 'cup', 'glass', 'bag', 'flower', 'boat',
    'vehicle', 'key', 'painting', 'cabinet', 'TV', 'bottle', 'cat', 'desk', 'shoe', 'mirror',
    'clock', 'bench', 'bike', 'lamp', 'lion', 'piano', 'crystal', 'toy', 'duck', 'sword',
    'sculpture', 'rod', 'truck', 'basket', 'bear', 'nest', 'sphere', 'bush', 'surgeon', 'poster',
    'throne', 'giant', 'trophy', 'hedge', 'log', 'tent', 'ladder', 'helicopter', 'barrel', 'yacht',
    'statue', 'bucket', 'skull', 'beast', 'lemon', 'whale', 'cage', 'gardner', 'fox', 'sink',
    'trainee', 'dragon', 'cylinder', 'monk', 'bat', 'headmaster', 'philosopher', 'foreigner', 'worm', 'chemist',
    'corpse', 'wolf', 'torch', 'sailor', 'valve', 'hammer', 'doll', 'genius', 'baron', 'murderer',
    'bicycle', 'keyboard', 'stool', 'pepper', 'warrior', 'pillar', 'monkey', 'cassette', 'broker', 'bin'

]

only_seen_as_subject = 'hedgehog'
only_seen_as_noun_prim = 'shark'
only_seen_as_object = 'cockroach'
only_seen_as_subject_proper_noun = 'Lina'
only_seen_as_proper_noun_prim = 'Paula'
only_seen_as_object_proper_noun =  'Charlie'
only_seen_as_transitive_obj_omissible = 'baked'
only_seen_as_unaccuative = 'shattered'
only_seen_as_verb_prim = 'crawl'
only_seen_as_transitive_subject_animate = 'cobra'
only_seen_as_unaccusative_subject_animate = 'hippo'
only_seen_as_active = 'blessed'
only_seen_as_passive = 'squeezed'
only_seen_as_double_object = 'teleported'
only_seen_as_pp = 'shipped'


proper_nouns = [
    'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'William', 'Isabella', 'James', 'Sophia', 'Oliver',
    'Charlotte', 'Benjamin', 'Mia', 'Elijah', 'Amelia', 'Lucas', 'Harper', 'Mason', 'Evelyn', 'Logan',
    'Abigail', 'Alexander', 'Emily', 'Ethan', 'Elizabeth', 'Jacob', 'Mila', 'Michael', 'Ella', 'Daniel',
    'Avery', 'Henry', 'Sofia', 'Jackson', 'Camila', 'Sebastian', 'Aria', 'Aiden', 'Scarlett', 'Matthew',
    'Victoria', 'Samuel', 'Madison', 'David', 'Luna', 'Joseph', 'Grace', 'Carter', 'Chloe', 'Owen',
    'Penelope', 'Wyatt', 'Layla', 'John', 'Riley', 'Jack', 'Zoey', 'Luke', 'Nora', 'Jayden',
    'Lily', 'Dylan', 'Eleanor', 'Grayson', 'Hannah', 'Levi', 'Lillian', 'Isaac', 'Addison', 'Gabriel',
    'Aubrey', 'Julian', 'Ellie', 'Mateo', 'Stella', 'Anthony', 'Natalie', 'Jaxon', 'Zoe', 'Lincoln',
    'Leah', 'Joshua', 'Hazel', 'Christopher', 'Violet', 'Andrew', 'Aurora', 'Theodore', 'Savannah', 'Caleb',
    'Audrey', 'Ryan', 'Brooklyn', 'Asher', 'Bella', 'Nathan', 'Claire', 'Thomas', 'Skylar', 'Leo'
]

target_item_nouns = [only_seen_as_subject, only_seen_as_noun_prim, only_seen_as_object,
                    only_seen_as_transitive_subject_animate, only_seen_as_unaccusative_subject_animate]

target_item_props = [only_seen_as_subject_proper_noun, only_seen_as_proper_noun_prim,
                    only_seen_as_object_proper_noun]

noun_set = set(animate_nouns + inanimate_nouns + on_nouns + in_nouns + beside_nouns + target_item_nouns)
proper_nouns_set = set ( proper_nouns + target_item_props )

trans_v_lemma = set([ verbs_lemmas[v] for v in (V_trans_omissible + V_trans_not_omissible)])
V_unacc_lemma = [verbs_lemmas[v] for v in V_unacc]
V_unerg_lemma = [verbs_lemmas[v] for v in V_unerg] + [only_seen_as_verb_prim]
with open('nouns.json', 'w') as file:
    # Write the list to the file
    json.dump(list(noun_set), file)

with open('proper_nouns.json', 'w') as file:
    json.dump(list(proper_nouns_set), file)

with open('trans_V.json', 'w') as file:
    json.dump(list(trans_v_lemma), file)

with open('V_unacc.json', 'w') as file:
    json.dump(V_unacc_lemma, file)

with open('V_unerg.json', 'w') as file:
    json.dump(V_unerg_lemma, file)
