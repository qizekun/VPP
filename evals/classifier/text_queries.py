# Text Queries

id_to_label = {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4, '03211117': 5, '03636649': 6,
               '03691459': 7, '04090263': 8, '04256520': 9, '04379243': 10, '04401088': 11, '04530566': 12}

label_to_category = {0: 'airplane', 1: 'bench', 2: 'cabinet', 3: 'car', 4: 'chair', 5: 'monitor', 6: 'lamp',
                     7: 'loudspeaker', 8: 'gun', 9: 'sofa', 10: 'table', 11: 'phone', 12: 'boat'}

id_to_sub_category = {

    "02691156": ["airplane", "jet", "fighter plane", "biplane", "seaplane", "space shuttle", "supersonic plane",
                 "rocket plane", "delta wing", "swept wing plane", "straight wing plane", "propeller plane"],

    "02828884": ["bench", "pew", "flat bench", "settle", "back bench", "laboratory bench", "storage bench"],

    "02933112": ["cabinet", "garage cabinet", "desk cabinet"],

    "02958343": ["car", "bus", "shuttle-bus", "pickup car", "truck", "suv", "sports car", "limo", "jeep", "van",
                 "gas guzzler", "race car", "monster truck", "armored", "atv", "microbus", "muscle car", "retro car",
                 "wagon car", "hatchback", "sedan", "ambulance", "roadster car", "beach wagon"],

    "03001627": ["chair", "arm chair", "bowl chair", "rocking chair", "egg chair", "swivel chair", "bar stool",
                 "ladder back chair", "throne", "office chair", "wheelchair", "stool", "barber chair", "folding chair",
                 "lounge chair", "vertical back chair", "recliner", "wing chair", "sling"],

    "03211117": ["monitor", "crt monitor"],

    "03636649": ["lamp", "street lamp", "fluorescent lamp", "gas lamp", "bulb"],

    "03691459": ["loudspeaker", "subwoofer speaker"],

    "04090263": ["gun", "machine gun", "sniper rifle", "pistol", "shotgun"],

    "04256520": ["sofa", "double couch", "love seat", "chesterfield", "convertiable sofa", "L shaped sofa",
                 "settee sofa", "daybed", "sofa bed", "ottoman"],

    "04379243": ["table", "dressing table", "desk", "refactory table", "counter", "operating table", "stand",
                 "billiard table", "pool table", "ping-pong table", "console table"],

    "04401088": ["phone", "desk phone", "flip-phone"],

    "04530566": ["boat", "war ship", "sail boat", "speedboat", "cabin cruiser", "yacht"],

}

id_to_shape_attribute = {

    "02691156": ["triangular"],

    "02828884": ["square", "round", "circular", "rectangular", "thick", "thin"],

    "02933112": ["cuboid", "round", "rectangular", "thick", "thin"],

    "02958343": ["square", "round", "rectangular", "thick", "thin"],

    "03001627": ["square", "round", "rectangular", "thick", "thin"],

    "03211117": ["square", "round", "rectangular", "thick", "thin"],

    "03636649": ["square", "round", "rectangular", "cuboid", "circular", "thick", "thin"],

    "03691459": ["square", "round", "rectangular", "circular", "thick", "thin"],

    "04090263": ["thick", "thin"],

    "04256520": ["square", "round", "rectangular", "thick", "thin"],

    "04379243": ["square", "round", "circular", "rectangular", "thick", "thin"],

    "04401088": ["square", "rectangular", "thick", "thin"],

    "04530566": ["square", "round", "rectangular", "thick", "thin"],

}

id_to_other_stuff = {

    "02691156": ["boeing", "airbus", "f-16", "plane", "aeroplane", "aircraft", "commerical plane"],

    "02828884": ["park bench"],

    "02933112": ["dresser", "cupboard", "container", "case", "locker", "cupboard", "closet", "sideboard"],

    "02958343": ["auto", "automobile", "motor car"],

    "03001627": ["seat", "cathedra"],

    "03211117": ["TV", "digital display", "flat panel display", "screen", "television", "telly", "video"],

    "03636649": ["lantern", "table lamp", "torch"],

    "03691459": ["speaker", "speaker unit", "tannoy"],

    "04090263": ["ak-47", "uzi", "M1 Garand", "M-16", "firearm", "shooter", "weapon"],

    "04256520": ["couch", "lounge", "divan", "futon"],

    "04379243": ["altar table", "worktop", "workbench"],

    "04401088": ["telephone", "telephone set", "cellular telephone", "cellular phone", "cellphone", "cell",
                 "mobile phone", "iphone"],

    "04530566": ["rowing boat", "watercraft", "ship", "canal boat", "ferry", "steamboat", "barge"],

}


def generate_all_queries(prefix="a"):
    all_queries = []
    all_labels = []

    for category_id in id_to_sub_category:
        sub_category_queries = id_to_sub_category[category_id]
        main_category = sub_category_queries[0]

        new_prefix = prefix
        for shape_attributes_query in id_to_shape_attribute[category_id]:
            if prefix == "a" and shape_attributes_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"

            query = new_prefix + " " + shape_attributes_query + " " + main_category
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])

        for sub_category_query in sub_category_queries:
            if prefix == "a" and sub_category_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"

            query = new_prefix + " " + sub_category_query
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])

        for other_query in id_to_other_stuff[category_id]:
            if prefix == "a" and other_query[0] in ["a", "e", "i", "o", "u"]:
                new_prefix = "an"
            elif prefix == "a":
                new_prefix = "a"

            query = new_prefix + " " + other_query
            all_queries.append(query)
            all_labels.append(id_to_label[category_id])

    return all_queries, all_labels
