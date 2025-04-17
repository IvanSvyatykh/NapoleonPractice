


def gen_rec_label(input_path, out_label):
    with open(out_label, "w") as out_file:
        with open(input_path, "r") as f:
            for line in f.readlines():
                tmp = line.strip("\n").replace(" ", "").split(",")
                img_path, label = tmp[0], tmp[1]
                label = label.replace('"', "")
                out_file.write(img_path + "\t" + label + "\n")
