# final running

import sys, os

def get_conf(s):
    data_dir = s["data"]
    embed_file = "../../data/glove/glove.6B.50d.txt" if s["embed"] else ""
    loss_mode = s["loss_mode"]
    beam_div = s["beam_div"]
    recomb_ss = {"nope": "recomb_mode:0\nrecomb_divL:0\nrecomb_divU:0",
                 "merge": "recomb_mode:4\nrecomb_divL:0\nrecomb_divU:0",
                 "unlabel": "recomb_mode:4\nrecomb_divL:0\nrecomb_divU:4",
                 "top": "recomb_mode:4\nrecomb_divL:0\nrecomb_divU:6",
        }[s["recomb_mode"]]

    ss = """file_train:%s/train.auto
file_dev:%s/dev.auto
file_test:%s/test.auto
embed_file:%s
embed_scale:1.0
mss:o-update_mode-0
fss:zc0
tr_iters:30
tr_lrate_lbound:0.001
mss:h0-d0.0|h1-d0.0
tr_nocut_iters:10
tr_cut_iters:4
tr_lrate:0.05
mss:o-blstm_layer-1
mss:o-blstm_fsize-0
mss:o-blstm_size-160
mss:o-blstm_drop-0.2
margin:2
loss_mode:%s
beam_div:%s
%s
beam_all:12""" % (data_dir,data_dir,data_dir,embed_file,loss_mode,beam_div,recomb_ss)
    return ss
    
def get_confs(data_dir):
    to_init_embed = "ptb" in data_dir
    tasks = [
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":4, "recomb_mode":"nope"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":4, "recomb_mode":"merge"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":4, "recomb_mode":"unlabel"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":4, "recomb_mode":"top"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":1, "recomb_mode":"unlabel"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":1, "recomb_mode":"top"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":0, "beam_div":4, "recomb_mode":"unlabel"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":1, "beam_div":4, "recomb_mode":"unlabel"},
        {"data":data_dir, "embed":to_init_embed, "loss_mode":2, "beam_div":4, "recomb_mode":"unlabel"},
    ]
    confs = [get_conf(s) for s in tasks]
    return confs

def main():
    data_dir = sys.argv[1]
    print("#Creating tasks for %s." % data_dir)
    for cur_id, conf in enumerate(get_confs(data_dir)):
        dir_name = "task%s" % cur_id
        os.system("mkdir %s" % dir_name)
        with open("%s/_conf" % dir_name, 'w') as fd:
            fd.write(conf)
        print("cd %s; bash ../../run.sh _conf >z.log 2>&1 & cd .." % dir_name)

if __name__ == "__main__":
    main()

data_dirs = [
    "../../data/ptb", 
    "../../data/ctb",
    "../../data/ud/ud2/UD_Arabic/",
    "../../data/ud/ud2/UD_Catalan/",
    "../../data/ud/ud2/UD_Czech/",
    "../../data/ud/ud2/UD_Finnish/",
    "../../data/ud/ud2/UD_French-FTB/"
    "../../data/ud/ud2/UD_German/",
    "../../data/ud/ud2/UD_Italian/",
    "../../data/ud/ud2/UD_Latin-ITTB/",
    "../../data/ud/ud2/UD_Russian-SynTagRus/",
    "../../data/ud/ud2/UD_Spanish/",
]
# python3 ../ef2.py ../data/ptb ../data/ctb ../data/ud/ud2/UD_English/

# script for cross-running, on .. dir
#BASE_DIR=`pwd`
#for i in 0 1; do
#    for j in 0 1; do
#        cd $BASE_DIR;
#        mkdir $BASE_DIR/devtask$i$j;
#        cd $BASE_DIR/devtask$i$j;
#        echo -e "\niftrain:0" | cat ../task$i/_conf - | sed "s/test.auto/dev.auto/g" > _conf;
#        ln -s ../task$j/model.mach .;
#        ln -s ../task$j/model.mach.spec .;
#        ln -s ../task$j/dictionary.txt .;
#        bash ../../run.sh _conf >z.log 2>&1 &
#    done
#done
