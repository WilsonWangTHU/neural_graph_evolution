#!/bin/bash


# A command to remove multiple files that are smaller of some size
# under the directory
# find . -maxdepth 1 -type d | grep -v ^\\.$ | xargs -n 1 du -s | 
# while read size name; do if [ $size -lt 4096 ]; then rm -rf $name; fi done


# first sync with the directories in guppies
i = 0

while true; do
    echo '@iteration: $i'

    # sync with remote guppy servers
    # rsync -r henryzhou@cluster7:/ais/gobi6/henryzhou/topology_rl/evolution_data/* \
    #          ../evolution_data/ \
    #          --delete --exclude='*species_video*' 
    # sync with tingwu's guppy server
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_bayesianop/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_pruning/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_finetune_baseline/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0513_finetune/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0513_walker/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_fishes/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0512_walker/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_finetune/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0511_walker/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_ablation/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_circle/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    # rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0510_walker/topology_rl/evolution_data/*\
    #            ../evolution_data \
    #            --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0510_noon/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0509_night/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0509/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 
    rsync -avz henryzhou@cluster12:/ais/gobi6/tingwuwang/topology_rl_0508/topology_rl/evolution_data/*\
               ../evolution_data \
               --delete --exclude='*species_video*' 

    # start to burst video
    for trn_sess in ../evolution_data/*; do
	echo 'Starting on $trn_sess'
        python visualize_species.py -i ${trn_sess}/species_data -v 0

    done
    sleep 300
done

    
