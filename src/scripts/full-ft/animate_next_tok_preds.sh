for i in {1..5};
    do 
        echo Example $i;
        python src/animate_next_token_probs.py --path_to_probs results/pythia-1.4b/full-ft/lr_2e-6/early_stopping/val/bsize_128/next_token_probs/ex_$i.json --animation_save_path "results/plots/pythia-1.4b/full-ft/lr_2e-6/early_stopping/val/ex_$i.gif"
    done