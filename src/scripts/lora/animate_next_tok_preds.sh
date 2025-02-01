for i in {1..5};
    do 
        echo $i;
        python src/animate_next_token_probs.py --path_to_probs results/pythia-1.4b/lora/r_16/lr_2e-4/early_stopping/val/bsize_128/ex_$i.json --animation_save_path results/plots/pythia-1.4b/lora/lr_2e-4/early_stopping/val/ex_$i.gif
    done

