#!/bin/bash


max_deg_ls=(6 8 16 32)
dim_inner_ls=(256)
num_layer_ls=(4)
dropout_ls=(0.0)
layers_post_mp_ls=(3)

base_lr_ls=(0.0015)
weight_decay_ls=(0.0)
bundle_dim_ls=(8)
num_bundle_ls=(8)

tau_ls=(10.)
tau_method_ls=(fixed)  # not implemented yet

seed=$(date +%s | awk '{print substr($1,length($1)-9)}')

for max_deg in "${max_deg_ls[@]}"; do
    job_file="all_kids/job_func_$seed.job"
    echo "#!/bin/bash" > "$job_file"
    echo "#SBATCH -N 1" >> "$job_file"
    echo "#SBATCH --job-name=search-bun" >> "$job_file"
    echo "#SBATCH --output=${search-bun}.out" >> "$job_file"
    echo "#SBATCH --error=${search-bun}.err" >> "$job_file"
    echo "#SBATCH --mem=36000" >> "$job_file"
    echo "#SBATCH --qos=short" >> "$job_file"
    echo "#SBATCH --gres=gpu:1" >> "$job_file"
    echo "" >> "$job_file"
    for dim_inner in "${dim_inner_ls[@]}"; do
        for num_layer in "${num_layer_ls[@]}"; do
            for dropout in "${dropout_ls[@]}"; do
                for layers_post_mp in "${layers_post_mp_ls[@]}"; do
                    for base_lr in "${base_lr_ls[@]}"; do
                        for bundle_dim in "${bundle_dim_ls[@]}"; do
                            for tau in "${tau_ls[@]}"; do
                                for weight_decay in "${weight_decay_ls[@]}"; do
                                    for tau_method in "${tau_method_ls[@]}"; do
                                        for num_bundle in "${num_bundle_ls[@]}"; do
                                            ((seed++))
                                            echo "srun -u /slurm-storage/jacbam/.conda/envs/graphgps/bin/python main.py --cfg configs/bundle/peptides-func-bunn.yaml wandb.use True seed $seed bundle.max_deg $max_deg gnn.dim_inner $dim_inner gnn.layers_mp $num_layer gnn.dropout $dropout gnn.layers_post_mp $layers_post_mp optim.base_lr $base_lr optim.weight_decay $weight_decay bundle.bundle_dim $bundle_dim bundle.tau $tau bundle.tau_method $tau_method bundle.num_bundle $num_bundle" >> "$job_file"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
