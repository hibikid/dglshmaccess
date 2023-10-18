# dglshmaccess

python=3.8  
pytorch=1.13  
cuda=11.6  
dgl=1.1.2+cu116  


python3 {path to dgl tools}/dgl/tools/launch.py \  
--workspace {path to file} \  
--num_trainers 2 \  
--num_samplers 0 \  
--num_servers 1 \  
--part_config {path to data}/ogb-product.json \  
--ip_config ip_config.txt \  
"~/anaconda3/envs/stableDgl/bin/python train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 2 --batch_size 1000 --num_gpus 2"
