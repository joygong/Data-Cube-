

#zs=$(for i in $(seq 20); do bc <<< "scale=3; ($RANDOM*0.7/32767)+0.3"; done)
zs=$(for i in $(seq 1); do bc <<< "scale=3; ($RANDOM*1.7/32767)+0.3"; done)
for z in $zs; do sbatch run_z.sl $z; done 
