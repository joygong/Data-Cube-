
zs=$(for i in $(seq 1); do bc <<< "scale=3; ($RANDOM*1.7/32767)+0.3"; done)
for z in $zs; do ./datacube.sh $z; done

# echo done simulating! 
