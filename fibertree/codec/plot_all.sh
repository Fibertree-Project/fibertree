# plot
rm *.png

python3 plot-cache.py stats/500_5_dsds_unif_500_15/
python3 plot-cache.py stats/500_1_dsds_unif_500_15/ 
python3 plot-cache.py stats/500_all_dsds_unif_500_15/ 100000
python3 plot-cache.py stats/500_100_dsds_unif_500_15/

# without B
# python3 plot-cache.py stats/500_5_dsds_unif_500_15/
# python3 plot-cache.py stats/500_1_dsds_unif_500_15/
# python3 plot-cache.py stats/500_all_dsds_unif_500_15/ 100000
# python3 plot-cache.py stats/500_100_dsds_unif_500_15/

python3 plot-energy.py stats/500_5_dsds_unif_500_15/ 150000
python3 plot-energy.py stats/500_1_dsds_unif_500_15/ 150000
python3 plot-energy.py stats/500_all_dsds_unif_500_15/ 200000
python3 plot-energy.py stats/500_100_dsds_unif_500_15/ 150000

# copy into windows
cp *.png /mnt/c/Users/helenx/plots/
