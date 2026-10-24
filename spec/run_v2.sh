set -x
python3 sweep.py 32,64,128,256,512 20 > results_table.txt
CF=---,S--,SA-,SAZ WS=0,0.5,1 TAG=contest python3 contest.py 32,64,128 20 > contest_table.txt
INIT_G=3 GSIG=0.6 CF=SA-,SAZ WS=0,0.5,1 TAG=stress python3 contest.py 64 20 > stress_table.txt
FCOST=2 CF=---,S--,SAZ WS=0,0.5,1 TAG=fcost python3 contest.py 64,128 20 > fcost_table.txt
CF=---,S--,SA-,SAZ WS=0.5 TAG=contest256 python3 contest.py 256 20 > contest256_table.txt
CF=---,SAZ WS=0.5 TAG=contest512 python3 contest.py 512 20 > contest512_table.txt
