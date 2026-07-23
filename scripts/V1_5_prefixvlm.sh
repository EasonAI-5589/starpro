#常用的命令：
cd LLaVA-STAR-Pro2
source /mnt/eason/fa_xf/bin/activate
export http_proxy=http://192.168.32.28:18000  
export https_proxy=http://192.168.32.28:18000  

bash scripts/v1_5/7b/mme.sh prefixvlm 128
bash scripts/v1_5/7b/mmbench.sh prefixvlm 128
bash scripts/v1_5/7b/mmbench_cn.sh prefixvlm 128
bash scripts/v1_5/7b/mmvet.sh prefixvlm 128
bash scripts/v1_5/7b/textvqa.sh prefixvlm 128
bash scripts/v1_5/7b/sqa.sh prefixvlm 128
bash scripts/v1_5/7b/gqa.sh prefixvlm 128
bash scripts/v1_5/7b/pope.sh prefixvlm 128

bash scripts/v1_5/7b/mme.sh prefixvlm 64
bash scripts/v1_5/7b/mmbench.sh prefixvlm 64
bash scripts/v1_5/7b/mmbench_cn.sh prefixvlm 64
bash scripts/v1_5/7b/mmvet.sh prefixvlm 64
bash scripts/v1_5/7b/textvqa.sh prefixvlm 64
bash scripts/v1_5/7b/sqa.sh prefixvlm 64
bash scripts/v1_5/7b/gqa.sh prefixvlm 64
bash scripts/v1_5/7b/pope.sh prefixvlm 64

bash scripts/v1_5/7b/mme.sh prefixvlm 32
bash scripts/v1_5/7b/mmbench.sh prefixvlm 32
bash scripts/v1_5/7b/mmbench_cn.sh prefixvlm 32
bash scripts/v1_5/7b/mmvet.sh prefixvlm 32
bash scripts/v1_5/7b/textvqa.sh prefixvlm 32
bash scripts/v1_5/7b/sqa.sh prefixvlm 32
bash scripts/v1_5/7b/gqa.sh prefixvlm 32
bash scripts/v1_5/7b/pope.sh prefixvlm 32

mme:2374
mmbench:4377
mmbench_cn:4329
mmvet:218
textvqa:5000
sqa:2017
gqa:12578
pope:9000

