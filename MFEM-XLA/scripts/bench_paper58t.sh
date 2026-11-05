#!/bin/bash
# Paper table rerun: native vs xla2p58c (rt/paper58: as paper57, element-chunk barrier form in uneven chunks), paired, as bench_paper48t.sh.
E=/home/wmoses/ejwork; cd $E
tag=paper58t; a=native; b=xla2p58c
busy() { [ -n "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader)" ]; }
gpus=""; for g in 0 1; do busy $g || gpus="$gpus $g"; done
[ -z "$gpus" ] && gpus=0
set -- $gpus
if [ $# -eq 2 ]; then cp scratch/paper_cases_gpu0.txt scratch/${tag}_cases_gpu0.txt; cp scratch/paper_cases_gpu1.txt scratch/${tag}_cases_gpu1.txt
else cat scratch/paper_cases_gpu0.txt scratch/paper_cases_gpu1.txt > scratch/${tag}_cases_gpu$1.txt; fi
echo "PAPER58T-START $(date -Is) gpus:$gpus load $(cut -d' ' -f1 /proc/loadavg)"
for g in $gpus; do
  REACTANT_EXEC_DOT_FORMATION=0 REPS=3 MAX_LOAD=25 bash scripts/mfem3_bench_pair.sh $a $b $g scratch/${tag}_cases_gpu$g.txt ${tag}_g$g > logs/bench_${tag}_g$g.log 2>&1 &
done
wait
for g in $gpus; do
  : > scratch/${tag}_rerun_gpu$g.txt
  for s in $(cat results/mfem3/${tag}_g$g/{$a,$b}/summary.txt | awk -F'|' '$4 ~ /load/ {split($4,x," "); if (x[2]+0 >= 25) print $1}' | sort -u); do
    grep "^$s " scratch/${tag}_cases_gpu$g.txt >> scratch/${tag}_rerun_gpu$g.txt; done
  echo "gpu$g rerun: $(cut -d' ' -f1 scratch/${tag}_rerun_gpu$g.txt | tr '\n' ' ')"
  [ -s scratch/${tag}_rerun_gpu$g.txt ] && REACTANT_EXEC_DOT_FORMATION=0 REPS=3 MAX_LOAD=25 bash scripts/mfem3_bench_pair.sh $a $b $g scratch/${tag}_rerun_gpu$g.txt ${tag}r_g$g > logs/bench_${tag}r_g$g.log 2>&1 &
done
wait
for bl in $a $b; do O=results/mfem3/$tag/$bl; mkdir -p $O; : > $O/summary.txt
  for g in $gpus; do cp results/mfem3/${tag}_g$g/$bl/bench_*.{json,txt} $O/ 2>/dev/null; grep -v MFEM3-BENCH-DONE results/mfem3/${tag}_g$g/$bl/summary.txt >> $O/summary.txt
    if [ -d results/mfem3/${tag}r_g$g ]; then cp results/mfem3/${tag}r_g$g/$bl/bench_*.{json,txt} $O/ 2>/dev/null; grep -v MFEM3-BENCH-DONE results/mfem3/${tag}r_g$g/$bl/summary.txt | sed 's/$/ (rerun)/' >> $O/summary.txt; fi; done; done
cat results/mfem3/$tag/*/summary.txt
python3 scripts/paper_table.py $tag $b
echo "== median compare"; python3 scripts/cmp_median.py results/mfem3/$tag/$a results/mfem3/$tag/$b 12
echo "PAPER58T-BENCH-DONE $(date -Is)"
