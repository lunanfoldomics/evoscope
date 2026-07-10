for SEED in $(seq 25 124); do
  evoscope \
    --width 60 \
    --height 40 \
    --seed "$SEED" \
    --epochs 120 \
    --initial_cells 30 \
    --nutrient 6.9 \
    --outdir "runs/figure4_seed_${SEED}"
done
