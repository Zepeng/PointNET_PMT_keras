for enc_r_bits in 8 12
do
    for enc_l_bits in 0 8 12
    do
        for o_bits in 0 8
        do
            python train_keras.py --epochs 50 \
            --save_ver "enc_r$enc_r_bits enc_l$enc_l_bits dec_r$enc_r_bits dec_l$enc_l_bits o_bits$o_bits" \
            --enc_a $enc_r_bits --enc_b $enc_l_bits --dec_a $enc_r_bits --dec_b $enc_l_bits --o_int_bits $o_bits
        done
    done
done

# python train_qkeras.py --epochs 50 --enc_dropout 0 --dec_dropout 0.1 \
        # --weight_decay 1e-3 --lr 1e-3 --save_ver "enc_r$enc_r_bits enc_l$enc_l_bits dec_r$dec_r_bits dec_l$dec_l_bits" --dim_reduce_factor 3 \
        # --batch_size 64 --mean_only --save_best --patience 15 --xyz_energy \
        # --enc_a $enc_r_bits --enc_b $enc_l_bits --dec_a $dec_r_bits --dec_b $dec_l_bits
        # python train_qkeras.py --epochs 30 --enc_dropout 0 --dec_dropout 0.1 \
        # --weight_decay 1e-3 --lr 1e-3 --save_ver "enc_r$enc_r_bits enc_l$enc_l_bits dec_r$enc_r_bits dec_l$enc_l_bits" --dim_reduce_factor 3 \
        # --batch_size 16 --mean_only --save_best --patience 15 --xyz_energy \
        # --enc_a $enc_r_bits --enc_b $enc_l_bits --dec_a $enc_r_bits --dec_b $enc_l_bits
        