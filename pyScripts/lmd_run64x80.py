#cd /Users/Tlacael/NYU/RhythmData
import lmd2 as L
import score_save
## params for first run

x=64
y=80
#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,800)),
            L.AffineArgs(output_shape=(800,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,800)),
            L.AffineArgs(output_shape=(400,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,800)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,800)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

