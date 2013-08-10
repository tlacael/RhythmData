#cd /Users/Tlacael/NYU/RhythmData
import lmd as L
## params for first run

#1##############################################################
arch = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)
##################################################
arch = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]

score_save.gather_scores(cur_arch=arch)
##################################################
arch = [L.AffineArgs(weight_shape=(3600,7200)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]

score_save.gather_scores(cur_arch=arch)
##################################################
arch = [L.AffineArgs(weight_shape=(3600,5400)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]


score_save.gather_scores(cur_arch=arch)
##################################################
arch = [L.AffineArgs(weight_shape=(3600,7200)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]

score_save.gather_scores(cur_arch=arch)
##################################################
arch = [L.AffineArgs(weight_shape=(3600,5400)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]

score_save.gather_scores(cur_arch=arch)

