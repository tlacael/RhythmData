#cd /Users/Tlacael/NYU/RhythmData

import lmd2 as L
import score_save
## params for first run

x=32
y=60
#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,3840)),
            L.AffineArgs(output_shape=(1920,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,3840)),
            L.AffineArgs(output_shape=(1920,)),
            L.AffineArgs(output_shape=(480,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,1920)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

#1##############################################################
arch = [L.AffineArgs(weight_shape=(x*y,1920)),
            L.AffineArgs(output_shape=(480,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)
########################
arch = [L.AffineArgs(weight_shape=(x*y,960)),
            L.AffineArgs(output_shape=(480,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

########################
arch = [L.AffineArgs(weight_shape=(x*y,256)),
            L.AffineArgs(output_shape=(64,)),
            L.SoftmaxArgs(output_shape=(10,))]
score_save.gather_scores(cur_arch=arch)

arch = [L.AffineArgs(weight_shape=(1600,256)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]

