

dset = L.RhythmDataset(valid=8,test=9)

#1##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,512)),
            L.AffineArgs(output_shape=(64,)),
            L.SoftmaxArgs(output_shape=(10,))]

clf1 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
'clf1': 83.072100313479623}


#2##############################################################


clf2.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)
'clf2': 83.385579937304072, 


#3##############################################################

clf3.fit(dset,0.1,batch_size=100,print_freq=1000,EVAL_ALL=False, n_iter=3000)

'clf3': 81.818181818181827, 

#4##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(20,))]


clf4.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)
'clf4': 84.32601880877742, 

#5##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]



clf5.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)
'clf5': 82.445141065830711, 

#6##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.AffineArgs(output_shape=(20,)),
            L.SoftmaxArgs(output_shape=(10,))]



clf6.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)
'clf6': 79.937304075235105, 

#7##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,400)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]



clf7.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

'clf7': 82.445141065830711, 

#8##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,400)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]



clf8.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

clf8': 84.012539184952985, 


##############################################################
##############################################################


