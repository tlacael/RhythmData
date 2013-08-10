#cd /Users/Tlacael/NYU/RhythmData
import lmd as L
import pickle
## params for first run

dset = L.RhythmDataset(valid=8,test=9)
'''
#1##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(20,))]


clf1 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf1.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf1,dset,'valid'))
pickle.dump(clf1,file('clfPickles/clf1.3','w'))
#2##############################################################

clf2 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf2.fit(dset,0.1,batch_size=5,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf2,dset,'valid'))
pickle.dump(clf2,file('clfPickles/clf2.3','w'))
#3##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]



clf3 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf3.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf3,dset,'valid'))

pickle.dump(clf3,file('clfPickles/clf3.3','w'))
#4######################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,1600)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(20,))]


########################
clf4 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf4.fit(dset,0.1,batch_size=50,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf4,file('clfPickles/clf4.3','w'))
L.score(L.pred_dset_mean(clf4,dset,'valid'))
#5##############################################################

clf5 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf5.fit(dset,0.1,batch_size=70,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf5,file('clfPickles/clf5.3','w'))

L.score(L.pred_dset_mean(clf5,dset,'valid'))
#6##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,1600)),
            L.AffineArgs(output_shape=(800,)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.AffineArgs(output_shape=(100,)),
            L.SoftmaxArgs(output_shape=(20,))]

clf6 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf6.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf6,file('clfPickles/clf6.3','w'))
L.score(L.pred_dset_mean(clf6,dset,'valid'))
#7##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.AffineArgs(output_shape=(100,)),
            L.SoftmaxArgs(output_shape=(10,))]


clf7 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf7.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf7,file('clfPickles/clf7.3','w'))
L.score(L.pred_dset_mean(clf7,dset,'valid'))
#8##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,1600)),
            L.AffineArgs(output_shape=(400,)),
            L.SoftmaxArgs(output_shape=(10,))]


clf8 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf8.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf8,dset,'valid'))
pickle.dump(clf8,file('clfPickles/clf8.3','w'))

#scores##############################################################
'''
clf1=pickle.load(open('clfPickles/clf1.3','rb'))
clf2=pickle.load(open('clfPickles/clf2.3','rb'))
clf3=pickle.load(open('clfPickles/clf3.3','rb'))
clf4=pickle.load(open('clfPickles/clf4.3','rb'))
clf5=pickle.load(open('clfPickles/clf5.3','rb'))
#clf5=pickle.load(open('clfPickles/clf6.3','rb'))

#scores##############################################################
scores = []

scores.append(["clf1",L.score(L.pred_dset_mean(clf1,dset,'valid'))]);
scores.append(["clf2",L.score(L.pred_dset_mean(clf2,dset,'valid'))]);
scores.append(["clf3",L.score(L.pred_dset_mean(clf3,dset,'valid'))]);
scores.append(["clf4",L.score(L.pred_dset_mean(clf4,dset,'valid'))]);
scores.append(["clf5",L.score(L.pred_dset_mean(clf5,dset,'valid'))]);
#scores.append(["clf6",L.score(L.pred_dset_mean(clf6,dset,'valid'))]);
#scores.append(["clf7",L.score(L.pred_dset_mean(clf7,dset,'valid'))]);
#scores.append(["clf8",L.score(L.pred_dset_mean(clf8,dset,'valid'))]);

scores = dict(scores)

f = open( 'scores_3.txt', 'w' )
f.write( 'scores = ' + repr(scores) + '\n' )
f.close()
'''
scores = {
        'clf8': 84.012539184952985, 
        'clf7': 82.445141065830711, 
        'clf6': 79.937304075235105, 
        'clf5': 82.445141065830711, 
        'clf4': 84.32601880877742, 
        'clf3': 81.818181818181827, 
        'clf2': 83.385579937304072, 
        'clf1': 83.072100313479623}
'''
