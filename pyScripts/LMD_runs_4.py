#cd /Users/Tlacael/NYU/RhythmData
import lmd as L
import pickle
## params for first run

ext = "4"
dset = L.RhythmDataset(valid=8,test=9)
#1##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(20,))]


clf1 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf1.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf1,dset,'valid'))
filename = "clf1." + ext
pickle.dump(clf1,file(filename,'w'))
#2##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]

clf2 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf2.fit(dset,0.1,batch_size=10,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf2,dset,'valid'))
filename = "clf2" + ext
pickle.dump(clf2,file(filename,'w'))
#3##############################################################

#scores##############################################################
'''
clf1=pickle.load(open('clfPickles/clf1.3','rb'))
clf2=pickle.load(open('clfPickles/clf2.3','rb'))
clf3=pickle.load(open('clfPickles/clf3.3','rb'))
clf4=pickle.load(open('clfPickles/clf4.3','rb'))
clf5=pickle.load(open('clfPickles/clf5.3','rb'))
#clf5=pickle.load(open('clfPickles/clf6.3','rb'))
'''
#scores##############################################################
scores = []

scores.append(["clf1",L.score(L.pred_dset_mean(clf1,dset,'valid'))]);
scores.append(["clf2",L.score(L.pred_dset_mean(clf2,dset,'valid'))]);
#scores.append(["clf3",L.score(L.pred_dset_mean(clf3,dset,'valid'))]);
#scores.append(["clf4",L.score(L.pred_dset_mean(clf4,dset,'valid'))]);
#scores.append(["clf5",L.score(L.pred_dset_mean(clf5,dset,'valid'))]);
#scores.append(["clf6",L.score(L.pred_dset_mean(clf6,dset,'valid'))]);
#scores.append(["clf7",L.score(L.pred_dset_mean(clf7,dset,'valid'))]);
#scores.append(["clf8",L.score(L.pred_dset_mean(clf8,dset,'valid'))]);

scores = dict(scores)

f = open( 'scores_4.txt', 'w' )
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
