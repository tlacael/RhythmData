#cd /Users/Tlacael/NYU/RhythmData
import lmd as L
import pickle
## params for first run

dset = L.RhythmDataset(valid=8,test=9)

#1##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,800)),
            L.AffineArgs(output_shape=(400,)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(20,))]


clf1 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf1.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf1,dset,'valid'))
pickle.dump(clf1,file('clfPickles/clf1.2','w'))
#2##############################################################

clf2 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf2.fit(dset,0.1,batch_size=30,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf2,dset,'valid'))
pickle.dump(clf1,file('clfPickles/clf1.2','w'))
pickle.dump(clf2,file('clfPickles/clf2.2','w'))
#3##############################################################
clf3 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf3.fit(dset,0.1,batch_size=40,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf3,dset,'valid'))

pickle.dump(clf3,file('clfPickles/clf3.2','w'))
#4##############################################################
clf4 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf4.fit(dset,0.1,batch_size=50,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf4,file('clfPickles/clf4.2','w'))
L.score(L.pred_dset_mean(clf4,dset,'valid'))
#5##############################################################

clf5 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf5.fit(dset,0.1,batch_size=70,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf5,file('clfPickles/clf5.2','w'))

L.score(L.pred_dset_mean(clf5,dset,'valid'))
#6##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,512)),
            L.AffineArgs(output_shape=(375,)),
            L.SoftmaxArgs(output_shape=(10,))]


clf6 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf6.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf6,file('clfPickles/clf6.clf','w'))
L.score(L.pred_dset_mean(clf6,dset,'valid'))
#7##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,512)),
            L.AffineArgs(output_shape=(375,)),
            L.SoftmaxArgs(output_shape=(20,))]


clf7 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf7.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)
pickle.dump(clf7,file('clfPickles/clf7.clf','w'))
L.score(L.pred_dset_mean(clf7,dset,'valid'))
#8##############################################################
L.arch_lmd = [L.AffineArgs(weight_shape=(3600,512)),
            L.AffineArgs(output_shape=(200,)),
            L.SoftmaxArgs(output_shape=(10,))]


clf8 = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
clf8.fit(dset,0.1,batch_size=20,print_freq=1000,EVAL_ALL=False, n_iter=3000)

L.score(L.pred_dset_mean(clf8,dset,'valid'))
pickle.dump(clf8,file('clfPickles/clf8.clf','w'))

#scores##############################################################
'''
clf1=pickle.load(open('clfPickles/clf1.clf','rb'))
clf2=pickle.load(open('clfPickles/clf2.clf','rb'))
clf3=pickle.load(open('clfPickles/clf3.clf','rb'))
clf4=pickle.load(open('clfPickles/clf4.clf','rb'))
clf5=pickle.load(open('clfPickles/clf5.clf','rb'))
'''
#scores##############################################################
scores = []

scores.append(["clf1",L.score(L.pred_dset_mean(clf1,dset,'valid'))]);
scores.append(["clf2",L.score(L.pred_dset_mean(clf2,dset,'valid'))]);
scores.append(["clf3",L.score(L.pred_dset_mean(clf3,dset,'valid'))]);
scores.append(["clf4",L.score(L.pred_dset_mean(clf4,dset,'valid'))]);
scores.append(["clf5",L.score(L.pred_dset_mean(clf5,dset,'valid'))]);
scores.append(["clf6",L.score(L.pred_dset_mean(clf6,dset,'valid'))]);
scores.append(["clf7",L.score(L.pred_dset_mean(clf7,dset,'valid'))]);
scores.append(["clf8",L.score(L.pred_dset_mean(clf8,dset,'valid'))]);

scores = dict(scores)

f = open( 'scores.txt', 'w' )
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
